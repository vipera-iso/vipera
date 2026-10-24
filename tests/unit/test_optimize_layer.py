"""The optimization layer as the HTTP surface sees it."""

import types

from fastapi.testclient import TestClient

from vipera.optimize.layer import OptimizeLayer
from vipera.server import build_app

SYSTEM = {"role": "system", "content": "you are a careful assistant."}


class FakeEngine:
    def __init__(self, *, context=512, cached=0):
        self.calls = []
        self.plan = types.SimpleNamespace(context=context)
        self.cached = cached

    def chat(self, messages, stream=False, **kw):
        self.calls.append({"messages": messages, "stream": stream, **kw})
        usage = {"prompt_tokens": 100, "completion_tokens": 5}
        if self.cached:
            usage["prompt_tokens_details"] = {"cached_tokens": self.cached}
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": usage,
        }

    def tokenize(self, text):
        return [0] * (len(str(text)) // 3)


def _history(turns: int, chars: int = 400) -> list[dict]:
    messages = [SYSTEM]
    for i in range(turns):
        messages.append({"role": "user", "content": f"q{i} " + "x" * chars})
        messages.append({"role": "assistant", "content": f"a{i} " + "y" * chars})
    messages.append({"role": "user", "content": "final question"})
    return messages


def _client(engine):
    return TestClient(build_app(engine, "test-model"))


def test_a_request_that_fits_reaches_the_engine_untouched():
    engine = FakeEngine()
    client = _client(engine)
    messages = [SYSTEM, {"role": "user", "content": "hi"}]

    client.post("/v1/chat/completions", json={"messages": messages})

    assert engine.calls[-1]["messages"] == messages


def test_an_overlong_history_is_trimmed_before_the_engine_sees_it():
    engine = FakeEngine()
    client = _client(engine)
    messages = _history(20)

    r = client.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 64})

    assert r.status_code == 200
    seen = engine.calls[-1]["messages"]
    assert len(seen) < len(messages)
    assert seen[0] == SYSTEM
    assert seen[-1]["content"] == "final question"
    stats = client.get("/health").json()["optimize"]
    assert stats["messages_dropped"] > 0


def test_health_reports_prefix_reuse_and_cache_hits():
    engine = FakeEngine(cached=80)
    client = _client(engine)
    first = [SYSTEM, {"role": "user", "content": "first"}]
    second = [SYSTEM, {"role": "user", "content": "second"}]

    client.post("/v1/chat/completions", json={"messages": first})
    client.post("/v1/chat/completions", json={"messages": second})

    stats = client.get("/health").json()["optimize"]
    assert stats["requests"] == 2
    # Same system prompt and no history: the reusable prefix is identical.
    assert stats["prefix_hits"] == 1
    assert stats["cached_tokens"] == 160
    assert stats["cache_rate"] > 0


def test_the_tool_schema_reaches_the_shaper():
    # The layer used to accept `tools` and drop it, so a large schema pushed the
    # rendered prompt past the context without anything being trimmed.
    engine = FakeEngine(context=1024)
    client = _client(engine)
    messages = _history(3, chars=400)
    tools = [
        {
            "type": "function",
            "function": {"name": "read", "description": "d" * 1200, "parameters": {}},
        }
    ]

    client.post(
        "/v1/chat/completions",
        json={"messages": messages, "max_tokens": 128, "tools": tools},
    )

    assert engine.calls[-1]["tools"] == tools
    assert len(engine.calls[-1]["messages"]) < len(messages)


def test_the_ollama_layer_shapes_and_counts_too():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model", protocols=["ollama"]))

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": _history(20),
            "stream": False,
            "options": {"num_predict": 64},
        },
    )

    assert r.status_code == 200
    assert len(engine.calls[-1]["messages"]) < len(_history(20))
    assert client.get("/health").json()["optimize"]["requests"] == 1


def test_shaping_can_be_switched_off_in_the_config(monkeypatch):
    import vipera.config as config_mod

    original = config_mod.load_config
    monkeypatch.setattr(config_mod, "load_config", lambda: {**original(), "shaping": False})

    engine = FakeEngine()
    client = _client(engine)
    messages = _history(20)

    client.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 64})

    assert engine.calls[-1]["messages"] == messages
    assert client.get("/health").json()["optimize"]["messages_dropped"] == 0


def test_a_gateway_without_the_layer_still_serves(monkeypatch):
    import vipera.server as server_mod

    monkeypatch.setattr(
        server_mod, "OptimizeLayer", types.SimpleNamespace(from_config=lambda: None)
    )

    engine = FakeEngine()
    client = _client(engine)
    messages = _history(20)

    r = client.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 64})

    assert r.status_code == 200
    assert engine.calls[-1]["messages"] == messages
    # No layer, no counters to report.
    assert "optimize" not in client.get("/health").json()


class _TokenizerEngine:
    """An engine whose tokenizer disagrees with the 3-chars/token estimate."""

    def __init__(self, context=4096, chars_per_token=1.8):
        self.plan = types.SimpleNamespace(context=context)
        self.chars_per_token = chars_per_token
        self.calls = 0

    def tokenize(self, text):
        self.calls += 1
        return [0] * int(len(text) / self.chars_per_token)


def test_shaping_defers_to_the_tokenizer_when_the_estimate_is_near_the_limit():
    # A log dump tokenizes at ~1.8 chars/token; the 3-chars estimate would leave
    # the prompt 1.7x over the window and llama-server would answer 400.
    engine = _TokenizerEngine()
    blob = "abcdefgh " * 2600  # ~23.4k chars, estimates to ~7.8k tokens
    messages = [{"role": "user", "content": blob}]

    shaping = OptimizeLayer(shaping=True).prepare(engine, messages, max_tokens=32)

    assert engine.calls >= 1
    assert shaping.truncated == 1
    real = len(engine.tokenize("\n".join(m["content"] for m in shaping.messages)))
    assert real <= 4096 - 32


def test_shaping_does_not_tokenize_a_request_that_fits_comfortably():
    engine = _TokenizerEngine()
    messages = [{"role": "user", "content": "short question"}]

    OptimizeLayer(shaping=True).prepare(engine, messages, max_tokens=32)

    assert engine.calls == 0
