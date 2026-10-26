import asyncio
import json
import logging
import time
import types
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from vipera.pool import ModelPool
from vipera.protocols import mount
from vipera.protocols.ollama import OllamaChatRequest, OllamaMessage, _ollama_tool_calls
from vipera.server import build_app


class FakeEngine:
    """OpenAI-shaped engine, optionally able to embed."""

    def __init__(self, *, embeddings=False):
        self.calls = []
        self.supports_embeddings = embeddings

    def chat(self, messages, stream=False, **kw):
        self.calls.append({"messages": messages, "stream": stream, **kw})
        if stream:

            def _gen():
                yield {"choices": [{"delta": {"content": "hi"}}]}
                yield {"choices": [{"delta": {"content": " there"}}]}

            return _gen()
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    def tokenize(self, text):
        return [0] * len(str(text).split())

    def embed(self, texts):
        self.calls.append({"embed": list(texts)})
        return [[float(len(t)), 0.5] for t in texts]


def _client(engine, *, protocols=None, model="test-model"):
    return TestClient(build_app(engine, model, protocols=protocols))


class PoolEngine(FakeEngine):
    """A fake engine the pool can load, close, and count."""

    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


def _pool_client(*, keep_alive=30):
    engines = []

    def factory(alias, path):
        engine = PoolEngine()
        engines.append(engine)
        return engine

    pool = ModelPool(
        {"test-model": "/tmp/test-model.gguf"},
        factory,
        max_loaded=1,
        keep_alive=keep_alive,
        reap_interval=60,
    )
    client = TestClient(build_app(pool=pool, protocols=["ollama"]))
    return client, pool, engines


def test_ollama_rejects_an_oversized_prompt_without_loading_the_model():
    """The prompt limit is checked before a load, not after it."""

    client, pool, engines = _pool_client()
    try:
        r = client.post(
            "/api/chat",
            json={
                "model": "test-model",
                "stream": False,
                "messages": [{"role": "user", "content": "x" * 500_001}],
            },
        )
        assert r.status_code == 400
        assert engines == []
    finally:
        pool.close()


# -- OpenAI embeddings -------------------------------------------------------


def test_openai_embeddings_returns_one_vector_per_input():
    client = _client(FakeEngine(embeddings=True))

    r = client.post("/v1/embeddings", json={"input": ["a b", "c"]})

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert body["data"][0]["embedding"] == [3.0, 0.5]
    assert body["model"] == "test-model"
    assert body["usage"]["prompt_tokens"] == 3


def test_openai_embeddings_accepts_a_single_string():
    client = _client(FakeEngine(embeddings=True))

    r = client.post("/v1/embeddings", json={"input": "hello"})

    assert r.status_code == 200
    assert len(r.json()["data"]) == 1


def test_openai_embeddings_explains_how_to_enable_them():
    engine = FakeEngine()  # not loaded with embedding support
    client = _client(engine)

    r = client.post("/v1/embeddings", json={"input": "hello"})

    assert r.status_code == 501
    assert "embeddings true" in r.json()["detail"]
    assert engine.calls == [], "the engine must not be asked to embed"


def test_openai_embeddings_rejects_base64():
    client = _client(FakeEngine(embeddings=True))

    r = client.post("/v1/embeddings", json={"input": "x", "encoding_format": "base64"})

    assert r.status_code == 400


def test_openai_embeddings_rejects_an_empty_input_list():
    client = _client(FakeEngine(embeddings=True))

    assert client.post("/v1/embeddings", json={"input": []}).status_code == 400


def test_openai_embeddings_is_not_mounted_on_the_wrong_model():
    client = _client(FakeEngine(embeddings=True))

    r = client.post("/v1/embeddings", json={"model": "other", "input": "x"})

    assert r.status_code == 404


# -- which layers are mounted ------------------------------------------------


def test_ollama_routes_are_not_mounted_by_default():
    client = _client(FakeEngine())

    assert client.get("/api/tags").status_code == 404


def test_both_layers_can_be_mounted_together():
    app = build_app(FakeEngine(embeddings=True), "test-model", protocols=["openai", "ollama"])

    assert app.state.protocols == ("openai", "ollama")
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 200
    assert client.get("/api/tags").status_code == 200


def test_duplicate_layers_are_mounted_once():
    app = build_app(FakeEngine(), "test-model", protocols=["openai", "openai", " "])

    assert app.state.protocols == ("openai",)


def test_unknown_protocol_names_the_available_ones():
    with pytest.raises(ValueError) as exc:
        mount(types.SimpleNamespace(), object(), ["anthropic"])

    assert "anthropic" in str(exc.value)
    assert "openai" in str(exc.value)


def test_plugin_protocol_is_loaded_from_entry_points(monkeypatch):
    import importlib.metadata

    class Plugin:
        @staticmethod
        def mount(app, gateway):
            @app.get("/custom")
            def custom():
                return {"default": gateway.default_model()}

    class Entry:
        name = "custom"

        @staticmethod
        def load():
            return Plugin

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: types.SimpleNamespace(select=lambda group: [Entry()]),
    )

    app = build_app(FakeEngine(), "test-model", protocols=["custom"])

    assert TestClient(app).get("/custom").json() == {"default": "test-model"}


# -- Ollama API --------------------------------------------------------------


def test_ollama_version_and_tags():
    client = _client(FakeEngine(), protocols=["ollama"])

    assert client.get("/api/version").json()["version"].startswith("0.5.")

    tags = client.get("/api/tags").json()["models"]
    assert [m["name"] for m in tags] == ["test-model"]
    assert tags[0]["details"]["format"] == "gguf"


def test_ollama_chat_non_stream():
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["message"]["content"] == "hi there"
    assert body["done"] is True
    assert body["done_reason"] == "stop"
    assert body["eval_count"] == 2
    assert engine.calls[-1]["stream"] is False


def test_ollama_options_are_translated():
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 7, "stop": ["END"]},
        },
    )

    call = engine.calls[-1]
    assert call["temperature"] == pytest.approx(0.1)
    assert call["max_tokens"] == 7
    assert call["stop"] == ["END"]


def test_ollama_chat_stream_is_ndjson_with_a_final_done_frame():
    client = _client(FakeEngine(), protocols=["ollama"])

    with client.stream(
        "POST",
        "/api/chat",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()

    frames = [json.loads(line) for line in body.splitlines() if line.strip()]
    assert [f["message"]["content"] for f in frames if not f["done"]] == ["hi", " there"]
    assert frames[-1]["done"] is True
    assert frames[-1]["message"] == {"role": "assistant", "content": ""}


def test_ollama_generate_non_stream():
    client = _client(FakeEngine(), protocols=["ollama"])

    r = client.post("/api/generate", json={"model": "test-model", "prompt": "hi", "stream": False})

    assert r.status_code == 200
    assert r.json()["response"] == "hi there"


def test_ollama_generate_sends_the_system_prompt():
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    client.post(
        "/api/generate",
        json={"model": "test-model", "prompt": "hi", "system": "be brief", "stream": False},
    )

    assert engine.calls[-1]["messages"][0] == {"role": "system", "content": "be brief"}


def test_ollama_unknown_model_uses_its_error_shape():
    client = _client(FakeEngine(), protocols=["ollama"])

    r = client.post(
        "/api/chat",
        json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 404
    assert "ghost" in r.json()["error"]


def test_ollama_latest_tag_resolves_to_the_alias():
    client = _client(FakeEngine(), protocols=["ollama"])

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model:latest",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert r.status_code == 200


def test_ollama_embed_returns_vectors():
    client = _client(FakeEngine(embeddings=True), protocols=["ollama"])

    r = client.post("/api/embed", json={"model": "test-model", "input": ["a", "b c"]})

    assert r.status_code == 200
    assert r.json()["embeddings"] == [[1.0, 0.5], [3.0, 0.5]]
    assert r.json()["prompt_eval_count"] == 3


def test_ollama_embed_explains_when_embeddings_are_off():
    client = _client(FakeEngine(), protocols=["ollama"])

    r = client.post("/api/embed", json={"model": "test-model", "input": "x"})

    assert r.status_code == 501
    assert "embeddings true" in r.json()["error"]


def test_ollama_chat_honours_format_json():
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "format": "json",
        },
    )

    assert r.status_code == 200
    assert engine.calls[-1]["response_format"] == {"type": "json_object"}


def test_ollama_chat_keeps_a_json_schema_format():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "format": schema,
        },
    )

    assert engine.calls[-1]["response_format"] == {"type": "json_object", "schema": schema}


def test_ollama_generate_honours_format_json():
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    client.post(
        "/api/generate",
        json={"model": "test-model", "prompt": "hi", "stream": False, "format": "json"},
    )

    assert engine.calls[-1]["response_format"] == {"type": "json_object"}


def test_ollama_rejects_an_unknown_format_before_loading_a_model():
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    r = client.post(
        "/api/chat",
        json={
            "model": "ghost",
            "messages": [{"role": "user", "content": "hi"}],
            "format": "yaml",
        },
    )

    assert r.status_code == 400
    assert "format" in r.json()["error"]
    assert engine.calls == []


def test_ollama_rejects_images_instead_of_dropping_them():
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "what is this?", "images": ["aGVsbG8="]}],
            "stream": False,
        },
    )

    assert r.status_code == 400
    assert "image" in r.json()["error"]
    assert engine.calls == []


def test_ollama_rejects_an_empty_message_list():
    engine = FakeEngine()
    client = _client(engine, protocols=["ollama"])

    r = client.post("/api/chat", json={"model": "test-model", "messages": []})

    assert r.status_code == 400
    assert "messages" in r.json()["error"]
    assert engine.calls == []


def test_ollama_non_stream_reports_prompt_tokens_without_usage():
    class NoUsageEngine(FakeEngine):
        def chat(self, messages, stream=False, **kw):
            self.calls.append({"messages": messages, "stream": stream, **kw})
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ]
            }

    client = _client(NoUsageEngine(), protocols=["ollama"])

    body = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "one two three"}],
            "stream": False,
        },
    ).json()

    # Ollama clients read these as the context the prompt occupied. Reporting 0
    # made them think the prompt was free.
    assert body["prompt_eval_count"] == 3
    assert body["eval_count"] == 2


def test_ollama_generate_non_stream_reports_prompt_tokens_without_usage():
    class NoUsageEngine(FakeEngine):
        def chat(self, messages, stream=False, **kw):
            self.calls.append({"messages": messages, "stream": stream, **kw})
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ]
            }

    client = _client(NoUsageEngine(), protocols=["ollama"])

    body = client.post(
        "/api/generate",
        json={"model": "test-model", "prompt": "one two three", "stream": False},
    ).json()

    assert body["prompt_eval_count"] == 3
    assert body["eval_count"] == 2


def test_ollama_tool_call_arguments_become_an_object():
    # OpenAI streams arguments as a JSON string; Ollama clients expect a map.
    calls = _ollama_tool_calls(
        [{"id": "call_1", "function": {"name": "get", "arguments": '{"city": "Hue"}'}}]
    )

    assert calls == [{"function": {"name": "get", "arguments": {"city": "Hue"}}}]


def test_streaming_passes_raw_sse_through_when_the_engine_speaks_it():
    """llama-server already speaks the OpenAI wire format; re-encoding each
    frame in Python is overhead on the token path."""

    class RawEngine(FakeEngine):
        passthrough_sse = True

        def chat_sse(self, messages, **kw):
            self.calls.append({"messages": messages, "raw": True, **kw})

            def _gen():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield b'data: {"choices":[],"usage":{"prompt_tokens":3}}\n\n'
                yield b"data: [DONE]\n\n"

            return _gen()

    engine = RawEngine()
    client = _client(engine)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        body = b"".join(r.iter_bytes()).decode()

    assert body.count("[DONE]") == 1
    assert '"content":"hi"' in body
    assert engine.calls[-1]["raw"] is True
    # The dict path must not have been used.
    assert all("stream" not in call for call in engine.calls)


def test_streaming_falls_back_to_re_encoding_for_other_engines():
    engine = FakeEngine()
    client = _client(engine)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        body = b"".join(r.iter_bytes()).decode()

    assert '"content": "hi"' in body
    assert body.rstrip().endswith("data: [DONE]")
    assert engine.calls[-1]["stream"] is True


# -- Ollama lifecycle --------------------------------------------------------


def test_ps_reports_the_keep_alive_deadline_after_a_chat():
    client, pool, _engines = _pool_client()

    response = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": "5m",
        },
    )

    assert response.status_code == 200
    loaded = client.get("/api/ps").json()["models"]
    assert [m["model"] for m in loaded] == ["test-model"]
    deadline = datetime.fromisoformat(loaded[0]["expires_at"].replace("Z", "+00:00"))
    assert deadline > datetime.now(UTC)
    assert loaded[0]["expires_at"] != "9999-12-31T23:59:59Z"
    pool.close()


def test_keep_alive_zero_unloads_after_the_response():
    client, pool, engines = _pool_client()

    response = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": 0,
        },
    )

    assert response.status_code == 200
    assert pool.loaded_ids() == []
    assert engines[0].closed is True
    pool.close()


def test_invalid_keep_alive_is_rejected_before_loading_a_model():
    client, pool, engines = _pool_client()

    response = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": "soon",
        },
    )

    assert response.status_code == 400
    assert "keep_alive" in response.json()["error"]
    assert engines == []
    pool.close()


def test_show_describes_a_model_without_loading_it():
    client, pool, engines = _pool_client()

    response = client.post("/api/show", json={"model": "test-model"})

    assert response.status_code == 200
    body = response.json()
    assert body["details"]["format"] == "gguf"
    assert body["capabilities"] == ["completion"]
    assert isinstance(body["model_info"], dict)
    assert engines == [], "show must not load the weights"
    pool.close()


def test_show_rejects_an_unknown_model():
    client, pool, _engines = _pool_client()

    response = client.post("/api/show", json={"model": "ghost"})

    assert response.status_code == 404
    assert "ghost" in response.json()["error"]
    pool.close()


# -- backend failures and request bounds -------------------------------------


def test_ollama_backend_failure_uses_its_error_shape(caplog):
    """Ollama clients read {"error": "..."}; a bodyless 500 tells them nothing."""

    class ExplodingEngine(FakeEngine):
        def chat(self, messages, stream=False, **kw):
            raise RuntimeError("child died mid-prefill")

    client = _client(ExplodingEngine(), protocols=["ollama"])
    with caplog.at_level(logging.ERROR, logger="vipera"):
        r = client.post(
            "/api/chat",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert r.status_code == 502
    assert "child died mid-prefill" in r.json()["error"]
    assert "child died mid-prefill" in caplog.text


def test_ollama_shaping_tokenizer_runs_off_the_event_loop(monkeypatch, tmp_path):
    """The /api/chat path must measure with the tokenizer off the event loop.

    Same trigger as the OpenAI layer: a prompt near the context limit makes the
    optimizer ask the engine for an exact count, which on the server backend is
    a blocking HTTP round trip.
    """
    monkeypatch.setattr("vipera.config.CONFIG_FILE", tmp_path / "config.json")

    class MeasuringEngine(FakeEngine):
        plan = types.SimpleNamespace(context=2048)

        def __init__(self):
            super().__init__()
            self.tokenize_calls = 0

        def tokenize(self, text):
            self.tokenize_calls += 1
            time.sleep(0.5)
            return [0] * 4000

    async def scenario():
        engine = MeasuringEngine()
        app = build_app(engine, "test-model", protocols=["ollama"])
        endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/chat")
        request = OllamaChatRequest(
            model="test-model",
            messages=[OllamaMessage(role="user", content="x" * 8000)],
            stream=False,
            options={"num_predict": 512},
        )

        task = asyncio.create_task(endpoint(request))
        t0 = time.perf_counter()
        await asyncio.sleep(0.1)
        elapsed = time.perf_counter() - t0
        await task
        return engine, elapsed

    engine, elapsed = asyncio.run(scenario())

    assert engine.tokenize_calls >= 1, "the exact-tokenizer path was never exercised"
    assert elapsed < 0.3, f"event loop blocked for {elapsed:.2f}s by the shaping tokenizer"


def test_ollama_generate_rejects_an_oversized_prompt_without_loading_the_model():
    client, pool, engines = _pool_client()
    try:
        r = client.post(
            "/api/generate",
            json={"model": "test-model", "prompt": "x" * 500_001, "stream": False},
        )
        assert r.status_code == 400
        assert engines == []
    finally:
        pool.close()


def test_ollama_embed_rejects_too_many_inputs_without_loading_the_model():
    client, pool, engines = _pool_client()
    try:
        r = client.post(
            "/api/embed",
            json={"model": "test-model", "input": ["x"] * 257},
        )
        assert r.status_code == 400
        assert "too many inputs" in r.json()["error"]
        assert engines == []
    finally:
        pool.close()


def test_ollama_embed_rejects_an_oversized_input():
    client = _client(FakeEngine(embeddings=True), protocols=["ollama"])

    r = client.post("/api/embed", json={"model": "test-model", "input": ["x" * 1_000_001]})

    assert r.status_code == 400
    assert "too large" in r.json()["error"]


def test_ollama_legacy_embeddings_rejects_an_oversized_prompt():
    client = _client(FakeEngine(embeddings=True), protocols=["ollama"])

    r = client.post(
        "/api/embeddings",
        json={"model": "test-model", "prompt": "x" * 1_000_001},
    )

    assert r.status_code == 400
    assert "too large" in r.json()["error"]
