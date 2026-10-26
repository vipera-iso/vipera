import asyncio
import contextlib
import json
import logging
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from vipera.server import ChatRequest, build_app


class FakeEngine:
    def __init__(self):
        self.calls = []

    def chat(self, messages, stream=False, **kw):
        self.calls.append({"messages": messages, "stream": stream, **kw})
        if stream:

            def _gen():
                for tok in ["hi", " ", "there"]:
                    yield {"choices": [{"delta": {"content": tok}}]}

            return _gen()
        return {
            "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }


def test_health():
    client = TestClient(build_app(FakeEngine(), "test-model"))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "test-model"
    # The optimization layer reports prefix reuse here; a fresh gateway has none.
    assert body["optimize"]["requests"] == 0
    assert body["optimize"]["prefix_hits"] == 0


def test_health_reports_the_engine_state_when_it_is_ok():
    class HealthyEngine(FakeEngine):
        def health(self):
            return "ok"

    client = TestClient(build_app(HealthyEngine(), "test-model"))

    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["engines"] == {"test-model": "ok"}


def test_health_is_503_when_the_engine_is_not_serving():
    class SickEngine(FakeEngine):
        def health(self):
            return "exited"

    client = TestClient(build_app(SickEngine(), "test-model"))

    r = client.get("/health")

    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "exited"
    assert body["engines"] == {"test-model": "exited"}


def test_health_in_pool_mode_only_probes_loaded_models():
    app, pool, _ = _pool_app()
    client = TestClient(app)

    # A health probe must not become the thing that loads a model.
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["engines"] == {}
    assert pool.loaded_ids() == []

    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert client.get("/health").json()["engines"] == {"alpha": "ok"}


def test_health_in_pool_mode_reports_a_model_that_stopped_responding():
    from vipera.pool import ModelPool

    class DeadEngine(_NamedEngine):
        def health(self):
            return "unreachable"

    pool = ModelPool({"alpha": "/tmp/alpha.gguf"}, lambda alias, path: DeadEngine(alias))
    client = TestClient(build_app(pool=pool))
    try:
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        r = client.get("/health")

        assert r.status_code == 503
        assert r.json()["status"] == "unhealthy"
        assert r.json()["engines"] == {"alpha": "unreachable"}
    finally:
        pool.close()


def test_models():
    client = TestClient(build_app(FakeEngine(), "test-model"))
    r = client.get("/v1/models")
    assert r.json()["data"][0]["id"] == "test-model"


def test_chat_non_stream():
    client = TestClient(build_app(FakeEngine(), "test-model"))
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert "hi there" in r.json()["choices"][0]["message"]["content"]


def test_chat_stream():
    client = TestClient(build_app(FakeEngine(), "test-model"))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = b"".join(r.iter_bytes()).decode()
        assert "hi" in body
        assert "[DONE]" in body


@pytest.mark.parametrize("usage_with_choices", [False, True])
def test_stream_emits_usage_frame_once(usage_with_choices):
    """llama-server already sends a trailing choices:[] usage chunk.

    Re-emitting it for include_usage would hand the client two usage frames:
    the backend's own plus the synthesized one. A backend that attaches usage
    to a chunk carrying content must not be duplicated either.
    """

    class UsageChunkEngine:
        def chat(self, messages, stream=False, **kw):
            if stream:

                def _gen():
                    yield {"choices": [{"delta": {"content": "hi"}}]}
                    usage = {"prompt_tokens": 3, "completion_tokens": 1}
                    if usage_with_choices:
                        yield {"choices": [{"delta": {}}], "usage": usage}
                    else:
                        # Exactly the shape llama-server ends a stream with.
                        yield {"choices": [], "usage": usage}

                return _gen()
            raise AssertionError("stream=True expected")

    client = TestClient(build_app(UsageChunkEngine(), "test-model"))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as r:
        body = b"".join(r.iter_bytes()).decode()

    frames = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    usage_frames = [f for f in frames if f.get("usage")]
    assert len(usage_frames) == 1
    assert usage_frames[0]["usage"] == {"prompt_tokens": 3, "completion_tokens": 1}


def test_stream_engine_error_becomes_sse_event():
    """A failure before the first chunk must not abort the body mid-stream."""

    class RaisingEngine:
        def chat(self, messages, stream=False, **kw):
            raise RuntimeError("engine exploded")

    client = TestClient(build_app(RaisingEngine(), "test-model"))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()

    assert "engine exploded" in body
    assert "[DONE]" in body


def test_backend_failure_is_an_openai_shaped_502(caplog):
    """A failed engine call used to reach the client as a bodyless 500.

    The cause (the child's own 400, a socket error, a dead process) has to
    survive to the client in OpenAI's error object, and it has to be logged:
    the failure is raised from a worker thread, so nothing else records it.
    """

    class ExplodingEngine(FakeEngine):
        def chat(self, messages, stream=False, **kw):
            raise RuntimeError("llama-server returned 400: prompt too long")

    client = TestClient(build_app(ExplodingEngine(), "test-model"))
    with caplog.at_level(logging.ERROR, logger="vipera"):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert r.status_code == 502
    error = r.json()["error"]
    assert error["message"] == "llama-server returned 400: prompt too long"
    assert error["type"] == "backend_error"
    assert "prompt too long" in caplog.text
    assert caplog.records[-1].exc_info is not None, "the traceback was dropped"


def test_embeddings_backend_failure_is_an_openai_shaped_502():
    class ExplodingEngine(FakeEngine):
        supports_embeddings = True

        def embed(self, texts):
            raise RuntimeError("child stopped answering")

    client = TestClient(build_app(ExplodingEngine(), "test-model"))

    r = client.post("/v1/embeddings", json={"input": ["a"]})

    assert r.status_code == 502
    assert "child stopped answering" in r.json()["error"]["message"]


def test_a_model_that_fails_to_load_is_a_structured_503():
    """A load failure in pool mode escaped as an unhandled exception."""
    from vipera.pool import ModelPool

    def factory(alias, path):
        raise RuntimeError("Not enough RAM for this plan")

    pool = ModelPool({"alpha": "/tmp/alpha.gguf"}, factory)
    client = TestClient(build_app(pool=pool))
    try:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503
        assert "Not enough RAM" in r.json()["error"]["message"]
    finally:
        pool.close()


def test_lock_not_released_while_worker_is_still_generating():
    """Client disconnect must not let a second request race the engine.

    Cancelling the streaming request leaves next() running in a worker thread.
    The lock has to stay held until that thread returns; releasing it on
    cancellation would allow two calls into the same llama.cpp context.
    """

    class SlowEngine:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        def chat(self, messages, stream=False, **kw):
            if stream:

                def _gen():
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    try:
                        for i in range(6):
                            time.sleep(0.15)
                            yield {"choices": [{"delta": {"content": str(i)}}]}
                    finally:
                        self.active -= 1

                return _gen()

            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.01)
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            finally:
                self.active -= 1

    async def scenario():
        engine = SlowEngine()
        app = build_app(engine, "test-model")
        endpoint = next(
            r.endpoint for r in app.routes if getattr(r, "path", None) == "/v1/chat/completions"
        )

        response = await endpoint(
            ChatRequest(
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )
        iterator = response.body_iterator
        await anext(iterator)  # first chunk; the lock stays held

        pending = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0.02)  # worker is now sleeping inside next()
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending

        t0 = time.perf_counter()
        await endpoint(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
        elapsed = time.perf_counter() - t0

        with contextlib.suppress(Exception):
            await iterator.aclose()
        return engine, elapsed

    engine, elapsed = asyncio.run(scenario())

    assert engine.max_active == 1, "two engine calls overlapped"
    # The abandoned worker still had ~0.13s of sleep left. If the lock had been
    # released on cancellation, the follow-up call would have returned at once.
    assert elapsed >= 0.1, f"lock released early (follow-up took {elapsed:.3f}s)"


def test_health_responsive_while_streaming():
    """The event loop must stay free while a stream is in flight.

    engine.chat(stream=True) returns a SYNC generator. Iterating it inline
    would block the event loop for the whole generation, so /health would not
    answer until the stream finished. Chunks are pulled via asyncio.to_thread
    instead; this asserts that actually holds.
    """

    class SlowEngine(FakeEngine):
        def chat(self, messages, stream=False, **kw):
            if stream:

                def _gen():
                    for i in range(4):
                        time.sleep(0.15)  # stand in for token generation
                        yield {"choices": [{"delta": {"content": str(i)}}]}

                return _gen()
            return super().chat(messages, stream=stream, **kw)

    with TestClient(build_app(SlowEngine(), "test-model")) as client:
        finished = threading.Event()

        def run_stream():
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ) as r:
                b"".join(r.iter_bytes())
            finished.set()

        worker = threading.Thread(target=run_stream, daemon=True)
        worker.start()
        time.sleep(0.1)  # let the stream get going

        t0 = time.perf_counter()
        r = client.get("/health")
        latency = time.perf_counter() - t0

        assert r.status_code == 200
        # The stream takes ~0.6s. If the loop were blocked, /health would have
        # waited for all of it; a free loop answers immediately.
        assert latency < 0.3, f"/health blocked for {latency:.2f}s during a stream"

        worker.join(timeout=5)
        assert finished.is_set()


def test_shaping_tokenizer_runs_off_the_event_loop(monkeypatch, tmp_path):
    """Measuring a near-limit prompt must not freeze the gateway.

    Once the shaped estimate lands near the context limit, the optimizer asks
    the engine's own tokenizer for an exact count. On the server backend that
    is a blocking HTTP round trip, so running it inline stops /health and every
    other request for its duration.
    """
    monkeypatch.setattr("vipera.config.CONFIG_FILE", tmp_path / "config.json")

    class MeasuringEngine(FakeEngine):
        plan = types.SimpleNamespace(context=2048)

        def __init__(self):
            super().__init__()
            self.tokenize_calls = 0

        def tokenize(self, text):
            self.tokenize_calls += 1
            time.sleep(0.5)  # stands in for llama-server's /tokenize
            return [0] * 4000

    async def scenario():
        engine = MeasuringEngine()
        app = build_app(engine, "test-model")
        endpoint = next(
            r.endpoint for r in app.routes if getattr(r, "path", None) == "/v1/chat/completions"
        )
        # 8k characters against a 2048-token window with 512 reserved for the
        # answer: large enough that shaping measures instead of estimating.
        request = ChatRequest(messages=[{"role": "user", "content": "x" * 8000}], max_tokens=512)

        task = asyncio.create_task(endpoint(request))
        t0 = time.perf_counter()
        await asyncio.sleep(0.1)  # resumes at once only if the loop is free
        elapsed = time.perf_counter() - t0
        await task
        return engine, elapsed

    engine, elapsed = asyncio.run(scenario())

    assert engine.tokenize_calls >= 1, "the exact-tokenizer path was never exercised"
    assert elapsed < 0.3, f"event loop blocked for {elapsed:.2f}s by the shaping tokenizer"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def test_chat_forwards_openai_params_to_engine():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model"))

    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 64,
            "top_p": 0.8,
            "top_k": 10,
            "min_p": 0.01,
            "repeat_penalty": 1.2,
            "presence_penalty": 0.1,
            "frequency_penalty": -0.1,
            "stop": ["\n\n"],
            "seed": 7,
            "logit_bias": {"42": -2.0},
            "response_format": {"type": "json_object"},
            "tools": TOOLS,
            "tool_choice": "auto",
        },
    )

    assert r.status_code == 200
    call = engine.calls[-1]
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 64
    assert call["top_p"] == 0.8
    assert call["top_k"] == 10
    assert call["min_p"] == 0.01
    assert call["repeat_penalty"] == 1.2
    assert call["presence_penalty"] == 0.1
    assert call["frequency_penalty"] == -0.1
    assert call["stop"] == ["\n\n"]
    assert call["seed"] == 7
    assert call["logit_bias"] == {"42": -2.0}
    assert call["response_format"] == {"type": "json_object"}
    assert call["tools"] == TOOLS
    assert call["tool_choice"] == "auto"


def test_unset_optional_params_are_not_forwarded_as_null():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model"))

    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "###",
        },
    )

    assert r.status_code == 200
    call = engine.calls[-1]
    assert call["stop"] == "###"
    assert "seed" not in call
    assert "tools" not in call
    assert "top_k" not in call


def test_tool_calls_pass_through_non_streaming():
    class ToolEngine(FakeEngine):
        def chat(self, messages, stream=False, **kw):
            self.calls.append({"messages": messages, "stream": stream, **kw})
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5},
            }

    engine = ToolEngine()
    client = TestClient(build_app(engine, "test-model"))
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": TOOLS,
            "tool_choice": "auto",
        },
    )

    assert r.status_code == 200
    tool_calls = r.json()["choices"][0]["message"]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_call_deltas_stream_and_usage_frame_is_emitted():
    class ToolStreamEngine(FakeEngine):
        def chat(self, messages, stream=False, **kw):
            self.calls.append({"messages": messages, "stream": stream, **kw})

            def _gen():
                yield {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}
                yield {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":',
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
                yield {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"Paris"}'},
                                    }
                                ]
                            },
                        }
                    ]
                }
                yield {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 5},
                }

            return _gen()

    client = TestClient(build_app(ToolStreamEngine(), "test-model"))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": TOOLS,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()

    assert '"tool_calls"' in body
    assert '"name": "get_weather"' in body
    assert '"finish_reason": "tool_calls"' in body
    # include_usage adds one choices:[] frame carrying the usage totals.
    assert '"choices": []' in body
    assert '"prompt_tokens": 9' in body
    assert body.rstrip().endswith("data: [DONE]")


def test_assistant_tool_call_turn_is_accepted_without_content():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model"))
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "22C sunny"},
            ],
            "tools": TOOLS,
        },
    )

    assert r.status_code == 200
    sent = engine.calls[-1]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "content" not in sent[1]
    assert sent[2]["tool_call_id"] == "call_1"


def test_message_without_content_or_tool_calls_is_rejected():
    client = TestClient(build_app(FakeEngine(), "test-model"))
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user"}],
        },
    )
    assert r.status_code == 422


def test_invalid_sampling_ranges_are_rejected():
    client = TestClient(build_app(FakeEngine(), "test-model"))

    for payload in (
        {"top_p": 0},
        {"presence_penalty": 3},
        {"seed": -1},
        {"top_k": -1},
        {"max_tokens": 0},
    ):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                **payload,
            },
        )
        assert r.status_code in {400, 422}, payload


def test_unknown_model_is_still_rejected():
    client = TestClient(build_app(FakeEngine(), "test-model"))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "other",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 404


def test_include_usage_is_forwarded_to_the_backend():
    # The trailing choices:[] frame is only useful when the backend actually
    # reports usage; llama-server does, llama-cpp-python 0.3.x does not.
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model"))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as r:
        assert r.status_code == 200
        b"".join(r.iter_bytes())

    assert engine.calls[-1]["stream_options"] == {"include_usage": True}


def test_include_usage_is_not_sent_when_not_requested():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model"))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        b"".join(r.iter_bytes())

    assert "stream_options" not in engine.calls[-1]


class _NamedEngine:
    """Echo the model it was built for, so routing mistakes are visible."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def chat(self, messages, stream=False, **kw):
        self.calls.append({"messages": messages, "stream": stream, **kw})
        if stream:

            def _gen():
                yield {"choices": [{"delta": {"content": self.name}}]}

            return _gen()
        return {
            "choices": [{"message": {"role": "assistant", "content": self.name}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def _pool_app():
    from vipera.pool import ModelPool

    engines = {alias: _NamedEngine(alias) for alias in ("alpha", "beta")}
    pool = ModelPool(
        {alias: f"/tmp/{alias}.gguf" for alias in engines},
        lambda alias, path: engines[alias],
    )
    return build_app(pool=pool), pool, engines


def test_pool_lists_every_model():
    app, pool, _ = _pool_app()
    client = TestClient(app)
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["alpha", "beta"]
    assert pool.loaded_ids() == []  # nothing loaded just from listing


def test_pool_routes_to_the_requested_model():
    app, _pool, engines = _pool_app()
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "beta", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "beta"
    assert engines["alpha"].calls == []


def test_an_invalid_request_is_rejected_without_loading_the_model():
    """A rejected request must not pay for a multi-minute GGUF load first."""

    from vipera.pool import ModelPool

    loads = []

    def factory(alias, path):
        loads.append(alias)
        return _NamedEngine(alias)

    pool = ModelPool({"alpha": "/tmp/alpha.gguf"}, factory, max_loaded=1)
    client = TestClient(build_app(pool=pool))
    try:
        bad_bodies = [
            {"messages": [{"role": "user", "content": "hi"}], "temperature": 5},
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
            {"messages": [{"role": "user", "content": "x" * 500_001}]},
        ]
        for body in bad_bodies:
            assert client.post("/v1/chat/completions", json=body).status_code == 400
        assert loads == []

        # A request that passes validation still loads the model and is served.
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert loads == ["alpha"]
    finally:
        pool.close()


def test_pool_defaults_to_the_first_model():
    app, _, _ = _pool_app()
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "alpha"


def test_pool_rejects_an_unknown_model_and_lists_alternatives():
    app, _, _ = _pool_app()
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gamma", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert "gamma" in r.json()["detail"]
    assert "alpha" in r.json()["detail"]


def test_pool_health_reports_loaded_models():
    app, _pool, _ = _pool_app()
    client = TestClient(app)
    assert client.get("/health").json()["loaded"] == []

    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    body = client.get("/health").json()
    assert body["model"] == "alpha"
    assert body["loaded"] == ["alpha"]


def test_build_app_requires_an_engine_or_a_pool():
    with pytest.raises(ValueError):
        build_app()


class _RecordingPool:
    """Minimal pool that records where acquire runs and how often release runs."""

    default = "alpha"

    def __init__(self):
        self.ran_on_loop = None
        self.in_flight = 0
        self.releases = 0

    def ids(self):
        return ["alpha"]

    def loaded_ids(self):
        return ["alpha"]

    def acquire(self, name):
        # A load takes minutes; if this runs on the event loop, /health and
        # every other request freeze behind it.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.ran_on_loop = False
        else:
            self.ran_on_loop = True
        self.in_flight += 1
        return _NamedEngine("alpha"), "alpha"

    def release(self, engine):
        self.in_flight -= 1
        self.releases += 1


def test_a_pool_load_does_not_run_on_the_event_loop():
    pool = _RecordingPool()
    client = TestClient(build_app(pool=pool))

    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 200
    assert pool.ran_on_loop is False


def test_a_finished_request_gives_its_pool_reference_back():
    pool = _RecordingPool()
    client = TestClient(build_app(pool=pool))

    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert pool.releases == 1
    assert pool.in_flight == 0


def test_a_finished_stream_gives_its_pool_reference_back():
    pool = _RecordingPool()
    client = TestClient(build_app(pool=pool))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        b"".join(r.iter_bytes())

    assert pool.releases == 1
    assert pool.in_flight == 0


# -- SSE relay ---------------------------------------------------------------


def _drain(agen):
    async def _collect():
        return [item async for item in agen]

    return asyncio.run(_collect())


def test_relay_yields_every_item_and_releases_the_slot():
    from vipera.protocols.streaming import relay_chunks

    closed = []

    async def _run():
        lock = asyncio.Lock()
        items = [b"a", b"b", b"c"]
        got = [
            item
            async for item in relay_chunks(
                lock, lambda: iter(items), on_close=lambda: closed.append(1)
            )
        ]
        return got, lock.locked()

    got, locked = asyncio.run(_run())

    assert got == [b"a", b"b", b"c"]
    assert locked is False
    assert closed == [1]


def test_relay_closes_a_parked_reader_before_releasing_the_slot():
    """A client that disconnects mid-stream must not hand a busy context on."""

    from vipera.protocols.streaming import relay_chunks

    events = []

    class _Stream:
        def __init__(self):
            self.first = True
            self.gate = threading.Event()

        def __iter__(self):
            return self

        def __next__(self):
            if self.first:
                self.first = False
                return b"data: one\n\n"
            # Parks here like a socket read with no data yet.
            assert self.gate.wait(5), "close() never unblocked the reader"
            raise StopIteration

        def close(self):
            events.append("closed")
            self.gate.set()

    async def _run():
        lock = asyncio.Lock()
        agen = relay_chunks(lock, _Stream)
        assert await agen.__anext__() == b"data: one\n\n"
        await agen.aclose()
        return lock.locked()

    locked = asyncio.run(_run())

    assert events == ["closed"]
    assert locked is False


def test_relay_holds_the_slot_when_close_cannot_unblock_the_reader():
    """A socket read already inside http.client survives close().

    llama-server's SSE reader is a blocking ``read1``: closing the response from
    another thread does not necessarily return it. Releasing the permit on a
    join timeout would admit a second request into a context that is still
    generating, so the permit must stay held until the reader returns.
    """

    from vipera.protocols.streaming import relay_chunks

    class _SocketLike:
        """``close()`` does nothing for the parked read, as http.client can be."""

        def __init__(self):
            self.first = True
            self.unpark = threading.Event()

        def __iter__(self):
            return self

        def __next__(self):
            if self.first:
                self.first = False
                return b"data: one\n\n"
            self.unpark.wait(15)
            raise StopIteration

        def close(self):
            pass

    stream = _SocketLike()

    async def _run():
        lock = asyncio.Lock()
        agen = relay_chunks(lock, lambda: stream)
        assert await agen.__anext__() == b"data: one\n\n"
        await agen.aclose()
        held_while_parked = lock.locked()
        stream.unpark.set()
        for _ in range(200):
            if not lock.locked():
                break
            await asyncio.sleep(0.05)
        return held_while_parked, lock.locked()

    held_while_parked, released_after = asyncio.run(_run())

    assert held_while_parked is True, "the slot was released while the reader was still parked"
    assert released_after is False


def test_relay_surfaces_a_reader_error():
    from vipera.protocols.streaming import relay_chunks

    class _Boom:
        def __iter__(self):
            raise RuntimeError("child died")

    async def _run():
        lock = asyncio.Lock()
        with pytest.raises(RuntimeError, match="child died"):
            async for _ in relay_chunks(lock, _Boom):
                pass
        return lock.locked()

    assert asyncio.run(_run()) is False
