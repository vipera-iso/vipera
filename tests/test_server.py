import asyncio
import contextlib
import threading
import time

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
    assert r.json() == {"status": "ok", "model": "test-model"}


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
