import threading
import time

from fastapi.testclient import TestClient

from vipera.server import build_app


class FakeEngine:
    def chat(self, messages, stream=False, **kw):
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
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert "hi there" in r.json()["choices"][0]["message"]["content"]


def test_chat_stream():
    client = TestClient(build_app(FakeEngine(), "test-model"))
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = b"".join(r.iter_bytes()).decode()
        assert "hi" in body
        assert "[DONE]" in body


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
            with client.stream("POST", "/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }) as r:
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
