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
