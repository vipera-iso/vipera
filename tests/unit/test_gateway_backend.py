import io
import json
import urllib.error
import urllib.request

import pytest

from vipera.backends import gateway


class _Payload:
    """A response body for the non-streaming path."""

    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


class _Lines:
    """An SSE response: iterable by line, and closeable."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


def _answer(monkeypatch, payload, *, recorded=None):
    """Make `urlopen` answer with `payload`; capture request/body if asked."""

    def fake_urlopen(request, timeout=None):
        if recorded is not None:
            recorded.append(request)
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, _Lines):
            return payload
        if isinstance(payload, _Payload):
            return payload
        return _Payload(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_served_models_reads_the_gateway_listing(monkeypatch):
    _answer(monkeypatch, {"data": [{"id": "gpt-oss-120b"}, {"id": "qwen"}]})

    assert gateway.served_models("http://127.0.0.1:8686") == ["gpt-oss-120b", "qwen"]


def test_served_models_returns_none_when_nothing_answers(monkeypatch):
    # None is "no gateway"; [] is "a gateway serving nothing". Attaching needs
    # the difference, so a dead port must not read as an empty one.
    _answer(monkeypatch, OSError("connection refused"))
    assert gateway.served_models("http://127.0.0.1:8686") is None

    _answer(monkeypatch, _Payload(b"<html>not a gateway</html>"))
    assert gateway.served_models("http://127.0.0.1:8686") is None


def test_served_models_sends_the_api_key(monkeypatch):
    seen = []
    _answer(monkeypatch, {"data": [{"id": "m"}]}, recorded=seen)

    gateway.served_models("http://127.0.0.1:8686", api_key="secret")

    assert seen[0].headers["Authorization"] == "Bearer secret"


def test_attach_only_takes_a_gateway_that_serves_this_model(monkeypatch):
    _answer(monkeypatch, {"data": [{"id": "other"}]})
    assert gateway.attach("http://127.0.0.1:8686", "m") is None

    _answer(monkeypatch, {"data": [{"id": "m"}]})
    engine = gateway.attach("http://127.0.0.1:8686", "m")
    assert isinstance(engine, gateway.GatewayEngine)
    assert engine.model == "m"


def test_base_url_reaches_a_wildcard_bind(monkeypatch):
    # `host: 0.0.0.0` is where serve binds, not an address to connect to.
    assert gateway.base_url("0.0.0.0", 8686) == "http://127.0.0.1:8686"
    assert gateway.base_url("127.0.0.1", 8686) == "http://127.0.0.1:8686"


def test_chat_posts_the_openai_body_with_the_model(monkeypatch):
    seen = []
    _answer(monkeypatch, {"choices": [{"message": {"content": "hi"}}]}, recorded=seen)

    engine = gateway.GatewayEngine("http://127.0.0.1:8686", "gpt-oss-120b")
    out = engine.chat([{"role": "user", "content": "hello"}], max_tokens=4, temperature=0)

    assert out["choices"][0]["message"]["content"] == "hi"
    body = json.loads(seen[0].data)
    # The model has to travel: a gateway serving several models picks by it.
    assert body["model"] == "gpt-oss-120b"
    assert body["max_tokens"] == 4
    assert body["stream"] is False


def test_chat_streams_frames_and_stops_at_done(monkeypatch):
    lines = _Lines(
        [
            b'data: {"choices":[{"delta":{"content":"he"}}]}\n',
            b": keep-alive\n",
            b'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
            b"data: [DONE]\n",
            b'data: {"choices":[{"delta":{"content":"never"}}]}\n',
        ]
    )
    _answer(monkeypatch, lines)

    engine = gateway.GatewayEngine("http://127.0.0.1:8686", "m")
    frames = list(engine.chat([{"role": "user", "content": "hi"}], stream=True))

    assert [frame["choices"][0]["delta"]["content"] for frame in frames] == ["he", "llo"]
    assert lines.closed is True


def test_close_does_not_touch_the_gateway():
    engine = gateway.GatewayEngine("http://127.0.0.1:8686", "m")
    engine.close()  # the gateway owns the model, not the client


def test_an_http_error_names_the_gateway_and_the_body(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 500, "boom", {}, io.BytesIO(b"model not loaded")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    engine = gateway.GatewayEngine("http://127.0.0.1:8686", "m")

    with pytest.raises(RuntimeError, match="returned 500: model not loaded"):
        engine.chat([{"role": "user", "content": "hi"}])
