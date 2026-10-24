"""The real Ollama Python SDK against a live gateway.

``tests/unit/test_protocols.py`` calls the route functions through
``TestClient``, which never touches the wire. This starts uvicorn on loopback
and drives it with ``ollama.Client``, so the framing, the NDJSON stream and the
``{"error": ...}`` body are what an Ollama client actually parses.

A fake engine stands in for the model: only the HTTP layer is under test, so
this needs no weights and no Ollama daemon.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from ollama import Client, ResponseError

from vipera.server import build_app

pytestmark = pytest.mark.integration


class FakeEngine:
    supports_embeddings = True

    def chat(self, messages, stream=False, **kw):
        if stream:

            def _gen():
                for piece in ("Hello", ", ", "world"):
                    yield {"choices": [{"delta": {"content": piece}}]}
                yield {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 3}}

            return _gen()
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello, world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 3},
        }

    def tokenize(self, text):
        return [0] * len(str(text).split())

    def embed(self, texts):
        return [[0.5, 1.5] for _ in texts]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def gateway():
    app = build_app(FakeEngine(), "test-model", protocols=["openai", "ollama"])
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("gateway did not start")
        time.sleep(0.02)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def client(gateway):
    http = Client(host=gateway)
    try:
        yield http
    finally:
        http.close()


def test_sdk_lists_the_served_model(client):
    models = client.list().models

    assert [m.model for m in models] == ["test-model"]
    assert models[0].details.format == "gguf"


def test_sdk_chat_without_streaming(client):
    response = client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )

    assert response.message.content == "Hello, world"
    assert response.done is True
    assert response.done_reason == "stop"
    assert response.eval_count == 3


def test_sdk_chat_streams_every_chunk(client):
    chunks = list(
        client.chat(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    )

    # The SDK parses one JSON object per line; a framing mistake would fail here
    # rather than inside the route.
    text = "".join(chunk.message.content for chunk in chunks if not chunk.done)
    assert text == "Hello, world"
    assert chunks[-1].done is True
    assert chunks[-1].eval_count == 3
    assert all(chunk.model == "test-model" for chunk in chunks)


def test_sdk_generate(client):
    response = client.generate(model="test-model", prompt="hi", stream=False)

    assert response.response == "Hello, world"
    assert response.done is True


def test_sdk_embed(client):
    response = client.embed(model="test-model", input=["a", "b"])

    assert response.embeddings == [[0.5, 1.5], [0.5, 1.5]]


def test_sdk_ps_lists_loaded_models(client):
    assert [m.model for m in client.ps().models] == ["test-model"]


def test_sdk_raises_response_error_for_an_unknown_model(client):
    with pytest.raises(ResponseError) as exc:
        client.chat(model="ghost", messages=[{"role": "user", "content": "hi"}])

    assert exc.value.status_code == 404
    assert "ghost" in str(exc.value)
