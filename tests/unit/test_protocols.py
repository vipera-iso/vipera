import json
import types

import pytest
from fastapi.testclient import TestClient

from vipera.protocols import mount
from vipera.protocols.ollama import _ollama_tool_calls
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


def test_ollama_tool_call_arguments_become_an_object():
    # OpenAI streams arguments as a JSON string; Ollama clients expect a map.
    calls = _ollama_tool_calls(
        [{"id": "call_1", "function": {"name": "get", "arguments": '{"city": "Hue"}'}}]
    )

    assert calls == [{"function": {"name": "get", "arguments": {"city": "Hue"}}}]
