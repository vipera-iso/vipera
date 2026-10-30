"""The opt-in API key and CORS layer, tested through the HTTP surface."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vipera.protocols import security


def _app(*, api_key=None, cors=None, max_body_bytes=None) -> FastAPI:
    """A stand-in gateway: the two protocol shapes plus the open probe path."""
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def openai_chat():
        return {"ok": True}

    @app.post("/api/chat")
    def ollama_chat():
        return {"ok": True}

    @app.post("/v1/echo")
    async def echo(payload: dict):
        # Declaring a body is what makes FastAPI read the request stream; a route
        # that ignores its body can never trip the cap.
        return {"ok": True}

    security.install(app, api_key=api_key, cors=cors, max_body_bytes=max_body_bytes)
    return app


def test_nothing_is_checked_when_no_key_is_configured():
    # The default install keeps working exactly as before: this is the
    # compatibility guarantee, not a convenience.
    client = TestClient(_app())

    assert client.post("/v1/chat/completions").status_code == 200
    assert client.post("/api/chat").status_code == 200


def test_a_missing_key_is_rejected_in_the_openai_shape():
    client = TestClient(_app(api_key="secret"))

    response = client.post("/v1/chat/completions")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    body = response.json()["error"]
    assert body["type"] == "invalid_request_error"
    assert body["code"] == "missing_api_key"
    assert "Authorization: Bearer" in body["message"]


def test_a_rejected_ollama_call_gets_a_string_error():
    # Ollama clients read `error` as a string; handing them an object makes
    # them report "unexpected response" instead of "bad key".
    client = TestClient(_app(api_key="secret"))

    response = client.post("/api/chat")

    assert response.status_code == 401
    assert isinstance(response.json()["error"], str)


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer secret"},
        {"Authorization": "bearer secret"},
        {"Authorization": "secret"},
        {"x-api-key": "secret"},
        {"X-API-Key": "secret"},
    ],
)
def test_every_accepted_way_of_carrying_the_key(headers):
    client = TestClient(_app(api_key="secret"))

    assert client.post("/v1/chat/completions", headers=headers).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Basic secret"},
        {"x-api-key": "sec"},
        {"x-api-key": ""},
    ],
)
def test_wrong_or_malformed_keys_are_rejected(headers):
    client = TestClient(_app(api_key="secret"))

    assert client.post("/v1/chat/completions", headers=headers).status_code == 401


def test_health_stays_open_for_probes():
    client = TestClient(_app(api_key="secret"))

    assert client.get("/health").status_code == 200


def test_an_empty_key_in_the_config_means_no_check():
    # A half-written key must not become "the empty string is the password".
    client = TestClient(_app(api_key="   "))

    assert client.post("/v1/chat/completions").status_code == 200


def test_cors_is_absent_unless_configured():
    client = TestClient(_app())

    response = client.options(
        "/v1/chat/completions",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )

    assert "access-control-allow-origin" not in response.headers


def test_cors_echoes_only_the_configured_origin():
    client = TestClient(_app(cors="http://localhost:3000, http://localhost:5173/"))

    allowed = client.options(
        "/v1/chat/completions",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    refused = client.options(
        "/v1/chat/completions",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in refused.headers


def test_cors_wildcard_allows_any_origin():
    client = TestClient(_app(cors="*"))

    response = client.get("/health", headers={"Origin": "http://anywhere.example"})

    assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ("http://a", ["http://a"]),
        ("http://a/, http://b", ["http://a", "http://b"]),
        (["http://a", "http://a", " "], ["http://a"]),
        (42, []),
    ],
)
def test_cors_origins_normalizes_the_config_value(value, expected):
    assert security.cors_origins(value) == expected


def test_key_check_and_cors_work_together():
    client = TestClient(_app(api_key="secret", cors="*"))
    rejected = client.post("/v1/chat/completions", headers={"Origin": "http://anywhere.example"})
    accepted = client.post(
        "/v1/chat/completions",
        headers={"Origin": "http://anywhere.example", "Authorization": "Bearer secret"},
    )

    # A rejected request is still a CORS response, or the browser hides the 401
    # behind a network error and the user never learns the key is missing.
    assert rejected.status_code == 401
    assert rejected.headers["access-control-allow-origin"] == "*"
    assert accepted.status_code == 200


def test_an_oversized_body_is_refused_before_it_is_read():
    client = TestClient(_app(max_body_bytes=64))

    response = client.post("/v1/echo", json={"text": "x" * 200})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_a_streamed_body_is_cut_off_at_the_cap():
    # A chunked request carries no Content-Length, so counting the bytes the app
    # actually reads is the only way to bound it.
    client = TestClient(_app(max_body_bytes=64))

    def chunks():
        for _ in range(10):
            yield b"x" * 32

    response = client.post("/v1/echo", content=chunks())

    assert response.status_code == 413


def test_a_body_within_the_cap_is_untouched():
    client = TestClient(_app(max_body_bytes=4096))

    assert client.post("/v1/echo", json={"text": "hello"}).status_code == 200


def test_a_413_still_carries_the_cors_headers():
    # Same reason as the 401 above: a browser must be able to read the refusal.
    client = TestClient(_app(max_body_bytes=64, cors="*"))

    response = client.post(
        "/v1/echo", content=b"x" * 200, headers={"Origin": "http://anywhere.example"}
    )

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "*"
