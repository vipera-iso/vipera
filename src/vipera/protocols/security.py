"""Opt-in HTTP hardening for the gateway: a shared API key and CORS.

Both are off unless the config asks for them, so an install that upgrades keeps
serving exactly as before. They exist because the gateway binds to loopback and
says so in the docs — the moment someone passes `--host 0.0.0.0` to reach it
from another machine, "no authentication" stops being a design note and becomes
an open port onto a language model.

The key check is a middleware rather than a per-route dependency: the protocol
layers (`openai`, `ollama`) mount their own routers and a new route must not be
able to forget the check. `hmac.compare_digest` compares, so a wrong key costs
the same time whatever its prefix.

Errors keep the envelope of the protocol the caller is speaking: an Ollama
client reads a plain string under ``error``, an OpenAI one reads an error
object, and a client that gets the other shape reports "unexpected response"
instead of "bad key".
"""

from __future__ import annotations

import hmac
from collections.abc import Iterable, Mapping

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Liveness stays open: container probes, `vipera system doctor` and the pool
# monitor all poll it, and none of them has the key. It answers for the process,
# not for the model: no prompt, no tokens.
OPEN_PATHS = frozenset({"/health"})

# Schemes that carry a bare token in `Authorization`. Anything else (Digest,
# Basic) is treated as a mistake and rejected rather than compared verbatim.
_TOKEN_SCHEMES = frozenset({"bearer", "token"})


def cors_origins(value) -> list[str]:
    """Normalize the `cors` config into a list of origins.

    Accepts a comma-separated string (what `vipera system config set cors`
    writes) or a JSON list, strips trailing slashes, and drops empties while
    keeping the caller's order. `None` means "no CORS middleware", which is the
    default and the behaviour every install had before this existed.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, Iterable):
        parts = list(value)
    else:
        return []
    seen: list[str] = []
    for part in parts:
        origin = str(part).strip().rstrip("/")
        if origin and origin not in seen:
            seen.append(origin)
    return seen


def supplied_key(headers: Mapping[str, str]) -> str | None:
    """The key the request carries, from either header a client might use.

    Ollama and the OpenAI SDKs both send `Authorization: Bearer <key>`; `x-api-key`
    is what most proxy configs reach for. A bare `Authorization: <key>` (no
    scheme) is accepted too, because it is what `curl -H` usually produces.
    """
    auth = headers.get("authorization")
    if auth:
        scheme, _, token = auth.strip().partition(" ")
        if token and scheme.lower() in _TOKEN_SCHEMES:
            return token.strip()
        if not token:
            return auth.strip()
    return headers.get("x-api-key")


def error_body(path: str, message: str, code: str = "invalid_api_key") -> dict:
    """The 401 body in the shape the caller's protocol uses."""
    if path.startswith("/api/"):
        return {"error": message}
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": code,
        }
    }


def install(
    app: FastAPI,
    *,
    api_key: str | None,
    cors: Iterable[str] | str | None,
) -> None:
    """Guard `app` with CORS and/or an API key, when either is configured."""
    key = (api_key or "").strip()
    if key:
        expected = key.encode()

        @app.middleware("http")
        async def _require_key(request, call_next):
            path = request.url.path
            if path in OPEN_PATHS:
                return await call_next(request)
            supplied = supplied_key(request.headers)
            if supplied is None or not hmac.compare_digest(supplied.encode(), expected):
                return JSONResponse(
                    security_error(path, supplied is None),
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

    # Added last so it is outermost: a 401 has to carry the CORS headers too, or
    # the browser hides it behind a network error and the caller never learns
    # the key was missing. (Starlette wraps in reverse order of addition.)
    origins = cors_origins(cors)
    if origins:
        wildcard = "*" in origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            # Browsers reject `*` together with credentials; a wildcard means
            # "any page", which is not a credentialled origin.
            allow_credentials=not wildcard,
            allow_methods=["*"],
            allow_headers=["*"],
        )


def security_error(path: str, missing: bool) -> dict:
    """Body for a rejected request: say which failure it was, not just 401."""
    if missing:
        return error_body(
            path,
            "No API key was supplied. Send 'Authorization: Bearer <key>' or 'x-api-key: <key>'.",
            "missing_api_key",
        )
    return error_body(path, "The API key does not match this gateway.", "invalid_api_key")
