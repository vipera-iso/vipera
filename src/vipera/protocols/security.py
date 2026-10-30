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

The body cap is the one part that is not opt-in: no client legitimately sends a
16 MiB request, and the per-field bounds in ``vipera.protocols.limits`` can only
be checked once the body has been parsed, by which point it has been read.

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
from starlette.types import Message, Receive, Scope, Send

from vipera.protocols.limits import MAX_BODY_BYTES

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


class _BodyLimit:
    """Refuse a request body over `max_bytes`, declared or streamed.

    A `Content-Length` over the cap is refused before the body is read at all,
    which is the point: the memory is never allocated. A chunked request carries
    no such header, so its body is read here, up to the cap, and replayed to the
    app — buffering is not a new cost, because FastAPI reads the whole body into
    memory to parse it anyway. Refusing by raising from inside the read does not
    work: FastAPI wraps any failure while parsing a body into a 400, so the cap
    has to be enforced before the app is called.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope["headers"]:
            if name != b"content-length":
                continue
            try:
                declared = int(value)
            except ValueError:
                break
            if declared > self.max_bytes:
                response = _too_large(scope["path"], self.max_bytes)
                await response(scope, receive, send)
                return
            break

        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                response = _too_large(scope["path"], self.max_bytes)
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay, send)


def _too_large(path: str, max_bytes: int) -> JSONResponse:
    """The 413 body, in the envelope of the protocol the caller spoke."""
    return JSONResponse(
        error_body(path, f"request body is larger than {max_bytes} bytes", "request_too_large"),
        status_code=413,
    )


def install(
    app: FastAPI,
    *,
    api_key: str | None,
    cors: Iterable[str] | str | None,
    max_body_bytes: int | None = None,
) -> None:
    """Guard `app` with a body cap, CORS and/or an API key.

    `max_body_bytes=None` uses the shared default; `0` disables the cap, which
    only a caller that enforces its own bound should ask for.
    """
    limit = MAX_BODY_BYTES if max_body_bytes is None else int(max_body_bytes)
    if limit > 0:
        # Added before the key middleware, so the key is checked first: that is a
        # header comparison, while the body cap may have to read the stream.
        app.add_middleware(_BodyLimit, max_bytes=limit)
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
