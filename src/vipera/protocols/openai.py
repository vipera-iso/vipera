"""The OpenAI-compatible protocol: chat, models and embeddings.

This is the default layer, so it keeps the request models and the exact
validation the gateway has always had; only how it reaches an engine changed
(it now goes through the Gateway handle rather than a closure over server.py).
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from starlette.background import BackgroundTask

from vipera.errors import log_failure
from vipera.pool import UnknownModelError
from vipera.protocols.gateway import Gateway
from vipera.protocols.limits import (
    MAX_EMBED_CHARS,
    MAX_EMBED_INPUTS,
    MAX_MESSAGES,
    MAX_PROMPT_CHARS,
    MAX_TOKENS,
)
from vipera.protocols.streaming import (
    Lease,
    iter_chunks,
    leased_iterator,
    relay_chunks,
    run_leased,
)

NAME = "openai"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # Assistant tool-call turns legitimately have no text content.
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _require_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls:
            raise ValueError("content is required unless the message carries tool_calls")
        return self


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(..., min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    # None means "llama.cpp's default" (1.0). Sending 1.1 here would put ~10%
    # of decode time into a sampler the client never asked for.
    repeat_penalty: float | None = None
    top_k: int | None = Field(None, ge=0)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    seed: int | None = Field(None, ge=0)
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(None, ge=0, le=20)
    response_format: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = None


def _error(status_code: int, message: str, *, error_type: str) -> JSONResponse:
    """OpenAI's error object: clients read ``error.message``, not ``detail``."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def _backend_failure(exc: Exception, operation: str) -> JSONResponse:
    """Report a failed engine call as a 502 that says why.

    Letting the exception escape produced a bodyless 500: the cause (the
    child's own 400, a dead process, a socket timeout) never reached the
    client, and nothing recorded it here either.
    """
    log_failure(operation, exc)
    return _error(502, str(exc) or exc.__class__.__name__, error_type="backend_error")


async def _engine(gateway: Gateway, name: str | None):
    """Resolve the engine for a request: ``(engine, None)`` or ``(None, error)``.

    An unknown name stays a plain 404 (that shape predates this layer), but a
    load that did not finish is a backend failure and is reported as one: it
    used to escape as a bodyless 500.
    """
    try:
        return await gateway.engine_for(name), None
    except UnknownModelError as exc:
        raise HTTPException(
            404,
            f"model '{exc.name}' is not loaded (available: {', '.join(exc.available)})",
        ) from exc
    except Exception as exc:
        log_failure(f"model load ({name or 'default'})", exc)
        return None, _error(503, f"model failed to load: {exc}", error_type="backend_error")


def _supports_embeddings(engine) -> bool:
    return bool(getattr(engine, "supports_embeddings", False))


def _validate_chat(req: ChatRequest) -> None:
    """Reject an unusable request before any engine is acquired.

    In pool mode acquiring an engine plans and loads a GGUF, which takes
    minutes; a 400 that arrives after that load has already paid for it.
    """
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")
    if len(req.messages) > MAX_MESSAGES:
        raise HTTPException(400, f"too many messages (max {MAX_MESSAGES})")
    if sum(len(m.content or "") for m in req.messages) > MAX_PROMPT_CHARS:
        raise HTTPException(400, "prompt too large")
    if not 0 < req.max_tokens <= MAX_TOKENS:
        raise HTTPException(400, f"max_tokens must be in [1, {MAX_TOKENS}]")
    if not 0.0 <= req.temperature <= 2.0:
        raise HTTPException(400, "temperature must be in [0, 2]")
    if not 0.0 < req.top_p <= 1.0:
        raise HTTPException(400, "top_p must be in (0, 1]")


def _embedding_texts(req: EmbeddingRequest) -> list[str]:
    """Validate an embedding request and return its inputs, before a load."""
    if req.encoding_format == "base64":
        raise HTTPException(400, "encoding_format 'base64' is not supported")
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        raise HTTPException(400, "input must not be empty")
    if len(texts) > MAX_EMBED_INPUTS:
        raise HTTPException(400, f"too many inputs (max {MAX_EMBED_INPUTS})")
    if sum(len(t) for t in texts) > MAX_EMBED_CHARS:
        raise HTTPException(400, "input too large")
    return texts


def mount(app: FastAPI, gateway: Gateway) -> None:
    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "vipera"} for name in gateway.ids()
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        _validate_chat(req)
        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        lease = Lease(lambda: gateway.release(active))
        streaming = False
        try:
            # exclude_none keeps unset optional knobs out of the llama.cpp call
            # so the backend's own defaults apply (e.g. seed=None is not 0).
            messages = [m.model_dump(exclude_none=True) for m in req.messages]
            kwargs = req.model_dump(
                exclude={"model", "messages", "stream", "stream_options"},
                exclude_none=True,
            )
            # The in-process engine drops this (llama-cpp-python 0.3.x does not
            # report usage), but llama-server returns real usage when asked,
            # which is what turns the trailing choices:[] frame into data.
            if req.stream and req.stream_options and req.stream_options.include_usage:
                kwargs["stream_options"] = {"include_usage": True}

            # Fit the request into the context the model was loaded with. A
            # request that already fits is passed through untouched. prepare()
            # is off the event loop because it may measure with the engine's own
            # tokenizer, which on the server backend is a blocking round trip.
            shaping = None
            if gateway.optimizer is not None:
                shaping = await asyncio.to_thread(
                    lambda: gateway.optimizer.prepare(
                        active,
                        messages,
                        max_tokens=int(kwargs.get("max_tokens") or 0),
                        tools=kwargs.get("tools"),
                    )
                )
                messages = shaping.messages

            # A backend that already speaks this wire format (llama-server does)
            # can have its bytes forwarded unchanged: re-encoding every frame in
            # Python is pure overhead on the token path.
            raw_stream = (
                req.stream
                and bool(getattr(active, "passthrough_sse", False))
                and hasattr(active, "chat_sse")
            )

            if req.stream:

                async def _stream():
                    usage = None
                    try:
                        if raw_stream:
                            # One reader thread for the whole body: the bytes go
                            # out as they arrive, without a thread round trip
                            # per frame (measured 0.8% of decode time against
                            # 9.6% for the per-item path).
                            frames = relay_chunks(
                                gateway.lock,
                                lambda: active.chat_sse(messages, **kwargs),
                                on_start=lease.start,
                                on_close=lease.release,
                            )
                            async for frame in frames:
                                yield frame
                            if gateway.optimizer is not None:
                                gateway.optimizer.note(
                                    messages, tools=kwargs.get("tools"), shaping=shaping
                                )
                            return
                        chunks = iter_chunks(
                            gateway.lock,
                            leased_iterator(
                                lease,
                                lambda: active.chat(messages, stream=True, **kwargs),
                            ),
                            on_start=lease.start,
                        )
                        async for chunk in chunks:
                            if isinstance(chunk, dict) and chunk.get("usage"):
                                usage = chunk["usage"]
                                # Never forward usage inside a chunk: the OpenAI
                                # shape is one trailing choices:[] frame, emitted
                                # below. llama-server already terminates with
                                # such a frame, so forwarding it as well would
                                # duplicate usage for the client.
                                chunk = {k: v for k, v in chunk.items() if k != "usage"}
                                if not chunk.get("choices"):
                                    continue
                            payload = json.dumps(chunk, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    except Exception as exc:
                        # engine.chat() itself may raise before the first chunk;
                        # report that as an SSE error event instead of aborting
                        # the response body mid-stream.
                        log_failure("chat.completions stream", exc)
                        err = {"error": {"message": str(exc)}}
                        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                        # The upstream frames already carried their own DONE (or
                        # none at all), so this path must not add another.
                        if raw_stream:
                            return
                    if gateway.optimizer is not None:
                        gateway.optimizer.note(
                            messages,
                            tools=kwargs.get("tools"),
                            payload={"usage": usage} if usage else None,
                            shaping=shaping,
                        )
                    # OpenAI emits a trailing choices:[] frame when the client
                    # asks for usage. Only possible when the backend reported
                    # any; llama-cpp-python 0.3.x does not, so it is omitted.
                    if (
                        req.stream_options is not None
                        and req.stream_options.include_usage
                        and usage is not None
                    ):
                        payload = json.dumps({"choices": [], "usage": usage}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"
                    yield "data: [DONE]\n\n"

                response = StreamingResponse(
                    _stream(),
                    media_type="text/event-stream",
                    background=BackgroundTask(lease.release_if_unstarted),
                )
                streaming = True
                return response

            try:
                result = await run_leased(
                    gateway.lock, lease, active.chat, messages, False, **kwargs
                )
            except Exception as exc:
                return _backend_failure(exc, "chat.completions")
            if gateway.optimizer is not None:
                gateway.optimizer.note(
                    messages, tools=kwargs.get("tools"), payload=result, shaping=shaping
                )
            return result
        finally:
            # Non-stream paths release here when the engine was never reached;
            # a started call releases from its own worker thread. A streaming
            # response is released by its iterator (or by the background task if
            # the client never consumed it).
            if not streaming:
                lease.release_if_unstarted()

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest):
        texts = _embedding_texts(req)
        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        lease = Lease(lambda: gateway.release(active))
        try:
            if not _supports_embeddings(active):
                raise HTTPException(
                    501,
                    "this backend was not started with embedding support. "
                    "Enable it with `vipera system config embeddings true` and restart "
                    "(the server backend needs a llama-server build with --embeddings "
                    "and --pooling).",
                )

            def _embed_and_count():
                vectors = active.embed(texts)
                # Count while the lease still protects the engine: tokenize
                # would touch a closed context if another model evicted it.
                tokens = sum(len(active.tokenize(t)) for t in texts)
                return vectors, tokens

            try:
                vectors, tokens = await run_leased(gateway.lock, lease, _embed_and_count)
            except Exception as exc:
                return _backend_failure(exc, "embeddings")
            alias = req.model or gateway.default_model() or "vipera"
            return {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector}
                    for i, vector in enumerate(vectors)
                ],
                "model": alias,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        finally:
            lease.release_if_unstarted()
