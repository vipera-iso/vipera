import asyncio
import contextlib
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from vipera import __version__

_MAX_TOKENS = 32_768
_MAX_MESSAGES = 200
_MAX_PROMPT_CHARS = 500_000

_SENTINEL = object()


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
    repeat_penalty: float = 1.1
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


def build_app(engine, model_name: str) -> FastAPI:
    app = FastAPI(title="Vipera Gateway", version=__version__)
    lock = asyncio.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "vipera"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        if not req.messages:
            raise HTTPException(400, "messages must not be empty")
        if req.model and req.model != model_name:
            raise HTTPException(404, f"model '{req.model}' not loaded (active: '{model_name}')")
        if len(req.messages) > _MAX_MESSAGES:
            raise HTTPException(400, f"too many messages (max {_MAX_MESSAGES})")
        if sum(len(m.content or "") for m in req.messages) > _MAX_PROMPT_CHARS:
            raise HTTPException(400, "prompt too large")
        if not 0 < req.max_tokens <= _MAX_TOKENS:
            raise HTTPException(400, f"max_tokens must be in [1, {_MAX_TOKENS}]")
        if not 0.0 <= req.temperature <= 2.0:
            raise HTTPException(400, "temperature must be in [0, 2]")
        if not 0.0 < req.top_p <= 1.0:
            raise HTTPException(400, "top_p must be in (0, 1]")

        # exclude_none keeps unset optional knobs out of the llama.cpp call so
        # the backend's own defaults apply (e.g. seed=None is not seed=0).
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        kwargs = req.model_dump(
            exclude={"model", "messages", "stream", "stream_options"},
            exclude_none=True,
        )
        # The in-process engine drops this (llama-cpp-python 0.3.x does not
        # report usage), but llama-server returns real usage when asked, which
        # is what turns the trailing choices:[] frame below into data.
        if req.stream and req.stream_options and req.stream_options.include_usage:
            kwargs["stream_options"] = {"include_usage": True}

        if req.stream:

            async def _stream():
                # The engine's stream is a SYNC generator. Iterating it inline
                # would block the event loop for the whole generation, so pull
                # each chunk in a worker thread instead.
                #
                # The lock is acquired explicitly (not via `async with`) so a
                # client disconnect cannot release it while a worker thread is
                # still inside next(): cancellation would otherwise let the
                # next request enter the engine concurrently with generation
                # that is still running.
                await lock.acquire()
                released = False
                iterator = None
                usage = None
                try:
                    try:
                        iterator = engine.chat(messages, stream=True, **kwargs)
                        while True:
                            task = asyncio.create_task(asyncio.to_thread(next, iterator, _SENTINEL))
                            try:
                                # shield: cancelling this request must not
                                # cancel the task that wraps the worker thread.
                                chunk = await asyncio.shield(task)
                            except asyncio.CancelledError:
                                # The worker is still inside next(); closing the
                                # generator now would raise "already executing"
                                # and release the lock too early. Wait for the
                                # thread, then close and release.
                                def _finish(_t, it=iterator):
                                    with contextlib.suppress(Exception):
                                        it.close()
                                    lock.release()

                                task.add_done_callback(_finish)
                                released = True
                                iterator = None
                                raise
                            if chunk is _SENTINEL:
                                break
                            if isinstance(chunk, dict) and chunk.get("usage"):
                                usage = chunk["usage"]
                            payload = json.dumps(chunk, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # engine.chat() itself may raise before the first
                        # chunk; report that as an SSE error event instead of
                        # aborting the response body mid-stream.
                        err = {"error": {"message": str(exc)}}
                        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
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
                finally:
                    if iterator is not None:
                        with contextlib.suppress(Exception):
                            iterator.close()
                    if not released:
                        lock.release()

            return StreamingResponse(_stream(), media_type="text/event-stream")

        # Same cancellation rule as the streaming path: do not hand the lock to
        # another request until the in-flight worker thread has returned.
        await lock.acquire()
        released = False
        try:
            task = asyncio.create_task(asyncio.to_thread(engine.chat, messages, False, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                task.add_done_callback(lambda _t: lock.release())
                released = True
                raise
        finally:
            if not released:
                lock.release()

    return app
