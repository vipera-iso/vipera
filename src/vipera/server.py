import asyncio
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

_MAX_TOKENS = 32_768
_MAX_MESSAGES = 200
_MAX_PROMPT_CHARS = 500_000

_SENTINEL = object()


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(..., min_length=1)
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    repeat_penalty: float = 1.1


def build_app(engine, model_name: str) -> FastAPI:
    app = FastAPI(title="Vipera Gateway", version="1.0.0")
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
            raise HTTPException(
                404, f"model '{req.model}' not loaded (active: '{model_name}')"
            )
        if len(req.messages) > _MAX_MESSAGES:
            raise HTTPException(400, f"too many messages (max {_MAX_MESSAGES})")
        if sum(len(m.content) for m in req.messages) > _MAX_PROMPT_CHARS:
            raise HTTPException(400, "prompt too large")
        if not 0 < req.max_tokens <= _MAX_TOKENS:
            raise HTTPException(400, f"max_tokens must be in [1, {_MAX_TOKENS}]")
        if not 0.0 <= req.temperature <= 2.0:
            raise HTTPException(400, "temperature must be in [0, 2]")
        if not 0.0 < req.top_p <= 1.0:
            raise HTTPException(400, "top_p must be in (0, 1]")

        messages = [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in req.messages]
        kwargs = dict(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            repeat_penalty=req.repeat_penalty,
        )

        if req.stream:
            async def _stream():
                # The engine's stream is a SYNC generator. Iterating it inline
                # would block the event loop for the whole generation, so pull
                # each chunk in a worker thread instead. If the client
                # disconnects, aclose() cancels us here; the pending to_thread
                # call returns and the sync iterator is GC'd, which closes it.
                async with lock:
                    iterator = engine.chat(messages, stream=True, **kwargs)
                    try:
                        while True:
                            chunk = await asyncio.to_thread(next, iterator, _SENTINEL)
                            if chunk is _SENTINEL:
                                break
                            payload = json.dumps(chunk, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    except Exception as exc:
                        err = {"error": {"message": str(exc)}}
                        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        async with lock:
            return await run_in_threadpool(engine.chat, messages, stream=False, **kwargs)

    return app
