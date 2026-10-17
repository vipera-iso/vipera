import json
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


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
    lock = Lock()

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
            raise HTTPException(status_code=400, detail="messages must not be empty")

        messages = [m.model_dump() for m in req.messages]
        kwargs = dict(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            repeat_penalty=req.repeat_penalty,
        )

        if req.stream:
            def _stream():
                with lock:
                    try:
                        for chunk in engine.chat(messages, stream=True, **kwargs):
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        err = {"error": {"message": str(e)}}
                        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    finally:
                        yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        def _generate():
            with lock:
                return engine.chat(messages, stream=False, **kwargs)

        return await run_in_threadpool(_generate)

    return app
