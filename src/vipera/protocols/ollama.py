"""The Ollama-compatible protocol: ``/api/*``.

Only translation: every route ends up in the same engine the OpenAI layer
uses, so a client written for Ollama can point at Vipera and keep its
``/api/tags``, ``/api/chat`` and ``/api/embed`` calls. Errors use Ollama's
``{"error": "..."}`` shape rather than FastAPI's ``{"detail": ...}``, because
that is what those clients read.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from vipera.pool import UnknownModelError
from vipera.protocols.gateway import Gateway
from vipera.protocols.streaming import (
    Lease,
    iter_chunks,
    leased_iterator,
    run_leased,
)
from vipera.registry import get_model

NAME = "ollama"

# The compatibility level this layer implements. Ollama clients parse the
# version as semver, so build metadata is used instead of a made-up release.
VERSION = "0.5.0+vipera"

_MAX_PROMPT_CHARS = 500_000

_OPTION_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "num_predict": "max_tokens",
    "repeat_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


class OllamaMessage(BaseModel):
    role: str
    content: str = ""
    images: list[str] | None = None
    tool_calls: list[dict] | None = None


class OllamaOptions(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    num_predict: int | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    num_ctx: int | None = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = True
    options: OllamaOptions | None = None
    tools: list[dict] | None = None
    format: str | dict | None = None
    keep_alive: str | int | None = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    system: str | None = None
    stream: bool = True
    options: OllamaOptions | None = None
    format: str | dict | None = None
    keep_alive: str | int | None = None


class OllamaEmbedRequest(BaseModel):
    model: str
    input: str | list[str]
    truncate: bool = True
    options: dict | None = None


class OllamaLegacyEmbedRequest(BaseModel):
    model: str
    prompt: str


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


async def _engine(gateway: Gateway, name: str) -> tuple[Any, JSONResponse | None]:
    """Resolve a model name, accepting Ollama's ':latest' tag suffix."""
    try:
        return await gateway.engine_for(name), None
    except UnknownModelError:
        pass
    if name.endswith(":latest"):
        try:
            return await gateway.engine_for(name[: -len(":latest")]), None
        except UnknownModelError:
            pass
    available = ", ".join(gateway.ids()) or "none"
    return None, _error(404, f"model '{name}' not found (available: {available})")


def _chat_kwargs(options: OllamaOptions | None) -> dict:
    if options is None:
        return {}
    kwargs = {}
    for source, target in _OPTION_MAP.items():
        value = getattr(options, source)
        if value is not None:
            kwargs[target] = value
    if options.stop:
        kwargs["stop"] = options.stop
    return kwargs


def _response_format(value: str | dict | None) -> dict | None:
    """Map Ollama's `format` onto the response_format the engine understands.

    llama-cpp-python constrains `{"type": "json_object"}` and reads an optional
    sibling `schema`; that is also accepted by llama-server, so a JSON-schema
    `format` keeps its shape instead of being silently discarded.
    """
    if value == "json":
        return {"type": "json_object"}
    if isinstance(value, dict):
        return {"type": "json_object", "schema": value}
    return None


def _prompt_tokens(active, messages: list[dict]) -> int:
    """Best-effort prompt count when the backend does not report usage."""
    try:
        return sum(len(active.tokenize(message.get("content") or "")) for message in messages)
    except Exception:
        return 0


def _messages(messages: list[OllamaMessage], system: str | None = None) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        item: dict = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = message.tool_calls
        out.append(item)
    return out


def _ollama_tool_calls(calls: list[dict] | None) -> list[dict] | None:
    """OpenAI tool calls carry arguments as a JSON string; Ollama as an object."""
    if not calls:
        return None
    out = []
    for call in calls:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                function["arguments"] = {}
        out.append({"function": function})
    return out


def _with_tool_calls(message: dict) -> dict:
    calls = _ollama_tool_calls(message.get("tool_calls"))
    if calls:
        message = {**message, "tool_calls": calls}
    return message


def _supports_embeddings(engine) -> bool:
    return bool(getattr(engine, "supports_embeddings", False))


def _shape(gateway: Gateway, active, messages: list[dict], kwargs: dict):
    """Fit the request into the loaded context; None when the layer is off."""
    if gateway.optimizer is None:
        return None
    return gateway.optimizer.prepare(
        active,
        messages,
        max_tokens=int(kwargs.get("max_tokens") or 0),
        tools=kwargs.get("tools"),
    )


def _count(gateway: Gateway, messages: list[dict], kwargs: dict, payload, shaping) -> None:
    """Record prefix reuse for one finished request."""
    if gateway.optimizer is not None:
        gateway.optimizer.note(
            messages, tools=kwargs.get("tools"), payload=payload, shaping=shaping
        )


def mount(app: FastAPI, gateway: Gateway) -> None:
    @app.get("/api/version")
    def version():
        return {"version": VERSION}

    @app.get("/api/tags")
    def tags():
        models = []
        for alias in gateway.ids():
            spec = get_model(alias)
            meta = dict(spec.meta) if spec is not None else {}
            path = spec.path if spec is not None else None
            size = meta.get("size_bytes")
            if size is None and path is not None:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = None
            models.append(
                {
                    "name": alias,
                    "model": alias,
                    "modified_at": (spec.added if spec is not None else "") or _now(),
                    "size": int(size or 0),
                    "digest": hashlib.sha256(alias.encode()).hexdigest(),
                    "details": {
                        "parent_model": "",
                        "format": spec.format if spec is not None else "gguf",
                        "family": meta.get("architecture") or "",
                        "families": [meta["architecture"]] if meta.get("architecture") else None,
                        "parameter_size": _parameter_size(meta.get("n_params")),
                        "quantization_level": meta.get("quant") or "",
                    },
                }
            )
        return {"models": models}

    @app.get("/api/ps")
    def ps():
        models = []
        for alias in gateway.loaded_ids():
            spec = get_model(alias)
            meta = dict(spec.meta) if spec is not None else {}
            models.append(
                {
                    "name": alias,
                    "model": alias,
                    "size": int(meta.get("size_bytes") or 0),
                    "digest": hashlib.sha256(alias.encode()).hexdigest(),
                    "details": {"format": "gguf", "quantization_level": meta.get("quant") or ""},
                    "expires_at": "9999-12-31T23:59:59Z",
                }
            )
        return {"models": models}

    @app.post("/api/chat")
    async def chat(req: OllamaChatRequest):
        if not req.messages:
            return _error(400, "messages must not be empty")
        if any(message.images for message in req.messages):
            # Accepting the field and dropping the images would answer a
            # question about a picture the model never saw.
            return _error(
                400,
                "image input is not supported: this gateway serves text-only models",
            )
        response_format = _response_format(req.format)
        if req.format is not None and response_format is None:
            return _error(400, "format must be 'json' or a JSON schema object")

        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        lease = Lease(lambda: gateway.release(active))
        streaming = False
        try:
            messages = _messages(req.messages)
            if sum(len(m.get("content") or "") for m in messages) > _MAX_PROMPT_CHARS:
                return _error(400, "prompt too large")
            kwargs = _chat_kwargs(req.options)
            if req.tools:
                kwargs["tools"] = req.tools
            if response_format is not None:
                kwargs["response_format"] = response_format
            shaping = _shape(gateway, active, messages, kwargs)
            if shaping is not None:
                messages = shaping.messages

            started = time.perf_counter()
            if not req.stream:
                result, counts = await run_leased(
                    gateway.lock, lease, _nonstream_result, active, messages, kwargs
                )
                _count(gateway, messages, kwargs, result, shaping)
                elapsed = time.perf_counter() - started
                choice = (result.get("choices") or [{}])[0]
                message = _with_tool_calls(dict(choice.get("message") or {}))
                message.setdefault("role", "assistant")
                message.setdefault("content", "")
                return {
                    "model": req.model,
                    "created_at": _now(),
                    "message": message,
                    "done": True,
                    "done_reason": _done_reason(choice.get("finish_reason")),
                    "total_duration": _ns(elapsed),
                    "load_duration": 0,
                    "prompt_eval_count": counts[0],
                    "prompt_eval_duration": 0,
                    "eval_count": counts[1],
                    "eval_duration": _ns(elapsed),
                }

            response = StreamingResponse(
                _stream_chat(gateway, active, req.model, messages, kwargs, started, shaping, lease),
                media_type="application/x-ndjson",
                background=BackgroundTask(lease.release_if_unstarted),
            )
            streaming = True
            return response
        finally:
            if not streaming:
                lease.release_if_unstarted()

    @app.post("/api/generate")
    async def generate(req: OllamaGenerateRequest):
        response_format = _response_format(req.format)
        if req.format is not None and response_format is None:
            return _error(400, "format must be 'json' or a JSON schema object")

        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        lease = Lease(lambda: gateway.release(active))
        streaming = False
        try:
            messages = _messages([OllamaMessage(role="user", content=req.prompt)], req.system)
            kwargs = _chat_kwargs(req.options)
            if response_format is not None:
                kwargs["response_format"] = response_format
            shaping = _shape(gateway, active, messages, kwargs)
            if shaping is not None:
                messages = shaping.messages

            started = time.perf_counter()
            if not req.stream:
                result, counts = await run_leased(
                    gateway.lock, lease, _nonstream_result, active, messages, kwargs
                )
                _count(gateway, messages, kwargs, result, shaping)
                elapsed = time.perf_counter() - started
                choice = (result.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                return {
                    "model": req.model,
                    "created_at": _now(),
                    "response": content,
                    "done": True,
                    "done_reason": _done_reason(choice.get("finish_reason")),
                    "total_duration": _ns(elapsed),
                    "load_duration": 0,
                    "prompt_eval_count": counts[0],
                    "prompt_eval_duration": 0,
                    "eval_count": counts[1],
                    "eval_duration": _ns(elapsed),
                }

            response = StreamingResponse(
                _stream_generate(
                    gateway, active, req.model, messages, kwargs, started, shaping, lease
                ),
                media_type="application/x-ndjson",
                background=BackgroundTask(lease.release_if_unstarted),
            )
            streaming = True
            return response
        finally:
            if not streaming:
                lease.release_if_unstarted()

    @app.post("/api/embed")
    async def embed(req: OllamaEmbedRequest):
        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        lease = Lease(lambda: gateway.release(active))
        try:
            if not _supports_embeddings(active):
                return _error(501, _EMBEDDINGS_OFF)
            texts = [req.input] if isinstance(req.input, str) else list(req.input)
            if not texts:
                return _error(400, "input must not be empty")
            started = time.perf_counter()
            vectors, tokens = await run_leased(gateway.lock, lease, _embed_and_count, active, texts)
            elapsed = time.perf_counter() - started
            return {
                "model": req.model,
                "embeddings": vectors,
                "total_duration": _ns(elapsed),
                "load_duration": 0,
                "prompt_eval_count": tokens,
            }
        finally:
            lease.release_if_unstarted()

    @app.post("/api/embeddings")
    async def legacy_embeddings(req: OllamaLegacyEmbedRequest):
        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        lease = Lease(lambda: gateway.release(active))
        try:
            if not _supports_embeddings(active):
                return _error(501, _EMBEDDINGS_OFF)
            vectors = await run_leased(gateway.lock, lease, active.embed, [req.prompt])
            return {"embedding": vectors[0]}
        finally:
            lease.release_if_unstarted()


def _nonstream_result(active, messages: list[dict], kwargs: dict):
    """Run one non-streaming turn, returning the result and Ollama token counts."""
    result = active.chat(messages, False, **kwargs)
    choice = (result.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    return result, _usage_counts(result.get("usage"), content, active, messages)


def _embed_and_count(active, texts: list[str]):
    """Embed texts and count their prompt tokens on the same worker thread."""
    vectors = active.embed(texts)
    tokens = 0
    with contextlib.suppress(Exception):
        tokens = sum(len(active.tokenize(text)) for text in texts)
    return vectors, tokens


def _usage_counts(usage, text: str, active, messages: list[dict]) -> tuple[int, int]:
    """Ollama counts, falling back for backends that report no prompt usage."""
    prompt_tokens, completion_tokens = _counts(usage, text, active)
    if prompt_tokens <= 0:
        prompt_tokens = _prompt_tokens(active, messages)
    return prompt_tokens, completion_tokens


_EMBEDDINGS_OFF = (
    "embeddings are disabled. Enable them with `vipera system config embeddings true` "
    "and restart the gateway."
)


def _done_reason(finish_reason: str | None) -> str:
    return "length" if finish_reason == "length" else "stop"


def _parameter_size(n_params) -> str:
    try:
        value = float(n_params)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    return f"{value / 1e9:.1f}B"


async def _stream_chat(gateway, active, model, messages, kwargs, started, shaping=None, lease=None):
    text = ""
    usage = None
    finish_reason = None
    tool_calls: dict[int, dict] = {}
    try:
        chunks = iter_chunks(
            gateway.lock,
            _stream_factory(lease, lambda: active.chat(messages, stream=True, **kwargs)),
            on_start=lease.start if lease is not None else None,
        )
        async for chunk in chunks:
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                for call in delta.get("tool_calls") or []:
                    _accumulate_tool_call(tool_calls, call)
                piece = delta.get("content") or ""
                if not piece:
                    continue
                text += piece
                yield _line(
                    {
                        "model": model,
                        "created_at": _now(),
                        "message": {"role": "assistant", "content": piece},
                        "done": False,
                    }
                )
    except Exception as exc:
        yield _line({"error": str(exc)})
        return

    calls = _ollama_tool_calls(list(tool_calls.values()))
    if calls:
        yield _line(
            {
                "model": model,
                "created_at": _now(),
                "message": {"role": "assistant", "content": "", "tool_calls": calls},
                "done": False,
            }
        )
    _count(gateway, messages, kwargs, {"usage": usage} if usage else None, shaping)
    elapsed = time.perf_counter() - started
    yield _line(
        _final_body(
            model,
            usage,
            finish_reason,
            elapsed,
            text,
            active,
            key="message",
            value={"role": "assistant", "content": ""},
        )
    )


async def _stream_generate(
    gateway, active, model, messages, kwargs, started, shaping=None, lease=None
):
    text = ""
    usage = None
    finish_reason = None
    try:
        chunks = iter_chunks(
            gateway.lock,
            _stream_factory(lease, lambda: active.chat(messages, stream=True, **kwargs)),
            on_start=lease.start if lease is not None else None,
        )
        async for chunk in chunks:
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                piece = (choice.get("delta") or {}).get("content") or ""
                if not piece:
                    continue
                text += piece
                yield _line(
                    {
                        "model": model,
                        "created_at": _now(),
                        "response": piece,
                        "done": False,
                    }
                )
    except Exception as exc:
        yield _line({"error": str(exc)})
        return

    _count(gateway, messages, kwargs, {"usage": usage} if usage else None, shaping)
    elapsed = time.perf_counter() - started
    yield _line(
        _final_body(
            model,
            usage,
            finish_reason,
            elapsed,
            text,
            active,
            key="response",
            value="",
        )
    )


def _stream_factory(lease, factory):
    """Release the engine lease when the streamed generator finishes or closes."""
    return leased_iterator(lease, factory) if lease is not None else factory


def _final_body(model, usage, finish_reason, elapsed, text, active, *, key, value) -> dict:
    prompt_tokens, completion_tokens = _counts(usage, text, active)
    return {
        "model": model,
        "created_at": _now(),
        key: value,
        "done": True,
        "done_reason": _done_reason(finish_reason),
        "total_duration": _ns(elapsed),
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": completion_tokens,
        "eval_duration": _ns(elapsed),
    }


def _counts(usage: dict | None, text: str, active) -> tuple[int, int]:
    """Usage when the backend reports it, a tokenizer count otherwise."""
    if usage:
        return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    try:
        return 0, len(active.tokenize(text))
    except Exception:
        return 0, 0


def _accumulate_tool_call(acc: dict[int, dict], call: dict) -> None:
    index = int(call.get("index") or 0)
    slot = acc.setdefault(index, {"function": {"name": "", "arguments": ""}})
    function = call.get("function") or {}
    if function.get("name"):
        slot["function"]["name"] += function["name"]
    if function.get("arguments"):
        slot["function"]["arguments"] += function["arguments"]
    if call.get("id") and not slot.get("id"):
        slot["id"] = call["id"]


def _line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"
