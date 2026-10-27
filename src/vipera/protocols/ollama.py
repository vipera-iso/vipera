"""The Ollama-compatible protocol: ``/api/*``.

Only translation: every route ends up in the same engine the OpenAI layer
uses, so a client written for Ollama can point at Vipera and keep its
``/api/tags``, ``/api/chat`` and ``/api/embed`` calls. Errors use Ollama's
``{"error": "..."}`` shape rather than FastAPI's ``{"detail": ...}``, because
that is what those clients read.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from vipera.errors import log_failure
from vipera.keepalive import parse_keep_alive
from vipera.optimize.shaping import Shaping
from vipera.pool import UnknownModelError
from vipera.protocols.gateway import Gateway
from vipera.protocols.limits import MAX_EMBED_CHARS, MAX_EMBED_INPUTS, MAX_PROMPT_CHARS
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

#: Ollama reports a model that is never unloaded with this timestamp.
_NEVER = "9999-12-31T23:59:59Z"

#: Shown by /api/show when the registry has no template for the model.
_DEFAULT_TEMPLATE = "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}"

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
    keep_alive: str | int | float | None = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    system: str | None = None
    stream: bool = True
    options: OllamaOptions | None = None
    format: str | dict | None = None
    keep_alive: str | int | float | None = None


class OllamaShowRequest(BaseModel):
    model: str
    verbose: bool = False


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


def _backend_failure(exc: Exception, operation: str) -> JSONResponse:
    """Report a failed engine call in Ollama's shape, and record why.

    An escaped exception became a bodyless 500: Ollama clients read
    ``{"error": "..."}`` and got neither the shape nor the cause.
    """
    log_failure(operation, exc)
    return _error(502, str(exc) or exc.__class__.__name__)


def _details(spec, meta: dict) -> dict:
    """The model-details block Ollama clients read from tags/ps/show."""
    return {
        "parent_model": "",
        "format": spec.format if spec is not None else "gguf",
        "family": meta.get("architecture") or "",
        "families": [meta["architecture"]] if meta.get("architecture") else None,
        "parameter_size": _parameter_size(meta.get("n_params")),
        "quantization_level": meta.get("quant") or "",
    }


def _model_info(alias: str, meta: dict) -> dict:
    """GGUF metadata in Ollama's ``model_info`` shape.

    Ollama prefixes per-architecture keys with the architecture name. Vipera
    only has the analyzer's summary, so the keys it can populate are mapped;
    everything else stays absent rather than being invented.
    """
    arch = meta.get("architecture") or "unknown"
    info: dict[str, Any] = {
        "general.architecture": arch,
        "general.name": alias,
    }
    keys = {
        "context_length": "context_length",
        "n_layers": "block_count",
        "n_heads": "attention.head_count",
        "n_kv_heads": "attention.head_count_kv",
        "embedding_length": "embedding_length",
        "n_params": "parameter_count",
    }
    for source, target in keys.items():
        value = meta.get(source)
        if value:
            info[f"{arch}.{target}"] = value
    return info


def _capabilities(meta: dict) -> list[str]:
    """What this gateway can do with the model. Text generation only, for now."""
    return ["completion"]


def _resolve_alias(gateway: Gateway, name: str) -> str | None:
    """Match an alias exactly, or after stripping Ollama's implicit ``:latest``."""
    if name in gateway.ids():
        return name
    if name.endswith(":latest"):
        stripped = name[: -len(":latest")]
        if stripped in gateway.ids():
            return stripped
    return None


def _keep_alive(value) -> tuple[float | None, JSONResponse | None]:
    """Parse ``keep_alive`` before a model is loaded; 400 on bad input."""
    try:
        return parse_keep_alive(value), None
    except ValueError as exc:
        return None, _error(400, str(exc))


def _load_failure(name: str, exc: Exception) -> JSONResponse:
    """A load that did not finish is a 503 with its reason, not a bare 500."""
    log_failure(f"model load ({name})", exc)
    return _error(503, f"model failed to load: {exc}")


async def _engine(gateway: Gateway, name: str) -> tuple[Any, JSONResponse | None]:
    """Resolve a model name, accepting Ollama's ':latest' tag suffix.

    In pool mode this is where a model is loaded, so a failing load belongs in
    the response rather than escaping as an unhandled exception.
    """
    try:
        return await gateway.engine_for(name), None
    except UnknownModelError:
        pass
    except Exception as exc:
        return None, _load_failure(name, exc)
    if name.endswith(":latest"):
        try:
            return await gateway.engine_for(name[: -len(":latest")]), None
        except UnknownModelError:
            pass
        except Exception as exc:
            return None, _load_failure(name, exc)
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


async def _shape(gateway: Gateway, active, messages: list[dict], kwargs: dict):
    """Fit the request into the loaded context; None when the layer is off.

    Off the event loop: prepare() may measure with the engine's own tokenizer,
    which on the server backend is a blocking HTTP round trip.
    """
    if gateway.optimizer is None:
        return None
    return await asyncio.to_thread(
        lambda: gateway.optimizer.prepare(
            active,
            messages,
            max_tokens=int(kwargs.get("max_tokens") or 0),
            tools=kwargs.get("tools"),
        )
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
                    "details": _details(spec, meta),
                }
            )
        return {"models": models}

    @app.get("/api/ps")
    def ps():
        expiries = gateway.expiries()
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
                    "details": {
                        "format": spec.format if spec is not None else "gguf",
                        "quantization_level": meta.get("quant") or "",
                    },
                    # Ollama reports a far-future stamp for a model that will
                    # never be unloaded; the pool reports None for that case.
                    "expires_at": expiries.get(alias) or _NEVER,
                }
            )
        return {"models": models}

    @app.post("/api/show")
    def show(req: OllamaShowRequest):
        """Describe an installed model without loading its weights."""
        alias = _resolve_alias(gateway, req.model)
        if alias is None:
            available = ", ".join(gateway.ids()) or "none"
            return _error(404, f"model '{req.model}' not found (available: {available})")
        spec = get_model(alias)
        meta = dict(spec.meta) if spec is not None else {}
        return {
            "modified_at": (spec.added if spec is not None else "") or _now(),
            "details": _details(spec, meta),
            "model_info": _model_info(alias, meta),
            "capabilities": _capabilities(meta),
            "template": meta.get("template") or _DEFAULT_TEMPLATE,
            "parameters": meta.get("parameters") or "",
            "license": meta.get("license") or "",
        }

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
        ttl, problem = _keep_alive(req.keep_alive)
        if problem is not None:
            return problem
        # Prompt size is a property of the request, so it is settled before a
        # model load is paid for; only what needs the engine comes after.
        messages = _messages(req.messages)
        if sum(len(m.get("content") or "") for m in messages) > MAX_PROMPT_CHARS:
            return _error(400, "prompt too large")

        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        gateway.keep_alive(active, ttl)
        lease = Lease(lambda: gateway.release(active))
        streaming = False
        try:
            kwargs = _chat_kwargs(req.options)
            if req.tools:
                kwargs["tools"] = req.tools
            if response_format is not None:
                kwargs["response_format"] = response_format
            shaping = await _shape(gateway, active, messages, kwargs)
            if shaping is not None:
                messages = shaping.messages

            started = time.perf_counter()
            if not req.stream:
                try:
                    result, counts = await run_leased(
                        gateway.lock, lease, _nonstream_result, active, messages, kwargs
                    )
                except Exception as exc:
                    return _backend_failure(exc, "chat")
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
        ttl, problem = _keep_alive(req.keep_alive)
        if problem is not None:
            return problem
        # Same bound as /api/chat, and for the same reason: a prompt the backend
        # cannot swallow must not first pay for a model load.
        if len(req.prompt) + len(req.system or "") > MAX_PROMPT_CHARS:
            return _error(400, "prompt too large")

        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        gateway.keep_alive(active, ttl)
        lease = Lease(lambda: gateway.release(active))
        streaming = False
        try:
            messages = _messages([OllamaMessage(role="user", content=req.prompt)], req.system)
            kwargs = _chat_kwargs(req.options)
            if response_format is not None:
                kwargs["response_format"] = response_format
            shaping = await _shape(gateway, active, messages, kwargs)
            if shaping is not None:
                messages = shaping.messages

            started = time.perf_counter()
            if not req.stream:
                try:
                    result, counts = await run_leased(
                        gateway.lock, lease, _nonstream_result, active, messages, kwargs
                    )
                except Exception as exc:
                    return _backend_failure(exc, "chat")
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
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        if not texts:
            return _error(400, "input must not be empty")
        if len(texts) > MAX_EMBED_INPUTS:
            return _error(400, f"too many inputs (max {MAX_EMBED_INPUTS})")
        if sum(len(text) for text in texts) > MAX_EMBED_CHARS:
            return _error(400, "input too large")
        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        lease = Lease(lambda: gateway.release(active))
        try:
            if not _supports_embeddings(active):
                return _error(501, _EMBEDDINGS_OFF)
            started = time.perf_counter()
            try:
                vectors, tokens = await run_leased(
                    gateway.lock, lease, _embed_and_count, active, texts
                )
            except Exception as exc:
                return _backend_failure(exc, "embed")
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
        if len(req.prompt) > MAX_EMBED_CHARS:
            return _error(400, "prompt too large")
        active, problem = await _engine(gateway, req.model)
        if problem is not None:
            return problem
        lease = Lease(lambda: gateway.release(active))
        try:
            if not _supports_embeddings(active):
                return _error(501, _EMBEDDINGS_OFF)
            try:
                vectors = await run_leased(gateway.lock, lease, active.embed, [req.prompt])
            except Exception as exc:
                return _backend_failure(exc, "embeddings")
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


async def _stream_chat(
    gateway: Gateway,
    active: Any,
    model: str,
    messages: list[dict],
    kwargs: dict[str, Any],
    started: float,
    shaping: Shaping | None = None,
    lease: Lease | None = None,
) -> AsyncIterator[str]:
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
        log_failure("chat stream", exc)
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
    gateway: Gateway,
    active: Any,
    model: str,
    messages: list[dict],
    kwargs: dict[str, Any],
    started: float,
    shaping: Shaping | None = None,
    lease: Lease | None = None,
) -> AsyncIterator[str]:
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
        log_failure("generate stream", exc)
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
