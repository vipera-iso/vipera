import gc
from dataclasses import replace
from typing import Any, Protocol

import psutil

from vipera.planner import Plan


class ChatEngine(Protocol):
    """What the commands need from either backend.

    `Engine` (in-process) and `ServerEngine` (llama-server child) both satisfy
    this; callers must not reach for `.llm`, which only the in-process one has.
    """

    plan: Plan

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any): ...

    def tokenize(self, text: str | bytes) -> list[int]: ...

    def close(self) -> None: ...


# ggml type ids accepted by llama.cpp for KV caches.
_KV_TYPES = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
    "iq4_nl": 20,
    "bf16": 30,
}

# Sampling and tool-calling parameters accepted by
# llama_cpp.Llama.create_chat_completion. Anything else is dropped so a typo in
# an HTTP request cannot turn into a TypeError from the backend.
_CHAT_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "max_tokens",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
        "tfs_z",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "grammar",
        "response_format",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
    }
)

_DEFAULT_SAMPLING = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
}


def check_host_ram(plan: Plan) -> None:
    """Refuse a plan whose host-side footprint clearly exceeds free RAM.

    An OOM here is an uncatchable SIGKILL, not a Python exception, so the
    GPU->CPU fallback never gets a chance to run. Shared by both backends: the
    llama-server child dies the same way.
    """
    if plan.ram_estimate_gb <= 0:
        return
    available_gb = psutil.virtual_memory().available / 1024**3
    if plan.ram_estimate_gb > available_gb + 0.5:
        raise RuntimeError(
            f"Not enough RAM for this plan: ~{plan.ram_estimate_gb:.1f} GB "
            f"needed on the host, {available_gb:.1f} GB available. "
            f"Lower --ctx, use kv_cache=q4_0, or close other applications."
        )


def create_engine(plan: Plan, model_path: str, backend: str = "inprocess") -> ChatEngine:
    """Build the engine selected by the ``backend`` config key."""
    if backend == "server":
        from vipera.server_backend import ServerEngine

        return ServerEngine(plan, model_path)
    return Engine(plan, model_path)


class Engine:
    def __init__(self, plan: Plan, model_path: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
            ) from None

        self.plan = plan
        self._model_path = model_path
        self._Llama = Llama
        self._closed = False

        check_host_ram(plan)

        gpu_exc: Exception | None = None
        try:
            self.llm = self._load(plan)
            return
        except Exception as exc:  # need to try CPU fallback
            gpu_exc = exc

        print("\n[!] Warning: failed to load with GPU. Falling back to CPU...")

        # Fall back in stages rather than jumping straight to f16/2048. Callers
        # such as Hermes require a 64K context floor, and quantized KV is the
        # only way that fits in RAM on this class of machine. Silently dropping
        # to 2048 would load "successfully" and then be useless.
        cpu_plan = replace(
            plan,
            gpu_layers=0,
            offload_kqv=False,
            n_batch=min(plan.n_batch, 128),
        )
        reduced_plan = replace(cpu_plan, context=min(plan.context, 2048))
        safe_plan = replace(reduced_plan, kv_cache="f16", flash_attn=False)

        attempts = [
            (cpu_plan, f"CPU, ctx={cpu_plan.context}, kv={cpu_plan.kv_cache}"),
            (reduced_plan, f"CPU, ctx={reduced_plan.context}, kv={reduced_plan.kv_cache}"),
            (safe_plan, f"CPU safe mode, ctx={safe_plan.context}, kv={safe_plan.kv_cache}"),
        ]

        last_exc: Exception = gpu_exc
        for i, (candidate, label) in enumerate(attempts):
            if i > 0:
                print(f"[!] Warning: fallback failed; retrying with {label}. ")
                if candidate.context < plan.context:
                    print(
                        f"[!] Warning: requested ctx={plan.context} will NOT be met "
                        f"(using {candidate.context})."
                    )
            try:
                self.llm = self._load(candidate)
                self.plan = candidate
                return
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"Failed to initialize model even on CPU: {last_exc}") from gpu_exc

    def _load(self, p: Plan):
        kv = _KV_TYPES.get(p.kv_cache)

        def _do(use_mlock: bool):
            return self._Llama(
                model_path=self._model_path,
                n_gpu_layers=p.gpu_layers,
                n_ctx=p.context,
                n_threads=p.cpu_threads,
                n_batch=p.n_batch,
                n_ubatch=min(p.n_batch, 512),
                flash_attn=p.flash_attn,
                offload_kqv=p.offload_kqv,
                type_k=kv,
                type_v=kv,
                use_mmap=True,
                use_mlock=use_mlock,
                verbose=False,
            )

        try:
            return _do(p.use_mlock)
        except Exception:
            if not p.use_mlock:
                raise
            return _do(False)

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any):
        """Run a chat completion, forwarding every supported OpenAI parameter.

        Only keys in ``_CHAT_PARAMS`` reach llama.cpp. ``None`` values are
        treated as "not set" so the backend keeps its own defaults for
        optional knobs such as ``seed`` or ``logit_bias``.
        """
        params = {k: v for k, v in kw.items() if k in _CHAT_PARAMS and v is not None}
        for key, default in _DEFAULT_SAMPLING.items():
            params.setdefault(key, default)
        return self.llm.create_chat_completion(messages=messages, stream=stream, **params)

    def tokenize(self, text: str | bytes) -> list[int]:
        """Tokenize with the loaded model's vocabulary.

        Callers must go through this instead of ``engine.llm.tokenize`` so the
        same code works against the server backend, which has no ``llm``.
        """
        data = text.encode() if isinstance(text, str) else text
        return self.llm.tokenize(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        llm = self.llm
        self.llm = None
        del llm
        gc.collect()
