"""The in-process backend: llama-cpp-python drives llama.cpp in this process."""

from __future__ import annotations

import gc
from typing import Any

from vipera import config
from vipera.backends.loading import check_host_ram, load_attempts
from vipera.backends.protocol import CHAT_PARAMS, DEFAULT_SAMPLING
from vipera.errors import BackendError
from vipera.planner import Plan

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


class Engine:
    def __init__(self, plan: Plan, model_path: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise BackendError(
                "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
            ) from None

        self.plan = plan
        self._model_path = model_path
        self._Llama = Llama
        self._closed = False
        # llama.cpp has to build the context for embeddings up front; there is
        # no cheap way to enable it later, so it is opt-in through the config.
        self.supports_embeddings = bool(config.load_config().get("embeddings", False))

        check_host_ram(plan)

        # Stage the fallback instead of jumping straight to CPU-only: first walk
        # the offload down one layer at a time, then give up the placement, then
        # the context and finally the KV type. Callers such as Hermes require a
        # 64K context floor, and quantized KV is the only way that fits in RAM on
        # this class of machine, so dropping to 2048 must stay the last resort.
        attempts = load_attempts(plan)
        last_exc: Exception | None = None
        for i, (candidate, label) in enumerate(attempts):
            try:
                self.llm = self._load(candidate)
                self.plan = candidate
                if i > 0:
                    print(f"\n[!] Warning: failed to load with GPU. Falling back to {label}...")
                if i > 0 and candidate.context < plan.context:
                    print(
                        f"[!] Warning: requested ctx={plan.context} will NOT be met "
                        f"(using {candidate.context})."
                    )
                return
            except Exception as exc:  # try the next rung
                last_exc = exc

        raise BackendError(f"Failed to initialize model even on CPU: {last_exc}") from last_exc

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
                embedding=self.supports_embeddings,
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in the order they were given."""
        if not self.supports_embeddings:
            raise BackendError(
                "the model was not loaded with embedding support; "
                "set `embeddings true` in the config and restart"
            )
        result = self.llm.create_embedding(texts)
        return [list(item["embedding"]) for item in result["data"]]

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any):
        """Run a chat completion, forwarding every supported OpenAI parameter.

        Only keys in ``CHAT_PARAMS`` reach llama.cpp. ``None`` values are
        treated as "not set" so the backend keeps its own defaults for
        optional knobs such as ``seed`` or ``logit_bias``.
        """
        params = {k: v for k, v in kw.items() if k in CHAT_PARAMS and v is not None}
        for key, default in DEFAULT_SAMPLING.items():
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
