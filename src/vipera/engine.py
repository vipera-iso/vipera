from dataclasses import replace
from typing import Any

from vipera.planner import Plan

_KV_TYPES = {"f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q8_0": 8}


class Engine:
    def __init__(self, plan: Plan, model_path: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python is not installed.")

        self.plan = plan
        self._model_path = model_path
        self._Llama = Llama

        try:
            self.llm = self._load(plan)
            return
        except Exception:
            pass

        print(f"\n[!] Warning: failed to load with GPU. Falling back to CPU...")
        safe_plan = replace(
            plan,
            gpu_layers=0,
            flash_attn=False,
            offload_kqv=False,
            kv_cache="f16",
            n_batch=128,
            context=min(plan.context, 2048),
        )

        try:
            self.llm = self._load(safe_plan)
            self.plan = safe_plan
        except Exception as e2:
            raise RuntimeError(f"Không thể khởi tạo model ngay cả trên CPU: {e2}") from e2

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
        return self.llm.create_chat_completion(
            messages=messages,
            stream=stream,
            temperature=kw.get("temperature", 0.7),
            max_tokens=kw.get("max_tokens", 2048),
            top_p=kw.get("top_p", 0.95),
            repeat_penalty=kw.get("repeat_penalty", 1.1),
        )
