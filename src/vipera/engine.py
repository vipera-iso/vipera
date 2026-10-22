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


def layer_ladder(layers: int) -> list[int]:
    """Offload counts to try for a plan that asks for ``layers``, best first.

    A split that is one or two layers too optimistic used to fall straight to
    ``gpu_layers=0``, which is the slowest outcome on the box this was measured
    on: 3.90 t/s CPU-only against 4.97 t/s at the last split that loaded
    (14B/Q4, 8K, GTX 1650). Each rung costs one failed context creation (~2 s
    for a 14B at 8K), and a failed attempt does not hand its VRAM back before
    the next one: after three failures on a 4 GB card even a 10-layer rung
    failed, which is how asking for 20 layers ended up on the CPU. The ladder is
    therefore short and its last rung is a quarter of the plan, not half — it
    exists to land anywhere other than CPU-only.
    """
    if layers <= 0:
        return []
    steps = (layers, layers - 1, layers - 2, max(2, layers // 4))
    return [n for n in dict.fromkeys(steps) if 0 < n <= layers]


def with_gpu_layers(plan: Plan, layers: int) -> Plan:
    """Same plan at a different offload depth, with a truthful note."""
    note = plan.note
    if note.startswith("Hybrid: "):
        total = note.split("/", 1)[1].split()[0]
        note = f"Hybrid: {layers}/{total} layers"
    elif note == "Full GPU":
        # A large card can plan a full offload and still fail to create the
        # context (the driver holds more than the weights). Reporting "Full
        # GPU" while running a reduced split would repeat the lie the ladder
        # exists to stop.
        note = f"Hybrid: {layers}/all layers"
    return replace(plan, gpu_layers=layers, note=note)


def load_attempts(plan: Plan) -> list[tuple[Plan, str]]:
    """Candidate plans for loading, most offload first, ending CPU-only."""
    attempts: list[tuple[Plan, str]] = [(plan, "planned split")]
    if plan.requires_server and plan.n_layers > 0:
        # The server ladder moves one expert layer at a time to the CPU, which
        # frees roughly 190-220 MB on the reference model. The last rung keeps
        # every expert on the CPU instead of failing the load entirely.
        rungs = (plan.moe_cpu_layers + 1, plan.moe_cpu_layers + 2, plan.n_layers)
        for cpu_layers in dict.fromkeys(rungs):
            if cpu_layers <= plan.moe_cpu_layers or cpu_layers > plan.n_layers:
                continue
            gpu_experts = plan.n_layers - cpu_layers
            attempts.append(
                (
                    replace(
                        plan,
                        moe_cpu_layers=cpu_layers,
                        note=f"MoE: {gpu_experts}/{plan.n_layers} expert layers on GPU",
                    ),
                    f"MoE, {gpu_experts}/{plan.n_layers} expert layers",
                )
            )
    else:
        hybrid_total = (
            plan.note.split("/", 1)[1].split()[0] if plan.note.startswith("Hybrid: ") else None
        )
        for layers in layer_ladder(plan.gpu_layers)[1:]:
            label = f"GPU, {layers}/{hybrid_total} layers" if hybrid_total else f"GPU, {layers}"
            attempts.append((with_gpu_layers(plan, layers), label))

    # CPU stages: keep the caller's context and KV type first, then reduce.
    # Clearing the MoE fields matters: -ncmoe with -ngl 0 is not a valid
    # CPU-only plan.
    cpu_plan = replace(
        plan,
        gpu_layers=0,
        offload_kqv=False,
        n_batch=min(plan.n_batch, 128),
        moe_cpu_layers=-1,
        n_ubatch=0,
    )
    reduced_plan = replace(cpu_plan, context=min(plan.context, 2048))
    safe_plan = replace(reduced_plan, kv_cache="f16", flash_attn=False)
    attempts.append((cpu_plan, f"CPU, ctx={cpu_plan.context}, kv={cpu_plan.kv_cache}"))
    attempts.append((reduced_plan, f"CPU, ctx={reduced_plan.context}, kv={reduced_plan.kv_cache}"))
    attempts.append((safe_plan, f"CPU safe mode, ctx={safe_plan.context}, kv={safe_plan.kv_cache}"))
    return attempts


def resolve_backend(plan: Plan, backend: str) -> tuple[str, Plan]:
    """Return a backend that can actually execute ``plan``.

    llama-cpp-python has no ``n_cpu_moe`` argument, so a MoE plan is silently
    ignored by the in-process engine (and its classic ``gpu_layers`` split is
    the only safe fallback). If a llama-server with ``-ncmoe`` is available,
    switch to it; otherwise clear the MoE fields so the banner and bench report
    what is really running.
    """
    if backend == "server" or not plan.requires_server:
        return backend, plan

    try:
        from vipera.optimizer import OptimizerError, find_llama_tool
        from vipera.server_backend import supports_n_cpu_moe

        binary = find_llama_tool("server")
        supported = supports_n_cpu_moe(binary)
    except (OptimizerError, OSError):
        binary = None
        supported = False

    if supported:
        gpu_experts = plan.n_layers - plan.moe_cpu_layers
        print(
            "[i] MoE plan: using llama-server for expert placement "
            f"({gpu_experts}/{plan.n_layers} expert layers on GPU, "
            f"ubatch {plan.n_ubatch})."
        )
        return "server", plan

    where = f"{binary} has no -ncmoe" if binary else "llama-server was not found"
    print(
        f"[!] Warning: {where}; the MoE plan cannot run in-process and is "
        "falling back to the classic layer split. Install a newer llama-server "
        "to use expert offload."
    )
    return "inprocess", replace(plan, moe_cpu_layers=-1, n_ubatch=0)


def create_engine(plan: Plan, model_path: str, backend: str = "inprocess") -> ChatEngine:
    """Build the engine selected by the ``backend`` config key."""
    backend, plan = resolve_backend(plan, backend)
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

        raise RuntimeError(f"Failed to initialize model even on CPU: {last_exc}") from last_exc

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
