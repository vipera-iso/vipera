from dataclasses import dataclass

from vipera.errors import PlannerError
from vipera.hardware import HardwareInfo


@dataclass
class Plan:
    gpu_layers: int
    cpu_threads: int
    context: int
    kv_cache: str
    flash_attn: bool
    n_batch: int
    offload_kqv: bool
    use_mlock: bool
    note: str
    # Rough host-RAM requirement: CPU-resident weights + CPU-resident KV plus
    # room for compute buffers. 0.0 means "not estimated" (callers skip the
    # preflight check), which keeps hand-built Plans in tests working.
    ram_estimate_gb: float = 0.0
    # KV cache size this plan expects, and how many layers actually keep KV.
    # 0.0 / 0 mean "not estimated" (hand-built Plans in tests stay valid).
    kv_cache_gb: float = 0.0
    kv_layers: int = 0
    # --n-cpu-moe value: the first N layers keep their routed experts on the
    # CPU. -1 means "no MoE plan"; 0 means every expert stays on the GPU.
    # The classic gpu_layers field stays untouched so an in-process fallback
    # still loads a split that fits.
    moe_cpu_layers: int = -1
    # llama.cpp ubatch for the server backend. 0 keeps the historical
    # min(n_batch, 512) behaviour.
    n_ubatch: int = 0
    n_layers: int = 0

    @property
    def requires_server(self) -> bool:
        """True when the plan needs llama-server flags the binding lacks."""
        return self.moe_cpu_layers >= 0


_VALID_PROFILES = ("auto", "cpu", "gpu", "lowmem")


def profile_names() -> tuple[str, ...]:
    """The planner profiles a config or CLI flag may name."""
    return _VALID_PROFILES


# Room left on the card for the CUDA/driver context and llama.cpp's compute
# buffers. This used to be a flat 1.0 GB (driver) + 0.3 GB (compute), which on a
# 4 GB card gave away four layers: measured on a GTX 1650 with 3.64 GB free, a
# 14B/Q4 model at 8K offloads 14 layers (3.37 GB of weights+KV) and 15 fail, so
# the real reserve is ~0.3 GB. A flat 1.3 GB reserve measured 5.0 t/s where 14
# layers measured 5.96 t/s. Scale it with the card instead: small cards get the
# measured figure, big cards still keep ~1 GB for their larger compute buffers.
_VRAM_RESERVE_BASE_GB = 0.3
_VRAM_RESERVE_FRACTION = 0.07
# Extra reserve when the plan cannot use flash attention. Without FA llama.cpp
# allocates a full KQ matrix per batch instead of streaming it, which on the same
# 4 GB card cuts the offload from 14 layers to 10 (0.9 GB of weights).
_NO_FA_RESERVE_GB = 0.9

# Bytes per element for each supported kv_cache dtype. The values for the
# original five are the rounded ones the VRAM budget was measured against
# (11 layers load, 12 fail, on a 4 GB GTX 1650); the newer dtypes use their
# exact ggml block size so a q5_0 plan does not inherit a q4_0 estimate.
_KV_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0,
    "q5_0": 0.6875,  # 22 B / 32 elems
    "q5_1": 0.75,  # 24 B / 32 elems
    "q4_0": 0.5,
    "q4_1": 0.5,
    "iq4_nl": 0.5,
}

# llama.cpp can only create a context with these cache types when flash
# attention is on. f32/f16/bf16 do not need it, and the CPU flash-attention
# path is the slow one once the attention work grows with context, so it is
# requested only where it is required.
_FA_REQUIRED_KV = {"q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"}


def _kv_layer_count(meta: dict) -> int:
    """Number of layers that carry a KV cache.

    Dense models: every layer. Hybrid models report the attention subset in
    ``kv_layers``; anything missing or nonsensical falls back to the full stack,
    which is the conservative direction for a VRAM budget.
    """
    n_layers = int(meta.get("n_layers") or 0)
    kv_layers = int(meta.get("kv_layers") or 0)
    if kv_layers <= 0 or (n_layers > 0 and kv_layers > n_layers):
        return n_layers
    return kv_layers


def _estimate_kv_cache_gb(meta: dict, ctx: int, kv_cache: str) -> float:
    """Estimate KV-cache size in GB for a given context and cache dtype.

    Prefers exact per-model head counts when the metadata carries them. Falls
    back to a no-GQA worst-case estimate from embedding_length, then to the
    legacy 1.5 GB @ 8k heuristic.
    """
    # Hybrid stacks (KDA/BCSA/Mamba) keep a KV cache only on their attention
    # layers: the recurrent layers carry a fixed-size state instead. Using
    # n_layers here overestimates a 72-layer hybrid by roughly 4x.
    kv_layers = _kv_layer_count(meta)
    bytes_per_elem = _KV_BYTES_PER_ELEM.get(kv_cache, 2.0)

    if kv_layers > 0:
        n_heads = int(meta.get("n_heads") or 0)
        n_kv_heads = int(meta.get("n_kv_heads") or 0) or n_heads
        head_dim = int(meta.get("head_dim") or 0)
        embed = int(meta.get("embedding_length") or 0)

        if head_dim <= 0 and embed > 0 and n_heads > 0:
            head_dim = embed // n_heads

        if n_kv_heads > 0 and head_dim > 0:
            # K + V, kv-bearing layers x kv_heads x head_dim x ctx x bytes
            elems = 2 * kv_layers * n_kv_heads * head_dim * ctx
            return elems * bytes_per_elem / 1024**3

        if embed > 0:
            # No GQA info — assume n_kv_heads == n_heads (worst case)
            elems = 2 * kv_layers * embed * ctx
            return elems * bytes_per_elem / 1024**3

    # Last-resort heuristic (conservative for ~7-8B models), scaled by the KV
    # dtype: 1.5 GB @ 8k assumes f16. Without the scale factor a q4_0 plan
    # inherits the f16 number and under-offloads 1-2 layers.
    return (ctx / 8192.0) * 1.5 * (bytes_per_elem / 2.0)


# --- MoE placement ----------------------------------------------------------
#
# `--n-cpu-moe N` keeps the routed experts of the FIRST N layers on the CPU and
# leaves the rest on the GPU. This is the opposite direction from the Hugging
# Face guide, and was verified on a GTX 1650 by reading VRAM at different N:
# N=24 (all experts CPU) used 972 MiB, N=20 used 1844 MiB (the four experts
# layers 20-23, not 0-3), and N=12 used 3412 MiB (layers 12-23). The help text
# in llama.cpp is right; the guide is not.
#
# The budget separates the non-expert weights (attention, embeddings, shared
# experts, router) from the routed experts, because only the latter move with
# --n-cpu-moe. A shared expert or router must stay on the GPU with attention.
#
# Runtime overhead (CUDA context + compute buffers) measured on the reference
# box with LFM2.5-8B-A1B Q4_K_M, ctx 65536, q4_0 KV:
#   ub 512  -> 321 MiB
#   ub 896  -> 405 MiB
#   ub 1024 -> 432 MiB
# A linear fit (0.20 GB + 0.00022 GB per ubatch token) reproduces all three.
_MOE_UBATCH = 896
_MOE_OVERHEAD_BASE_GB = 0.20
_MOE_OVERHEAD_PER_UB_GB = 0.00022
# Driver/fragmentation headroom. The 0.3 GB dense reserve already covers the
# compute buffer, so charging it again here would drop one expert layer.
_MOE_VRAM_RESERVE_GB = 0.15


def _moe_cpu_layers(
    meta: dict,
    hw: HardwareInfo,
    n_layers: int,
    ctx: int,
    kv_cache: str,
    n_ubatch: int,
) -> int | None:
    """Pick the --n-cpu-moe value that fits, or None when MoE placement is unsafe.

    Greedily keeps the suffix of layers whose routed experts fit in VRAM.
    Because --n-cpu-moe counts from layer 0, the suffix is exactly what stays on
    the GPU. Requires the per-layer expert sizes from ``analyze_gguf``; without
    them the estimate would be a guess that can offload one layer too many and
    fail context creation.
    """
    expert_bytes = int(meta.get("expert_bytes") or 0)
    layer_experts = meta.get("expert_layer_bytes")
    if expert_bytes <= 0 or not isinstance(layer_experts, list) or len(layer_experts) != n_layers:
        return None
    sizes = [int(value or 0) for value in layer_experts]
    if sum(sizes) <= 0:
        return None

    size_bytes = int(meta.get("size_bytes") or 4_000_000_000)
    non_expert_gb = max(0.0, (size_bytes - expert_bytes) / 1024**3)
    kv_gb = _estimate_kv_cache_gb(meta, ctx, kv_cache)
    overhead_gb = _MOE_OVERHEAD_BASE_GB + _MOE_OVERHEAD_PER_UB_GB * n_ubatch
    budget_gb = hw.vram_free_gb - _MOE_VRAM_RESERVE_GB - non_expert_gb - kv_gb - overhead_gb
    if budget_gb <= 0:
        return None

    budget_bytes = int(budget_gb * 1024**3)
    used = 0
    gpu_layers = 0
    for size in reversed(sizes):
        if used + size > budget_bytes:
            break
        used += size
        gpu_layers += 1

    # At least one layer that actually carries routed experts must stay on the
    # GPU. "All experts on CPU" measured 40.8 t/s vs 56.7 for the MoE split on
    # the reference card, so the classic split is the better fallback.
    if not any(size > 0 for size in sizes[n_layers - gpu_layers :]):
        return None
    return n_layers - gpu_layers


def plan(
    meta: dict,
    hw: HardwareInfo,
    profile: str = "auto",
    ctx: int = 8192,
    kv_cache: str = "q8_0",
    n_batch: int = 512,
    n_ubatch: int = 0,
) -> Plan:
    if profile not in _VALID_PROFILES:
        raise PlannerError(f"unknown profile '{profile}' (valid: {', '.join(_VALID_PROFILES)})")

    n_layers = int(meta.get("n_layers") or 32)
    kv_layers = _kv_layer_count(meta) or n_layers
    size_bytes = int(meta.get("size_bytes") or 4_000_000_000)

    # One thread per physical core. `cores - 1` used to leave a core idle: on a
    # 4C/8T box, threads=4 measured +19-21% decode over threads=3 (median-3,
    # both CPU-only and with 12 GPU layers), while threads=8 lost throughput to
    # SMT contention.
    cpu_threads = max(1, hw.cpu_cores)
    has_gpu = getattr(hw, "vram_total_gb", 0) > 0

    model_size_gb = size_bytes / 1024**3

    # Low-memory profile: never offload, small ctx, small batch.
    if profile == "lowmem":
        return Plan(
            gpu_layers=0,
            cpu_threads=cpu_threads,
            context=min(ctx, 2048),
            kv_cache="f16",
            flash_attn=False,
            n_batch=min(n_batch, 64),
            offload_kqv=False,
            use_mlock=False,
            note="CPU ONLY (Low Memory Profile)",
            kv_layers=kv_layers,
        )

    want_gpu = has_gpu and profile != "cpu"
    # Flash attention is required by quantized KV, and on the GPU it is also the
    # fast path. Decided before the VRAM budget because it changes how much
    # scratch space llama.cpp needs on the card.
    flash_attn_on_gpu = kv_cache in _FA_REQUIRED_KV or hw.vram_total_gb > 4
    reserve_gb = max(_VRAM_RESERVE_BASE_GB, min(1.0, _VRAM_RESERVE_FRACTION * hw.vram_total_gb))
    if want_gpu and not flash_attn_on_gpu:
        reserve_gb += _NO_FA_RESERVE_GB
    safe_vram = max(0.0, hw.vram_free_gb - reserve_gb)

    # llama.cpp allocates the KV cache per layer: only the layers offloaded to
    # the GPU take their slice of KV with them, the rest stays in system RAM.
    # Subtracting the FULL cache from VRAM before offloading anything made a
    # 14B/Q4 model on a 4 GB card look like it had 0.2 GB free (0 layers) even
    # though ~9 layers actually fit. Budget both weights and KV per layer.
    kv_cache_gb = _estimate_kv_cache_gb(meta, ctx, kv_cache)
    # Charge each KV-bearing layer, not each layer: on a hybrid stack most
    # layers hold a fixed recurrent state and never pay for context.
    per_layer_kv_gb = kv_cache_gb / kv_layers if kv_layers else 0.0

    # Charge every layer the mean weight size (size_bytes / n_layers). An
    # earlier version preferred meta["layer_bytes"] (the mean size of one
    # repeating blk.N block, which excludes token_embd/output.weight) on the
    # theory that those tensors never follow a layer to the GPU. Measured, that
    # is optimistic: llama.cpp's per-layer CUDA compute buffer grows with the
    # number of offloaded layers, and charging the embedding/output bytes to
    # each layer happens to cover it.
    #   Q4_K_M 9.00 GB, 8K q8_0: layer_bytes predicts 10+ layers too many (the
    #                            budget below lands on 13); measured ceiling 14.
    #                            With a 0.28 GB reserve it predicted 14 exactly,
    #                            so the base reserve keeps a one-layer margin.
    #   Q3_K_M 6.82 GB, 8K q8_0: layer_bytes predicts 20 against a measured 17.
    # Over-estimating is the expensive direction: a failed context creation
    # leaves its VRAM allocated, so the ladder can end up CPU-only.
    layer_gb = model_size_gb / n_layers if n_layers else 0.4
    layer_total_gb = layer_gb + per_layer_kv_gb
    gpu_layers = 0

    # profile=cpu forces no offload even when a GPU is present.
    if want_gpu and safe_vram > 0 and layer_total_gb > 0:
        gpu_layers = min(n_layers, int(safe_vram / layer_total_gb))

    # The classic split offloads whole layers, so a layer's experts travel with
    # it. MoE models do better by keeping every attention block on the GPU and
    # moving only routed experts, which is a flag the Python binding does not
    # expose. Only propose that plan when at least one classic layer fits (the
    # in-process fallback needs a valid gpu_layers) and at least one expert
    # layer fits.
    moe_cpu_layers = -1
    moe_ubatch = 0
    if want_gpu and gpu_layers > 0:
        candidate_ubatch = n_ubatch if n_ubatch > 0 else _MOE_UBATCH
        candidate = _moe_cpu_layers(meta, hw, n_layers, ctx, kv_cache, candidate_ubatch)
        if candidate is not None:
            moe_cpu_layers = candidate
            moe_ubatch = candidate_ubatch

    if gpu_layers == 0:
        offload_kqv = False
        n_batch = min(n_batch, 128)
        # An explicit quantized KV request survives the CPU-only path:
        # llama.cpp runs flash attention on the CPU backend too, and forcing
        # f16 here would make a 64K-context KV for a 14B model ~10 GiB, which
        # does not fit beside this gateway on a 15 GiB / no-swap box.
        # flash_attn is required for KV quantization, so enable it only for the
        # dtypes that need it (bf16/f16/f32 stay on the fast default path).
        flash_attn = kv_cache in _FA_REQUIRED_KV

        if profile == "gpu":
            note = "CPU ONLY (GPU profile requested, no usable GPU)"
        elif hw.ram_total_gb < 8:
            ctx = min(ctx, 2048)
            note = "CPU ONLY (Low RAM Safe Mode)"
        else:
            note = "CPU ONLY"
    else:
        # Quantized KV requires flash attention, regardless of VRAM size.
        flash_attn = flash_attn_on_gpu
        offload_kqv = True
        n_batch = min(n_batch, 512)
        note = "Full GPU" if gpu_layers == n_layers else f"Hybrid: {gpu_layers}/{n_layers} layers"
        if moe_cpu_layers >= 0:
            note = f"MoE: {n_layers - moe_cpu_layers}/{n_layers} expert layers on GPU"

    # mlock pins weights in RAM, so it is only safe when the machine can hold
    # the host-side weights AND the host-side KV cache with headroom. Checking
    # against model size alone ignores a 5 GB KV cache (Q8_0 at 64K), which
    # turns an mlock into an OOM on a 15 GB / no-swap box.
    if moe_cpu_layers >= 0:
        # A MoE plan offloads every layer for its dense tensors, so all KV
        # lives on the GPU; only the routed experts of the CPU prefix remain
        # in system RAM.
        layer_experts = meta.get("expert_layer_bytes") or []
        cpu_weights_gb = sum(int(value) for value in layer_experts[:moe_cpu_layers]) / 1024**3
        cpu_kv_gb = 0.0
    else:
        cpu_weights_gb = max(0.0, model_size_gb - layer_gb * gpu_layers)
        cpu_kv_gb = per_layer_kv_gb * max(0, kv_layers - gpu_layers)
    host_headroom_gb = hw.ram_free_gb - cpu_weights_gb - cpu_kv_gb
    # 3 GB of headroom leaves room for llama.cpp compute buffers; pinning with
    # less has OOM-killed this class of machine mid-load (no swap).
    use_mlock = host_headroom_gb > 3.0
    # llama.cpp needs more than weights+KV for compute buffers; 2 GB is a
    # conservative placeholder, calibrated against a 14B/Q8_0/64K load.
    ram_estimate_gb = cpu_weights_gb + cpu_kv_gb + 2.0

    return Plan(
        gpu_layers=gpu_layers,
        cpu_threads=cpu_threads,
        context=ctx,
        kv_cache=kv_cache,
        flash_attn=flash_attn,
        n_batch=n_batch,
        offload_kqv=offload_kqv,
        use_mlock=use_mlock,
        note=note,
        ram_estimate_gb=ram_estimate_gb,
        kv_cache_gb=kv_cache_gb,
        kv_layers=kv_layers,
        moe_cpu_layers=moe_cpu_layers,
        n_ubatch=moe_ubatch,
        n_layers=n_layers,
    )
