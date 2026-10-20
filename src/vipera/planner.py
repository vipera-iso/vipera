from dataclasses import dataclass

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


_VALID_PROFILES = ("auto", "cpu", "gpu", "lowmem")

# llama.cpp allocates CUDA compute buffers for the GPU-resident layers on top of
# their weights and KV. Measured on a GTX 1650 / 14B-Q4 / 8K f16: 11 layers
# (2.37 GB) load, 12 layers (2.59 GB) fail to create the context. Keeping a
# fixed slice free stops the planner from planning a split the GPU cannot run;
# the engine's only fallback is CPU-only, which is much slower than 11 layers.
_GPU_COMPUTE_RESERVE_GB = 0.3

# Bytes per element for each supported kv_cache dtype.
_KV_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
    "q4_1": 0.5,
}


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


def plan(
    meta: dict,
    hw: HardwareInfo,
    profile: str = "auto",
    ctx: int = 8192,
    kv_cache: str = "q8_0",
    n_batch: int = 512,
) -> Plan:
    if profile not in _VALID_PROFILES:
        raise ValueError(f"unknown profile '{profile}' (valid: {', '.join(_VALID_PROFILES)})")

    n_layers = int(meta.get("n_layers") or 32)
    kv_layers = _kv_layer_count(meta) or n_layers
    size_bytes = int(meta.get("size_bytes") or 4_000_000_000)

    cpu_threads = max(1, hw.cpu_cores - 1)
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

    # 1 GB for the OS/display driver, plus room for llama.cpp compute buffers.
    safe_vram = max(0.0, hw.vram_free_gb - 1.0 - _GPU_COMPUTE_RESERVE_GB)

    # llama.cpp allocates the KV cache per layer: only the layers offloaded to
    # the GPU take their slice of KV with them, the rest stays in system RAM.
    # Subtracting the FULL cache from VRAM before offloading anything made a
    # 14B/Q4 model on a 4 GB card look like it had 0.2 GB free (0 layers) even
    # though ~9 layers actually fit. Budget both weights and KV per layer.
    kv_cache_gb = _estimate_kv_cache_gb(meta, ctx, kv_cache)
    # Charge each KV-bearing layer, not each layer: on a hybrid stack most
    # layers hold a fixed recurrent state and never pay for context.
    per_layer_kv_gb = kv_cache_gb / kv_layers if kv_layers else 0.0

    # Repeating-layer weights, when the GGUF index is available. Embeddings and
    # output.weight follow no layer to the GPU: on a 14B/Q4 model they are ~1 GB
    # that size_bytes/n_layers wrongly charges to every layer, which under-
    # offloads by a layer or two on a small card.
    layer_bytes = int(meta.get("layer_bytes") or 0)
    if layer_bytes > 0:
        layer_gb = layer_bytes / 1024**3
    else:
        layer_gb = model_size_gb / n_layers if n_layers else 0.4
    layer_total_gb = layer_gb + per_layer_kv_gb
    gpu_layers = 0

    # profile=cpu forces no offload even when a GPU is present.
    want_gpu = has_gpu and profile != "cpu"
    if want_gpu and safe_vram > 0 and layer_total_gb > 0:
        gpu_layers = min(n_layers, int(safe_vram / layer_total_gb))

    if gpu_layers == 0:
        offload_kqv = False
        n_batch = min(n_batch, 128)
        # An explicit quantized KV request survives the CPU-only path:
        # llama.cpp runs flash attention on the CPU backend too, and forcing
        # f16 here would make a 64K-context KV for a 14B model ~10 GiB, which
        # does not fit beside this gateway on a 15 GiB / no-swap box.
        # flash_attn is required for KV quantization, so enable it whenever
        # the cache is not f16.
        flash_attn = kv_cache != "f16"

        if profile == "gpu":
            note = "CPU ONLY (GPU profile requested, no usable GPU)"
        elif hw.ram_total_gb < 8:
            ctx = min(ctx, 2048)
            note = "CPU ONLY (Low RAM Safe Mode)"
        else:
            note = "CPU ONLY"
    else:
        # Quantized KV requires flash attention, regardless of VRAM size.
        flash_attn = kv_cache != "f16" or hw.vram_total_gb > 4
        offload_kqv = True
        n_batch = min(n_batch, 512)
        note = "Full GPU" if gpu_layers == n_layers else f"Hybrid: {gpu_layers}/{n_layers} layers"

    # mlock pins weights in RAM, so it is only safe when the machine can hold
    # the host-side weights AND the host-side KV cache with headroom. Checking
    # against model size alone ignores a 5 GB KV cache (Q8_0 at 64K), which
    # turns an mlock into an OOM on a 15 GB / no-swap box.
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
    )
