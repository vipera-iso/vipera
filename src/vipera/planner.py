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


_VALID_PROFILES = ("auto", "cpu", "gpu", "lowmem")

# Bytes per element for each supported kv_cache dtype.
_KV_BYTES_PER_ELEM = {
    "f32": 4.0, "f16": 2.0, "q8_0": 1.0, "q4_0": 0.5, "q4_1": 0.5,
}


def _estimate_kv_cache_gb(meta: dict, ctx: int, kv_cache: str) -> float:
    """Estimate KV-cache size in GB for a given context and cache dtype.

    Prefers exact per-model head counts when the metadata carries them. Falls
    back to a no-GQA worst-case estimate from embedding_length, then to the
    legacy 1.5 GB @ 8k heuristic.
    """
    n_layers = int(meta.get("n_layers") or 0)
    bytes_per_elem = _KV_BYTES_PER_ELEM.get(kv_cache, 2.0)

    if n_layers > 0:
        n_heads = int(meta.get("n_heads") or 0)
        n_kv_heads = int(meta.get("n_kv_heads") or 0) or n_heads
        head_dim = int(meta.get("head_dim") or 0)
        embed = int(meta.get("embedding_length") or 0)

        if head_dim <= 0 and embed > 0 and n_heads > 0:
            head_dim = embed // n_heads

        if n_kv_heads > 0 and head_dim > 0:
            # K + V, layers x kv_heads x head_dim x ctx x bytes
            elems = 2 * n_layers * n_kv_heads * head_dim * ctx
            return elems * bytes_per_elem / 1024**3

        if embed > 0:
            # No GQA info — assume n_kv_heads == n_heads (worst case)
            elems = 2 * n_layers * embed * ctx
            return elems * bytes_per_elem / 1024**3

    # Last-resort heuristic (conservative for ~7-8B models only).
    return (ctx / 8192.0) * 1.5


def plan(
    meta: dict,
    hw: HardwareInfo,
    profile: str = "auto",
    ctx: int = 8192,
    kv_cache: str = "q8_0",
    n_batch: int = 512,
) -> Plan:
    if profile not in _VALID_PROFILES:
        raise ValueError(
            f"unknown profile '{profile}' (valid: {', '.join(_VALID_PROFILES)})"
        )

    n_layers = int(meta.get("n_layers") or 32)
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
        )

    use_mlock = (hw.ram_free_gb - model_size_gb) > 2.0

    safe_vram = max(0.0, hw.vram_free_gb - 1.0)  # reserve 1 GB for OS/driver
    kv_cache_gb = _estimate_kv_cache_gb(meta, ctx, kv_cache)
    usable_vram = max(0.0, safe_vram - kv_cache_gb)

    layer_gb = model_size_gb / n_layers if n_layers else 0.4
    gpu_layers = 0

    # profile=cpu forces no offload even when a GPU is present.
    want_gpu = has_gpu and profile != "cpu"
    if want_gpu and usable_vram > 0 and layer_gb > 0:
        gpu_layers = min(n_layers, int(usable_vram / layer_gb))

    if gpu_layers == 0:
        flash_attn = False
        offload_kqv = False
        n_batch = min(n_batch, 128)
        kv_cache = "f16"

        if profile == "gpu":
            note = "CPU ONLY (GPU profile requested, no usable GPU)"
        elif hw.ram_total_gb < 8:
            ctx = min(ctx, 2048)
            note = "CPU ONLY (Low RAM Safe Mode)"
        else:
            note = "CPU ONLY"
    else:
        flash_attn = hw.vram_total_gb > 4
        offload_kqv = True
        n_batch = min(n_batch, 512)
        note = (
            "Full GPU"
            if gpu_layers == n_layers
            else f"Hybrid: {gpu_layers}/{n_layers} layers"
        )

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
    )
