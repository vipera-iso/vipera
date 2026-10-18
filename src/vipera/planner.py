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


def plan(
    meta: dict,
    hw: HardwareInfo,
    profile: str = "auto",
    ctx: int = 8192,
    kv_cache: str = "q8_0",
    n_batch: int = 512,
) -> Plan:
    n_layers = int(meta.get("n_layers") or 32)
    size_bytes = int(meta.get("size_bytes") or 4_000_000_000)

    cpu_threads = max(1, hw.cpu_cores - 1)
    has_gpu = getattr(hw, "vram_total_gb", 0) > 0

    model_size_gb = size_bytes / 1024**3
    use_mlock = (hw.ram_free_gb - model_size_gb) > 2.0

    safe_vram = max(0.0, hw.vram_free_gb - 1.0)
    usable_vram = safe_vram * 0.90

    layer_gb = (size_bytes / 1024**3) / n_layers if n_layers else 0.4
    gpu_layers = 0

    if has_gpu and usable_vram > 0 and layer_gb > 0:
        gpu_layers = min(n_layers, int(usable_vram / layer_gb))

    if gpu_layers == 0:
        flash_attn = False
        offload_kqv = False
        n_batch = min(n_batch, 128)
        kv_cache = "f16"

        if hw.ram_total_gb < 8:
            ctx = min(ctx, 2048)
            note = "CPU ONLY (Low RAM Safe Mode)"
        else:
            note = "CPU ONLY"
    else:
        flash_attn = has_gpu and hw.vram_total_gb > 4
        offload_kqv = True
        n_batch = min(n_batch, 512)
        note = "Full GPU" if gpu_layers == n_layers else f"Hybrid: {gpu_layers}/{n_layers} layers"

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
