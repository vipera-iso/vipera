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
    # The share of ram_estimate_gb that cannot be paged out: the host-side KV
    # cache plus compute buffers. Weights are mmap'd and may live on disk, so
    # only this part has to be resident. 0.0 means "not estimated", and callers
    # then treat the whole footprint as resident.
    host_resident_gb: float = 0.0
    # KV cache size this plan expects, and how many layers actually keep KV.
    # 0.0 / 0 mean "not estimated" (hand-built Plans in tests stay valid).
    kv_cache_gb: float = 0.0
    kv_layers: int = 0
    # -ctk/-ctv when a side is overridden; "" means "same as kv_cache".
    kv_cache_k: str = ""
    kv_cache_v: str = ""
    # --n-cpu-moe value: the first N layers keep their routed experts on the
    # CPU. -1 means "no MoE plan"; 0 means every expert stays on the GPU.
    # The classic gpu_layers field stays untouched so an in-process fallback
    # still loads a split that fits.
    moe_cpu_layers: int = -1
    # llama.cpp ubatch for the server backend. 0 keeps the historical
    # min(n_batch, 512) behaviour.
    n_ubatch: int = 0
    # MiB of VRAM llama.cpp may use as a cache of the CPU-resident expert
    # tensors (--moe-cache-mib). 0 means off; see _moe_cache_mib for the gate.
    moe_cache_mib: int = 0
    n_layers: int = 0
    # --n-cpu-ffn value: the first N layers keep their dense FFN on the CPU
    # while every attention block (and the KV cache with it) stays on the card.
    # -1 means "no tensor split". Like moe_cpu_layers, this is a server flag, so
    # the classic gpu_layers field stays valid for the in-process fallback.
    n_cpu_ffn: int = -1

    @property
    def requires_server(self) -> bool:
        """True when the plan needs llama-server flags the binding lacks."""
        return self.moe_cpu_layers >= 0 or self.n_cpu_ffn >= 0

    @property
    def kv_types(self) -> tuple[str, str]:
        """The (K, V) cache dtypes to request, each falling back to kv_cache."""
        return self.kv_cache_k or self.kv_cache, self.kv_cache_v or self.kv_cache


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
# Compute buffers grow with the micro-batch, and they are allocated on the card
# for the whole load. On the reference 4 GB card a 16-layer plan at -ub 512
# loaded fine but died with "CUDA error: out of memory" while prefilling a
# ~3300-token prompt; 15 layers at -ub 896 took the same prompt at 37.8 t/s.
# Charging the ubatch here is what keeps the plan on the safe side of that
# cliff, and it moves in the right direction when a caller asks for a smaller
# batch. Same coefficient the MoE path measured (0.20 GB base + this per token).
_RESERVE_PER_UBATCH_GB = 0.00022
# The dense plan's micro-batch: config n_batch, capped the same way
# server_command caps it, so the reserve below charges the buffer the child will
# actually allocate.
_DENSE_UBATCH = 512
# Prefill micro-batch for a plan whose weights stay in RAM. Nothing on the card
# has to hold these buffers, and prefill is where such a plan spends its life.
# Measured on gpt-oss-120b Q2_K (58.3 GB, 4 cores, 15 GiB RAM, ctx 4096, -t 4,
# 341-token prompt, second request after a warm-up so the page cache is not
# part of the number): ub 128 -> 2.96 t/s, ub 512 -> 6.50, ub 2048 -> 6.57.
# Flat from 512 up, so 512 is the knee: it buys the whole 2.2x over 128 without
# the larger compute buffers (in RAM on this path) that 2048 asks for. Below it
# the CPU GEMMs are too short to pay for themselves. `lowmem` capped the batch
# at 64, which is how the same model ended up planned at 2 t/s prefill.
_HOST_UBATCH = 512
# Below this much free RAM the old conservative cap stays: the prefill buffers
# of a 100B-class CPU-only plan live in RAM, and no machine that small was part
# of the measurement above, so it is not the place to spend the gains.
_HOST_UBATCH_MIN_FREE_RAM_GB = 8.0
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
    "iq4_nl": 0.5625,  # 18 B / 32 elems
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

# `--moe-cache-mib N` keeps a pool of the CPU-resident expert tensors in VRAM so
# a decode step does not have to pull every selected expert over the bus. The
# win depends on the pool being large enough for the working set: on a 24 GB
# card it took decode from 11.55 to 21.21 t/s (+84%), while on a 4 GB card the
# same flag made the plan three times slower — the pool evicted itself on every
# step and the copy was paid twice. So the pool is only requested when the card
# has room left after placement for one worth having, and it is sized as half
# of that room: the other half has to hold the next prefill's compute buffers.
_MOE_CACHE_MIN_SPARE_GB = 8.0
_MOE_CACHE_FRACTION = 0.5
_MOE_CACHE_MAX_MIB = 4096
_MOE_CACHE_STEP_MIB = 128

# Host-side allowance for llama.cpp compute buffers, charged on top of the
# resident KV cache. Part of ram_estimate_gb since it was calibrated against a
# 14B/Q8_0/64K load, and the only part of that estimate a mmap'd model must keep
# in RAM.
_HOST_COMPUTE_GB = 2.0


def _host_ubatch(hw: HardwareInfo, small_box: int) -> int:
    """Prefill micro-batch for a plan that keeps its weights in RAM.

    `small_box` is the cap to keep on a machine with too little RAM to hold the
    larger compute buffers; see `_HOST_UBATCH_MIN_FREE_RAM_GB`.
    """
    if hw.ram_free_gb < _HOST_UBATCH_MIN_FREE_RAM_GB:
        return small_box
    return _HOST_UBATCH


def _lowmem_moe_plan(
    meta: dict,
    hw: HardwareInfo,
    *,
    n_layers: int,
    kv_layers: int,
    ctx: int,
    n_batch: int,
    cpu_threads: int,
) -> Plan | None:
    """`lowmem` for a MoE model: every routed expert in RAM, attention on the card.

    "No expert on the GPU" is not the same as "no GPU". On a MoE model the routed
    experts are the bulk of the file, and the attention side is small enough to fit
    beside the reserve of a 4 GB card. Measured on LFM2.5-8B-A1B (Q4_K weights, Q3_K
    experts, ctx 8192, GTX 1650): CPU-only **37.1 t/s**, experts-on-CPU **49.9 t/s**,
    the full MoE split 69.3 t/s. So the profile can honour its promise - no expert
    weight in VRAM - and still be 35% faster than the plan it used to return.

    `-ncmoe n_layers` is exactly llama.cpp's `-cmoe`.

    Returns None when the model has no routed experts, there is no GPU, or the dense
    side does not fit; the caller then keeps the CPU-only plan.
    """
    if getattr(hw, "vram_total_gb", 0) <= 0:
        return None

    expert_bytes = int(meta.get("expert_bytes") or 0)
    size_bytes = int(meta.get("size_bytes") or 0)
    if expert_bytes <= 0 or size_bytes <= expert_bytes:
        return None

    lowmem_ctx = min(ctx, 2048)
    ubatch = min(n_batch, 64)
    dense_gb = (size_bytes - expert_bytes) / 1024**3
    kv_gb = _estimate_kv_cache_gb(meta, lowmem_ctx, "f16")
    overhead_gb = _MOE_OVERHEAD_BASE_GB + _MOE_OVERHEAD_PER_UB_GB * ubatch
    spare_gb = hw.vram_free_gb - dense_gb - kv_gb - overhead_gb - _MOE_VRAM_RESERVE_GB
    if spare_gb < 0:
        return None

    return Plan(
        gpu_layers=n_layers,
        cpu_threads=cpu_threads,
        context=lowmem_ctx,
        kv_cache="f16",
        # Attention now runs on the card, where flash attention is the fast path.
        flash_attn=True,
        n_batch=ubatch,
        offload_kqv=True,
        use_mlock=False,
        note=f"MoE: experts on CPU, attention on GPU ({n_layers}/{n_layers} layers, Low Memory)",
        kv_cache_gb=kv_gb,
        kv_layers=kv_layers,
        moe_cpu_layers=n_layers,
        n_ubatch=ubatch,
        # This is the configuration the expert pool is for: every routed expert
        # is in RAM and the card has room left over after attention. On the
        # reference 4 GB card the spare is ~0 and the pool stays off.
        moe_cache_mib=_moe_cache_mib(spare_gb),
        n_layers=n_layers,
    )


def _moe_spare_vram_gb(
    meta: dict,
    hw: HardwareInfo,
    *,
    n_layers: int,
    moe_cpu_layers: int,
    kv_gb: float,
    n_ubatch: int,
) -> float | None:
    """VRAM left over once the planned expert placement is loaded.

    Mirrors the budget in `_moe_cpu_layers` from the other side: that function
    fills the card with expert layers, this one reports what is still free
    afterwards, which is what a `--moe-cache-mib` pool would live in.
    """
    expert_bytes = int(meta.get("expert_bytes") or 0)
    layer_experts = meta.get("expert_layer_bytes")
    if expert_bytes <= 0 or not isinstance(layer_experts, list) or len(layer_experts) != n_layers:
        return None
    sizes = [int(value or 0) for value in layer_experts]
    gpu_experts_gb = sum(sizes[moe_cpu_layers:]) / 1024**3
    size_bytes = int(meta.get("size_bytes") or 0)
    non_expert_gb = max(0.0, (size_bytes - expert_bytes) / 1024**3)
    overhead_gb = _MOE_OVERHEAD_BASE_GB + _MOE_OVERHEAD_PER_UB_GB * n_ubatch
    return (
        hw.vram_free_gb
        - _MOE_VRAM_RESERVE_GB
        - non_expert_gb
        - kv_gb
        - overhead_gb
        - gpu_experts_gb
    )


def _moe_cache_mib(spare_gb: float | None) -> int:
    """Pool size for `--moe-cache-mib`, or 0 when the spare VRAM is too small.

    Below `_MOE_CACHE_MIN_SPARE_GB` the pool cannot hold enough experts to cover
    a decode step, and llama.cpp spends the step copying experts in and out
    instead of computing with them (see the constant's comment).
    """
    if spare_gb is None or spare_gb < _MOE_CACHE_MIN_SPARE_GB:
        return 0
    mib = min(int(spare_gb * _MOE_CACHE_FRACTION * 1024), _MOE_CACHE_MAX_MIB)
    return (mib // _MOE_CACHE_STEP_MIB) * _MOE_CACHE_STEP_MIB


def _moe_cpu_layers(
    meta: dict,
    hw: HardwareInfo,
    n_layers: int,
    ctx: int,
    kv_gb: float,
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


# --- tensor-type split (dense models) ---------------------------------------
#
# `--n-cpu-ffn N` keeps the dense FFN of the FIRST N layers on the CPU and leaves
# every attention block on the card, KV cache included. It is a tensor override,
# so like `--n-cpu-moe` it is paired with a full `-ngl`: llama.cpp then streams
# only FFN weights from system RAM.
#
# Measured on a GTX 1650 4 GB, Hermes-4-14B Q3_K_M (attention 1193 MiB over 40
# layers, dense FFN 131.5 MiB per layer, token_embd+output 927 MiB), ctx 8192,
# q8_0 KV, against the classic contiguous split that fits the same card (18/40):
#   prompt  177 tokens: classic 6.34 t/s, tensor split 5.51 t/s
#   prompt 1840 tokens: classic 3.96 t/s, tensor split 5.22 t/s
#   prompt 3750 tokens: classic 3.44 t/s, tensor split 4.89 t/s
#   ctx 16384, 3750:    classic fails (KV buffer cudaMalloc OOM),
#                       tensor split 4.99 t/s
# The classic split pays CPU-side attention that grows with the window, while the
# tensor split's extra host bytes do not, so the crossover sits between 177 and
# 1840 tokens of prompt. The floor below is on the *configured* window, which is
# the only thing a plan knows: at 2048 the classic split is the better bet, and
# from 4096 up the tensor split won every measurement.
_TENSOR_SPLIT_MIN_CTX = 4096


def _ffn_layer_sizes(meta: dict, n_layers: int) -> list[int] | None:
    """Per-layer dense-FFN bytes, or None when the analyzer did not report them."""
    sizes = meta.get("ffn_layer_bytes")
    if not isinstance(sizes, list) or len(sizes) != n_layers:
        return None
    out = [max(0, int(value or 0)) for value in sizes]
    return out if sum(out) > 0 else None


def _tensor_split_ffn_layers(
    meta: dict,
    safe_vram_gb: float,
    *,
    n_layers: int,
    ctx: int,
    kv_gb: float,
) -> int | None:
    """The --n-cpu-ffn value for a dense model, or None when it does not apply.

    Fills what the card has left after attention (and the whole KV cache, since
    every attention block is on the card) with dense FFN blocks, counting from
    the last layer because --n-cpu-ffn keeps the FIRST N on the CPU.

    Returns None when the analyzer did not report the attention/FFN split, when
    the model has routed experts (--n-cpu-ffn moves dense FFN only, and the MoE
    path places experts better), when the context is below the measured
    crossover, or when not a single FFN block fits beside attention.
    """
    if ctx < _TENSOR_SPLIT_MIN_CTX:
        return None
    if int(meta.get("expert_bytes") or 0) > 0:
        return None
    attn_bytes = int(meta.get("attn_bytes") or 0)
    sizes = _ffn_layer_sizes(meta, n_layers)
    if attn_bytes <= 0 or sizes is None:
        return None
    fixed_bytes = int(meta.get("fixed_bytes") or 0)
    budget_bytes = int((safe_vram_gb - (attn_bytes + fixed_bytes) / 1024**3 - kv_gb) * 1024**3)
    if budget_bytes <= 0:
        return None

    used = 0
    gpu_ffn = 0
    for size in reversed(sizes):
        if used + size > budget_bytes:
            break
        used += size
        gpu_ffn += 1
    # At least one FFN block has to stay on the card: that is the configuration
    # measured above, and "no FFN on the GPU at all" is just a slower spelling of
    # the classic split.
    if gpu_ffn <= 0:
        return None
    return n_layers - gpu_ffn


def plan(
    meta: dict,
    hw: HardwareInfo,
    profile: str = "auto",
    ctx: int = 8192,
    kv_cache: str = "q8_0",
    n_batch: int = 512,
    n_ubatch: int = 0,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    no_kv_offload: bool = False,
) -> Plan:
    if profile not in _VALID_PROFILES:
        raise PlannerError(f"unknown profile '{profile}' (valid: {', '.join(_VALID_PROFILES)})")

    # -ctk and -ctv are independent flags; each side falls back to the combined
    # kv_cache, so a config that only sets kv_cache plans exactly as before.
    kv_k = kv_cache_k or kv_cache
    kv_v = kv_cache_v or kv_cache
    quantized_kv = kv_k in _FA_REQUIRED_KV or kv_v in _FA_REQUIRED_KV

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

    # Low-memory profile: no routed expert in VRAM, small ctx, small batch. A MoE
    # model can still put its attention on the card - that is a fraction of the
    # file and measured +35% decode, so the CPU-only plan is the fallback, not the
    # rule.
    if profile == "lowmem":
        lowmem_moe = _lowmem_moe_plan(
            meta,
            hw,
            n_layers=n_layers,
            kv_layers=kv_layers,
            ctx=ctx,
            n_batch=n_batch,
            cpu_threads=cpu_threads,
        )
        if lowmem_moe is not None:
            return lowmem_moe
        return Plan(
            gpu_layers=0,
            cpu_threads=cpu_threads,
            context=min(ctx, 2048),
            kv_cache="f16",
            flash_attn=False,
            # The profile's promise is "small context, no weights in VRAM". It
            # never promised a 64-token batch: that cap cost a 58 GB model most
            # of its prefill throughput (see _HOST_UBATCH), and a CPU-only plan
            # allocates nothing from it on the card.
            n_batch=min(n_batch, _host_ubatch(hw, 64)),
            n_ubatch=min(n_batch, _host_ubatch(hw, 64)),
            offload_kqv=False,
            use_mlock=False,
            note="CPU ONLY (Low Memory Profile)",
            kv_layers=kv_layers,
            kv_cache_k="f16",
            kv_cache_v="f16",
        )

    want_gpu = has_gpu and profile != "cpu"
    # Flash attention is required by quantized KV, and on the GPU it is also the
    # fast path. Decided before the VRAM budget because it changes how much
    # scratch space llama.cpp needs on the card.
    flash_attn_on_gpu = quantized_kv or hw.vram_total_gb > 4
    reserve_gb = max(_VRAM_RESERVE_BASE_GB, min(1.0, _VRAM_RESERVE_FRACTION * hw.vram_total_gb))
    # Micro-batch first: the prefill compute buffer has to fit beside the
    # weights, and that is what a long prompt runs out of room on.
    dense_ubatch = n_ubatch if n_ubatch > 0 else min(n_batch, _DENSE_UBATCH)
    # A MoE plan already charges its own ubatch overhead per expert layer
    # (`_moe_cpu_layers`); charging it again here would only drop expert layers.
    if want_gpu and not meta.get("expert_layer_bytes"):
        reserve_gb += _RESERVE_PER_UBATCH_GB * dense_ubatch
    if want_gpu and not flash_attn_on_gpu:
        reserve_gb += _NO_FA_RESERVE_GB
    safe_vram = max(0.0, hw.vram_free_gb - reserve_gb)

    # llama.cpp allocates the KV cache per layer: only the layers offloaded to
    # the GPU take their slice of KV with them, the rest stays in system RAM.
    # Subtracting the FULL cache from VRAM before offloading anything made a
    # 14B/Q4 model on a 4 GB card look like it had 0.2 GB free (0 layers) even
    # though ~9 layers actually fit. Budget both weights and KV per layer.
    kv_cache_gb = (
        _estimate_kv_cache_gb(meta, ctx, kv_k) + _estimate_kv_cache_gb(meta, ctx, kv_v)
    ) / 2
    # Charge each KV-bearing layer, not each layer: on a hybrid stack most
    # layers hold a fixed recurrent state and never pay for context. With
    # `--no-kv-offload` the whole cache stays in system RAM, so the card is
    # charged nothing for it and the freed room buys offload depth; the
    # host-side cost is added back when RAM is estimated below.
    per_layer_kv_gb = 0.0
    if not no_kv_offload and kv_layers:
        per_layer_kv_gb = kv_cache_gb / kv_layers

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
        candidate = _moe_cpu_layers(
            meta,
            hw,
            n_layers,
            ctx,
            0.0 if no_kv_offload else kv_cache_gb,
            candidate_ubatch,
        )
        if candidate is not None:
            moe_cpu_layers = candidate
            moe_ubatch = candidate_ubatch

    # What is left on the card after that placement buys a pool of the experts
    # that stayed in RAM; see _MOE_CACHE_MIN_SPARE_GB.
    moe_cache_mib = 0
    if moe_cpu_layers >= 0:
        moe_cache_mib = _moe_cache_mib(
            _moe_spare_vram_gb(
                meta,
                hw,
                n_layers=n_layers,
                moe_cpu_layers=moe_cpu_layers,
                kv_gb=0.0 if no_kv_offload else kv_cache_gb,
                n_ubatch=moe_ubatch,
            )
        )

    # A dense model that does not fit can still keep every attention block (and
    # therefore the whole KV cache) on the card and stream only FFN weights.
    # Only when the classic split is already partial: a full offload has nothing
    # left to improve, and a CPU-only plan has no card to split against.
    n_cpu_ffn = -1
    if want_gpu and 0 < gpu_layers < n_layers:
        candidate_ffn = _tensor_split_ffn_layers(
            meta,
            safe_vram,
            n_layers=n_layers,
            ctx=ctx,
            kv_gb=0.0 if no_kv_offload else kv_cache_gb,
        )
        if candidate_ffn is not None:
            n_cpu_ffn = candidate_ffn

    if gpu_layers == 0:
        offload_kqv = False
        # A CPU-only plan is a prefill machine: every token of the prompt runs
        # through weights that live in RAM, so -ub is the throughput lever and
        # there is no video memory to charge for it. The old cap of 128 (64
        # under `lowmem`) measured 1.8 t/s on gpt-oss-120b Q2_K, where a larger
        # physical batch is what makes the CPU GEMMs worth vectorising.
        n_batch = min(n_batch, _host_ubatch(hw, 128))
        dense_ubatch = min(n_batch, _host_ubatch(hw, 128))
        # An explicit quantized KV request survives the CPU-only path:
        # llama.cpp runs flash attention on the CPU backend too, and forcing
        # f16 here would make a 64K-context KV for a 14B model ~10 GiB, which
        # does not fit beside this gateway on a 15 GiB / no-swap box.
        # flash_attn is required for KV quantization, so enable it only for the
        # dtypes that need it (bf16/f16/f32 stay on the fast default path).
        flash_attn = quantized_kv

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
        offload_kqv = not no_kv_offload
        n_batch = min(n_batch, 512)
        note = "Full GPU" if gpu_layers == n_layers else f"Hybrid: {gpu_layers}/{n_layers} layers"
        if moe_cpu_layers >= 0:
            note = f"MoE: {n_layers - moe_cpu_layers}/{n_layers} expert layers on GPU"
        if n_cpu_ffn >= 0:
            note = (
                f"Tensor split: {n_layers - n_cpu_ffn}/{n_layers} dense FFN layers "
                "on GPU, attention on GPU"
            )

    # mlock pins weights in RAM, so it is only safe when the machine can hold
    # the host-side weights AND the host-side KV cache with headroom. Checking
    # against model size alone ignores a 5 GB KV cache (Q8_0 at 64K), which
    # turns an mlock into an OOM on a 15 GB / no-swap box.
    if n_cpu_ffn >= 0:
        # Same shape of plan as the MoE one: attention (and with it the whole KV
        # cache) is on the card, and the host keeps the FFN blocks of the CPU
        # prefix.
        sizes = _ffn_layer_sizes(meta, n_layers) or []
        cpu_weights_gb = sum(sizes[:n_cpu_ffn]) / 1024**3
        cpu_kv_gb = kv_cache_gb if no_kv_offload else 0.0
    elif moe_cpu_layers >= 0:
        # A MoE plan offloads every layer for its dense tensors, so all KV
        # lives on the GPU (unless it was parked in RAM); only the routed
        # experts of the CPU prefix remain in system RAM.
        layer_experts = meta.get("expert_layer_bytes") or []
        cpu_weights_gb = sum(int(value) for value in layer_experts[:moe_cpu_layers]) / 1024**3
        cpu_kv_gb = kv_cache_gb if no_kv_offload else 0.0
    else:
        cpu_weights_gb = max(0.0, model_size_gb - layer_gb * gpu_layers)
        cpu_kv_gb = (
            kv_cache_gb if no_kv_offload else per_layer_kv_gb * max(0, kv_layers - gpu_layers)
        )
    host_headroom_gb = hw.ram_free_gb - cpu_weights_gb - cpu_kv_gb
    # 3 GB of headroom leaves room for llama.cpp compute buffers; pinning with
    # less has OOM-killed this class of machine mid-load (no swap).
    use_mlock = host_headroom_gb > 3.0
    # llama.cpp needs more than weights+KV for compute buffers; 2 GB is a
    # conservative placeholder, calibrated against a 14B/Q8_0/64K load. Only
    # the KV cache and those buffers have to be resident, since the weights are
    # mmap'd and the kernel can page them (see loading.check_host_ram).
    ram_estimate_gb = cpu_weights_gb + cpu_kv_gb + _HOST_COMPUTE_GB
    host_resident_gb = cpu_kv_gb + _HOST_COMPUTE_GB

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
        host_resident_gb=host_resident_gb,
        kv_cache_gb=kv_cache_gb,
        kv_layers=kv_layers,
        kv_cache_k=kv_k,
        kv_cache_v=kv_v,
        moe_cpu_layers=moe_cpu_layers,
        moe_cache_mib=moe_cache_mib,
        # A MoE plan has its own -ub (it is the expert-placement lever); a dense
        # plan keeps the prefill batch it was budgeted for.
        n_ubatch=moe_ubatch if moe_cpu_layers >= 0 else dense_ubatch,
        n_layers=n_layers,
        n_cpu_ffn=n_cpu_ffn,
    )
