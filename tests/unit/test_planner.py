import pytest

from vipera.hardware import HardwareInfo
from vipera.planner import _KV_BYTES_PER_ELEM, _estimate_kv_cache_gb, plan

_GIB = 1024**3


def _hw(**overrides):
    base = dict(
        cpu_cores=8,
        cpu_threads=16,
        ram_total_gb=16,
        ram_free_gb=12,
        gpu_name="None",
        vram_total_gb=0.0,
        vram_free_gb=0.0,
        is_apple_silicon=False,
    )
    base.update(overrides)
    return HardwareInfo(**base)


def test_cpu_only_plan():
    p = plan({"n_layers": 32, "size_bytes": 4 * 1024**3}, _hw())
    assert p.gpu_layers == 0
    # The requested quantized KV survives the CPU-only path (llama.cpp runs
    # flash attention on CPU; f16 at large ctx can exceed free RAM).
    assert p.kv_cache == "q8_0"
    assert p.flash_attn is True
    assert p.note.startswith("CPU ONLY")
    # One thread per physical core: threads=cores measured +19-21% decode over
    # threads=cores-1 (median-3), and threads=2*cores loses to SMT contention.
    assert p.cpu_threads == 8


def test_cpu_only_plan_prefills_at_the_swept_batch():
    # Measured on gpt-oss-120b Q2_K: ub 128 -> 2.90 t/s, 512 -> 6.38, 2048 ->
    # 6.42. The knee is 512, and a CPU-only plan has no card to charge for it.
    p = plan({"n_layers": 36, "size_bytes": 58 * _GIB}, _hw())
    assert p.gpu_layers == 0
    assert p.n_batch == 512
    assert p.n_ubatch == 512


def test_cpu_only_plan_separates_streamed_weights_from_resident_footprint():
    # 58 GB of mmap'd weights on a 12 GB box. The whole footprint cannot be
    # resident, but the KV cache and compute buffers can, and that is all the
    # preflight may require (see loading.check_host_ram).
    p = plan({"n_layers": 36, "size_bytes": 58 * _GIB}, _hw())

    assert p.ram_estimate_gb > 55
    assert p.host_resident_gb < p.ram_estimate_gb / 10
    assert p.use_mlock is False


def test_cpu_only_plan_keeps_f16_without_flash_attention():
    p = plan({"n_layers": 32, "size_bytes": 4 * 1024**3}, _hw(), kv_cache="f16")
    assert p.gpu_layers == 0
    assert p.kv_cache == "f16"
    assert p.flash_attn is False


def test_full_gpu():
    hw = _hw(gpu_name="X", vram_total_gb=24.0, vram_free_gb=24.0)
    p = plan({"n_layers": 32, "size_bytes": 4 * 1024**3}, hw)
    assert p.gpu_layers == 32
    assert p.note == "Full GPU"


def test_hybrid_partial():
    hw = _hw(gpu_name="X", vram_total_gb=4.0, vram_free_gb=4.0)
    p = plan({"n_layers": 32, "size_bytes": 8 * 1024**3}, hw)
    assert 0 < p.gpu_layers < 32
    assert "Hybrid" in p.note


# --- per-layer VRAM budgeting ------------------------------------------------

# Hermes-4-14B Q4_K_M on a GTX 1650 (4 GB, ~3.7 GB free), 64K context, q4_0 KV.
# The old planner subtracted the full 2.5 GB KV from 2.7 GB usable VRAM and
# concluded 0 layers. llama.cpp keeps KV per layer, so ~9 layers actually fit.
_HERMES_META = {
    "n_layers": 40,
    "n_heads": 40,
    "n_kv_heads": 8,
    "head_dim": 128,
    "size_bytes": 9 * 1024**3,
}


def test_hybrid_offloads_layers_when_only_per_layer_kv_fits():
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.7)
    p = plan(_HERMES_META, hw, ctx=65536, kv_cache="q4_0")

    # safe_vram 3.42 GB (0.28 GB card reserve); layer 0.225 GB
    # + per-layer KV 0.0625 GB -> 11 layers.
    assert p.gpu_layers == 11
    assert p.offload_kqv is True
    assert p.note == "Hybrid: 11/40 layers"


def test_hybrid_offload_of_quantized_kv_requires_flash_attention():
    # A 4 GB card fails the old `vram_total_gb > 4` FA check; quantized KV then
    # makes llama.cpp reject the context. FA must stay on whenever KV is quantized.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.7)
    p = plan(_HERMES_META, hw, ctx=65536, kv_cache="q8_0")

    assert p.gpu_layers > 0
    assert p.flash_attn is True


def test_f16_kv_still_offloads_some_layers():
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.7)
    p = plan(_HERMES_META, hw, ctx=65536, kv_cache="f16")

    # 10 GB KV / 40 layers = 0.25 GB per layer on top of 0.225 GB of weights:
    # 2.7 / 0.475 -> 5 layers.
    assert p.gpu_layers == 5


def test_no_flash_attention_costs_layers():
    # Without FA llama.cpp allocates a full KQ matrix per batch, so the same card
    # offloads fewer layers. Measured on the reference box: 14 layers with FA at
    # 8K/q8_0, 10 without at 8K/f16.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    quantized = plan(_HERMES_META, hw, ctx=8192, kv_cache="q8_0")
    full_precision = plan(_HERMES_META, hw, ctx=8192, kv_cache="f16")

    assert quantized.flash_attn is True
    assert full_precision.flash_attn is False
    assert quantized.gpu_layers > full_precision.gpu_layers
    # 13 vs 9: the FA reserve is worth more than the f16 cache is cheaper.
    assert quantized.gpu_layers == 13
    assert full_precision.gpu_layers == 9


def test_layer_bytes_metadata_does_not_raise_the_offload():
    # meta["layer_bytes"] is the mean size of a repeating blk.N block, which
    # excludes token_embd/output.weight. Offloading on it measured optimistic on
    # both files it was tried with: a Q4_K_M plan asked for 6 layers more than
    # the card could create a context for, and a Q3_K_M plan for 3 more. A
    # failed attempt keeps its VRAM, so the ladder then ended up CPU-only. The
    # budget charges size_bytes/n_layers instead.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.7)
    lean = dict(_HERMES_META, layer_bytes=int(0.1844 * _GIB))

    assert plan(lean, hw, ctx=8192, kv_cache="f16").gpu_layers == 9
    assert plan(_HERMES_META, hw, ctx=8192, kv_cache="f16").gpu_layers == 9


def test_ram_estimate_reflects_q8_kv_at_64k():
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.7)
    q8 = plan(_HERMES_META, hw, ctx=65536, kv_cache="q8_0")
    q4 = plan(_HERMES_META, hw, ctx=65536, kv_cache="q4_0")

    # Host weights (~7.4 GB) + host KV (~4.1 GB) + compute headroom.
    assert q8.ram_estimate_gb > 12.0
    # q4_0 halves the KV and is what actually loads on a 10 GB-free box.
    assert q4.ram_estimate_gb < q8.ram_estimate_gb - 2.0


def test_mlock_disabled_when_host_kv_leaves_no_headroom():
    # Pinning ~7.4 GB of host weights plus ~4.1 GB of host KV on a 15 GB
    # no-swap box leaves nothing for compute buffers; the process gets OOM
    # killed instead of failing cleanly, so mlock must stay off.
    hw = _hw(
        gpu_name="GTX 1650",
        vram_total_gb=4.0,
        vram_free_gb=3.7,
        ram_total_gb=15.4,
        ram_free_gb=10.6,
    )
    p = plan(_HERMES_META, hw, ctx=65536, kv_cache="q8_0", n_batch=128)
    assert p.use_mlock is False


def test_mlock_enabled_with_generous_headroom():
    hw = _hw(ram_total_gb=32, ram_free_gb=24)
    meta = {
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "size_bytes": 4 * 1024**3,
    }
    p = plan(meta, hw)
    assert p.use_mlock is True


def test_low_ram_safe_mode():
    hw = _hw(ram_total_gb=4, ram_free_gb=2)
    p = plan({"n_layers": 32, "size_bytes": 2 * 1024**3}, hw, ctx=8192)
    assert p.gpu_layers == 0
    assert p.context <= 2048


# --- profiles ---------------------------------------------------------------


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        plan({"n_layers": 32, "size_bytes": 4 * 1024**3}, _hw(), profile="gpuu")


def test_cpu_profile_forces_no_offload_despite_gpu():
    hw = _hw(gpu_name="X", vram_total_gb=24.0, vram_free_gb=24.0)
    p = plan({"n_layers": 32, "size_bytes": 4 * 1024**3}, hw, profile="cpu")
    assert p.gpu_layers == 0
    assert p.note.startswith("CPU ONLY")


def test_gpu_profile_without_gpu_is_explained():
    p = plan({"n_layers": 32, "size_bytes": 4 * 1024**3}, _hw(), profile="gpu")
    assert p.gpu_layers == 0
    assert "GPU profile requested" in p.note


def test_lowmem_profile_caps_context_and_batch():
    hw = _hw(gpu_name="X", vram_total_gb=24.0, vram_free_gb=24.0)
    p = plan(
        {"n_layers": 32, "size_bytes": 4 * 1024**3},
        hw,
        profile="lowmem",
        ctx=8192,
        n_batch=512,
    )
    assert p.gpu_layers == 0
    assert p.context <= 2048
    # The profile's promise is a small context with no weights in VRAM; the
    # 64-token batch it used to cap on top of that cost a 58 GB model 2.2x its
    # prefill throughput (see _HOST_UBATCH), and a CPU-only plan allocates no
    # prefill buffer on the card.
    assert p.n_batch == 512
    assert p.n_ubatch == 512
    assert p.kv_cache == "f16"
    assert "Low Memory" in p.note


# --- KV-cache estimation ----------------------------------------------------


def test_kv_estimate_prefers_real_head_counts():
    # Llama-3-8B: 32 layers, 8 KV heads (GQA), head_dim 128.
    meta = {"n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128}
    expected = 2 * 32 * 8 * 128 * 8192 * 1.0 / _GIB  # q8_0 = 1 byte/elem
    assert _estimate_kv_cache_gb(meta, 8192, "q8_0") == pytest.approx(expected)
    # The legacy heuristic said 1.5 GB for this; GQA makes it 4x smaller.
    assert _estimate_kv_cache_gb(meta, 8192, "q8_0") == pytest.approx(0.5)


def test_kv_estimate_charges_only_kv_layers_on_hybrid():
    # LFM2.5-8B-A1B: 24 blocks, but only 6 carry attention (8 KV heads each,
    # head_dim 64) — the other 18 keep a conv state. This is the number that
    # set the offload: 0.1875 GB prices 14 layers on a 4 GB card, the old
    # dense/32-head reading of the same model reserved 3.00 GB and stopped at 10.
    meta = {
        "n_layers": 24,
        "kv_layers": 6,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "embedding_length": 2048,
    }
    expected = 2 * 6 * 8 * 64 * 65536 * 0.5 / _GIB  # q4_0 = 0.5 byte/elem
    assert _estimate_kv_cache_gb(meta, 65536, "q4_0") == pytest.approx(expected)
    assert _estimate_kv_cache_gb(meta, 65536, "q4_0") == pytest.approx(0.1875)


def test_kv_estimate_scales_with_cache_dtype():
    meta = {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128}
    q8 = _estimate_kv_cache_gb(meta, 8192, "q8_0")
    f16 = _estimate_kv_cache_gb(meta, 8192, "f16")
    assert f16 == pytest.approx(q8 * 2)


def test_kv_estimate_uses_embedding_when_gqa_unknown():
    # No head_count_kv — worst case assumes n_kv_heads == n_heads, derived
    # from embedding_length // n_heads.
    meta = {"n_layers": 80, "n_heads": 64, "embedding_length": 8192}
    expected = 2 * 80 * 8192 * 8192 * 2.0 / _GIB  # f16 = 2 bytes/elem
    assert _estimate_kv_cache_gb(meta, 8192, "f16") == pytest.approx(expected)


def test_kv_estimate_can_exceed_legacy_heuristic():
    # The dangerous direction: a wide f16 model genuinely needs MORE than the
    # 1.5 GB the old heuristic assumed, so the old planner would have
    # over-committed VRAM.
    meta = {"n_layers": 80, "n_heads": 64, "embedding_length": 8192}
    assert _estimate_kv_cache_gb(meta, 8192, "f16") > 1.5


def test_kv_estimate_falls_back_to_dtype_scaled_heuristic():
    # No metadata at all: the 1.5 GB @ 8k heuristic, scaled by ctx AND by the
    # KV dtype (1.5 GB assumes f16). Before the fix a q4_0 plan inherited the
    # f16 number, so the planner thought the KV cache was 4x its real size.
    assert _estimate_kv_cache_gb({}, 8192, "f16") == pytest.approx(1.5)
    assert _estimate_kv_cache_gb({}, 4096, "f16") == pytest.approx(0.75)
    assert _estimate_kv_cache_gb({}, 8192, "q8_0") == pytest.approx(0.75)
    assert _estimate_kv_cache_gb({}, 8192, "q4_0") == pytest.approx(0.375)
    assert _estimate_kv_cache_gb({}, 8192, "f32") == pytest.approx(3.0)


def test_head_dim_derived_from_embedding_when_absent():
    meta = {"n_layers": 32, "n_kv_heads": 8, "n_heads": 32, "embedding_length": 4096}
    expected = 2 * 32 * 8 * 128 * 8192 * 1.0 / _GIB  # 4096 // 32 == 128
    assert _estimate_kv_cache_gb(meta, 8192, "q8_0") == pytest.approx(expected)


# --- hybrid attention: KV footprint covers attention layers only ------------

_HYBRID = {
    "n_layers": 72,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "size_bytes": 20 * _GIB,
}


def test_kv_estimate_counts_attention_layers_only():
    hybrid = _estimate_kv_cache_gb({**_HYBRID, "kv_layers": 18}, 8192, "q8_0")
    dense = _estimate_kv_cache_gb(_HYBRID, 8192, "q8_0")

    # 18 of 72 layers keep a KV cache; the other 54 are recurrent and hold a
    # fixed-size state that does not grow with context.
    assert hybrid == pytest.approx(dense * 18 / 72)
    per_layer = 2 * 8 * 128 * 8192 * 1.0 / _GIB
    assert hybrid == pytest.approx(18 * per_layer)


def test_hybrid_estimate_ignores_nonsense_layer_counts():
    # More KV layers than layers: clamp instead of inflating the reservation.
    assert _estimate_kv_cache_gb({**_HYBRID, "kv_layers": 200}, 8192, "q8_0") == (
        pytest.approx(_estimate_kv_cache_gb(_HYBRID, 8192, "q8_0"))
    )
    assert _estimate_kv_cache_gb({**_HYBRID, "kv_layers": 0}, 8192, "q8_0") == (
        pytest.approx(_estimate_kv_cache_gb(_HYBRID, 8192, "q8_0"))
    )


def test_plan_exposes_hybrid_kv_split():
    hw = _hw(gpu_name="RTX 3050", vram_total_gb=8.0, vram_free_gb=7.6)
    p = plan({**_HYBRID, "kv_layers": 18}, hw, ctx=8192, kv_cache="q8_0")

    assert p.kv_layers == 18
    assert p.kv_cache_gb == pytest.approx(0.28125, abs=1e-4)
    assert p.kv_cache_gb < _estimate_kv_cache_gb(_HYBRID, 8192, "q8_0")


def test_plan_kv_layers_defaults_to_all_layers():
    p = plan({"n_layers": 32, "size_bytes": 4 * _GIB}, _hw())
    assert p.kv_layers == 32


# --- newer KV cache dtypes ---------------------------------------------------


def test_new_kv_dtypes_are_sized_by_their_block_size():
    meta = {"n_layers": 8, "n_heads": 8, "n_kv_heads": 2, "head_dim": 64}

    iq4 = _estimate_kv_cache_gb(meta, 8192, "iq4_nl")
    q5_0 = _estimate_kv_cache_gb(meta, 8192, "q5_0")
    q5_1 = _estimate_kv_cache_gb(meta, 8192, "q5_1")
    bf16 = _estimate_kv_cache_gb(meta, 8192, "bf16")

    assert iq4 < q5_0 < q5_1 < bf16
    assert bf16 == pytest.approx(_estimate_kv_cache_gb(meta, 8192, "f16"))


def test_iq4_nl_is_priced_at_its_ggml_block_size():
    # 18 bytes per 32 weights. Rounding down to 0.5 under-reserves 11% of the
    # KV cache, which is the direction that offloads one layer too many.
    assert _KV_BYTES_PER_ELEM["iq4_nl"] == pytest.approx(18 / 32)


def test_cpu_plan_requires_flash_attention_only_for_quantized_kv():
    meta = {"n_layers": 32, "size_bytes": 4 * _GIB}

    # Full-precision KV must not pay the (very slow) CPU flash-attention path.
    for kv in ("f32", "f16", "bf16"):
        assert plan(meta, _hw(), kv_cache=kv).flash_attn is False

    for kv in ("q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"):
        assert plan(meta, _hw(), kv_cache=kv).flash_attn is True


# --- MoE placement -----------------------------------------------------------

# LFM2.5-8B-A1B Q4_K_M, measured on the reference GTX 1650 (4 GB, 3.64 GB free)
# with ctx 65536 and q4_0 KV. Routed experts are 93% of the file; layers 0 and
# 1 have none. The last 12 expert layers are 2441 MiB, the last 13 are 2659,
# and --n-cpu-moe 12 used 3496 MiB, which is the measured fit.
_LFM_MOE_LAYER_MIB = [
    0,
    0,
    217.9,
    189.0,
    189.0,
    217.9,
    189.0,
    189.0,
    217.9,
    189.0,
    189.0,
    217.9,
    189.0,
    189.0,
    217.9,
    189.0,
    189.0,
    217.9,
    189.0,
    189.0,
    217.9,
    217.9,
    217.9,
    217.9,
]
_LFM_MOE_LAYER_BYTES = [int(value * 1024**2) for value in _LFM_MOE_LAYER_MIB]
_LFM_MOE_META = {
    "n_layers": 24,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 64,
    "kv_layers": 6,
    "context_length": 128000,
    "size_bytes": 5_155_564_768,
    "expert_bytes": sum(_LFM_MOE_LAYER_BYTES),
    "expert_layer_bytes": _LFM_MOE_LAYER_BYTES,
    "expert_layers": 22,
}


def test_lowmem_moe_keeps_experts_in_ram_and_attention_on_the_gpu():
    # The profile promises "no routed expert in VRAM"; it should not therefore
    # give up the card as well. Measured 37.1 -> 49.9 t/s on the reference box.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    p = plan(_LFM_MOE_META, hw, profile="lowmem", ctx=8192, n_batch=512)

    assert p.moe_cpu_layers == 24  # all experts in CPU RAM == llama.cpp -cmoe
    assert p.requires_server is True
    assert p.gpu_layers == 24
    assert p.context <= 2048
    assert p.n_batch <= 64
    assert p.kv_cache == "f16"
    assert p.offload_kqv is True
    assert p.note.startswith("MoE: experts on CPU")


def test_lowmem_keeps_the_small_batch_on_a_low_ram_box():
    # The prefill buffers of a CPU-only plan are system RAM, and nothing below
    # 8 GB free was measured; keep the old cap there.
    hw = _hw(ram_total_gb=4, ram_free_gb=3.5)
    p = plan({"n_layers": 32, "size_bytes": 4 * _GIB}, hw, profile="lowmem", ctx=8192)

    assert p.gpu_layers == 0
    assert p.n_batch == 64


def test_lowmem_moe_only_asks_for_an_expert_pool_when_the_card_has_spare():
    # `--moe-cache-mib` needs a pool large enough to hold the working set: on a
    # 4 GB card the spare after attention is ~0 and the flag made a plan three
    # times slower, so it must stay off there.
    small = plan(
        _LFM_MOE_META,
        _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64),
        profile="lowmem",
        ctx=8192,
    )
    assert small.moe_cache_mib == 0

    big = plan(
        _LFM_MOE_META,
        _hw(gpu_name="RTX 4090", vram_total_gb=24.0, vram_free_gb=23.5),
        profile="lowmem",
        ctx=8192,
    )
    assert big.moe_cache_mib > 0
    assert big.moe_cache_mib % 128 == 0
    assert big.moe_cache_mib <= 4096


def test_lowmem_moe_falls_back_to_cpu_only_when_the_dense_side_is_too_big():
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=0.30)
    p = plan(_LFM_MOE_META, hw, profile="lowmem", ctx=8192, n_batch=512)

    assert p.gpu_layers == 0
    assert p.moe_cpu_layers == -1
    assert "Low Memory" in p.note


def test_lowmem_stays_cpu_only_without_a_gpu():
    p = plan(_LFM_MOE_META, _hw(), profile="lowmem", ctx=8192)
    assert p.gpu_layers == 0
    assert p.moe_cpu_layers == -1


def test_moe_plan_fits_last_expert_layers_and_keeps_classic_split():
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    p = plan(_LFM_MOE_META, hw, ctx=65536, kv_cache="q4_0")

    assert p.requires_server is True
    assert p.moe_cpu_layers == 12
    assert p.n_ubatch == 896
    # The in-process fallback must still get the classic 14-layer split, not
    # the 24 layers the server uses.
    assert p.gpu_layers == 14
    assert p.note.startswith("MoE:")


def test_moe_plan_leaves_the_expert_pool_off_when_the_card_is_full():
    # The greedy placement fills the card with expert layers, so the plan that
    # needs a pool (spare VRAM beside CPU-resident experts) is the `-cmoe` one
    # above; here the spare is smaller than the pool threshold.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    p = plan(_LFM_MOE_META, hw, ctx=65536, kv_cache="q4_0")
    assert p.moe_cpu_layers >= 0
    assert p.moe_cache_mib == 0


def test_moe_plan_honours_explicit_ubatch():
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    p = plan(_LFM_MOE_META, hw, ctx=65536, kv_cache="q4_0", n_ubatch=512)

    assert p.moe_cpu_layers == 12
    assert p.n_ubatch == 512


def test_moe_plan_needs_per_layer_expert_sizes():
    meta = dict(_LFM_MOE_META)
    meta.pop("expert_layer_bytes")

    p = plan(meta, _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64))

    assert p.moe_cpu_layers == -1
    assert p.requires_server is False


def test_moe_plan_falls_back_when_non_expert_weights_do_not_fit():
    # 4 GB of non-expert weights on a 1.2 GB free card: the expert budget is
    # negative. The planner must keep the classic split instead of crashing or
    # emitting an impossible -ncmoe value.
    per_layer = 4 * _GIB // 24
    meta = {
        "n_layers": 24,
        "size_bytes": 8 * _GIB,
        "expert_bytes": per_layer * 24,
        "expert_layer_bytes": [per_layer] * 24,
        "expert_layers": 24,
    }
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=1.2)

    p = plan(meta, hw, ctx=8192, kv_cache="q4_0")

    assert p.gpu_layers > 0
    assert p.moe_cpu_layers == -1
    assert p.requires_server is False


# --- per-side KV dtype, and KV parked in system RAM --------------------------


def test_kv_sides_are_priced_separately():
    # -ctk and -ctv are independent in llama.cpp, so the planner prices K and V
    # separately instead of assuming one dtype for both halves.
    meta = {"n_layers": 40, "n_kv_heads": 8, "head_dim": 128, "size_bytes": 9 * _GIB}
    hw = _hw(gpu_name="X", vram_total_gb=4.0, vram_free_gb=4.0)

    both = plan(meta, hw, ctx=8192, kv_cache="q8_0", kv_cache_k="q8_0", kv_cache_v="q4_0")
    same = plan(meta, hw, ctx=8192, kv_cache="q8_0")

    assert both.kv_cache_k == "q8_0"
    assert both.kv_cache_v == "q4_0"
    assert both.kv_cache_gb == pytest.approx(
        (_estimate_kv_cache_gb(meta, 8192, "q8_0") + _estimate_kv_cache_gb(meta, 8192, "q4_0")) / 2
    )
    assert both.kv_cache_gb < same.kv_cache_gb


def test_kv_in_ram_trades_vram_for_layers():
    # 64K on the reference 4 GB card: the KV cache is what holds the layer count
    # down, so parking it in system RAM lets more layers fit. Opt-in, because it
    # pays with a PCIe read of the whole cache on every token.
    meta = {"n_layers": 40, "n_kv_heads": 8, "head_dim": 128, "size_bytes": 7 * _GIB}
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)

    classic = plan(meta, hw, ctx=65536, kv_cache="q4_0")
    parked = plan(meta, hw, ctx=65536, kv_cache="q4_0", no_kv_offload=True)

    assert parked.gpu_layers > classic.gpu_layers
    assert classic.offload_kqv is True
    assert parked.offload_kqv is False
    # The card's budget did not grow, it is now spent on weights instead of on
    # weights+KV, so the host side holds roughly the same total: the cache that
    # arrives (2.5 GB, all of it) roughly matches the weights that leave. What
    # changed is that the estimate must carry the whole cache, not the slice
    # left over after the offloaded layers took theirs.
    assert parked.ram_estimate_gb == pytest.approx(classic.ram_estimate_gb, rel=0.02)


# --- tensor-type split: attention on the GPU, dense FFN on the CPU ----------

# Hermes-4-14B Q3_K_M as the analyzer reports it (measured on the reference
# box): attention with its norms is ~17% of the file over 40 layers, dense FFN
# ~70% (132 MiB per layer), and token_embd/output sit outside every block.
_DENSE_SPLIT_META = {
    "n_layers": 40,
    "n_heads": 40,
    "n_kv_heads": 8,
    "head_dim": 128,
    "size_bytes": int(6.82 * _GIB),
    "attn_bytes": 1200 * 1024**2,
    "ffn_layer_bytes": [132 * 1024**2] * 40,
    "fixed_bytes": 900 * 1024**2,
}


def test_tensor_split_keeps_attention_on_the_gpu_and_fills_the_rest_with_ffn():
    # Measured on the reference 4 GB card with a filled window: the classic
    # split fits 17/40 layers and decodes 3.44 t/s, while this plan keeps every
    # attention block (and the whole KV cache) on the card plus 4 FFN blocks and
    # decoded 4.89 t/s.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    p = plan(_DENSE_SPLIT_META, hw, ctx=8192, kv_cache="q8_0")

    # safe_vram 3.227 GB - attention 1.172 - fixed 0.879 - KV 0.625 = 0.551 GB,
    # which holds 4 x 132 MiB of FFN and not 5.
    assert p.n_cpu_ffn == 36
    assert p.requires_server is True
    assert p.note == "Tensor split: 4/40 dense FFN layers on GPU, attention on GPU"
    # The classic field still describes what an in-process fallback would load.
    assert p.gpu_layers == 17


def test_tensor_split_is_skipped_below_the_measured_crossover():
    # At 177 tokens of prompt the classic split decoded 6.34 t/s against 5.51
    # for the tensor split; the crossover sits between 177 and 1840 tokens, so a
    # plan for a small window keeps to whole layers.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    p = plan(_DENSE_SPLIT_META, hw, ctx=2048, kv_cache="q8_0")

    assert p.n_cpu_ffn == -1
    assert p.requires_server is False
    assert p.note.startswith("Hybrid: ")


def test_tensor_split_needs_the_analyzer_split_in_metadata():
    # Without per-tensor bytes there is no way to know what fits beside
    # attention; the planner must fall back to whole layers rather than guess.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)

    for missing in ("attn_bytes", "ffn_layer_bytes"):
        meta = {key: value for key, value in _DENSE_SPLIT_META.items() if key != missing}
        p = plan(meta, hw, ctx=8192, kv_cache="q8_0")
        assert p.n_cpu_ffn == -1, missing
        assert p.note.startswith("Hybrid: "), missing

    # A short per-layer list is a stale analysis, not a model with no FFN.
    p = plan(dict(_DENSE_SPLIT_META, ffn_layer_bytes=[132 * 1024**2] * 12), hw, ctx=8192)
    assert p.n_cpu_ffn == -1


def test_tensor_split_gives_up_when_attention_itself_does_not_fit():
    # No FFN block fits beside a 3 GB attention side, and "no FFN on the card"
    # is just a slower spelling of the classic split.
    meta = dict(_DENSE_SPLIT_META, attn_bytes=3 * _GIB)
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    p = plan(meta, hw, ctx=8192, kv_cache="q8_0")

    assert p.n_cpu_ffn == -1


def test_tensor_split_stays_out_of_moe_models():
    # -ncffn moves dense FFN only; routed experts belong to --n-cpu-moe, which
    # measured 58.6 t/s against 41.2 for "all experts on the CPU" on the same
    # card. A MoE model must not get both flags.
    meta = dict(
        _LFM_MOE_META,
        attn_bytes=400 * 1024**2,
        ffn_layer_bytes=[12 * 1024**2] * 24,
        fixed_bytes=205 * 1024**2,
    )
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.64)
    p = plan(meta, hw, ctx=65536, kv_cache="q4_0")

    assert p.moe_cpu_layers == 12
    assert p.n_cpu_ffn == -1


def test_tensor_split_ram_estimate_charges_only_the_cpu_ffn_blocks():
    # Attention and the KV cache are on the card, so the host side is the FFN
    # blocks of the prefix plus compute headroom -- not the whole model.
    hw = _hw(
        gpu_name="GTX 1650",
        vram_total_gb=4.0,
        vram_free_gb=3.64,
        ram_total_gb=15.4,
        ram_free_gb=10.6,
    )
    p = plan(_DENSE_SPLIT_META, hw, ctx=8192, kv_cache="q8_0")

    assert p.n_cpu_ffn == 36
    assert p.ram_estimate_gb == pytest.approx(36 * 132 / 1024 + 2.0, abs=0.02)
