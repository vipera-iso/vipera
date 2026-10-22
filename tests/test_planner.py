import pytest

from vipera.hardware import HardwareInfo
from vipera.planner import _estimate_kv_cache_gb, plan

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
    assert p.n_batch <= 64
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

