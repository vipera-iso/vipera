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
    assert p.cpu_threads == 7


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

    # safe_vram 2.4 GB (1 GB driver + 0.3 GB compute buffers); layer 0.225 GB
    # + per-layer KV 0.0625 GB -> 8 layers.
    assert p.gpu_layers == 8
    assert p.offload_kqv is True
    assert p.note == "Hybrid: 8/40 layers"


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


def test_real_layer_bytes_offloads_more_than_size_divided_by_layers():
    # The GGUF index reports ~0.18 GB per repeating layer for this model; the
    # remaining ~1 GB is token_embd + output.weight, which never follow a layer
    # to the GPU. size_bytes/n_layers charges that 1 GB to every layer, so the
    # old arithmetic settled one layer lower.
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.7)
    meta = dict(_HERMES_META, layer_bytes=int(0.1844 * _GIB))

    measured = plan(meta, hw, ctx=8192, kv_cache="f16")
    estimated = plan(_HERMES_META, hw, ctx=8192, kv_cache="f16")

    # safe_vram 2.4 GB; 0.1844 + 0.03125 per layer -> 11 layers, versus 9 for
    # the 0.225 GB/layer the old size/n_layers arithmetic assumed.
    assert measured.gpu_layers == 11
    assert estimated.gpu_layers == 9
    assert measured.gpu_layers > estimated.gpu_layers


def test_layer_bytes_zero_falls_back_to_size_estimate():
    hw = _hw(gpu_name="GTX 1650", vram_total_gb=4.0, vram_free_gb=3.7)
    meta = dict(_HERMES_META, layer_bytes=0)
    assert plan(meta, hw, ctx=8192, kv_cache="f16").gpu_layers == 9


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


def test_kv_estimate_falls_back_to_legacy_heuristic():
    # No metadata at all: exactly the pre-patch behaviour, still scaled by ctx.
    assert _estimate_kv_cache_gb({}, 8192, "q8_0") == pytest.approx(1.5)
    assert _estimate_kv_cache_gb({}, 4096, "q8_0") == pytest.approx(0.75)


def test_head_dim_derived_from_embedding_when_absent():
    meta = {"n_layers": 32, "n_kv_heads": 8, "n_heads": 32, "embedding_length": 4096}
    expected = 2 * 32 * 8 * 128 * 8192 * 1.0 / _GIB  # 4096 // 32 == 128
    assert _estimate_kv_cache_gb(meta, 8192, "q8_0") == pytest.approx(expected)
