import pytest

from vipera.hardware import HardwareInfo
from vipera.planner import _estimate_kv_cache_gb, plan

_GIB = 1024**3


def _hw(**overrides):
    base = dict(
        cpu_cores=8, cpu_threads=16,
        ram_total_gb=16, ram_free_gb=12,
        gpu_name="None", vram_total_gb=0.0, vram_free_gb=0.0,
        is_apple_silicon=False,
    )
    base.update(overrides)
    return HardwareInfo(**base)


def test_cpu_only_plan():
    p = plan({"n_layers": 32, "size_bytes": 4 * 1024**3}, _hw())
    assert p.gpu_layers == 0
    assert p.kv_cache == "f16"
    assert p.note.startswith("CPU ONLY")
    assert p.cpu_threads == 7


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
        {"n_layers": 32, "size_bytes": 4 * 1024**3}, hw,
        profile="lowmem", ctx=8192, n_batch=512,
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
