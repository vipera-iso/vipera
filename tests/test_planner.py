from vipera.hardware import HardwareInfo
from vipera.planner import plan


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
