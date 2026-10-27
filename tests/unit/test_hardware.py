import json
import sys
import tomllib
import types
from pathlib import Path

import pytest

from vipera import hardware

GIB = 1024**3


def test_nvidia_extra_declared():
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    requirements = data["project"]["optional-dependencies"]["nvidia"]

    assert any(req.startswith("nvidia-ml-py") for req in requirements)


def _result(text: str):
    return types.SimpleNamespace(returncode=0, stdout=text, stderr="")


def _probe_only(monkeypatch, probe):
    """Run `detect()` against one probe, as if no other vendor tool existed."""
    # detect() answers Apple Silicon before it consults the probes at all, so a
    # probe test must not run on a Mac.
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_PROBES", (probe,))


def _amdgpu(root: Path, **files: str) -> Path:
    device = root / "card1" / "device"
    device.mkdir(parents=True)
    for name, value in files.items():
        (device / name).write_text(f"{value}\n")
    return device


def test_detect_reads_amdgpu_sysfs(monkeypatch, tmp_path):
    _amdgpu(
        tmp_path,
        mem_info_vram_total=str(8 * GIB),
        mem_info_vram_used=str(2 * GIB),
        product_name="AMD Radeon RX 6700 XT",
    )
    _probe_only(monkeypatch, hardware._from_amdgpu_sysfs)
    monkeypatch.setattr(hardware, "_DRM_ROOT", tmp_path)

    info = hardware.detect()

    assert info.gpu_vendor == "amd"
    assert info.gpu_name == "AMD Radeon RX 6700 XT"
    assert info.vram_total_gb == pytest.approx(8.0)
    assert info.vram_free_gb == pytest.approx(6.0)
    assert info.vram_source == "amdgpu-sysfs"
    assert info.unified_memory is False
    assert info.vram_free_is_estimate is False


def test_amdgpu_apu_counts_gtt_as_unified_memory(monkeypatch, tmp_path):
    # An APU has a small BIOS carve-out and a GTT window backed by system RAM.
    _amdgpu(
        tmp_path,
        mem_info_vram_total=str(GIB // 2),
        mem_info_vram_used=str(0),
        mem_info_gtt_total=str(8 * GIB),
        mem_info_gtt_used=str(GIB),
        product_name="AMD Radeon 780M",
    )
    _probe_only(monkeypatch, hardware._from_amdgpu_sysfs)
    monkeypatch.setattr(hardware, "_DRM_ROOT", tmp_path)

    info = hardware.detect()

    assert info.unified_memory is True
    assert info.vram_total_gb == pytest.approx(8.5)
    assert info.vram_free_gb == pytest.approx(7.5)


def test_amdgpu_one_gb_card_with_a_gtt_window_is_not_read_as_an_apu(monkeypatch, tmp_path):
    # A discrete card exposes a GTT aperture too, sized from system RAM. On a
    # 1 GB card that window is several times the VRAM, which used to read as an
    # APU and hand the planner ~9 GB to spend on a card that holds one.
    _amdgpu(
        tmp_path,
        mem_info_vram_total=str(GIB),
        mem_info_vram_used=str(0),
        mem_info_gtt_total=str(8 * GIB),
        mem_info_gtt_used=str(GIB),
        product_name="AMD Radeon R7 240",
    )
    _probe_only(monkeypatch, hardware._from_amdgpu_sysfs)
    monkeypatch.setattr(hardware, "_DRM_ROOT", tmp_path)

    info = hardware.detect()

    assert info.unified_memory is False
    assert info.vram_total_gb == pytest.approx(1.0)
    assert info.vram_free_gb == pytest.approx(1.0)


def test_detect_reads_rocm_smi_json(monkeypatch):
    payload = {
        "card0": {
            "Card series": "Radeon RX 7900 XTX",
            "VRAM Total Memory (B)": str(24 * GIB),
            "VRAM Total Used Memory (B)": str(2 * GIB),
        }
    }
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return _result(json.dumps(payload))

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    _probe_only(monkeypatch, hardware._from_rocm_smi)

    info = hardware.detect()

    assert commands[0][0] == "rocm-smi"
    assert info.gpu_vendor == "amd"
    assert info.gpu_name == "Radeon RX 7900 XTX"
    assert info.vram_total_gb == pytest.approx(24.0)
    assert info.vram_free_gb == pytest.approx(22.0)
    assert info.vram_source == "rocm-smi"


def test_rocm_smi_reads_capacity_even_when_used_is_listed_first(monkeypatch):
    # Both keys contain "vram total"; key order must not decide whether the
    # capacity or the usage becomes total_gb.
    payload = {
        "card0": {
            "VRAM Total Used Memory (B)": str(2 * GIB),
            "VRAM Total Memory (B)": str(24 * GIB),
        }
    }
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **kw: _result(json.dumps(payload)))
    _probe_only(monkeypatch, hardware._from_rocm_smi)

    info = hardware.detect()

    assert info.vram_total_gb == pytest.approx(24.0)
    assert info.vram_free_gb == pytest.approx(22.0)


def test_detect_reads_xpu_smi_json(monkeypatch):
    payload = {
        "device_list": [
            {
                "device_id": 0,
                "device_name": "Intel(R) Arc(TM) A770 Graphics",
                "memory_physical_size_byte": str(16 * GIB),
            }
        ]
    }
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **kw: _result(json.dumps(payload)))
    _probe_only(monkeypatch, hardware._from_xpu_smi)

    info = hardware.detect()

    assert info.gpu_vendor == "intel"
    assert info.gpu_name == "Intel(R) Arc(TM) A770 Graphics"
    assert info.vram_total_gb == pytest.approx(16.0)
    # xpu-smi's discovery call reports capacity, not free memory, so the probe
    # scales it down and says the number is an estimate.
    assert info.vram_free_gb == pytest.approx(12.8)
    assert info.vram_free_is_estimate is True


def test_detect_falls_back_to_the_llama_server_device_list(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "_find_server_binary",
        lambda: "/opt/llama/llama-server",
    )
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *a, **kw: _result(
            "Available devices:\n"
            "  SYCL0: Intel(R) Arc(TM) A770 Graphics (16384 MiB, 15000 MiB free)\n"
        ),
    )
    _probe_only(monkeypatch, hardware._from_llama_server)

    info = hardware.detect()

    assert info.gpu_vendor == "intel"
    assert info.vram_total_gb == pytest.approx(16.0)
    assert info.vram_free_gb == pytest.approx(15000 / 1024)
    assert info.vram_source == "llama-server"
    assert info.vram_free_is_estimate is False


def test_llama_server_estimates_free_memory_from_capacity_only_output(monkeypatch):
    # Older builds print "(3730 MiB)" with no free-memory reading. Falling
    # through here made a machine with a working GPU plan as CPU-only.
    monkeypatch.setattr(hardware, "_find_server_binary", lambda: "/opt/llama/llama-server")
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *a, **kw: _result(
            "Available devices:\n  CUDA0: NVIDIA GeForce GTX 1650 (3730 MiB)\n"
        ),
    )
    _probe_only(monkeypatch, hardware._from_llama_server)

    info = hardware.detect()

    assert info.gpu_name == "NVIDIA GeForce GTX 1650"
    assert info.vram_total_gb == pytest.approx(3730 / 1024)
    assert info.vram_free_gb == pytest.approx(3730 / 1024 * 0.8)
    assert info.vram_free_is_estimate is True


def test_detect_reports_cpu_only_when_every_probe_fails(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_PROBES", ())

    info = hardware.detect()

    assert info.gpu_name == "None"
    assert info.gpu_vendor == "none"
    assert info.vram_total_gb == 0.0
    assert info.unified_memory is False


def test_apple_silicon_reports_unified_memory(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "_PROBES", ())

    info = hardware.detect()

    assert info.is_apple_silicon is True
    assert info.unified_memory is True
    assert info.gpu_vendor == "apple"
    assert info.vram_total_gb == pytest.approx(info.ram_total_gb)


def test_detect_reads_nvml(monkeypatch):
    fake = types.ModuleType("pynvml")
    fake.nvmlInit = lambda: None
    fake.nvmlDeviceGetHandleByIndex = lambda index: object()
    fake.nvmlDeviceGetName = lambda handle: b"Fake GPU"
    fake.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(
        total=8 * 1024**3,
        free=6 * 1024**3,
    )

    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_PROBES", (hardware._from_nvml,))

    info = hardware.detect()

    assert info.gpu_name == "Fake GPU"
    assert info.vram_total_gb == pytest.approx(8.0)
    assert info.vram_free_gb == pytest.approx(6.0)
    assert info.gpu_vendor == "nvidia"
    assert info.vram_source == "nvml"
