import sys
import tomllib
import types
from pathlib import Path

import pytest

from vipera import hardware


def test_nvidia_extra_declared():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    requirements = data["project"]["optional-dependencies"]["nvidia"]

    assert any(req.startswith("nvidia-ml-py") for req in requirements)


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

    info = hardware.detect()

    assert info.gpu_name == "Fake GPU"
    assert info.vram_total_gb == pytest.approx(8.0)
    assert info.vram_free_gb == pytest.approx(6.0)
