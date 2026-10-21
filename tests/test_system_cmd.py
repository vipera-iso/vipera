import sys
import types

from vipera.commands import system_cmd
from vipera.hardware import HardwareInfo


def _gpu_box():
    return HardwareInfo(
        cpu_cores=4,
        cpu_threads=8,
        ram_total_gb=15.0,
        ram_free_gb=12.0,
        gpu_name="NVIDIA GeForce GTX 1650",
        vram_total_gb=4.0,
        vram_free_gb=3.6,
        is_apple_silicon=False,
    )


def test_doctor_prints_install_hints_when_runtime_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setitem(sys.modules, "gguf", None)

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "python -m pip install llama-cpp-python" in out
    assert "python -m pip install gguf" in out


def test_doctor_omits_install_hints_when_runtime_present(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "pip install llama-cpp-python" not in out
    assert "pip install gguf" not in out


def test_doctor_reports_a_cpu_only_binding_as_a_warning(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: False
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())

    system_cmd.doctor()
    out = capsys.readouterr().out

    # The hardware row still says a GPU exists; the binding row says the
    # runtime cannot use it, and the hint says what to do about it.
    assert "GPU was detected" in out
    assert "every layer runs on the CPU" in out


def test_doctor_reports_gpu_offload_when_the_binding_has_it(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: True
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "compiled in (CUDA/Metal)" in out
    assert "GPU was detected" not in out


def test_doctor_survives_a_binding_without_the_offload_probe(monkeypatch, capsys):
    # llama-cpp-python before the probe existed: attribute missing entirely.
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "unknown (binding too old)" in out
