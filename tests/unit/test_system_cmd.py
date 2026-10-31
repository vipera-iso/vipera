import subprocess
import sys
import types
from dataclasses import replace

from vipera.commands import system_cmd
from vipera.hardware import HardwareInfo
from vipera.optimizer import OptimizerError


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
    monkeypatch.setattr(system_cmd, "_cuda_venv_hint", lambda: None)

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

    assert "compiled in (GPU backend)" in out
    assert "GPU was detected" not in out


def test_doctor_survives_a_binding_without_the_offload_probe(monkeypatch, capsys):
    # llama-cpp-python before the probe existed: attribute missing entirely.
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "unknown (binding too old)" in out


def test_doctor_reports_the_selected_server_and_cuda0(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: True
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())
    monkeypatch.setattr(system_cmd, "find_llama_tool", lambda tool: "/opt/llama/llama-server")
    monkeypatch.setattr(
        system_cmd,
        "probe_devices",
        lambda binary: "Available devices:\n  CUDA0: NVIDIA GeForce GTX 1650\n",
    )

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "/opt/llama/llama-server" in out
    assert "CUDA0: NVIDIA GeForce GTX 1650" in out
    assert "llama-server sees the card" in out


def test_doctor_warns_when_the_server_sees_no_gpu_device(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: True
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())
    monkeypatch.setattr(system_cmd, "find_llama_tool", lambda tool: "/opt/llama/llama-server")
    monkeypatch.setattr(
        system_cmd,
        "probe_devices",
        lambda binary: "Available devices:\n  (none)\n",
    )

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "server sees no GPU device" in out
    assert "LD_LIBRARY_PATH" in out
    assert "CPU-only" in out


def test_doctor_points_at_a_cuda_venv_when_this_one_is_cpu_only(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: False
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())

    def missing(tool):
        raise OptimizerError("llama-server was not found.")

    monkeypatch.setattr(system_cmd, "find_llama_tool", missing)
    monkeypatch.setattr(system_cmd, "_cuda_venv_hint", lambda: "~/.venvs/vipera")

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "~/.venvs/vipera is GPU-enabled" in out
    assert "CPU-only" in out


def test_doctor_suggests_the_backend_for_an_amd_card(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: False
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    amd = replace(
        _gpu_box(),
        gpu_name="AMD Radeon RX 6700 XT",
        gpu_vendor="amd",
        vram_total_gb=12.0,
        vram_free_gb=11.0,
        vram_source="amdgpu-sysfs",
    )
    monkeypatch.setattr(system_cmd, "detect", lambda: amd)

    def missing(tool):
        raise OptimizerError("llama-server was not found.")

    monkeypatch.setattr(system_cmd, "find_llama_tool", missing)

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "AMD Radeon RX 6700 XT" in out
    assert "amdgpu-sysfs" in out
    assert "-DGGML_HIP=on" in out


def test_doctor_labels_an_estimated_vram_reading(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: True
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    intel = replace(
        _gpu_box(),
        gpu_name="Intel(R) Arc(TM) A770 Graphics",
        gpu_vendor="intel",
        vram_source="xpu-smi",
        vram_free_is_estimate=True,
    )
    monkeypatch.setattr(system_cmd, "detect", lambda: intel)
    monkeypatch.setattr(system_cmd, "find_llama_tool", lambda tool: "/opt/llama/llama-server")
    monkeypatch.setattr(
        system_cmd,
        "probe_devices",
        lambda binary: "Available devices:\n  SYCL0: Intel(R) Arc(TM) A770 Graphics\n",
    )

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "vram (estimated)" in out
    assert "xpu-smi" in out
    assert "SYCL0: Intel(R) Arc(TM) A770 Graphics" in out


def test_doctor_survives_a_missing_server_binary(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: True
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())

    def missing(tool):
        raise OptimizerError("llama-server was not found.")

    monkeypatch.setattr(system_cmd, "find_llama_tool", missing)

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "server binary" in out
    assert "not found" in out


class _FakeSubprocess:
    """Records what `setup` would run instead of running it."""

    SubprocessError = subprocess.SubprocessError

    def __init__(self):
        self.calls = []

    def run(self, cmd, **kwargs):
        self.calls.append({"cmd": list(cmd), "env": kwargs.get("env")})
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _missing_binding():
    """`sys.modules` entry that makes `import llama_cpp` raise ImportError."""
    return None


def test_setup_builds_a_cuda_backend_for_an_nvidia_card(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llama_cpp", _missing_binding())
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())
    fake = _FakeSubprocess()
    monkeypatch.setattr(system_cmd, "subprocess", fake)

    system_cmd.setup()

    assert len(fake.calls) == 1
    assert fake.calls[0]["cmd"][:3] == [sys.executable, "-m", "pip"]
    assert "llama-cpp-python" in fake.calls[0]["cmd"]
    assert "gguf" not in fake.calls[0]["cmd"]
    assert fake.calls[0]["env"]["CMAKE_ARGS"] == "-DGGML_CUDA=on"
    assert "pip" in capsys.readouterr().out


def test_setup_asks_for_the_backend_of_the_card_that_is_present(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llama_cpp", _missing_binding())
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    amd = replace(
        _gpu_box(),
        gpu_name="AMD Radeon RX 6700 XT",
        gpu_vendor="amd",
        vram_total_gb=12.0,
        vram_free_gb=11.0,
        vram_source="amdgpu-sysfs",
    )
    monkeypatch.setattr(system_cmd, "detect", lambda: amd)
    fake = _FakeSubprocess()
    monkeypatch.setattr(system_cmd, "subprocess", fake)

    system_cmd.setup()

    assert fake.calls[0]["env"]["CMAKE_ARGS"] == "-DGGML_HIP=on"


def test_setup_installs_gguf_without_rebuilding_the_binding(monkeypatch, capsys):
    # A CPU-capable binding that already offloads is not worth a rebuild: the
    # only missing piece is the metadata reader.
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: True
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", None)
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())
    fake = _FakeSubprocess()
    monkeypatch.setattr(system_cmd, "subprocess", fake)

    system_cmd.setup()

    assert len(fake.calls) == 1
    assert fake.calls[0]["cmd"][-1] == "gguf"
    assert "llama-cpp-python" not in fake.calls[0]["cmd"]
    assert fake.calls[0]["env"] is None or "CMAKE_ARGS" not in (fake.calls[0]["env"] or {})


def test_setup_rebuilds_a_cpu_only_binding_for_the_detected_card(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: False
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())
    fake = _FakeSubprocess()
    monkeypatch.setattr(system_cmd, "subprocess", fake)

    system_cmd.setup()

    assert len(fake.calls) == 1
    assert "--force-reinstall" in fake.calls[0]["cmd"]
    assert "llama-cpp-python" in fake.calls[0]["cmd"]
    assert fake.calls[0]["env"]["CMAKE_ARGS"] == "-DGGML_CUDA=on"


def test_setup_is_a_noop_when_the_runtime_is_complete(monkeypatch, capsys):
    module = types.ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: True
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())
    fake = _FakeSubprocess()
    monkeypatch.setattr(system_cmd, "subprocess", fake)

    system_cmd.setup()

    assert fake.calls == []
    assert "nothing to install" in capsys.readouterr().out


def test_setup_dry_run_prints_the_command_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llama_cpp", _missing_binding())
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(system_cmd, "detect", lambda: _gpu_box())
    fake = _FakeSubprocess()
    monkeypatch.setattr(system_cmd, "subprocess", fake)

    system_cmd.setup(True)

    assert fake.calls == []
    out = capsys.readouterr().out
    assert "pip" in out
    assert "llama-cpp-python" in out
