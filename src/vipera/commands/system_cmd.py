import contextlib
import os
import platform
import subprocess
import sys
from pathlib import Path

import typer
from rich.text import Text

from vipera.backends.server import accelerator_device, probe_devices
from vipera.config import CONFIG_FILE, DEFAULT_CONFIG, _coerce_strict, load_config, save_config
from vipera.hardware import detect
from vipera.optimizer import OptimizerError, find_llama_tool
from vipera.ui import (
    C_DIM,
    C_GREEN,
    C_RED,
    C_TEXT,
    C_YELLOW,
    console,
    ledger,
    note,
    section,
)

app = typer.Typer()

# How to rebuild llama-cpp-python for the card that was found. The binding is
# the common case: the CLI backend already ships working binaries.
_BACKEND_BUILD = {
    "nvidia": "CMAKE_ARGS='-DGGML_CUDA=on'",
    "amd": "CMAKE_ARGS='-DGGML_HIP=on'",
    # SYCL needs the oneAPI compilers on PATH before the build starts.
    "intel": "source /opt/intel/oneapi/setvars.sh\nCMAKE_ARGS='-DGGML_SYCL=on'",
}


def _short_path(path: str) -> str:
    home = str(Path.home())
    if path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


def _cuda_venv_hint() -> str | None:
    """Point at ~/.venvs/vipera when this interpreter is a CPU-only build.

    The repository .venv and the user's CUDA venv can disagree, and the
    binding's own probe is the only reliable way to tell them apart. This is
    deliberately a best-effort diagnostic: a missing or broken alternate venv
    must not make `system doctor` fail.
    """
    root = Path.home() / ".venvs" / "vipera"
    python = root / "bin" / "python"
    if not python.is_file():
        python = root / "Scripts" / "python.exe"
    if not python.is_file():
        return None
    if Path(sys.prefix).resolve() == root.resolve():
        return None
    code = "import llama_cpp; raise SystemExit(0 if llama_cpp.llama_supports_gpu_offload() else 1)"
    try:
        probe = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return str(root) if probe.returncode == 0 else None


@app.command("doctor")
def doctor():
    """Run system diagnostics."""
    counters = {"ok": 0, "warn": 0, "fail": 0}

    def row(label: str, val: str, status: str = "ok", col: str = C_GREEN):
        counters[status] += 1
        line = Text("   ")
        line.append(f"{label:<20}", style=C_DIM)
        line.append(f"{val:<38}", style=C_TEXT)
        line.append(f"{status:>4}", style=col)
        console.print(line)

    section("runtime")
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    row("OS", os_info)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    row("python", py_ver)

    llama_missing = False
    gpu_offload = None
    try:
        import llama_cpp

        row("llama-cpp-python", "installed")

        # A binding compiled without a GPU backend silently ignores
        # n_gpu_layers: the planner still reports "9/40 layers" while every
        # layer runs on the CPU. Two virtualenvs on one machine can disagree
        # here, so this row reports the binding, not the hardware.
        supports_offload = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        try:
            gpu_offload = supports_offload() if supports_offload else None
        except Exception:
            gpu_offload = None

        if gpu_offload is True:
            row("gpu offload", "compiled in (GPU backend)", "ok", C_GREEN)
        elif gpu_offload is False:
            row("gpu offload", "CPU-only build", "warn", C_YELLOW)
        else:
            row("gpu offload", "unknown (binding too old)", "warn", C_YELLOW)
    except ImportError:
        row("llama-cpp-python", "missing", "fail", C_RED)
        llama_missing = True

    gguf_missing = False
    try:
        import gguf  # noqa: F401

        row("gguf", "installed")
    except ImportError:
        row("gguf", "missing", "warn", C_YELLOW)
        gguf_missing = True

    section("hardware")
    hw = detect()
    row("cpu", f"{hw.cpu_cores}C / {hw.cpu_threads}T")

    if hw.unified_memory:
        label = "Apple Silicon (Metal)" if hw.is_apple_silicon else hw.gpu_name
        row("gpu", label, "ok", C_GREEN)
        row("unified memory", f"{hw.vram_free_gb:.1f} GB usable / {hw.ram_total_gb:.1f} GB total")
    elif hw.gpu_name != "None":
        row("gpu", hw.gpu_name, "ok", C_GREEN)
        label = "vram (estimated)" if hw.vram_free_is_estimate else "vram"
        row(label, f"{hw.vram_free_gb:.1f} GB free / {hw.vram_total_gb:.1f} GB total")
        row("vram source", hw.vram_source)
    else:
        row("gpu", "None (CPU Only)", "warn", C_YELLOW)
        row("ram available", f"{hw.ram_free_gb:.1f} GB free / {hw.ram_total_gb:.1f} GB total")

    section("llama.cpp")
    server_bin: str | None = None
    server_gpu = False
    with contextlib.suppress(OptimizerError):
        server_bin = find_llama_tool("server")

    if server_bin:
        row("server binary", _short_path(server_bin))
        devices = probe_devices(server_bin)
        device_line = accelerator_device(devices)
        server_gpu = device_line is not None
        if device_line:
            row("server devices", device_line)
        elif devices is None:
            row("server devices", "probe failed", "warn", C_YELLOW)
        else:
            row("server devices", "none (CPU-only)", "warn", C_YELLOW)
    else:
        row("server binary", "not found", "warn", C_YELLOW)

    row("VIPERA_LLAMA_BIN", os.environ.get("VIPERA_LLAMA_BIN") or "(unset)")
    ld_path = os.environ.get("LD_LIBRARY_PATH") or "(unset)"
    row("LD_LIBRARY_PATH", ld_path)

    planner_gpu = hw.gpu_name != "None" and getattr(hw, "vram_total_gb", 0) > 0
    if planner_gpu and server_gpu:
        row("planner offload", "GPU (llama-server sees the card)")
    elif planner_gpu and server_bin:
        row("planner offload", "GPU (server sees no GPU device)", "warn", C_YELLOW)
    elif planner_gpu:
        row("planner offload", "GPU (no server binary)", "warn", C_YELLOW)
    else:
        row("planner offload", "CPU only")

    total = sum(counters.values())
    summary = Text("   ")
    summary.append(f"{total} checks", style=C_DIM)
    summary.append("   ")
    summary.append(f"{counters['ok']} ok", style=C_GREEN)
    summary.append("   ")
    summary.append(f"{counters['warn']} warn", style=C_YELLOW)
    summary.append("   ")
    summary.append(f"{counters['fail']} fail\n", style=C_RED)
    console.print(summary)

    if llama_missing:
        note(
            "llama-cpp-python is required to load models.\n"
            "python -m pip install llama-cpp-python\n"
            "then restart the gateway.",
            tone="warn",
        )

    if gpu_offload is False and hw.gpu_name != "None":
        build = _BACKEND_BUILD.get(hw.gpu_vendor, "CMAKE_ARGS='-DGGML_CUDA=on'")
        note(
            "A GPU was detected but this llama-cpp-python build has no GPU backend.\n"
            "gpu_layers is planned but ignored: every layer runs on the CPU.\n"
            "Reinstall with the backend for this card:\n"
            f"{build} pip install --force-reinstall llama-cpp-python\n"
            "then re-run this command.",
            tone="warn",
        )
        alternate = _cuda_venv_hint()
        if alternate:
            note(
                f"This Python environment is CPU-only, but {alternate} is GPU-enabled.\n"
                f"Use {alternate}/bin/vipera (or activate that venv) to run on the GPU.",
                tone="warn",
            )

    if planner_gpu and server_bin and not server_gpu:
        note(
            "The planner can offload layers, but the selected llama-server reports "
            "no GPU device.\n"
            f"binary: {_short_path(server_bin)}\n"
            f"LD_LIBRARY_PATH={ld_path}\n"
            "This server will ignore --gpu-layers and serve CPU-only. Use a build "
            "for this card (CUDA, HIP/ROCm, SYCL or Vulkan), or point "
            "LD_LIBRARY_PATH at its runtime libraries, then re-run this command.",
            tone="warn",
        )

    if gguf_missing:
        note(
            "gguf is optional, but layer and parameter metadata need it.\n"
            "python -m pip install gguf",
            tone="warn",
        )


@app.command("config")
def config_cmd(
    key: str = typer.Argument(None),
    value: str = typer.Argument(None),
):
    """View or set configuration."""
    cfg = load_config()

    if key is None:
        ledger([(k, str(v)) for k, v in cfg.items()])
        return

    if key not in DEFAULT_CONFIG:
        valid = ", ".join(sorted(DEFAULT_CONFIG.keys()))
        note(f"there is no `{key}` key.\nvalid keys: {valid}", tone="error")
        raise typer.Exit(1)

    if value is None:
        ledger([(key, str(cfg[key]))])
        return

    cfg[key] = value
    try:
        cfg = _coerce_strict(cfg)
    except ValueError as exc:
        console.print(f"[{C_RED}] ✗ {exc}[/]")
        raise typer.Exit(1) from exc
    save_config(cfg)
    ledger([(key, str(cfg[key]))])
    note(f"written to {CONFIG_FILE}", tone="ok")
