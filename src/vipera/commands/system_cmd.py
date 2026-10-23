import platform
import sys

import typer
from rich.text import Text

from vipera.config import CONFIG_FILE, DEFAULT_CONFIG, _coerce_strict, load_config, save_config
from vipera.hardware import detect
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
            row("gpu offload", "compiled in (CUDA/Metal)", "ok", C_GREEN)
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

    if hw.is_apple_silicon:
        row("gpu", "Apple Silicon (Metal)", "ok", C_GREEN)
        row("unified memory", f"{hw.ram_free_gb:.1f} GB free / {hw.ram_total_gb:.1f} GB total")
    elif hw.gpu_name != "None":
        row("gpu", hw.gpu_name, "ok", C_GREEN)
        row("vram", f"{hw.vram_free_gb:.1f} GB free / {hw.vram_total_gb:.1f} GB total")
    else:
        row("gpu", "None (CPU Only)", "warn", C_YELLOW)
        row("ram available", f"{hw.ram_free_gb:.1f} GB free / {hw.ram_total_gb:.1f} GB total")

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
        note(
            "A GPU was detected but this llama-cpp-python build has no GPU backend.\n"
            "gpu_layers is planned but ignored: every layer runs on the CPU.\n"
            "Reinstall with the matching extra:\n"
            "CMAKE_ARGS='-DGGML_CUDA=on' pip install --force-reinstall llama-cpp-python\n"
            "then re-run this command.",
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
