import platform
import sys

import typer
from rich import box
from rich.panel import Panel
from rich.text import Text

from vipera.config import DEFAULT_CONFIG, _coerce, load_config, save_config
from vipera.hardware import detect
from vipera.ui import (
    C_ACCENT,
    C_BORDER,
    C_DIM,
    C_GREEN,
    C_RED,
    C_TEXT,
    C_YELLOW,
    console,
)

app = typer.Typer()


@app.command("doctor")
def doctor():
    """Run system diagnostics."""
    content = Text()
    counters = {"pass": 0, "warn": 0, "fail": 0}

    def add_sec(title: str):
        content.append(f"  ┌─[ {title} ]", style=C_ACCENT)
        content.append("─" * max(0, 62 - len(title)), style=C_BORDER)
        content.append("┐\n", style=C_BORDER)

    def end_sec():
        content.append("  └" + "─" * 72 + "┘\n\n", style=C_BORDER)

    def row(label: str, val: str, status: str = "PASS", col: str = C_GREEN):
        counters[status.lower()] += 1
        pad = "·" * max(0, 42 - len(label) - len(val))
        content.append(f"  │ {label:<20} {val} {pad} ", style=C_TEXT)
        content.append(f"[ {status:^4} ] │\n", style=col)

    add_sec("RUNTIME")
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    row("OS", os_info)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    row("python", py_ver)

    try:
        import llama_cpp  # noqa: F401
        row("llama-cpp-python", "installed")
    except ImportError:
        row("llama-cpp-python", "missing", "FAIL", C_RED)

    try:
        import gguf  # noqa: F401
        row("gguf", "installed")
    except ImportError:
        row("gguf", "missing", "WARN", C_YELLOW)
    end_sec()

    add_sec("HARDWARE")
    hw = detect()
    row("cpu", f"{hw.cpu_cores}C / {hw.cpu_threads}T")

    if hw.is_apple_silicon:
        row("gpu", "Apple Silicon (Metal)", "PASS", C_GREEN)
        row("unified memory",
            f"{hw.ram_free_gb:.1f} GB free / {hw.ram_total_gb:.1f} GB total")
    elif hw.gpu_name != "None":
        row("gpu", hw.gpu_name, "PASS", C_GREEN)
        row("vram",
            f"{hw.vram_free_gb:.1f} GB free / {hw.vram_total_gb:.1f} GB total")
    else:
        row("gpu", "None (CPU Only)", "WARN", C_YELLOW)
        row("ram available",
            f"{hw.ram_free_gb:.1f} GB free / {hw.ram_total_gb:.1f} GB total")
    end_sec()

    summary = Text("  SUMMARY   ", style=C_ACCENT)
    summary.append(f"[ {counters['pass']} PASSED ]", style=C_GREEN)
    summary.append("  ─  ", style=C_BORDER)
    summary.append(f"[ {counters['warn']} WARN ]", style=C_YELLOW)
    summary.append("  ─  ", style=C_BORDER)
    summary.append(f"[ {counters['fail']} FAIL ]\n", style=C_RED)
    content.append(summary)

    console.print(Panel(
        content,
        title=Text(" VIPERA DIAGNOSTICS ", style=C_ACCENT),
        border_style=C_ACCENT,
        box=box.ROUNDED,
        padding=(0, 1),
    ))


@app.command("config")
def config_cmd(
    key: str = typer.Argument(None),
    value: str = typer.Argument(None),
):
    """View or set configuration."""
    cfg = load_config()

    if key is None:
        for k, v in cfg.items():
            line = Text("  ")
            line.append(f"{k:<20}", style=C_DIM)
            line.append(f"{v}", style=C_TEXT)
            console.print(line)
        return

    if key not in DEFAULT_CONFIG:
        valid = ", ".join(sorted(DEFAULT_CONFIG.keys()))
        console.print(f"[{C_RED}] ✗ Unknown key: {key}[/]")
        console.print(f"[{C_DIM}]  Valid keys: {valid}[/]")
        raise typer.Exit(1)

    if value is None:
        console.print(f"[{C_ACCENT}]{key}[/] = [{C_GREEN}]{cfg[key]}[/]")
        return

    cfg[key] = value
    cfg = _coerce(cfg)
    save_config(cfg)
    console.print(f"[{C_GREEN}] ✓[/]  {key} = {cfg[key]}")
