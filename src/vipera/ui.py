import random
import textwrap
import time

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from vipera import __version__

console = Console(highlight=False, soft_wrap=True)

C_PRIMARY   = "#5B7C99"
C_ACCENT    = "#8FB4CC"
C_HIGHLIGHT = "#B8C7D3"
C_TEXT      = "bold #E4E8EC"
C_DIM       = "#6B7A87"
C_BORDER    = "#3E5568"

C_GREEN   = "#7BA05B"
C_RED     = "#C25B5B"
C_YELLOW  = "#D4A574"
C_SUCCESS = C_GREEN
C_ERROR   = C_RED
C_WARN    = C_YELLOW


VIPERA_ART = r"""
   ██╗   ██╗██╗██████╗ ███████╗██████╗  █████╗
   ██║   ██║██║██╔══██╗██╔════╝██╔══██╗██╔══██╗
   ██║   ██║██║██████╔╝█████╗  ██████╔╝███████║
   ╚██╗ ██╔╝██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║
    ╚████╔╝ ██║██║     ███████╗██║  ██║██║  ██║
     ╚═══╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
"""


def heartbeat_banner():
    lines = VIPERA_ART.strip("\n").split("\n")
    max_len = max(len(line) for line in lines)
    padded = [line.ljust(max_len) for line in lines]

    def render(center: int) -> Text:
        text = Text()
        for line in padded:
            for x, ch in enumerate(line):
                dist = abs(x - center)
                if dist <= 3:
                    text.append(ch, style=C_HIGHLIGHT)
                elif dist <= 12:
                    text.append(ch, style=C_ACCENT)
                elif dist <= 26:
                    text.append(ch, style=C_PRIMARY)
                else:
                    text.append(ch, style=C_DIM)
            text.append("\n")
        return text

    with Live(console=console, refresh_per_second=60, transient=True) as live:
        for x in range(-25, max_len + 25, 2):
            live.update(render(x))
            time.sleep(0.008)
        time.sleep(0.06)
        for x in range(max_len + 25, -25, -3):
            live.update(render(x))
            time.sleep(0.008)

        final = Text()
        for line in padded:
            final.append(line + "\n", style=C_PRIMARY)
        live.update(final)
        time.sleep(0.25)


def print_banner():
    console.print()
    heartbeat_banner()

    rule = Text()
    rule.append("   " + "▬" * 74 + "\n", style=C_BORDER)
    rule.append("   ◆ ", style=C_ACCENT)
    rule.append("NODE   ", style=C_DIM)
    rule.append("localhost", style=C_TEXT)
    rule.append("     ◆ ", style=C_ACCENT)
    rule.append("VERSION  ", style=C_DIM)
    rule.append(__version__, style=C_TEXT)
    rule.append("     ◆ ", style=C_ACCENT)
    rule.append("RUNTIME  ", style=C_DIM)
    rule.append("llama.cpp\n", style=C_TEXT)
    rule.append("   " + "▬" * 74, style=C_BORDER)
    console.print(rule)
    console.print()


def animated_loading_bar(prefix: str = "INITIALIZING", duration: float = 1.2):
    width = 40
    with Live(console=console, refresh_per_second=60, transient=True) as live:
        steps = max(1, int(duration * 60))
        for step in range(steps + 1):
            progress = step / steps
            filled = int(width * progress)
            empty = width - filled
            percent = int(progress * 100)

            line = Text("  ▸ ")
            line.append(f"{prefix:<15} ", style=C_TEXT)
            line.append("█" * filled, style=C_ACCENT)
            line.append("░" * empty, style=C_BORDER)
            line.append(f"  {percent:>3}%")
            live.update(line)
            time.sleep(duration / steps)

    final = Text("  ▸ ")
    final.append(f"{prefix:<15} ", style=C_TEXT)
    final.append("█" * width, style=C_ACCENT)
    final.append("  100%  ", style=C_TEXT)
    final.append("[ ", style=C_BORDER)
    final.append("DONE", style=C_SUCCESS)
    final.append(" ]", style=C_BORDER)
    console.print(final)
    console.print()


def _wrap_block(text: str, width: int = 68) -> str:
    out = []
    for line in text.split("\n"):
        if not line:
            out.append(" " * width)
        else:
            for w in textwrap.wrap(line, width=width) or [""]:
                out.append(f"{w:<{width}}")
    return "\n".join(out)


_LIVE_UPDATE_INTERVAL = 0.03


def plain_chat_stream(stream_generator, title_text: str = " AI "):
    full_response = ""
    start_time = time.time()
    chunk_count = 0
    last_paint = 0.0

    with Live(console=console, refresh_per_second=60, transient=True) as live:
        for chunk in stream_generator:
            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
            except (KeyError, IndexError, TypeError):
                continue
            if not delta:
                continue
            chunk_count += 1

            for ch in delta:
                full_response += ch
                now = time.time()

                if now - last_paint >= _LIVE_UPDATE_INTERVAL:
                    padded = _wrap_block(full_response)
                    cursor = "█" if int(now * 3) % 2 == 0 else " "
                    panel = Panel(
                        Text(padded + cursor, style=C_TEXT),
                        title=Text(title_text, style=C_ACCENT),
                        title_align="left",
                        border_style=C_ACCENT,
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                    live.update(panel)
                    last_paint = now

                if ch in ".!?":
                    sleep = 0.14
                elif ch in ",;:":
                    sleep = 0.05
                else:
                    sleep = random.uniform(0.004, 0.018)
                time.sleep(sleep)

    padded = _wrap_block(full_response)
    final_panel = Panel(
        Text(padded, style=C_TEXT),
        title=Text(title_text, style=C_ACCENT),
        title_align="left",
        border_style=C_ACCENT,
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(final_panel)

    dt = time.time() - start_time
    tps = chunk_count / dt if dt > 0 else 0
    stat = Text("    ")
    stat.append("└─ ", style=C_BORDER)
    stat.append(f"{chunk_count} chunks  ·  {tps:.1f} c/s  ·  {dt:.2f}s\n", style=C_DIM)
    console.print(stat)
    return full_response


def pick_model(title: str = "SELECT A MODEL") -> str:
    from vipera.config import load_config
    from vipera.registry import list_models

    installed = list_models()
    if not installed:
        console.print(f"[{C_RED}] ✗ No models installed. Run `vipera model search` first.[/]")
        raise typer.Exit(1)

    cfg = load_config()
    default_model = cfg.get("default_model")

    t = Table(box=box.ROUNDED, title=title, title_justify="left")
    t.add_column("ID", style=C_DIM, justify="center")
    t.add_column("ALIAS", style=C_TEXT)
    t.add_column("SIZE", style=C_DIM, justify="right")
    t.add_column("QUANT", style=C_ACCENT)

    default_idx = 1
    for i, m in enumerate(installed):
        meta = m.get("meta", {})
        size_str = f"{meta.get('size_bytes', 0) / 1e9:>5.1f} G"
        is_default = " (default)" if m["name"] == default_model else ""
        t.add_row(
            f"[{i + 1:02d}]",
            f"{m['name']}{is_default}",
            size_str,
            meta.get("quant", "?"),
        )
        if m["name"] == default_model:
            default_idx = i + 1

    console.print(t)
    console.print()

    choice = IntPrompt.ask(
        f"[{C_ACCENT}] ▸ Select a model ID[/]",
        default=default_idx,
    )

    if choice < 1 or choice > len(installed):
        console.print(f"[{C_RED}] ✗ Invalid selection.[/]")
        raise typer.Exit(1)

    return installed[choice - 1]["name"]
