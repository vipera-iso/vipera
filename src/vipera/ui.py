"""Terminal presentation for Vipera.

The rules this file implements are written down in
``docs/design/visual-system.md``. In short: the terminal shares one identity
with ``docs/assets`` (ink / paper / cyan), numbers are right aligned and always
carry units, and motion never delays work.
"""

import os
import textwrap
import time

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from vipera import __version__

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------

_PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "paper": "#F1F5F9",
        "rule": "#1E293B",
        "muted": "#94A3B8",
        "accent": "#22D3EE",
        "ok": "#34D399",
        "warn": "#F5A524",
        "error": "#F87171",
    },
    # Cyan-400 (#22D3EE) is 1.6:1 on white; the light theme uses cyan-700 so
    # the brand colour stays readable instead of turning into a highlight.
    "light": {
        "paper": "#0B0F14",
        "rule": "#CBD5E1",
        "muted": "#52606D",
        "accent": "#0E7490",
        "ok": "#047857",
        "warn": "#B45309",
        "error": "#B91C1C",
    },
    # NO_COLOR, or VIPERA_THEME=none: every token becomes Rich's `default`
    # style, so `[<style>]…[/]` markup in command modules stays valid but the
    # output is plain enough to grep.
    "none": {
        "paper": "default",
        "rule": "default",
        "muted": "default",
        "accent": "default",
        "ok": "default",
        "warn": "default",
        "error": "default",
    },
}


def _resolve_theme() -> str:
    """Pick a palette: explicit override, then NO_COLOR, then the terminal."""
    forced = os.environ.get("VIPERA_THEME", "").strip().lower()
    if forced in _PALETTES:
        return forced
    if os.environ.get("NO_COLOR"):
        return "none"
    # COLORFGBG is "fg;bg" with ANSI indexes. 7+ means a light background.
    fgbg = os.environ.get("COLORFGBG", "")
    if ";" in fgbg:
        try:
            if int(fgbg.rsplit(";", 1)[1]) >= 7:
                return "light"
        except ValueError:
            pass
    return "dark"


THEME = _resolve_theme()
_P = _PALETTES[THEME]

# Semantic tokens. The C_* names are the ones command modules import, so they
# stay put even though the palette behind them is now theme-dependent.
C_TEXT = f"bold {_P['paper']}"
C_DIM = _P["muted"]
C_ACCENT = _P["accent"]
C_BORDER = _P["rule"]
C_GREEN = _P["ok"]
C_RED = _P["error"]
C_YELLOW = _P["warn"]
C_SUCCESS = C_GREEN
C_ERROR = C_RED
C_WARN = C_YELLOW
C_PRIMARY = C_ACCENT
C_HIGHLIGHT = C_TEXT

console = Console(highlight=False, soft_wrap=True)

GUTTER = "▏"
RULE_CHAR = "─"
INDENT = "   "


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


def rule(width: int | None = None, indent: str = INDENT) -> None:
    """A hairline. Used to close a section, never to decorate a gap."""
    span = width if width is not None else min(72, max(8, console.width - len(indent)))
    console.print(f"{indent}{RULE_CHAR * span}", style=C_BORDER)


def section(title: str) -> None:
    """Sentence-case heading with the hairline directly beneath it."""
    console.print()
    console.print(f"{INDENT}{title}", style=C_DIM)
    rule()


def ledger(rows: list[tuple[str, str]], indent: str = INDENT, key_width: int | None = None) -> None:
    """Key/value block with a fixed label column.

    The label column is as wide as the widest label, so values line up no
    matter which block printed them.
    """
    if not rows:
        return
    width = key_width or max(len(key) for key, _ in rows)
    for key, value in rows:
        line = Text(indent)
        line.append(key.ljust(width), style=C_DIM)
        line.append("  ")
        line.append(value, style=C_TEXT)
        console.print(line)


def catalog(rows: list[tuple[str, str]], indent: str = INDENT) -> None:
    """A command list: the thing you type in paper, what it does in muted.

    ``ledger`` puts the value in paper because values are the answer; here the
    command is the answer and the description is the aside.
    """
    if not rows:
        return
    width = max(len(cmd) for cmd, _ in rows)
    for command, description in rows:
        line = Text(indent)
        line.append(command.ljust(width), style=C_TEXT)
        line.append("  ")
        line.append(description, style=C_DIM)
        console.print(line)


def note(body: str, tone: str = "accent") -> None:
    """A left-ruled block: guidance, warnings, next steps.

    Boxes are reserved for things that stand on their own (a chat reply); a
    note is attached to whatever came before it, so it gets a gutter, not a
    frame.
    """
    colour = {"accent": C_ACCENT, "warn": C_YELLOW, "error": C_RED, "ok": C_GREEN}.get(
        tone, C_ACCENT
    )
    for index, line in enumerate(body.split("\n")):
        prefix = Text(f"{INDENT}{GUTTER} ", style=colour)
        prefix.append(line, style=C_TEXT if index == 0 else C_DIM)
        console.print(prefix)


def gauge(
    label: str,
    used: float,
    total: float,
    unit: str = "GB",
    width: int = 24,
    fmt: str = "{:.2f}",
) -> None:
    """A capacity bar. The one place in the CLI where colour encodes amount."""
    total = total or 1.0
    fraction = max(0.0, min(1.0, used / total))
    filled = round(fraction * width)
    line = Text(INDENT)
    line.append(f"{label:<10}", style=C_DIM)
    line.append("█" * filled, style=C_ACCENT)
    line.append("░" * (width - filled), style=C_BORDER)
    line.append(f"  {fmt.format(used)} / {fmt.format(total)} {unit}", style=C_TEXT)
    console.print(line)


def _guttered(text: str, style: str) -> Text:
    """Render multi-line text behind a hairline gutter."""
    lines = text.split("\n")
    out = Text()
    for index, line in enumerate(lines):
        out.append(f"{INDENT}{GUTTER} ", style=C_BORDER)
        out.append(line, style=style)
        if index < len(lines) - 1:
            out.append("\n")
    return out


# --------------------------------------------------------------------------
# Brand
# --------------------------------------------------------------------------

VIPERA_ART = r"""
   ██╗   ██╗██╗██████╗ ███████╗██████╗  █████╗
   ██║   ██║██║██╔══██╗██╔════╝██╔══██╗██╔══██╗
   ██║   ██║██║██████╔╝█████╗  ██████╔╝███████║
   ╚██╗ ██╔╝██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║
    ╚████╔╝ ██║██║     ███████╗██║  ██║██║  ██║
     ╚═══╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
"""


def print_banner(host: str = "localhost", *, backend: str | None = None) -> None:
    """Print the wordmark and the three facts needed before the first load.

    This used to sweep a highlight across the letters for ~1.9 s. Nothing is
    loaded during that time, so the animation only delayed the prompt.
    """
    console.print()
    for line in VIPERA_ART.strip("\n").split("\n"):
        console.print(line, style=C_ACCENT)
    console.print()
    rows = [("node", host), ("version", __version__)]
    if backend:
        rows.append(("backend", backend))
    ledger(rows)
    rule()
    console.print()


# --------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------


def animated_loading_bar(prefix: str = "initialising", duration: float = 1.2) -> None:
    """Fill a gauge while a blocking step runs (import, quantize, bench)."""
    width = 36
    steps = max(1, int(duration * 30))
    with Live(console=console, refresh_per_second=30, transient=True) as live:
        for step in range(steps + 1):
            progress = step / steps
            filled = round(width * progress)
            line = Text(INDENT)
            line.append(f"{prefix:<14} ", style=C_DIM)
            line.append("█" * filled, style=C_ACCENT)
            line.append("░" * (width - filled), style=C_BORDER)
            line.append(f"  {round(progress * 100):>3}%")
            live.update(line)
            time.sleep(duration / steps)

    final = Text(INDENT)
    final.append(f"{prefix:<14} ", style=C_DIM)
    final.append("█" * width, style=C_ACCENT)
    final.append("   100%", style=C_DIM)
    console.print(final)
    console.print()


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

_WRAP_WIDTH = 68
_PAINT_INTERVAL = 0.03


def _wrap_block(text: str, width: int = _WRAP_WIDTH) -> str:
    """Wrap text without destroying the indentation of code blocks."""
    out = []
    for line in text.split("\n"):
        if not line:
            out.append("")
            continue
        line = line.expandtabs(4)
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        for wrapped in wrapper.wrap(stripped) or [indent]:
            out.append(wrapped.rstrip())
    return "\n".join(out)


def _reply(text: str, cursor: bool = False, reasoning: str = "") -> Text:
    block = Text()
    if reasoning:
        block.append_text(_guttered(_wrap_block(reasoning), C_DIM))
    if text:
        if reasoning:
            block.append("\n")
        block.append_text(_guttered(_wrap_block(text), C_TEXT))
    if cursor:
        block.append("█" if int(time.time() * 3) % 2 == 0 else " ", style=C_ACCENT)
    return block


def _stats(tokens: int, seconds: float) -> Text:
    rate = tokens / seconds if seconds > 0 else 0.0
    line = Text(INDENT)
    line.append(f"{GUTTER} ", style=C_BORDER)
    line.append(f"{tokens} tokens", style=C_DIM)
    line.append("  ")
    line.append(f"{rate:.1f} tok/s", style=C_DIM)
    line.append("  ")
    line.append(f"{seconds:.2f} s", style=C_DIM)
    return line


def _stream(stream_generator, title_text: str, refresh: int) -> str:
    answer: list[str] = []
    reasoning: list[str] = []
    chunk_count = 0
    last_paint = 0.0
    started = time.time()

    with Live(console=console, refresh_per_second=refresh, transient=True) as live:
        for chunk in stream_generator:
            try:
                delta = chunk["choices"][0].get("delta") or {}
            except (KeyError, IndexError, TypeError):
                continue
            # Reasoning models stream `reasoning_content` before `content`.
            # Paint both: dropping the reasoning left the screen blank (and the
            # token count at 0) until the answer started.
            content = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if not content and not thought:
                continue
            chunk_count += 1
            if content:
                answer.append(content)
            if thought:
                reasoning.append(thought)

            # Painting is throttled; the model is not. The old code slept up to
            # 18 ms per character, which capped a fast model at ~120 tok/s.
            now = time.time()
            if now - last_paint >= _PAINT_INTERVAL:
                live.update(_reply("".join(answer), cursor=True, reasoning="".join(reasoning)))
                last_paint = now

    text = "".join(answer)
    if title_text:
        console.print(f"{INDENT}{title_text}", style=C_DIM)
    console.print(_reply(text, reasoning="".join(reasoning)))
    console.print(_stats(chunk_count, time.time() - started))
    console.print()
    return text


def typewriter_chat_stream(stream_generator, title_text: str = "assistant") -> str:
    """Stream a reply, repainting as tokens arrive."""
    return _stream(stream_generator, title_text, refresh=30)


def plain_chat_stream(stream_generator, title_text: str = "assistant") -> str:
    """Stream a reply with a slower repaint, for slow terminals."""
    return _stream(stream_generator, title_text, refresh=10)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def pick_model(title: str = "select a model") -> str:
    from vipera.config import load_config
    from vipera.registry import list_models

    installed = list_models()
    if not installed:
        note("no models installed. run `vipera model search` first.", tone="error")
        raise typer.Exit(1)

    cfg = load_config()
    default_model = cfg.get("default_model")
    default_idx = 1

    section(title)
    t = Table(box=box.SIMPLE_HEAD, padding=(0, 2), pad_edge=False, show_edge=False)
    t.add_column("#", style=C_DIM, justify="right")
    t.add_column("model", style=C_TEXT)
    t.add_column("size", style=C_DIM, justify="right")
    t.add_column("quant", style=C_ACCENT, justify="right")

    for i, model in enumerate(installed):
        meta = model.meta
        marker = "" if model.alias != default_model else "  default"
        t.add_row(
            f"{i + 1}",
            f"{model.alias}{marker}",
            f"{meta.get('size_bytes', 0) / 1e9:.2f} GB",
            meta.get("quant", "?"),
        )
        if model.alias == default_model:
            default_idx = i + 1

    console.print(t)
    console.print()

    choice = IntPrompt.ask(f"{INDENT}{GUTTER} select a model", default=default_idx)

    if choice < 1 or choice > len(installed):
        note("that number is not in the list.", tone="error")
        raise typer.Exit(1)

    return installed[choice - 1].alias
