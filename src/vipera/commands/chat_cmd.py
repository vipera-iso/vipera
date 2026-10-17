import typer
from rich import box
from rich.panel import Panel
from rich.text import Text

from vipera.config import load_config
from vipera.engine import Engine
from vipera.hardware import detect
from vipera.planner import plan
from vipera.registry import get_model
from vipera.ui import (
    C_ACCENT,
    C_DIM,
    C_RED,
    C_SUCCESS,
    console,
    pick_model,
    typewriter_chat_stream,
)

app = typer.Typer()


def _build(model: str) -> Engine:
    m = get_model(model)
    if not m:
        console.print(f"[{C_RED}] ✗ Model not found: {model}[/]")
        raise typer.Exit(1)

    cfg = load_config()
    hw = detect()
    p = plan(m["meta"], hw,
             ctx=cfg.get("ctx", 8192),
             kv_cache=cfg.get("kv_cache", "q8_0"),
             n_batch=cfg.get("n_batch", 512))
    with console.status(f"[{C_ACCENT}] Loading {model}…"):
        return Engine(p, m["path"])


def repl_loop(engine: Engine):
    console.print(Panel(
        Text("  System Ready. Type 'exit' or Ctrl+C to quit.", style=C_SUCCESS),
        border_style=C_ACCENT, box=box.ROUNDED,
    ))

    history = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        console.print(f"\n  [{C_ACCENT}]User >[/] ", end="")
        try:
            user = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n  [{C_DIM}]Disconnected.[/]")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit", ":q"):
            break

        history.append({"role": "user", "content": user})

        try:
            stream = engine.chat(history, stream=True)
            reply = typewriter_chat_stream(stream, title_text=" AI ")
            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            console.print(f"\n  [{C_RED}]✗ Error: {e}[/]")
            history.pop()


@app.command("repl")
def repl(model: str = typer.Option(None, "--model", "-m", help="Model alias")):
    """Start interactive chat REPL."""
    cfg = load_config()
    model = model or cfg.get("default_model")
    if not model:
        model = pick_model("SELECT A MODEL TO CHAT")
    engine = _build(model)
    repl_loop(engine)


@app.command("oneshot")
def oneshot(
    prompt: str = typer.Argument(..., help="Prompt to send"),
    model: str = typer.Option(None, "--model", "-m", help="Model alias"),
):
    """Ask a single question and exit."""
    cfg = load_config()
    model = model or cfg.get("default_model")
    if not model:
        model = pick_model("SELECT A MODEL TO CHAT")
    engine = _build(model)
    stream = engine.chat([{"role": "user", "content": prompt}], stream=True)
    typewriter_chat_stream(stream, title_text=" AI ")
