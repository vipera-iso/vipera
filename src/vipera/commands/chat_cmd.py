import typer

from vipera.backends.protocol import ChatEngine
from vipera.backends.registry import create_engine, serves_slots
from vipera.config import load_config
from vipera.hardware import detect
from vipera.optimize.tuning import apply_tuning
from vipera.planner import plan
from vipera.registry import get_model
from vipera.ui import (
    C_ACCENT,
    C_DIM,
    C_RED,
    console,
    note,
    pick_model,
    plain_chat_stream,
    typewriter_chat_stream,
)

app = typer.Typer()


def _chat_kwargs(cfg: dict) -> dict:
    """Pull the sampling overrides the engine understands out of the config."""
    keys = ("temperature", "max_tokens", "top_p", "repeat_penalty")
    return {k: cfg[k] for k in keys if k in cfg and cfg[k] is not None}


def _build(model: str) -> ChatEngine:
    m = get_model(model)
    if not m:
        console.print(f"[{C_RED}] ✗ Model not found: {model}[/]")
        raise typer.Exit(1)

    cfg = load_config()
    hw = detect()
    p = plan(
        m.meta,
        hw,
        profile=cfg.get("profile", "auto"),
        ctx=cfg.get("ctx", 8192),
        kv_cache=cfg.get("kv_cache", "q8_0"),
        kv_cache_k=cfg.get("kv_cache_k"),
        kv_cache_v=cfg.get("kv_cache_v"),
        no_kv_offload=cfg.get("no_kv_offload", False),
        n_batch=cfg.get("n_batch", 512),
        n_ubatch=cfg.get("n_ubatch", 0),
        # One slot per window: only the server backend splits `-c`, and its plan
        # is the one that has to price the extra caches.
        parallel=(
            max(1, int(cfg.get("parallel", 1) or 1))
            if serves_slots(cfg.get("backend", "inprocess"))
            else 1
        ),
    )
    if cfg.get("tuning", True):
        p = apply_tuning(p, model, hw)
    with console.status(f"[{C_ACCENT}] Loading {model}…"):
        return create_engine(p, m.path, cfg.get("backend", "inprocess"))


def repl_loop(engine: ChatEngine, typewriter: bool = False, chat_kwargs: dict | None = None):
    note("ready. type `exit` or press ctrl-c to quit.", tone="ok")

    history = [{"role": "system", "content": "You are a helpful assistant."}]
    kwargs = chat_kwargs or {}

    while True:
        console.print(f"\n   [{C_ACCENT}]>[/] ", end="")
        try:
            user = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n   [{C_DIM}]disconnected[/]")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit", ":q"):
            break

        history.append({"role": "user", "content": user})

        try:
            stream = engine.chat(history, stream=True, **kwargs)
            reply = typewriter_chat_stream(stream) if typewriter else plain_chat_stream(stream)
            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            note(f"the model returned an error: {e}", tone="error")
            history.pop()


@app.command("repl")
def repl(
    model: str | None = typer.Option(None, "--model", "-m", help="Model alias"),
    typewriter: bool = typer.Option(
        False, "--typewriter", help="Animate output (slower, nicer for demos)"
    ),
):
    """Start interactive chat REPL."""
    cfg = load_config()
    model = model or cfg.get("default_model")
    if not model:
        model = pick_model("select a model to chat")
    engine = _build(model)
    try:
        repl_loop(engine, typewriter=typewriter, chat_kwargs=_chat_kwargs(cfg))
    finally:
        engine.close()


@app.command("oneshot")
def oneshot(
    prompt: str = typer.Argument(..., help="Prompt to send"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model alias"),
    typewriter: bool = typer.Option(
        False, "--typewriter", help="Animate output (slower, nicer for demos)"
    ),
):
    """Ask a single question and exit."""
    cfg = load_config()
    model = model or cfg.get("default_model")
    if not model:
        model = pick_model("select a model to chat")
    engine = _build(model)
    try:
        stream = engine.chat(
            [{"role": "user", "content": prompt}], stream=True, **_chat_kwargs(cfg)
        )
        if typewriter:
            typewriter_chat_stream(stream)
        else:
            plain_chat_stream(stream)
    finally:
        engine.close()
