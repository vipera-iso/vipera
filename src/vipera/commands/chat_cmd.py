import os
import re
from typing import Any, Protocol

import psutil
import typer

from vipera.backends import gateway
from vipera.backends.protocol import ChatEngine
from vipera.backends.registry import create_engine, serves_slots
from vipera.config import load_config
from vipera.hardware import detect
from vipera.optimize.tuning import apply_tuning
from vipera.planner import Plan, plan
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


class _Chat(Protocol):
    """What the REPL needs, and no more.

    A local engine and a client attached to a running gateway both satisfy it;
    `ChatEngine` additionally promises a `plan`, which a client that never
    loaded the model cannot honestly report.
    """

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any): ...

    def close(self) -> None: ...


def _as_text(value: object) -> str | None:
    """A string option's value, or None for its OptionInfo default."""
    return value if isinstance(value, str) and value else None


def _chat_kwargs(cfg: dict) -> dict:
    """Pull the sampling overrides the engine understands out of the config."""
    keys = ("temperature", "max_tokens", "top_p", "repeat_penalty")
    return {k: cfg[k] for k in keys if k in cfg and cfg[k] is not None}


def _other_servers() -> list[tuple[str, str]]:
    """`llama-server` processes already running, as (port, cmdline).

    Another child means VRAM the plan cannot see the *reason* for: it only sees
    a small `vram_free_gb` and correctly falls back to CPU. That fallback is
    right but silent, and on a 62 GB model it is the difference between 1.8 and
    0.6 t/s - so the number needs a name attached to it.
    """
    found: list[tuple[str, str]] = []
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
        except psutil.Error:
            continue
        if not cmd or "llama-server" not in os.path.basename(str(cmd[0])):
            continue
        if len(cmd) > 1 and cmd[1] == "--help":
            continue
        line = " ".join(cmd)
        match = re.search(r"--port\s+(\d+)", line)
        found.append((match.group(1) if match else "?", line))
    return found


def _warn_card_in_use(p: Plan, hw) -> None:
    """Explain a CPU-only plan on a machine whose card is taken."""
    puts_nothing_on_card = p.gpu_layers == 0 and p.moe_cpu_layers < 0 and p.n_cpu_ffn < 0
    total = getattr(hw, "vram_total_gb", 0) or 0
    if not puts_nothing_on_card or total <= 0:
        return
    others = _other_servers()
    if not others:
        return
    free = getattr(hw, "vram_free_gb", 0) or 0
    where = ", ".join(f"port {port}" for port, _ in others)
    note(
        f"[!] another llama-server is already running ({where}) and holds "
        f"{max(0.0, total - free):.1f} of {total:.1f} GB of VRAM, "
        "so this plan keeps every layer on the CPU.\n"
        "    Stop it first (`vipera serve stop -y`) to load with attention on the GPU; "
        "as it stands, expect CPU speed.",
        tone="warn",
    )


def _gateway_url(cfg: dict, override: str | None = None) -> str:
    """Where to look for a gateway: `--gateway`, else the config's own bind.

    `--gateway` takes what a person would paste - `http://127.0.0.1:9000`,
    `127.0.0.1:9000`, or a bare port - because a gateway started with
    `--port 9000` is not reachable at the default one.
    """
    if override:
        value = override.strip()
        if "://" in value:
            return value.rstrip("/")
        host, _, port = value.rpartition(":")
        if not host:
            return gateway.base_url("127.0.0.1", int(port or value))
        return gateway.base_url(host, int(port))
    return gateway.base_url(cfg.get("host") or "127.0.0.1", int(cfg.get("port") or 8686))


def _attach(model: str, cfg: dict, *, gateway_url: str | None = None):
    """A client for a gateway that already serves `model`, if one does.

    Checked before planning or loading anything: `vipera serve` holds the
    weights, and a chat that loads them again pays for a second copy on a card
    that is already full - and, for a model larger than RAM, re-reads 62 GB
    from disk to ask one question.
    """
    return gateway.attach(_gateway_url(cfg, gateway_url), model, api_key=cfg.get("api_key"))


def _engine_for(
    model: str, cfg: dict, *, local: bool = False, gateway_url: str | None = None
) -> _Chat:
    """The running gateway if it serves `model`, else a local engine."""
    if local:
        note("--local: loading a local copy even if a gateway serves this alias.", tone="ok")
        return _build(model)
    attached = _attach(model, cfg, gateway_url=gateway_url)
    if attached is not None:
        note(f"using the gateway already serving {model} at {attached.base_url}", tone="ok")
        return attached
    return _build(model)


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
    _warn_card_in_use(p, hw)
    with console.status(f"[{C_ACCENT}] Loading {model}…"):
        return create_engine(p, m.path, cfg.get("backend", "inprocess"))


def repl_loop(engine: _Chat, typewriter: bool = False, chat_kwargs: dict | None = None):
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
    local: bool = typer.Option(
        False, "--local", help="Load the model here even if a gateway serves it"
    ),
    gateway_url: str | None = typer.Option(
        None, "--gateway", help="Gateway URL to attach to (default: config host:port)"
    ),
):
    """Start interactive chat REPL."""
    # Direct callers pass the OptionInfo default, not a bool or a string (see
    # `model_cmd.import_model`); treating an object as truthy loaded the model
    # locally and ignored a gateway that was serving it.
    cfg = load_config()
    model = model or cfg.get("default_model")
    if not model:
        model = pick_model("select a model to chat")
    engine = _engine_for(model, cfg, local=local is True, gateway_url=_as_text(gateway_url))
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
    local: bool = typer.Option(
        False, "--local", help="Load the model here even if a gateway serves it"
    ),
    gateway_url: str | None = typer.Option(
        None, "--gateway", help="Gateway URL to attach to (default: config host:port)"
    ),
):
    """Ask a single question and exit."""
    cfg = load_config()
    model = model or cfg.get("default_model")
    if not model:
        model = pick_model("select a model to chat")
    engine = _engine_for(model, cfg, local=local is True, gateway_url=_as_text(gateway_url))
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
