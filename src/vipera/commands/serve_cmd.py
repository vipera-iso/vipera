import socket
import time

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
    C_BORDER,
    C_DIM,
    C_GREEN,
    C_RED,
    C_SUCCESS,
    C_TEXT,
    C_YELLOW,
    animated_loading_bar,
    console,
)


def _resolve_option(val, default):
    """Typer returns OptionInfo when a function is called directly via ctx.invoke()."""
    if isinstance(val, typer.models.OptionInfo):
        return default
    return val if val is not None else default


def _quick_bench(engine: Engine) -> None:
    """Informational speed check after load."""
    prompt = "Write a short sentence about AI."

    with console.status(f"[{C_ACCENT}] Running quick benchmark..."):
        try:
            t0 = time.perf_counter()
            out = engine.chat(
                [{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=20,
            )
            dt = time.perf_counter() - t0

            usage = out.get("usage", {}) or {}
            gen_tok = usage.get("completion_tokens", 0)
            prompt_tok = usage.get("prompt_tokens", 0)

            if not usage:
                # Backend didn't report usage; fall back to a rough char/4 estimate.
                content = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
                gen_tok = max(1, len(content) // 4) if content else 0

            tps = gen_tok / dt if dt > 0 else 0

            if tps >= 20:
                speed_color, speed_status = C_SUCCESS, "FAST"
            elif tps >= 8:
                speed_color, speed_status = C_YELLOW, "OK"
            else:
                speed_color, speed_status = C_RED, "SLOW"

            bench_text = Text()
            bench_text.append("  ● ", style=speed_color)
            bench_text.append("SPEED TEST: ", style=C_TEXT)
            bench_text.append(f"{tps:.1f} T/s", style=speed_color)
            bench_text.append(f"  [{speed_status}]  ", style=speed_color)
            bench_text.append(
                f"({gen_tok} tokens in {dt:.2f}s, Prompt: {prompt_tok} t)",
                style=C_DIM,
            )

            console.print(Panel(
                bench_text, border_style=speed_color, box=box.ROUNDED, padding=(0, 1),
            ))
            console.print()

        except Exception as e:
            console.print(f"[{C_YELLOW}] ⚠ Benchmark failed: {e}[/]\n")


app = typer.Typer()


@app.command("start")
def start(
    model: str = typer.Option(None, "--model", "-m"),
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port", "-p"),
    profile: str = typer.Option("auto", "--profile"),
    ctx: int = typer.Option(None, "--ctx"),
):
    """Start the HTTP Gateway."""
    cfg = load_config()
    model = _resolve_option(model, None) or cfg.get("default_model")
    host = _resolve_option(host, cfg["host"])
    port = _resolve_option(port, cfg["port"])
    profile = _resolve_option(profile, "auto")
    ctx = _resolve_option(ctx, cfg["ctx"])

    if not model:
        console.print(f"[{C_RED}] ✗ No model specified. Use --model <name>[/]")
        raise typer.Exit(1)
    m = get_model(model)
    if not m:
        console.print(f"[{C_RED}] ✗ Model not found: {model}[/]")
        raise typer.Exit(1)

    if host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        console.print(
            f"[{C_RED}] ⚠ WARNING: binding to {host} exposes the model to the "
            f"network with no authentication.[/]"
        )
        console.print(f"[{C_DIM}]  Aborting in 5s — press Ctrl+C to stop now.[/]")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            raise typer.Exit(1) from None

    hw = detect()
    try:
        p = plan(
            m["meta"], hw, profile=profile, ctx=ctx,
            kv_cache=cfg.get("kv_cache", "q8_0"),
            n_batch=cfg.get("n_batch", 512),
        )
    except ValueError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    n_layers_total = int(m["meta"].get("n_layers") or 0) or max(p.gpu_layers, 1)
    quant = m["meta"].get("quant") or "?"

    content = Text()
    content.append("  ▸ MOUNTING MODEL\n", style=C_ACCENT)
    content.append(f"    ├─ {model}\n", style=C_TEXT)
    content.append(f"    └─ {quant}\n\n", style=C_DIM)

    content.append("  ▸ ALLOCATING VRAM\n", style=C_ACCENT)
    content.append(
        "    ├─ layer offload " + "·" * 28 + f" {p.gpu_layers}/{n_layers_total}\n",
        style=C_TEXT,
    )
    content.append(
        f"    ├─ kv cache ({p.kv_cache}) " + "·" * 24 + " [ OK ]\n",
        style=C_TEXT,
    )
    content.append(
        "    └─ flash attention " + "·" * 27 + f" [ {'ON' if p.flash_attn else 'OFF'} ]\n",
        style=C_TEXT,
    )

    console.print(Panel(
        content,
        title=Text(" VIPERA DAEMON · BOOT ", style=C_ACCENT),
        border_style=C_ACCENT,
        box=box.ROUNDED,
        padding=(0, 1),
    ))

    try:
        with console.status(f"[{C_ACCENT}] Loading model into memory…"):
            engine = Engine(p, m["path"])
    except RuntimeError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    animated_loading_bar("KERNEL INIT", 1.2)

    if cfg.get("show_bench_on_start", True):
        _quick_bench(engine)

    try:
        import uvicorn

        from vipera.server import build_app
        fastapi_app = build_app(engine, model)

        ready = Text()
        ready.append("  ● ", style=C_SUCCESS)
        ready.append("SYSTEM READY", style=C_TEXT)
        ready.append("   ·   ", style=C_BORDER)
        ready.append(f"http://{host}:{port}\n\n", style=C_ACCENT)
        ready.append("  ▸ ", style=C_ACCENT)
        ready.append("POST  ", style=C_DIM)
        ready.append("/v1/chat/completions\n", style=C_TEXT)
        ready.append("  ▸ ", style=C_ACCENT)
        ready.append("GET   ", style=C_DIM)
        ready.append("/v1/models\n", style=C_TEXT)
        ready.append("  ▸ ", style=C_ACCENT)
        ready.append("GET   ", style=C_DIM)
        ready.append("/health", style=C_TEXT)

        console.print(Panel(ready, border_style=C_SUCCESS, box=box.ROUNDED, padding=(0, 1)))

        uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")

    except ImportError:
        console.print(f"[{C_YELLOW}] ⚠ FastAPI is not installed. Falling back to Chat REPL.[/]\n")
        from vipera.commands.chat_cmd import _chat_kwargs, repl_loop

        repl_loop(engine, chat_kwargs=_chat_kwargs(cfg))
    finally:
        engine.close()


@app.command("status")
def status():
    """Check Gateway status."""
    cfg = load_config()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    alive = s.connect_ex((cfg["host"], cfg["port"])) == 0
    s.close()

    if alive:
        console.print(f"[{C_GREEN}] ● ONLINE[/]  — http://{cfg['host']}:{cfg['port']}")
    else:
        console.print(f"[{C_RED}] ● OFFLINE[/] — port {cfg['port']}")
