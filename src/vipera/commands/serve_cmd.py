import os
import socket
import time
from pathlib import Path

import psutil
import typer

from vipera.config import VIPERA_HOME, load_config
from vipera.engine import ChatEngine, create_engine, resolve_backend
from vipera.hardware import detect
from vipera.planner import plan
from vipera.registry import get_model
from vipera.ui import (
    C_ACCENT,
    C_DIM,
    C_GREEN,
    C_RED,
    C_YELLOW,
    catalog,
    console,
    gauge,
    ledger,
    note,
    section,
)


def _resolve_option(val, default):
    """Typer returns OptionInfo when a function is called directly via ctx.invoke()."""
    if isinstance(val, typer.models.OptionInfo):
        return default
    return val if val is not None else default


def _pid_file(port: int) -> Path:
    return VIPERA_HOME / f"serve-{port}.pid"


_LOOPBACK_HOSTS = {"localhost", "::1"}


def _is_loopback(host: str) -> bool:
    """Only loopback binds keep the unauthenticated API on this machine."""
    h = (host or "").strip().lower()
    return h in _LOOPBACK_HOSTS or h.startswith("127.")


def _quick_bench(engine: ChatEngine) -> None:
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

            status = "fast" if tps >= 20 else "ok" if tps >= 8 else "slow"

            console.print()
            ledger(
                [
                    ("speed", f"{tps:.1f} tok/s  {status}"),
                    ("sample", f"{gen_tok} tokens in {dt:.2f} s, {prompt_tok} prompt tokens"),
                ]
            )
            console.print()

        except Exception as e:
            console.print(f"[{C_YELLOW}] ⚠ Benchmark failed: {e}[/]\n")


app = typer.Typer()


@app.command("start")
def start(
    model: str | None = typer.Option(None, "--model", "-m"),
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port", "-p"),
    profile: str = typer.Option("auto", "--profile"),
    ctx: int = typer.Option(None, "--ctx"),
    bench: bool = typer.Option(
        False,
        "--bench",
        help="Run a 20-token speed check before serving (delays readiness)",
    ),
):
    """Start the HTTP Gateway."""
    cfg = load_config()
    model = _resolve_option(model, None) or cfg.get("default_model")
    host = _resolve_option(host, cfg["host"])
    port = _resolve_option(port, cfg["port"])
    profile = _resolve_option(profile, "auto")
    ctx = _resolve_option(ctx, cfg["ctx"])
    bench = _resolve_option(bench, False)
    backend = cfg.get("backend", "inprocess")

    if not model:
        console.print(f"[{C_RED}] ✗ No model specified. Use --model <name>[/]")
        raise typer.Exit(1)
    m = get_model(model)
    if not m:
        console.print(f"[{C_RED}] ✗ Model not found: {model}[/]")
        raise typer.Exit(1)

    if not _is_loopback(host):
        console.print(
            f"[{C_RED}] ⚠ WARNING: binding to {host} exposes the model to the "
            f"network with no authentication.[/]"
        )
        console.print(f"[{C_DIM}]  Serving in 5s — press Ctrl+C to abort now.[/]")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            raise typer.Exit(1) from None

    hw = detect()
    try:
        p = plan(
            m["meta"],
            hw,
            profile=profile,
            ctx=ctx,
            kv_cache=cfg.get("kv_cache", "q8_0"),
            n_batch=cfg.get("n_batch", 512),
            n_ubatch=cfg.get("n_ubatch", 0),
        )
    except ValueError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    # Decide before the banner: an MoE plan either upgrades to llama-server or
    # falls back to the classic split, and the panel must show the truth.
    backend, p = resolve_backend(p, backend)

    n_layers_total = int(m["meta"].get("n_layers") or 0) or max(p.gpu_layers, 1)
    quant = m["meta"].get("quant") or "?"

    offload_layers = p.n_layers if p.requires_server else p.gpu_layers

    section("plan")
    rows = [
        ("model", model),
        ("quant", quant),
        ("backend", backend),
        ("layers", f"{offload_layers}/{n_layers_total}"),
        ("kv cache", f"{p.kv_cache}  {p.kv_cache_gb:.2f} GB" if p.kv_cache_gb > 0 else p.kv_cache),
        ("flash attn", "on" if p.flash_attn else "off"),
    ]
    if p.requires_server:
        gpu_experts = p.n_layers - p.moe_cpu_layers
        ubatch = f"  ubatch {p.n_ubatch}" if p.n_ubatch else ""
        rows.insert(4, ("moe experts", f"{gpu_experts}/{p.n_layers} layers{ubatch}"))
    if p.kv_layers and p.kv_layers < n_layers_total:
        rows.append(("hybrid attn", f"{p.kv_layers}/{n_layers_total} layers keep KV"))
    ledger(rows)
    console.print()
    gauge("on the gpu", offload_layers, n_layers_total, unit="layers", fmt="{:.0f}")
    console.print()

    try:
        with console.status(f"[{C_ACCENT}] Loading model into memory…"):
            engine = create_engine(p, m["path"], backend)
    except RuntimeError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    # The gateway is ready as soon as uvicorn binds. The old unconditional
    # 20-token speed check ran here and cost minutes on slow hardware (a 0.2
    # tok/s model needed ~100s before the first request could be served), and
    # the loading bar added a fixed 1.2s sleep. The check is now opt-in.
    if bench:
        _quick_bench(engine)

    try:
        import uvicorn

        from vipera.server import build_app

        fastapi_app = build_app(engine, model)

        section("ready")
        ledger([("openai", f"http://{host}:{port}/v1")])
        catalog(
            [
                ("POST /v1/chat/completions", "chat, streaming or not"),
                ("GET  /v1/models", "the loaded alias"),
                ("GET  /health", "liveness"),
            ]
        )
        note("ctrl-c stops the server.", tone="ok")

        _pid_file(port).write_text(str(os.getpid()))
        uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")

    except ImportError:
        console.print(f"[{C_YELLOW}] ⚠ FastAPI is not installed. Falling back to Chat REPL.[/]\n")
        from vipera.commands.chat_cmd import _chat_kwargs, repl_loop

        repl_loop(engine, chat_kwargs=_chat_kwargs(cfg))
    finally:
        _pid_file(port).unlink(missing_ok=True)
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


@app.command("stop")
def stop(
    port: int = typer.Option(None, "--port", "-p", help="Gateway port (default: config)"),
    force: bool = typer.Option(False, "--force", help="SIGKILL if SIGTERM is ignored"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation"),
):
    """Stop a gateway started by `vipera serve start`."""
    cfg = load_config()
    port = _resolve_option(port, cfg["port"])
    pid_file = _pid_file(port)

    if not pid_file.exists():
        console.print(
            f"[{C_RED}] ● OFFLINE[/] — no pid file for port {port}. "
            f"If it is still running, kill it manually."
        )
        raise typer.Exit(1)

    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        pid_file.unlink(missing_ok=True)
        console.print(f"[{C_RED}] ✗ unreadable pid file for port {port}; removed.[/]")
        raise typer.Exit(1) from None

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        pid_file.unlink(missing_ok=True)
        console.print(f"[{C_YELLOW}] ⚠ stale pid file (pid {pid} is gone); removed.[/]")
        raise typer.Exit(1) from None

    # Guard against a recycled PID: only stop a process that looks like vipera.
    try:
        cmdline = " ".join(proc.cmdline())
    except psutil.Error:
        cmdline = ""
    if "vipera" not in cmdline.lower():
        console.print(f"[{C_RED}] ✗ pid {pid} does not look like vipera; refusing to stop it.[/]")
        raise typer.Exit(1)

    if not yes and not typer.confirm(f"Stop gateway on port {port} (pid {pid})?"):
        raise typer.Abort()

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except psutil.TimeoutExpired:
        if not force:
            console.print(
                f"[{C_YELLOW}] ⚠ pid {pid} did not stop within 10s; retry with --force.[/]"
            )
            raise typer.Exit(1) from None
        proc.kill()
        proc.wait(timeout=5)

    pid_file.unlink(missing_ok=True)
    console.print(f"[{C_GREEN}] ✓ STOPPED[/] — port {port} (pid {pid})")
