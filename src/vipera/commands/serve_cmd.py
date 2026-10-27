import os
import signal
import socket
import time
from pathlib import Path

import psutil
import typer

from vipera.backends.protocol import ChatEngine
from vipera.backends.registry import create_engine, resolve_backend
from vipera.config import VIPERA_HOME, load_config
from vipera.hardware import detect
from vipera.keepalive import parse_keep_alive
from vipera.optimize.tuning import apply_tuning
from vipera.planner import plan
from vipera.pool import ModelPool
from vipera.protocols.names import is_known as _is_known_protocol
from vipera.protocols.names import names as _protocol_names
from vipera.registry import ModelSpec, get_model, list_models
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


def _protocols(option: str | None, cfg: dict) -> list[str]:
    """Mount order for the HTTP layers: the option, else the config value."""
    # Direct callers (tests) pass the OptionInfo default, as with every other
    # option in this command.
    raw = _resolve_option(option, None) or cfg.get("protocols", "openai")
    parts = raw.split(",") if isinstance(raw, str) else list(raw)
    return [name.strip().lower() for name in parts if name.strip()]


def _run_server(uvicorn, app, host: str, port: int) -> None:
    """Run uvicorn so SIGTERM unwinds instead of killing the process.

    This uvicorn exits on SIGTERM by default, which skips both the `finally`
    below and the engines' atexit hooks: with `backend: server` the
    llama-server child then keeps the model resident after `serve stop`. A
    handler that raises SystemExit gives the process a normal stack unwind,
    so the child, the pid file and the in-process model are all released.
    """

    def _terminate(_signum, _frame):
        raise SystemExit(0)

    try:
        previous = signal.signal(signal.SIGTERM, _terminate)
    except ValueError:
        # Not the main thread (embedded use): nothing to install.
        previous = None
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _pid_file(port: int) -> Path:
    return VIPERA_HOME / f"serve-{port}.pid"


def _release_pid_file(port: int, previous: str | None = None) -> None:
    """Release our pid file, or put back the running gateway's.

    A second `serve start` that fails to bind used to write its own pid and then
    unlink the file of the gateway that was already running: `serve stop` then
    reported OFFLINE and the real server could no longer be stopped. On a failed
    start the previous content is restored; on a clean shutdown ours is removed.
    """
    path = _pid_file(port)
    try:
        if path.read_text().strip() != str(os.getpid()):
            return
        if previous is not None:
            path.write_text(previous)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        return


def _model_aliases(option, cfg: dict) -> list[str]:
    """Aliases to serve together: --models, else cfg["models"], else none."""
    if isinstance(option, typer.models.OptionInfo):
        option = None
    raw = option or cfg.get("models")
    if isinstance(raw, str):
        raw = raw.split(",")
    return [alias.strip() for alias in (raw or []) if alias and alias.strip()]


def _announce_plan(label: str, info: ModelSpec, p, backend: str) -> None:
    n_layers_total = int(info.meta.get("n_layers") or 0) or max(p.gpu_layers, 1)
    offload_layers = p.n_layers if p.requires_server else p.gpu_layers

    # Each half is named when the two differ, and the row says when the cache
    # is not on the card at all — both are decisions the plan made for the user.
    cache_k, cache_v = p.kv_types
    kv_label = cache_k if cache_k == cache_v else f"K {cache_k} / V {cache_v}"
    if p.kv_cache_gb > 0:
        kv_label = f"{kv_label}  {p.kv_cache_gb:.2f} GB"
    if not p.offload_kqv and offload_layers > 0:
        kv_label = f"{kv_label}  (system RAM)"

    section("plan")
    rows = [
        ("model", label),
        ("quant", info.meta.get("quant") or "?"),
        ("backend", backend),
        ("layers", f"{offload_layers}/{n_layers_total}"),
        ("kv cache", kv_label),
        ("flash attn", "on" if p.flash_attn else "off"),
    ]
    if p.n_cpu_ffn >= 0:
        # The tensor split puts every attention block on the card and keeps N
        # layers' dense FFN in RAM, so the layer row above is about attention and
        # this row is about the weights that stream. Reusing the MoE row here
        # would read "41/40 expert layers": there are no experts in a dense plan.
        gpu_ffn = p.n_layers - p.n_cpu_ffn
        rows.insert(
            4,
            ("dense ffn", f"{gpu_ffn}/{p.n_layers} layers on GPU, attention on GPU"),
        )
    elif p.requires_server:
        gpu_experts = p.n_layers - p.moe_cpu_layers
        ubatch = f"  ubatch {p.n_ubatch}" if p.n_ubatch else ""
        rows.insert(4, ("moe experts", f"{gpu_experts}/{p.n_layers} layers{ubatch}"))
    if p.kv_layers and p.kv_layers < n_layers_total:
        rows.append(("hybrid attn", f"{p.kv_layers}/{n_layers_total} layers keep KV"))
    ledger(rows)
    console.print()
    gauge("on the gpu", offload_layers, n_layers_total, unit="layers", fmt="{:.0f}")
    console.print()


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
    profile: str | None = typer.Option(None, "--profile"),
    ctx: int = typer.Option(None, "--ctx"),
    bench: bool = typer.Option(
        False,
        "--bench",
        help="Run a 20-token speed check before serving (delays readiness)",
    ),
    models: str = typer.Option(
        None, "--models", help="Comma-separated aliases to serve together (lazy-loaded)"
    ),
    max_loaded: int = typer.Option(
        None, "--max-loaded", help="How many of --models stay in memory (default 2)"
    ),
    keep_alive: str = typer.Option(
        None,
        "--keep-alive",
        help="Unload idle models after this long: 5m, 1h, seconds, 0, or -1 forever",
    ),
    protocol: str = typer.Option(
        None, "--protocol", help="HTTP layers to mount: openai, ollama, or openai,ollama"
    ),
    api_key: str = typer.Option(
        None,
        "--api-key",
        help="Require this key on every request (Authorization: Bearer). A key on "
        "the command line is visible in `ps`; prefer `api_key` in the config.",
    ),
    cors: str = typer.Option(
        None,
        "--cors",
        help="Browser origins allowed to call the gateway, comma-separated, or * for any",
    ),
):
    """Start the HTTP Gateway."""
    cfg = load_config()
    model = _resolve_option(model, None) or cfg.get("default_model")
    host = _resolve_option(host, cfg["host"])
    port = _resolve_option(port, cfg["port"])
    profile = _resolve_option(profile, cfg.get("profile", "auto")) or "auto"
    ctx = _resolve_option(ctx, cfg["ctx"])
    bench = _resolve_option(bench, False)
    backend = cfg.get("backend", "inprocess")
    max_loaded = int(_resolve_option(max_loaded, cfg.get("max_loaded", 2)))
    keep_alive = _resolve_option(keep_alive, cfg.get("keep_alive", "5m"))
    api_key = _resolve_option(api_key, cfg.get("api_key"))
    cors = _resolve_option(cors, cfg.get("cors"))
    try:
        keep_alive_seconds = parse_keep_alive(keep_alive)
    except ValueError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    # Validate before loading a model: an unknown layer would otherwise fail
    # only after the (possibly minute-long) load finished.
    protocols = _protocols(protocol, cfg)
    unknown = [name for name in protocols if not _is_known_protocol(name)]
    if unknown or not protocols:
        found = ", ".join(unknown) or "(empty)"
        console.print(
            f"[{C_RED}] ✗ Unknown protocol: {found}[/]\n"
            f"[{C_DIM}]  available: {', '.join(_protocol_names())}[/]"
        )
        raise typer.Exit(1)

    aliases = _model_aliases(models, cfg)
    if model and model not in aliases:
        aliases.insert(0, model)
    if not aliases:
        console.print(f"[{C_RED}] ✗ No model specified. Use --model <name>[/]")
        raise typer.Exit(1)

    # Resolve every alias up front so a typo fails at startup, not on the first
    # request hours later.
    metas: dict[str, ModelSpec] = {}
    for alias in aliases:
        info = get_model(alias)
        if not info:
            installed = ", ".join(m.alias for m in list_models()) or "none installed"
            console.print(
                f"[{C_RED}] ✗ Model not found: {alias}[/]\n[{C_DIM}]  installed: {installed}[/]"
            )
            raise typer.Exit(1)
        metas[alias] = info
    model = aliases[0]
    m = metas[model]

    if not _is_loopback(host):
        if str(api_key or "").strip():
            # Reachable from the network, but every request needs the key; the
            # client-side reminder is what stops "it 401s from my phone".
            console.print(
                f"[{C_YELLOW}] ⚠ Binding to {host}: every request needs the API "
                f"key (Authorization: Bearer).[/]"
            )
        else:
            console.print(
                f"[{C_RED}] ⚠ WARNING: binding to {host} exposes the model to the "
                f"network with no authentication. Set `api_key` or pass --api-key.[/]"
            )
            console.print(f"[{C_DIM}]  Serving in 5s — press Ctrl+C to abort now.[/]")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                raise typer.Exit(1) from None

    hw = detect()

    def _mount(info: ModelSpec, label: str, *, announce: bool):
        """Plan and load one model; shared by single-model and lazy pool paths."""
        try:
            p = plan(
                info.meta,
                hw,
                profile=profile,
                ctx=ctx,
                kv_cache=cfg.get("kv_cache", "q8_0"),
                kv_cache_k=cfg.get("kv_cache_k"),
                kv_cache_v=cfg.get("kv_cache_v"),
                no_kv_offload=cfg.get("no_kv_offload", False),
                n_batch=cfg.get("n_batch", 512),
                n_ubatch=cfg.get("n_ubatch", 0),
            )
        except ValueError as e:
            console.print(f"[{C_RED}] ✗ {e}[/]")
            raise typer.Exit(1) from e

        # A split this machine already measured beats the VRAM formula. Losing
        # the profile (or never having one) changes nothing.
        if cfg.get("tuning", True):
            tuned = apply_tuning(p, info.alias, hw)
            if tuned is not p:
                p = tuned
                if announce:
                    note(f"layers from the measured {info.alias} profile: {p.gpu_layers}")

        # Decide before the banner: an MoE plan either upgrades to llama-server
        # or falls back to the classic split, and the panel must show the truth.
        backend_used, p = resolve_backend(p, backend)
        if announce:
            _announce_plan(label, info, p, backend_used)

        try:
            with console.status(f"[{C_ACCENT}] Loading {label} into memory…"):
                return p, create_engine(p, info.path, backend_used), backend_used
        except RuntimeError as e:
            console.print(f"[{C_RED}] ✗ {e}[/]")
            raise typer.Exit(1) from e

    pool = None
    engine = None
    if len(metas) > 1:
        # Lazy: a model is planned and mounted on its first request, so a
        # three-model gateway does not pay for three models up front.
        pool = ModelPool(
            {alias: info.path for alias, info in metas.items()},
            lambda alias, _path: _mount(metas[alias], alias, announce=False)[1],
            max_loaded=max_loaded,
            keep_alive=keep_alive_seconds,
        )
        section("models")
        ledger(
            [
                ("serving", ", ".join(metas)),
                ("max loaded", str(max_loaded)),
                ("keep alive", str(keep_alive)),
                ("backend", backend),
            ]
        )
        note("each model is planned and loaded on its first request.", tone="ok")
        console.print()
    else:
        _, engine, backend = _mount(m, model, announce=True)

    # The gateway is ready as soon as uvicorn binds. The old unconditional
    # 20-token speed check ran here and cost minutes on slow hardware (a 0.2
    # tok/s model needed ~100s before the first request could be served), and
    # the loading bar added a fixed 1.2s sleep. The check is now opt-in.
    if bench and engine is not None:
        _quick_bench(engine)

    # Only touch the pid file once this process owns it, and restore a running
    # gateway's file if this start fails before uvicorn ever binds.
    pid_written = False
    previous_pid: str | None = None
    clean_shutdown = False
    try:
        import uvicorn

        from vipera.server import build_app

        fastapi_app = build_app(
            engine, model, pool=pool, protocols=protocols, api_key=api_key, cors=cors
        )

        section("ready")
        ledger(
            [
                (name, f"http://{host}:{port}/{prefix}")
                for name, prefix in (("openai", "v1"), ("ollama", "api"))
                if name in protocols
            ]
        )
        routes = []
        if "openai" in protocols:
            routes += [
                ("POST /v1/chat/completions", "chat, streaming or not"),
                ("POST /v1/embeddings", "vectors (needs embeddings enabled)"),
                ("GET  /v1/models", "every served alias" if pool else "the loaded alias"),
            ]
        if "ollama" in protocols:
            routes += [
                ("POST /api/chat", "Ollama clients, streaming or not"),
                ("POST /api/embed", "vectors (needs embeddings enabled)"),
                ("GET  /api/tags", "every served alias"),
            ]
        routes.append(("GET  /health", "liveness (no key needed)" if api_key else "liveness"))
        catalog(routes)
        if api_key:
            note("auth: every route needs `Authorization: Bearer <api_key>`.", tone="ok")
        note("ctrl-c stops the server.", tone="ok")

        pid_path = _pid_file(port)
        try:
            previous_pid = pid_path.read_text()
        except OSError:
            previous_pid = None
        pid_path.write_text(str(os.getpid()))
        pid_written = True
        try:
            _run_server(uvicorn, fastapi_app, host, port)
        except SystemExit as exc:
            # SIGTERM unwinds through SystemExit(0) (see _run_server) and that
            # is a normal stop, so this process's pid file must go. uvicorn
            # also exits with SystemExit(STARTUP_FAILURE=3) when it cannot
            # bind, which is a failed start: the running gateway keeps its
            # file. So the exit code decides, not the exception type.
            clean_shutdown = exc.code in (0, None)
            raise
        clean_shutdown = True

    except ImportError:
        console.print(f"[{C_YELLOW}] ⚠ FastAPI is not installed. Falling back to Chat REPL.[/]\n")
        from vipera.commands.chat_cmd import _chat_kwargs, repl_loop

        active = engine
        if active is None:
            # Pool mode: the REPL has no request to name a model, so load the
            # first one. `pool` is set whenever `engine` is None here.
            assert pool is not None
            active, _ = pool.get(None)
        repl_loop(active, chat_kwargs=_chat_kwargs(cfg))
    finally:
        if pid_written:
            _release_pid_file(port, None if clean_shutdown else previous_pid)
        if pool is not None:
            pool.close()
        if engine is not None:
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
