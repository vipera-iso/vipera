# vipera — Full Source Code

```python
# __init__.py
__version__ = "1.2.0"

# cli.py
import typer
from rich import box
from rich.panel import Panel
from rich.text import Text

from vipera import __version__
from vipera.commands import bench_cmd, chat_cmd, model_cmd, serve_cmd, system_cmd
from vipera.config import load_config
from vipera.registry import list_models
from vipera.ui import (
    C_ACCENT,
    C_DIM,
    C_RED,
    C_TEXT,
    console,
    pick_model,
    print_banner,
)

app = typer.Typer(
    name="vipera",
    help="Vipera — Modern Local AI Gateway",
    no_args_is_help=False,
    invoke_without_command=True,
)

app.add_typer(model_cmd.app, name="model", help="Manage GGUF models")
app.add_typer(serve_cmd.app, name="serve", help="Start HTTP Gateway")
app.add_typer(chat_cmd.app, name="chat", help="Chat with AI")
app.add_typer(bench_cmd.app, name="bench", help="Benchmark performance")
app.add_typer(system_cmd.app, name="system", help="Diagnostics & Config")


@app.command("help")
def show_help():
    """Detailed usage guide."""
    guide = Text()

    guide.append("  ▸ QUICK START\n", style=C_ACCENT)
    guide.append(
        "    • vipera                          Start gateway (default model)\n", style=C_TEXT
    )
    guide.append(
        "    • vipera -s                       Start gateway with model picker\n", style=C_TEXT
    )
    guide.append(
        "    • vipera model search             Search & download from HuggingFace\n", style=C_TEXT
    )
    guide.append("    • vipera chat repl                Chat in the terminal\n\n", style=C_TEXT)

    guide.append("  ▸ MODELS (vipera model ...)\n", style=C_ACCENT)
    guide.append("    • vipera model list               List installed models\n", style=C_TEXT)
    guide.append("    • vipera model import <path>      Import a local .gguf file\n", style=C_TEXT)
    guide.append(
        "    • vipera model inspect <path>     Analyze GGUF layout & quants\n", style=C_TEXT
    )
    guide.append("    • vipera model imatrix <path>     Build an importance matrix\n", style=C_TEXT)
    guide.append(
        "    • vipera model quantize <path>    Quantize with llama-quantize\n", style=C_TEXT
    )
    guide.append(
        "    • vipera model optimize <path>    Inspect → imatrix → quantize\n", style=C_TEXT
    )
    guide.append("    • vipera model rm <name>          Remove a model\n\n", style=C_TEXT)

    guide.append("  ▸ INFERENCE (vipera chat / bench ...)\n", style=C_ACCENT)
    guide.append("    • vipera chat repl                Interactive chat\n", style=C_TEXT)
    guide.append('    • vipera chat oneshot "<prompt>"  One-shot query\n', style=C_TEXT)
    guide.append("    • vipera bench run                Benchmark model\n\n", style=C_TEXT)

    guide.append("  ▸ SYSTEM (vipera system ...)\n", style=C_ACCENT)
    guide.append(
        "    • vipera system doctor            Hardware & library diagnostics\n", style=C_TEXT
    )
    guide.append("    • vipera system config            View/edit config\n\n", style=C_TEXT)

    guide.append("  💡 TIP: use `vipera -s` to switch the running model.\n", style=C_DIM)

    console.print(
        Panel(
            guide,
            title=Text(" VIPERA MANUAL ", style=C_ACCENT),
            border_style=C_ACCENT,
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    select: bool = typer.Option(False, "--select", "-s", help="Force model selection menu"),
):
    """Vipera — Modern Local AI Gateway"""
    if version:
        console.print(f"vipera {__version__}")
        raise typer.Exit(0)

    if ctx.invoked_subcommand is not None:
        return

    cfg = load_config()
    print_banner(host=cfg.get("host", "localhost"))

    installed = list_models()
    if not installed:
        console.print(
            Panel(
                Text(
                    "  [ NO MODELS INSTALLED ]\n\n"
                    "  Import a model first:\n"
                    "  ► vipera model import <path_to_gguf> --name my-model\n\n"
                    "  Or search Hugging Face:\n"
                    "  ► vipera model search",
                    style=C_RED,
                ),
                title=Text(" WARNING ", style=C_RED),
                border_style=C_RED,
                box=box.ROUNDED,
            )
        )
        raise typer.Exit(1)

    cfg = load_config()
    model = cfg.get("default_model")

    if select or not model:
        if len(installed) == 1 and not select:
            model = installed[0]["name"]
            console.print(f"[{C_ACCENT}]► Auto-selected the only installed model: {model}[/]\n")
        else:
            model = pick_model("SELECT A MODEL TO START")
    else:
        console.print(f"[{C_ACCENT}]► Starting Gateway with default model: {model}[/]")
        console.print(f"[{C_DIM}]  (Use `vipera -s` to choose a different model)[/]\n")

    ctx.invoke(serve_cmd.start, model=model)


if __name__ == "__main__":
    app()

# commands/__init__.py
# Subcommand package.

# commands/bench_cmd.py
import time

import typer
from rich import box
from rich.table import Table

from vipera.config import load_config
from vipera.engine import ChatEngine, create_engine
from vipera.hardware import detect
from vipera.planner import plan
from vipera.registry import get_model
from vipera.ui import (
    C_ACCENT,
    C_DIM,
    C_GREEN,
    C_RED,
    C_TEXT,
    animated_loading_bar,
    console,
    pick_model,
)

app = typer.Typer()

PROMPT = (
    "Write a highly detailed and comprehensive guide on how to build a modern "
    "web application using Python. Include sections on backend architecture, "
    "database design, API routing, and frontend integration. Make sure to explain "
    "the concepts clearly for a beginner."
)


def _run_bench(engine: ChatEngine, prompt: str, max_tokens: int) -> dict:
    try:
        # Go through the engine: the server backend has no `.llm`.
        prompt_tok = len(engine.tokenize(prompt))
    except Exception:
        prompt_tok = 0

    chunk_count = 0
    final_usage: dict | None = None
    first_token_at = None
    t0 = time.perf_counter()

    try:
        stream = engine.chat(
            [{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=max_tokens,
            # In-process drops this (llama-cpp-python 0.3.x does not report
            # usage); llama-server returns the exact completion_tokens with it.
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            usage = chunk.get("usage") if isinstance(chunk, dict) else None
            if usage:
                final_usage = usage
            try:
                delta = (
                    chunk["choices"][0]["delta"].get("content", "") if chunk.get("choices") else ""
                )
            except (KeyError, IndexError, TypeError):
                delta = ""
            if delta:
                chunk_count += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
    except Exception as e:
        return {"error": str(e), "prompt_tok": prompt_tok, "gen_tok": chunk_count}

    dt = time.perf_counter() - t0
    ttft = (first_token_at - t0) if first_token_at else dt
    gen_time = dt - ttft
    prompt_tps = prompt_tok / ttft if ttft > 0 and prompt_tok > 0 else 0.0

    # Prefer the model's own token count when the backend reports it; llama.cpp
    # only does so when asked via stream_options, which this build may not
    # support, so the streamed-chunk count remains the fallback.
    gen_tok = chunk_count
    token_count_is_exact = False
    if final_usage and final_usage.get("completion_tokens"):
        gen_tok = int(final_usage["completion_tokens"])
        token_count_is_exact = True

    if gen_time > 0 and gen_tok > 1:
        gen_tps = (gen_tok - 1) / gen_time
    elif dt > 0:
        gen_tps = gen_tok / dt
    else:
        gen_tps = 0.0

    return {
        "prompt_tok": prompt_tok,
        "gen_tok": gen_tok,
        "token_count_is_exact": token_count_is_exact,
        "ttft_ms": ttft * 1000.0,
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "total_s": dt,
    }


@app.command("run")
def run(
    model: str | None = typer.Argument(None, help="Model name (default: current default model)"),
    ctx: str = typer.Option("512,2048", "--ctx", help="Comma-separated context sizes"),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max tokens to generate"),
):
    """Benchmark model speed (prefill + generation) across context sizes."""
    cfg = load_config()
    model = model or cfg.get("default_model")
    if not model:
        model = pick_model("SELECT A MODEL TO BENCHMARK")

    m = get_model(model)
    if not m:
        console.print(f"[{C_RED}] ✗ Model not found: {model}[/]")
        raise typer.Exit(1)

    hw = detect()
    try:
        ctx_list = [int(x) for x in ctx.split(",") if x.strip()]
    except ValueError as exc:
        console.print(
            f"[{C_RED}] ✗ --ctx must be comma-separated integers "
            f"(example: 512,8192,65536), got: {ctx!r}[/]"
        )
        raise typer.Exit(1) from exc
    if not ctx_list or any(c <= 0 for c in ctx_list):
        console.print(f"[{C_RED}] ✗ --ctx values must be positive integers, got: {ctx!r}[/]")
        raise typer.Exit(1)

    backend = cfg.get("backend", "inprocess")
    # The in-process engine cannot do speculative decoding (llama-cpp-python's
    # prompt-lookup draft measured -29%), so the setting only exists here as
    # context for reading the numbers: the server backend applies it itself.
    spec = cfg.get("spec_type", "none") if backend == "server" else "off"
    console.print(
        f"\n[{C_ACCENT}]▸ Benchmarking model:[/] [{C_TEXT}]{model}[/]"
        f"  [{C_DIM}](backend: {backend}, spec: {spec})[/]"
    )
    console.print(
        f"[{C_DIM}]  (Hardware: {hw.cpu_cores} cores, "
        f"{hw.ram_free_gb:.1f} GB RAM free, GPU: {hw.gpu_name})[/]\n"
    )

    t = Table(box=box.ROUNDED, title="BENCHMARK RESULTS", title_justify="left")
    t.add_column("CTX", style=C_TEXT, justify="right")
    t.add_column("GPU LAYERS", style=C_DIM, justify="center")
    t.add_column("PROMPT TOK", style=C_DIM, justify="right")
    t.add_column("TTFT", style=C_DIM, justify="right")
    t.add_column("PREFILL T/s", style=C_GREEN, justify="right")
    t.add_column("GEN TOK", style=C_DIM, justify="right")
    t.add_column("GEN T/s", style=C_GREEN, justify="right")
    t.add_column("TOTAL", style=C_TEXT, justify="right")

    for c in ctx_list:
        p = plan(
            m["meta"],
            hw,
            ctx=c,
            kv_cache=cfg.get("kv_cache", "q8_0"),
            n_batch=cfg.get("n_batch", 512),
            n_ubatch=cfg.get("n_ubatch", 0),
        )

        engine = None
        try:
            display_layers = p.n_layers if p.requires_server else p.gpu_layers
            with console.status(
                f"[{C_ACCENT}] Loading model with ctx={c} (GPU: {display_layers} layers)…"
            ):
                try:
                    engine = create_engine(p, m["path"], backend)
                except Exception as e:
                    console.print(f"[{C_RED}] ✗ Failed to load model with ctx={c}: {e}[/]")
                    continue

            animated_loading_bar(f"RUNNING CTX {c}", 0.5)
            result = _run_bench(engine, PROMPT, max_tokens)

            if "error" in result:
                console.print(f"[{C_RED}] ✗ Benchmark error at ctx={c}: {result['error']}[/]")
                continue

            # Report the plan the engine actually loaded: a GPU load can fail
            # and fall back to CPU, which would otherwise still print the
            # optimistic planned layer count.
            layers_total = int(m["meta"].get("n_layers") or 0) or max(engine.plan.gpu_layers, 1)

            gen_tok_str = str(result["gen_tok"])
            if not result.get("token_count_is_exact"):
                gen_tok_str = f"~{result['gen_tok']}"

            t.add_row(
                str(c),
                f"{engine.plan.n_layers if engine.plan.requires_server else engine.plan.gpu_layers}/{layers_total}",
                str(result["prompt_tok"]),
                f"{result['ttft_ms']:.0f} ms",
                f"{result['prompt_tps']:.1f}",
                gen_tok_str,
                f"{result['gen_tps']:.2f}",
                f"{result['total_s']:.2f} s",
            )
        finally:
            if engine is not None:
                engine.close()

    console.print(t)
    console.print(
        f"[{C_DIM}]  * TTFT = time to first token. PREFILL T/s is prompt_tokens / TTFT "
        f"(approximate). GEN T/s is measured after the first token. "
        f"A '~' prefix on GEN TOK means the backend did not report usage and the "
        f"count is a streamed-chunk approximation.[/]\n"
    )

# commands/chat_cmd.py
import typer
from rich import box
from rich.panel import Panel
from rich.text import Text

from vipera.config import load_config
from vipera.engine import ChatEngine, create_engine
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
        m["meta"],
        hw,
        ctx=cfg.get("ctx", 8192),
        kv_cache=cfg.get("kv_cache", "q8_0"),
        n_batch=cfg.get("n_batch", 512),
        n_ubatch=cfg.get("n_ubatch", 0),
    )
    with console.status(f"[{C_ACCENT}] Loading {model}…"):
        return create_engine(p, m["path"], cfg.get("backend", "inprocess"))


def repl_loop(engine: ChatEngine, typewriter: bool = False, chat_kwargs: dict | None = None):
    console.print(
        Panel(
            Text("  System Ready. Type 'exit' or Ctrl+C to quit.", style=C_SUCCESS),
            border_style=C_ACCENT,
            box=box.ROUNDED,
        )
    )

    history = [{"role": "system", "content": "You are a helpful assistant."}]
    kwargs = chat_kwargs or {}

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
            stream = engine.chat(history, stream=True, **kwargs)
            if typewriter:
                reply = typewriter_chat_stream(stream, title_text=" AI ")
            else:
                reply = plain_chat_stream(stream, title_text=" AI ")
            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            console.print(f"\n  [{C_RED}]✗ Error: {e}[/]")
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
        model = pick_model("SELECT A MODEL TO CHAT")
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
        model = pick_model("SELECT A MODEL TO CHAT")
    engine = _build(model)
    try:
        stream = engine.chat(
            [{"role": "user", "content": prompt}], stream=True, **_chat_kwargs(cfg)
        )
        if typewriter:
            typewriter_chat_stream(stream, title_text=" AI ")
        else:
            plain_chat_stream(stream, title_text=" AI ")
    finally:
        engine.close()

# commands/model_cmd.py
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import typer
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from rich import box
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from vipera.config import MODELS_DIR, ensure_dirs, load_config, save_config
from vipera.gguf_analyzer import GGUFAnalysisError, analyze_gguf
from vipera.optimizer import (
    OptimizerError,
    build_imatrix_command,
    build_quantize_command,
    default_output_path,
    estimate_quant_for_target,
    estimate_size_for_quant,
    find_llama_tool,
    normalize_quant,
    tool_supports_flag,
    unique_output_path,
)
from vipera.registry import add_model, get_model, list_models, remove_model
from vipera.ui import (
    C_ACCENT,
    C_DIM,
    C_GREEN,
    C_RED,
    C_TEXT,
    C_YELLOW,
    animated_loading_bar,
    console,
)

app = typer.Typer()

api = HfApi()

_QUANT_RE = re.compile(
    r"(Q\d+_K(?:_[A-Z]+)?|Q\d+_\d|IQ\d+_\w+|F16|BF16|F32|Q\d+)",
    re.IGNORECASE,
)


def _read_meta(path: Path, name: str | None = None) -> dict:
    meta = {
        "size_bytes": path.stat().st_size,
        "n_layers": 0,
        "n_params": 0,
        "context_length": 0,
        "quant": "?",
        "architecture": None,
        "embedding_length": 0,
        "n_heads": 0,
        "n_kv_heads": 0,
        "head_dim": 0,
        "layer_bytes": 0,
        "hybrid_attention": False,
        "kv_layers": 0,
        "recurrent_layers": 0,
        "expert_bytes": 0,
        "shared_expert_bytes": 0,
        "expert_layer_bytes": [],
        "expert_layers": 0,
    }

    m = _QUANT_RE.search(name or path.name)
    if m:
        meta["quant"] = m.group(1).upper()

    try:
        # Imported here (not only at module level) so tests and callers can
        # substitute the analyzer and so the optional gguf dependency is not
        # required merely to import this module.
        from vipera.gguf_analyzer import analyze_gguf

        analysis = analyze_gguf(str(path))
        am = analysis["metadata"]
        meta["n_layers"] = int(am.get("n_layers") or 0)
        meta["n_params"] = int(am.get("n_params") or 0)
        meta["context_length"] = int(am.get("context_length") or 0)
        meta["architecture"] = am.get("architecture")
        # The planner sizes the KV cache from these; without them it silently
        # falls back to the fixed 1.5 GB @ 8k heuristic.
        meta["embedding_length"] = int(am.get("embedding_length") or 0)
        meta["n_heads"] = int(am.get("n_heads") or 0)
        meta["n_kv_heads"] = int(am.get("n_kv_heads") or 0)
        meta["head_dim"] = int(am.get("head_dim") or 0)
        # Hybrid stacks (KDA/BCSA/Mamba) keep KV only on attention layers; the
        # planner needs the split to size the cache instead of assuming dense.
        meta["hybrid_attention"] = bool(am.get("hybrid_attention"))
        meta["kv_layers"] = int(am.get("kv_layers") or 0)
        meta["recurrent_layers"] = int(am.get("recurrent_layers") or 0)
        meta["expert_bytes"] = int(am.get("expert_bytes") or 0)
        meta["shared_expert_bytes"] = int(am.get("shared_expert_bytes") or 0)
        meta["expert_layers"] = int(am.get("expert_layers") or 0)
        layer_experts = am.get("expert_layer_bytes") or []
        if isinstance(layer_experts, list):
            meta["expert_layer_bytes"] = [int(value) for value in layer_experts]
        # Size of one repeating transformer layer, excluding the non-repeating
        # tensors (token_embd, output.weight, ...). The planner budgets VRAM per
        # layer from this; size_bytes/n_layers would charge those tensors to
        # every layer and offload one or two layers fewer than actually fit.
        per_layer = [
            size
            for layer, size in (analysis.get("layer_distribution") or {}).items()
            if isinstance(layer, int)
        ]
        if per_layer:
            meta["layer_bytes"] = int(sum(per_layer) / len(per_layer))
        if meta["quant"] == "?" and am.get("dominant_quant"):
            meta["quant"] = am["dominant_quant"]
    except Exception as exc:  # analyzer is best-effort, but never silent
        console.print(
            f"[{C_YELLOW}] ⚠ Could not read GGUF metadata from {path.name}: {exc}[/]\n"
            f"[{C_DIM}]  Planner will fall back to worst-case KV/architecture estimates.[/]"
        )

    return meta


def _unique_alias(base: str) -> str:
    alias = base
    n = 2
    while get_model(alias) or (MODELS_DIR / f"{alias}.gguf").exists():
        alias = f"{base}-{n}"
        n += 1
    return alias


@app.command("import")
def import_model(
    src: str = typer.Argument(..., help="Path to .gguf file"),
    name: str = typer.Option(None, "--name", "-n", help="Alias for the model"),
):
    """Import a local .gguf file."""
    ensure_dirs()
    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        console.print(f"[{C_RED}] ✗ File not found: {src_path}[/]")
        raise typer.Exit(1)

    base_alias = (name or src_path.stem).removesuffix(".gguf")
    alias = _unique_alias(base_alias)
    dst = MODELS_DIR / f"{alias}.gguf"

    if src_path.resolve() != dst.resolve():
        with console.status(f"[{C_ACCENT}] Copying {src_path.name}…"):
            shutil.copy2(src_path, dst)

    animated_loading_bar("IMPORTING GGUF", 1.0)
    meta = _read_meta(dst, src_path.name)
    add_model(alias, dst, meta)

    cfg = load_config()
    cfg["default_model"] = alias
    save_config(cfg)

    param_str = f"{meta.get('n_params', 0) / 1e9:.2f}B · " if meta.get("n_params") else ""
    console.print(f"  [{C_GREEN}]✓ IMPORTED & SET AS DEFAULT[/]  [{C_TEXT}]{alias}[/]")
    console.print(
        f"    [{C_DIM}]└─ {param_str}{meta['size_bytes'] / 1e9:.2f} GB  ·  "
        f"{meta.get('n_layers', 0)} layers  ·  {meta['quant']}[/]\n"
    )


@app.command("search")
def search_model():
    """Search and download GGUF models from Hugging Face."""
    ensure_dirs()
    console.print(
        Panel(
            "  Enter a keyword to search for GGUF models on Hugging Face.",
            title="[ SEARCH HUB ]",
            border_style=C_ACCENT,
            box=box.ROUNDED,
        )
    )
    query = Prompt.ask(f"[{C_ACCENT}]  ▸ Keyword[/]")
    if not query:
        return

    with console.status(f"[{C_ACCENT}] Searching for '{query}'…"):
        try:
            models = list(api.list_models(search=query, filter="gguf", limit=10))
        except Exception as e:
            console.print(f"[{C_RED}] ✗ Connection failed: {e}[/]")
            raise typer.Exit(1) from e

    if not models:
        console.print(f"[{C_YELLOW}] ⚠ No models found.[/]")
        raise typer.Exit(0)

    t = Table(box=box.ROUNDED, title=f"SEARCH RESULTS · '{query}'", title_justify="left")
    t.add_column("ID", style=C_DIM, justify="center")
    t.add_column("REPO ID", style=C_TEXT)
    t.add_column("DOWNLOADS", style=C_ACCENT, justify="right")
    for i, m in enumerate(models):
        t.add_row(f"[{i + 1}]", m.id, f"{m.downloads:,}" if m.downloads else "0")
    console.print(t)

    choice = IntPrompt.ask(f"\n[{C_ACCENT}] ▸ Select a model ID[/]", default=1)
    if choice < 1 or choice > len(models):
        raise typer.Exit(1)
    selected_repo = models[choice - 1].id

    with console.status(f"[{C_ACCENT}] Fetching files…"):
        try:
            files = list_repo_files(repo_id=selected_repo)
            gguf_files = [f for f in files if f.endswith(".gguf")]
        except Exception as e:
            raise typer.Exit(1) from e

    if not gguf_files:
        console.print(f"[{C_RED}] ✗ No .gguf files found.[/]")
        raise typer.Exit(1)

    ft = Table(box=box.ROUNDED, title=f"FILES · {selected_repo}", title_justify="left")
    ft.add_column("ID", style=C_DIM, justify="center")
    ft.add_column("FILE NAME", style=C_TEXT)
    for i, f in enumerate(gguf_files):
        ft.add_row(f"[{i + 1}]", f)
    console.print(ft)

    fc = IntPrompt.ask(f"\n[{C_ACCENT}] ▸ Select a file ID to download[/]", default=1)
    if fc < 1 or fc > len(gguf_files):
        raise typer.Exit(1)
    selected_file = gguf_files[fc - 1]

    console.print(f"\n[{C_ACCENT}] ▸ Downloading: {selected_file}[/]")
    try:
        downloaded_path = hf_hub_download(
            repo_id=selected_repo,
            filename=selected_file,
            local_dir=MODELS_DIR,
        )
        console.print(f"\n[{C_GREEN}] ✓ Download complete![/]")
        animated_loading_bar("FINALISING", 0.5)

        base_alias = Path(downloaded_path).stem
        alias = _unique_alias(base_alias)
        meta = _read_meta(Path(downloaded_path))
        add_model(alias, downloaded_path, meta)

        cfg = load_config()
        cfg["default_model"] = alias
        save_config(cfg)

        param_str = f"{meta.get('n_params', 0) / 1e9:.2f}B · " if meta.get("n_params") else ""
        console.print(f"  [{C_GREEN}]✓ REGISTERED & SET AS DEFAULT[/]  [{C_TEXT}]{alias}[/]")
        console.print(
            f"    [{C_DIM}]└─ {param_str}{meta['size_bytes'] / 1e9:.2f} GB  ·  {meta['quant']}[/]"
        )
        console.print(
            f"\n[{C_ACCENT}] ▸ Just run [bold]vipera[/bold] and this model will be pre-selected.\n"
        )
    except KeyboardInterrupt:
        console.print(f"\n[{C_YELLOW}] ⚠ Download cancelled.[/]")
    except Exception as e:
        console.print(f"\n[{C_RED}] ✗ Download failed: {e}[/]")


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


@app.command("inspect")
def inspect_model(
    path: str = typer.Argument(..., help="Path to a .gguf file"),
    tensors: bool = typer.Option(False, "--tensors", help="List every tensor"),
):
    """Analyze GGUF metadata and tensor layout without loading weights."""
    try:
        result = analyze_gguf(path)
    except GGUFAnalysisError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    meta = result["metadata"]
    summary = Table(box=box.ROUNDED, title="GGUF ANATOMY", title_justify="left")
    summary.add_column("PROPERTY", style=C_DIM)
    summary.add_column("VALUE", style=C_TEXT)
    summary.add_row("file", result["path"])
    summary.add_row("architecture", str(meta.get("architecture") or "?"))
    summary.add_row("name", str(meta.get("name") or "?"))
    summary.add_row("layers", str(meta.get("n_layers") or "?"))
    summary.add_row("context", f"{meta['context_length']:,}" if meta.get("context_length") else "?")
    summary.add_row(
        "embedding",
        f"{meta['embedding_length']:,}" if meta.get("embedding_length") else "?",
    )
    summary.add_row("quant (file name)", str(meta.get("quant") or "?"))
    summary.add_row("quant (dominant)", str(meta.get("dominant_quant") or "?"))
    if meta.get("hybrid_attention"):
        summary.add_row("attention", "hybrid")
        summary.add_row(
            "kv layers",
            f"{meta.get('kv_layers', 0)}/{meta.get('n_layers', 0)}"
            f" (recurrent: {meta.get('recurrent_layers', 0)})",
        )
    summary.add_row("tensors", str(meta.get("tensor_count", 0)))
    summary.add_row("size", _format_bytes(int(meta.get("file_size_bytes", 0))))
    console.print(summary)

    total = max(1, int(meta.get("file_size_bytes", 0)))
    tensor_counts: dict[str, int] = {}
    for tensor in result["tensor_map"]:
        tensor_counts[tensor["category"]] = tensor_counts.get(tensor["category"], 0) + 1

    categories = Table(box=box.ROUNDED, title="SIZE BY CATEGORY", title_justify="left")
    categories.add_column("CATEGORY", style=C_TEXT)
    categories.add_column("TENSORS", style=C_DIM, justify="right")
    categories.add_column("SIZE", style=C_ACCENT, justify="right")
    categories.add_column("SHARE", style=C_DIM, justify="right")
    for category, size in sorted(
        result["category_sizes"].items(), key=lambda item: item[1], reverse=True
    ):
        categories.add_row(
            category,
            str(tensor_counts.get(category, 0)),
            _format_bytes(size),
            f"{size / total * 100:.1f}%",
        )
    console.print(categories)

    quant_table = Table(box=box.ROUNDED, title="QUANTIZATION MIX", title_justify="left")
    quant_table.add_column("QUANT", style=C_ACCENT)
    quant_table.add_column("TENSORS", style=C_DIM, justify="right")
    quant_table.add_column("SIZE", style=C_TEXT, justify="right")
    for quant, size in sorted(
        result["quant_sizes"].items(), key=lambda item: item[1], reverse=True
    ):
        quant_table.add_row(
            quant,
            str(result["quant_counts"].get(quant, 0)),
            _format_bytes(size),
        )
    console.print(quant_table)

    warnings = result["warnings"]
    if warnings:
        lines = []
        for warning in warnings[:10]:
            lines.append(f"  • {warning['tensor']}: {warning['quant']}")
        if len(warnings) > 10:
            lines.append(f"  … and {len(warnings) - 10} more")
        console.print(
            Panel(
                "\n".join(lines),
                title=Text(" QUALITY WARNINGS ", style=C_YELLOW),
                border_style=C_YELLOW,
                box=box.ROUNDED,
            )
        )

    if tensors:
        tensor_table = Table(box=box.SIMPLE, title="TENSORS", title_justify="left")
        tensor_table.add_column("LAYER", style=C_DIM, justify="right")
        tensor_table.add_column("NAME", style=C_TEXT)
        tensor_table.add_column("QUANT", style=C_ACCENT)
        tensor_table.add_column("SHAPE", style=C_DIM, justify="right")
        tensor_table.add_column("SIZE", style=C_DIM, justify="right")
        for tensor in result["tensor_map"]:
            layer = tensor["layer"]
            tensor_table.add_row(
                str(layer) if layer is not None else "—",
                tensor["name"],
                tensor["quant"],
                "x".join(str(dim) for dim in tensor["shape"]),
                _format_bytes(tensor["n_bytes"]),
            )
        console.print(tensor_table)


def _require_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        console.print(f"[{C_RED}] ✗ {label} not found: {path}[/]")
        raise typer.Exit(1)
    return path


def _prepare_output(value: str | None, fallback: Path) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.exists():
            console.print(f"[{C_RED}] ✗ output already exists: {path}[/]")
            raise typer.Exit(1)
        return path
    return unique_output_path(fallback)


def _run_tool(command: list[str], dry_run: bool, cleanup: Path | None = None) -> bool:
    console.print(f"[{C_DIM}]$ {shlex.join(command)}[/]")
    if dry_run:
        return True
    ok = True
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        ok = False
    except subprocess.CalledProcessError as e:
        console.print(f"[{C_RED}] ✗ command failed with exit code {e.returncode}[/]")
        ok = False
    except KeyboardInterrupt:
        console.print(f"\n[{C_YELLOW}] ⚠ Interrupted.[/]")
        ok = False
    if not ok and cleanup is not None:
        cleanup.unlink(missing_ok=True)
    return ok


@app.command("imatrix")
def imatrix_cmd(
    path: str = typer.Argument(..., help="Path to a .gguf file"),
    calibration_data: str = typer.Option(
        ..., "--calibration-data", "-f", help="Calibration text file"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output imatrix path"),
    ctx: int = typer.Option(512, "--ctx", help="Calibration context size"),
    threads: int = typer.Option(None, "--threads", "-t", help="CPU threads"),
    bin_dir: str = typer.Option(None, "--bin-dir", help="Directory with llama.cpp tools"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command only"),
):
    """Build an importance matrix with llama-imatrix."""
    model = _require_file(path, "model")
    calibration = _require_file(calibration_data, "calibration data")
    if ctx <= 0:
        console.print(f"[{C_RED}] ✗ --ctx must be greater than zero[/]")
        raise typer.Exit(1)

    try:
        binary = find_llama_tool("imatrix", bin_dir)
        default_out = model.with_suffix(".imatrix.gguf")
        out = _prepare_output(output, default_out)
        command = build_imatrix_command(binary, model, calibration, out, ctx, threads)
    except OptimizerError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    if not _run_tool(command, dry_run, cleanup=out):
        raise typer.Exit(1)
    if not dry_run:
        console.print(f"[{C_GREEN}] ✓ imatrix written:[/] {out}")


@app.command("quantize")
def quantize_cmd(
    path: str = typer.Argument(..., help="Path to a .gguf file"),
    target_quant: str = typer.Option(
        "q4_k_m", "--target-quant", "-q", help="Target llama.cpp quantization type"
    ),
    imatrix: str = typer.Option(None, "--imatrix", help="Importance matrix file"),
    output: str = typer.Option(None, "--output", "-o", help="Output .gguf path"),
    prune_layers: str = typer.Option(
        None, "--prune-layers", help="Comma-separated layer numbers to prune"
    ),
    prior_weight: int = typer.Option(
        1, "--prior-weight", help="MoE prior weight (requires a supporting build)"
    ),
    allow_requantize: bool = typer.Option(
        False, "--allow-requantize", help="Allow requantizing an already quantized model"
    ),
    output_tensor: str = typer.Option(
        None,
        "--output-tensor",
        help="Mixed-precision type for output layer (e.g. q6_k)",
    ),
    token_embedding: str = typer.Option(
        None,
        "--token-embedding",
        help="Mixed-precision type for token embeddings (e.g. q6_k)",
    ),
    bin_dir: str = typer.Option(None, "--bin-dir", help="Directory with llama.cpp tools"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command only"),
):
    """Quantize a GGUF file with llama-quantize."""
    model = _require_file(path, "model")
    imatrix_path = _require_file(imatrix, "imatrix") if imatrix else None
    if prior_weight < 1:
        console.print(f"[{C_RED}] ✗ --prior-weight must be at least 1[/]")
        raise typer.Exit(1)

    try:
        binary = find_llama_tool("quantize", bin_dir)
        supports_prior = prior_weight != 1 and tool_supports_flag(binary, "--prior-weight")
        out = _prepare_output(output, default_output_path(model, target_quant))
        command = build_quantize_command(
            binary,
            model,
            out,
            target_quant,
            imatrix=imatrix_path,
            prune_layers=prune_layers,
            prior_weight=prior_weight,
            allow_requantize=allow_requantize,
            supports_prior_weight=supports_prior,
            output_tensor_type=output_tensor,
            token_embedding_type=token_embedding,
        )
    except OptimizerError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    if prune_layers:
        console.print(f"[{C_YELLOW}] ⚠ Pruning layers can substantially reduce quality.[/]")
    if not _run_tool(command, dry_run, cleanup=out):
        raise typer.Exit(1)
    if not dry_run:
        console.print(
            f"[{C_GREEN}] ✓ quantized model:[/] {out} ({_format_bytes(out.stat().st_size)})"
        )


@app.command("optimize")
def optimize_cmd(
    path: str = typer.Argument(..., help="Path to a .gguf file"),
    target_size: float = typer.Option(
        None, "--target-size", help="Target model size in GB (decimal)"
    ),
    target_quant: str = typer.Option(None, "--target-quant", help="Override the chosen quant"),
    calibration_data: str = typer.Option(
        None, "--calibration-data", "-f", help="Optional calibration text file"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output .gguf path"),
    prune_layers: str = typer.Option(None, "--prune-layers"),
    prior_weight: int = typer.Option(1, "--prior-weight"),
    output_tensor: str = typer.Option(
        None,
        "--output-tensor",
        help="Mixed-precision type for output layer (e.g. q6_k)",
    ),
    token_embedding: str = typer.Option(
        None,
        "--token-embedding",
        help="Mixed-precision type for token embeddings (e.g. q6_k)",
    ),
    bin_dir: str = typer.Option(None, "--bin-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Analyze, optionally build an imatrix, then quantize toward a size target."""
    model = _require_file(path, "model")
    if target_size is None and target_quant is None:
        console.print(f"[{C_RED}] ✗ provide --target-size or --target-quant[/]")
        raise typer.Exit(1)
    if target_size is not None and target_size <= 0:
        console.print(f"[{C_RED}] ✗ --target-size must be greater than zero[/]")
        raise typer.Exit(1)
    if prior_weight < 1:
        console.print(f"[{C_RED}] ✗ --prior-weight must be at least 1[/]")
        raise typer.Exit(1)

    try:
        analysis = analyze_gguf(str(model))
        meta = analysis["metadata"]
        current_size = int(meta["file_size_bytes"])
        current_quant = str(meta.get("quant") or meta.get("dominant_quant") or "Q4_K_M")
        if target_quant:
            quant = normalize_quant(target_quant)
            estimated = estimate_size_for_quant(current_size, current_quant, quant)
        else:
            choice = estimate_quant_for_target(
                current_size, current_quant, int(target_size * 1_000_000_000)
            )
            quant = choice.quant
            estimated = choice.estimated_bytes
    except (GGUFAnalysisError, OptimizerError) as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    try:
        quantize_bin = find_llama_tool("quantize", bin_dir)
        supports_prior = prior_weight != 1 and tool_supports_flag(quantize_bin, "--prior-weight")
    except OptimizerError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    stem = model.stem
    out = _prepare_output(output, default_output_path(model, quant))
    imatrix_out = model.with_name(f"{stem}.imatrix.gguf")
    if imatrix_out.exists():
        imatrix_out = unique_output_path(imatrix_out)

    plan = Table(box=box.ROUNDED, title="OPTIMIZATION PLAN", title_justify="left")
    plan.add_column("PROPERTY", style=C_DIM)
    plan.add_column("VALUE", style=C_TEXT)
    plan.add_row("model", str(model))
    plan.add_row("current", f"{current_quant} · {_format_bytes(current_size)}")
    plan.add_row("target quant", quant)
    plan.add_row("estimated size", _format_bytes(estimated))
    if target_size is not None:
        plan.add_row("target size", f"{target_size:.2f} GB")
    plan.add_row("output", str(out))
    plan.add_row("imatrix", str(imatrix_out) if calibration_data else "not requested")
    console.print(plan)

    calibration = None
    if calibration_data:
        calibration = _require_file(calibration_data, "calibration data")
        try:
            imatrix_bin = find_llama_tool("imatrix", bin_dir)
            imatrix_cmd = build_imatrix_command(
                imatrix_bin, model, calibration, imatrix_out, ctx_size=512
            )
        except OptimizerError as e:
            console.print(f"[{C_RED}] ✗ {e}[/]")
            raise typer.Exit(1) from e
        if not _run_tool(imatrix_cmd, dry_run, cleanup=imatrix_out):
            raise typer.Exit(1)

    try:
        command = build_quantize_command(
            quantize_bin,
            model,
            out,
            quant,
            imatrix=imatrix_out if calibration else None,
            prune_layers=prune_layers,
            prior_weight=prior_weight,
            allow_requantize=True,
            supports_prior_weight=supports_prior,
            output_tensor_type=output_tensor,
            token_embedding_type=token_embedding,
        )
    except OptimizerError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    if prune_layers:
        console.print(f"[{C_YELLOW}] ⚠ Pruning layers can substantially reduce quality.[/]")
    console.print(
        f"[{C_YELLOW}] ⚠ Optimizing an already-quantized model requantizes it; quality can drop.[/]"
    )
    if not _run_tool(command, dry_run, cleanup=out):
        raise typer.Exit(1)
    if not dry_run:
        console.print(
            f"[{C_GREEN}] ✓ optimized model:[/] {out} ({_format_bytes(out.stat().st_size)})"
        )


@app.command("list")
def list_cmd():
    """List all installed models."""
    models = list_models()
    if not models:
        console.print(f"[{C_DIM}]No models. Run `vipera model import` or `vipera model search`.[/]")
        return

    t = Table(box=box.ROUNDED, title="LOCAL MODELS", title_justify="left")
    t.add_column("ID", style=C_DIM, justify="center")
    t.add_column("ALIAS", style=C_TEXT)
    t.add_column("SIZE", style=C_DIM)
    t.add_column("QUANT", style=C_ACCENT)
    t.add_column("LAYERS", style=C_DIM, justify="right")
    t.add_column("CTX", style=C_DIM, justify="right")

    for i, m in enumerate(models):
        meta = m.get("meta", {})
        size_str = f"{meta.get('size_bytes', 0) / 1e9:>5.1f} G"
        ctx_val = meta.get("context_length") or 0
        ctx_str = f"{ctx_val:,}" if ctx_val else "?"
        t.add_row(
            f"[{i + 1:02d}]",
            m["name"],
            size_str,
            meta.get("quant", "?"),
            f"{meta.get('n_layers', 0) or '?'}",
            ctx_str,
        )
    console.print(t)
    console.print()


@app.command("rm")
def rm_cmd(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Remove a model."""
    m = get_model(name)
    if not m:
        console.print(f"[{C_RED}] ✗ Not found: {name}[/]")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Remove model '{name}'?"):
        raise typer.Abort()

    # Registry first: a dangling file is easy to reclaim, a dangling entry is not.
    remove_model(name)

    cfg = load_config()
    if cfg.get("default_model") == name:
        cfg["default_model"] = None
        save_config(cfg)

    path = Path(m["path"])
    if any(other["path"] == m["path"] for other in list_models()):
        # Two aliases can share one file (`model search` for the same download
        # twice); deleting it here would break the alias that remains.
        console.print(f"[{C_DIM}]  Kept {path}: another alias still uses it.[/]")
    else:
        path.unlink(missing_ok=True)

    console.print(f"[{C_GREEN}] ✓ Removed: {name}[/]")


@app.command("recommend")
def recommend_quantization(
    path: str = typer.Argument(..., help="Path to a .gguf file"),
):
    """Analyze a model and recommend mixed-precision quantization settings."""
    try:
        result = analyze_gguf(path)
    except GGUFAnalysisError as e:
        console.print(f"[{C_RED}] ✗ {e}[/]")
        raise typer.Exit(1) from e

    meta = result["metadata"]
    current_quant = meta.get("quant", "Unknown")
    warnings = result["warnings"]

    console.print(f"\n[{C_ACCENT}]▸ Optimization Recommendations for {path}[/]\n")

    if not warnings:
        console.print(f"  [{C_GREEN}]✓ This model has no critical low-precision tensors.[/]")
        console.print(f"  [{C_DIM}]You can safely quantize to {current_quant} or lower.[/]\n")
        return

    console.print(f"  [{C_YELLOW}]⚠ Found {len(warnings)} critical tensors quantized below Q4_K[/]")
    console.print(
        f"  [{C_DIM}]To preserve quality, use mixed-precision for output & embeddings:[/]\n"
    )

    console.print(f"  [{C_ACCENT}]Command to run:[/]")
    console.print(
        f"  vipera model quantize {path} \\\n"
        f"    --target-quant {current_quant} \\\n"
        f"    --output-tensor q6_k \\\n"
        f"    --token-embedding q6_k \\\n"
        f"    --imatrix <your_imatrix.gguf>\n"
    )

# commands/serve_cmd.py
import os
import socket
import time
from pathlib import Path

import psutil
import typer
from rich import box
from rich.panel import Panel
from rich.text import Text

from vipera.config import VIPERA_HOME, load_config
from vipera.engine import ChatEngine, create_engine, resolve_backend
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
    console,
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

            console.print(
                Panel(
                    bench_text,
                    border_style=speed_color,
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
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

    content = Text()
    content.append("  ▸ MOUNTING MODEL\n", style=C_ACCENT)
    content.append(f"    ├─ {model}\n", style=C_TEXT)
    content.append(f"    ├─ {quant}\n", style=C_DIM)
    content.append(f"    └─ backend {backend}\n\n", style=C_DIM)

    content.append("  ▸ ALLOCATING VRAM\n", style=C_ACCENT)
    offload_layers = p.n_layers if p.requires_server else p.gpu_layers
    content.append(
        "    ├─ layer offload " + "·" * 28 + f" {offload_layers}/{n_layers_total}\n",
        style=C_TEXT,
    )
    if p.requires_server:
        gpu_experts = p.n_layers - p.moe_cpu_layers
        ubatch = f" · ub {p.n_ubatch}" if p.n_ubatch else ""
        content.append(
            "    ├─ moe experts " + "·" * 27 + f" {gpu_experts}/{p.n_layers} layers{ubatch}\n",
            style=C_TEXT,
        )
    kv_suffix = f" · {p.kv_cache_gb:.2f} GB" if p.kv_cache_gb > 0 else ""
    content.append(
        f"    ├─ kv cache ({p.kv_cache}) " + "·" * 24 + f" [ OK ]{kv_suffix}\n",
        style=C_TEXT,
    )
    if p.kv_layers and p.kv_layers < n_layers_total:
        content.append(
            "    ├─ hybrid attention " + "·" * 21 + f" {p.kv_layers}/{n_layers_total} kv layers\n",
            style=C_TEXT,
        )
    content.append(
        "    └─ flash attention " + "·" * 27 + f" [ {'ON' if p.flash_attn else 'OFF'} ]\n",
        style=C_TEXT,
    )

    console.print(
        Panel(
            content,
            title=Text(" VIPERA DAEMON · BOOT ", style=C_ACCENT),
            border_style=C_ACCENT,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

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

# commands/system_cmd.py
import platform
import sys

import typer
from rich import box
from rich.panel import Panel
from rich.text import Text

from vipera.config import DEFAULT_CONFIG, _coerce_strict, load_config, save_config
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

    ROW_W = 78

    def add_sec(title: str):
        inner = f"[ {title} ]"
        rest = ROW_W - 5 - len(inner)
        content.append(f"  ┌─{inner}", style=C_ACCENT)
        content.append("─" * max(0, rest) + "┐\n", style=C_BORDER)

    def end_sec():
        content.append("  └" + "─" * (ROW_W - 4) + "┘\n\n", style=C_BORDER)

    def row(label: str, val: str, status: str = "PASS", col: str = C_GREEN):
        counters[status.lower()] += 1
        status_s = f"[ {status:^4} ]"
        dots = "·" * max(1, 41 - len(val))
        content.append(f"  │ {label:<20} ", style=C_TEXT)
        content.append(val, style=C_TEXT)
        content.append(f" {dots} ", style=C_TEXT)
        content.append(status_s, style=col)
        content.append(" │\n", style=C_TEXT)

    add_sec("RUNTIME")
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
            row("gpu offload", "compiled in (CUDA/Metal)", "PASS", C_GREEN)
        elif gpu_offload is False:
            row("gpu offload", "CPU-only build", "WARN", C_YELLOW)
        else:
            row("gpu offload", "unknown (binding too old)", "WARN", C_YELLOW)
    except ImportError:
        row("llama-cpp-python", "missing", "FAIL", C_RED)
        llama_missing = True

    gguf_missing = False
    try:
        import gguf  # noqa: F401

        row("gguf", "installed")
    except ImportError:
        row("gguf", "missing", "WARN", C_YELLOW)
        gguf_missing = True
    end_sec()

    add_sec("HARDWARE")
    hw = detect()
    row("cpu", f"{hw.cpu_cores}C / {hw.cpu_threads}T")

    if hw.is_apple_silicon:
        row("gpu", "Apple Silicon (Metal)", "PASS", C_GREEN)
        row("unified memory", f"{hw.ram_free_gb:.1f} GB free / {hw.ram_total_gb:.1f} GB total")
    elif hw.gpu_name != "None":
        row("gpu", hw.gpu_name, "PASS", C_GREEN)
        row("vram", f"{hw.vram_free_gb:.1f} GB free / {hw.vram_total_gb:.1f} GB total")
    else:
        row("gpu", "None (CPU Only)", "WARN", C_YELLOW)
        row("ram available", f"{hw.ram_free_gb:.1f} GB free / {hw.ram_total_gb:.1f} GB total")
    end_sec()

    summary = Text("  SUMMARY   ", style=C_ACCENT)
    summary.append(f"[ {counters['pass']} PASSED ]", style=C_GREEN)
    summary.append("  ─  ", style=C_BORDER)
    summary.append(f"[ {counters['warn']} WARN ]", style=C_YELLOW)
    summary.append("  ─  ", style=C_BORDER)
    summary.append(f"[ {counters['fail']} FAIL ]\n", style=C_RED)
    content.append(summary)

    console.print(
        Panel(
            content,
            title=Text(" VIPERA DIAGNOSTICS ", style=C_ACCENT),
            border_style=C_ACCENT,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    if llama_missing:
        console.print(f"[{C_YELLOW}] ⚠ llama-cpp-python is required to load models.[/]")
        console.print(f"[{C_DIM}]   Install:[/] python -m pip install llama-cpp-python")
        console.print(f"[{C_DIM}]   Then restart the gateway.[/]\n")

    if gpu_offload is False and hw.gpu_name != "None":
        console.print(
            f"[{C_YELLOW}] ⚠ A GPU was detected but this llama-cpp-python build has no "
            f"GPU backend.[/]"
        )
        console.print(
            f"[{C_DIM}]   gpu_layers is planned but ignored: every layer runs on the CPU. "
            f"Reinstall with the matching extra:[/]"
        )
        console.print(
            f"[{C_DIM}]   CMAKE_ARGS='-DGGML_CUDA=on' "
            f"pip install --force-reinstall llama-cpp-python"
        )
        console.print(f"[{C_DIM}]   Then re-run this command.[/]\n")

    if gguf_missing:
        console.print(
            f"[{C_YELLOW}] ⚠ gguf is optional, but layer and parameter metadata need it.[/]"
        )
        console.print(f"[{C_DIM}]   Install:[/] python -m pip install gguf\n")


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
    try:
        cfg = _coerce_strict(cfg)
    except ValueError as exc:
        console.print(f"[{C_RED}] ✗ {exc}[/]")
        raise typer.Exit(1) from exc
    save_config(cfg)
    console.print(f"[{C_GREEN}] ✓[/]  {key} = {cfg[key]}")

# config.py
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

VIPERA_HOME = Path(os.environ.get("VIPERA_HOME") or (Path.home() / ".vipera"))
MODELS_DIR = VIPERA_HOME / "models"
LOGS_DIR = VIPERA_HOME / "logs"
REGISTRY_FILE = VIPERA_HOME / "registry.json"
CONFIG_FILE = VIPERA_HOME / "config.json"

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8686,
    "default_model": None,
    "backend": "inprocess",
    "profile": "auto",
    "ctx": 8192,
    "kv_cache": "q8_0",
    "flash_attn": True,
    "n_batch": 512,
    # 0 = auto (server backend uses 896 for MoE plans, min(n_batch, 512)
    # otherwise). -ub is the prefill lever; -b is raised to at least this.
    "n_ubatch": 0,
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
    # llama-server's own still-free speed lever: --spec-type ngram-*. The
    # in-process engine has no equivalent (llama-cpp-python's prompt-lookup
    # draft measured -29% decode on this box), so the value only applies to the
    # `server` backend. "none" turns it off.
    "spec_type": "ngram-map-k",
}

_KV_CACHE_TYPES = {"f32", "bf16", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}

# "inprocess" drives llama-cpp-python in this process; "server" runs the
# llama-server binary as a child and talks OpenAI/SSE to it. The default stays
# inprocess until the server backend has been validated on real hardware.
_BACKENDS = {"inprocess", "server"}

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off", ""}

_TYPE_MAP = {
    "port": int,
    "ctx": int,
    "n_batch": int,
    "n_ubatch": int,
    "max_tokens": int,
    "temperature": float,
    "top_p": float,
    "repeat_penalty": float,
    "flash_attn": bool,
}


def ensure_dirs() -> None:
    for d in (VIPERA_HOME, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _coerce(cfg: dict) -> dict:
    for k, cast in _TYPE_MAP.items():
        if k in cfg and cfg[k] is not None:
            try:
                if cast is bool:
                    cfg[k] = str(cfg[k]).lower() in ("1", "true", "yes", "on")
                else:
                    cfg[k] = cast(cfg[k])
            except (ValueError, TypeError):
                pass
    if cfg.get("kv_cache") not in _KV_CACHE_TYPES:
        cfg["kv_cache"] = DEFAULT_CONFIG["kv_cache"]
    if cfg.get("backend") not in _BACKENDS:
        cfg["backend"] = DEFAULT_CONFIG["backend"]
    return cfg


def _parse_bool(value) -> bool:
    """Strictly parse a boolean, rejecting anything that is not unambiguous."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _TRUE_STRINGS:
        return True
    if s in _FALSE_STRINGS:
        return False
    raise ValueError(f"expected a boolean, got '{value}'")


def _coerce_strict(cfg: dict) -> dict:
    """Raise ValueError on any invalid typed value."""
    for k, cast in _TYPE_MAP.items():
        if k in cfg and cfg[k] is not None:
            try:
                if cast is bool:
                    cfg[k] = _parse_bool(cfg[k])
                else:
                    cfg[k] = cast(cfg[k])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{k}: invalid {cast.__name__} value '{cfg[k]}'") from exc
    if "kv_cache" in cfg and cfg["kv_cache"] not in _KV_CACHE_TYPES:
        valid = ", ".join(sorted(_KV_CACHE_TYPES))
        raise ValueError(f"kv_cache: must be one of {valid}")
    if "backend" in cfg and cfg["backend"] not in _BACKENDS:
        valid = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"backend: must be one of {valid}")
    return cfg


def quarantine_corrupt(path: Path, exc: Exception) -> Path:
    """Keep an unreadable JSON file before anything overwrites it.

    Loads fall back to defaults, so without this copy the next
    ``system config``/``model import`` would silently erase the user's file.
    """
    backup = path.with_name(path.name + ".corrupt")
    with contextlib.suppress(OSError):
        shutil.copy2(path, backup)
    print(
        f"vipera: {path} is unreadable ({exc}); using defaults. Original kept at {backup}",
        file=sys.stderr,
    )
    return backup


def load_config() -> dict:
    ensure_dirs()
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            quarantine_corrupt(CONFIG_FILE, exc)
        else:
            if isinstance(data, dict):
                return _coerce({**DEFAULT_CONFIG, **data})
            quarantine_corrupt(CONFIG_FILE, ValueError("not a JSON object"))
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    ensure_dirs()
    # Write-then-rename so a crash mid-write cannot leave a truncated config
    # that load_config() would silently replace with defaults.
    tmp = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, CONFIG_FILE)

# engine.py
import gc
from dataclasses import replace
from typing import Any, Protocol

import psutil

from vipera.planner import Plan


class ChatEngine(Protocol):
    """What the commands need from either backend.

    `Engine` (in-process) and `ServerEngine` (llama-server child) both satisfy
    this; callers must not reach for `.llm`, which only the in-process one has.
    """

    plan: Plan

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any): ...

    def tokenize(self, text: str | bytes) -> list[int]: ...

    def close(self) -> None: ...


# ggml type ids accepted by llama.cpp for KV caches.
_KV_TYPES = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
    "iq4_nl": 20,
    "bf16": 30,
}

# Sampling and tool-calling parameters accepted by
# llama_cpp.Llama.create_chat_completion. Anything else is dropped so a typo in
# an HTTP request cannot turn into a TypeError from the backend.
_CHAT_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "max_tokens",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
        "tfs_z",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "grammar",
        "response_format",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
    }
)

_DEFAULT_SAMPLING = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
}


def check_host_ram(plan: Plan) -> None:
    """Refuse a plan whose host-side footprint clearly exceeds free RAM.

    An OOM here is an uncatchable SIGKILL, not a Python exception, so the
    GPU->CPU fallback never gets a chance to run. Shared by both backends: the
    llama-server child dies the same way.
    """
    if plan.ram_estimate_gb <= 0:
        return
    available_gb = psutil.virtual_memory().available / 1024**3
    if plan.ram_estimate_gb > available_gb + 0.5:
        raise RuntimeError(
            f"Not enough RAM for this plan: ~{plan.ram_estimate_gb:.1f} GB "
            f"needed on the host, {available_gb:.1f} GB available. "
            f"Lower --ctx, use kv_cache=q4_0, or close other applications."
        )


def layer_ladder(layers: int) -> list[int]:
    """Offload counts to try for a plan that asks for ``layers``, best first.

    A split that is one or two layers too optimistic used to fall straight to
    ``gpu_layers=0``, which is the slowest outcome on the box this was measured
    on: 3.90 t/s CPU-only against 4.97 t/s at the last split that loaded
    (14B/Q4, 8K, GTX 1650). Each rung costs one failed context creation (~2 s
    for a 14B at 8K), and a failed attempt does not hand its VRAM back before
    the next one: after three failures on a 4 GB card even a 10-layer rung
    failed, which is how asking for 20 layers ended up on the CPU. The ladder is
    therefore short and its last rung is a quarter of the plan, not half — it
    exists to land anywhere other than CPU-only.
    """
    if layers <= 0:
        return []
    steps = (layers, layers - 1, layers - 2, max(2, layers // 4))
    return [n for n in dict.fromkeys(steps) if 0 < n <= layers]


def with_gpu_layers(plan: Plan, layers: int) -> Plan:
    """Same plan at a different offload depth, with a truthful note."""
    note = plan.note
    if note.startswith("Hybrid: "):
        total = note.split("/", 1)[1].split()[0]
        note = f"Hybrid: {layers}/{total} layers"
    elif note == "Full GPU":
        # A large card can plan a full offload and still fail to create the
        # context (the driver holds more than the weights). Reporting "Full
        # GPU" while running a reduced split would repeat the lie the ladder
        # exists to stop.
        note = f"Hybrid: {layers}/all layers"
    return replace(plan, gpu_layers=layers, note=note)


def load_attempts(plan: Plan) -> list[tuple[Plan, str]]:
    """Candidate plans for loading, most offload first, ending CPU-only."""
    attempts: list[tuple[Plan, str]] = [(plan, "planned split")]
    if plan.requires_server and plan.n_layers > 0:
        # The server ladder moves one expert layer at a time to the CPU, which
        # frees roughly 190-220 MB on the reference model. The last rung keeps
        # every expert on the CPU instead of failing the load entirely.
        rungs = (plan.moe_cpu_layers + 1, plan.moe_cpu_layers + 2, plan.n_layers)
        for cpu_layers in dict.fromkeys(rungs):
            if cpu_layers <= plan.moe_cpu_layers or cpu_layers > plan.n_layers:
                continue
            gpu_experts = plan.n_layers - cpu_layers
            attempts.append(
                (
                    replace(
                        plan,
                        moe_cpu_layers=cpu_layers,
                        note=f"MoE: {gpu_experts}/{plan.n_layers} expert layers on GPU",
                    ),
                    f"MoE, {gpu_experts}/{plan.n_layers} expert layers",
                )
            )
    else:
        hybrid_total = (
            plan.note.split("/", 1)[1].split()[0] if plan.note.startswith("Hybrid: ") else None
        )
        for layers in layer_ladder(plan.gpu_layers)[1:]:
            label = f"GPU, {layers}/{hybrid_total} layers" if hybrid_total else f"GPU, {layers}"
            attempts.append((with_gpu_layers(plan, layers), label))

    # CPU stages: keep the caller's context and KV type first, then reduce.
    # Clearing the MoE fields matters: -ncmoe with -ngl 0 is not a valid
    # CPU-only plan.
    cpu_plan = replace(
        plan,
        gpu_layers=0,
        offload_kqv=False,
        n_batch=min(plan.n_batch, 128),
        moe_cpu_layers=-1,
        n_ubatch=0,
    )
    reduced_plan = replace(cpu_plan, context=min(plan.context, 2048))
    safe_plan = replace(reduced_plan, kv_cache="f16", flash_attn=False)
    attempts.append((cpu_plan, f"CPU, ctx={cpu_plan.context}, kv={cpu_plan.kv_cache}"))
    attempts.append((reduced_plan, f"CPU, ctx={reduced_plan.context}, kv={reduced_plan.kv_cache}"))
    attempts.append((safe_plan, f"CPU safe mode, ctx={safe_plan.context}, kv={safe_plan.kv_cache}"))
    return attempts


def resolve_backend(plan: Plan, backend: str) -> tuple[str, Plan]:
    """Return a backend that can actually execute ``plan``.

    llama-cpp-python has no ``n_cpu_moe`` argument, so a MoE plan is silently
    ignored by the in-process engine (and its classic ``gpu_layers`` split is
    the only safe fallback). If a llama-server with ``-ncmoe`` is available,
    switch to it; otherwise clear the MoE fields so the banner and bench report
    what is really running.
    """
    if backend == "server" or not plan.requires_server:
        return backend, plan

    try:
        from vipera.optimizer import OptimizerError, find_llama_tool
        from vipera.server_backend import supports_n_cpu_moe

        binary = find_llama_tool("server")
        supported = supports_n_cpu_moe(binary)
    except (OptimizerError, OSError):
        binary = None
        supported = False

    if supported:
        gpu_experts = plan.n_layers - plan.moe_cpu_layers
        print(
            "[i] MoE plan: using llama-server for expert placement "
            f"({gpu_experts}/{plan.n_layers} expert layers on GPU, "
            f"ubatch {plan.n_ubatch})."
        )
        return "server", plan

    where = f"{binary} has no -ncmoe" if binary else "llama-server was not found"
    print(
        f"[!] Warning: {where}; the MoE plan cannot run in-process and is "
        "falling back to the classic layer split. Install a newer llama-server "
        "to use expert offload."
    )
    return "inprocess", replace(plan, moe_cpu_layers=-1, n_ubatch=0)


def create_engine(plan: Plan, model_path: str, backend: str = "inprocess") -> ChatEngine:
    """Build the engine selected by the ``backend`` config key."""
    backend, plan = resolve_backend(plan, backend)
    if backend == "server":
        from vipera.server_backend import ServerEngine

        return ServerEngine(plan, model_path)
    return Engine(plan, model_path)


class Engine:
    def __init__(self, plan: Plan, model_path: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
            ) from None

        self.plan = plan
        self._model_path = model_path
        self._Llama = Llama
        self._closed = False

        check_host_ram(plan)

        # Stage the fallback instead of jumping straight to CPU-only: first walk
        # the offload down one layer at a time, then give up the placement, then
        # the context and finally the KV type. Callers such as Hermes require a
        # 64K context floor, and quantized KV is the only way that fits in RAM on
        # this class of machine, so dropping to 2048 must stay the last resort.
        attempts = load_attempts(plan)
        last_exc: Exception | None = None
        for i, (candidate, label) in enumerate(attempts):
            try:
                self.llm = self._load(candidate)
                self.plan = candidate
                if i > 0:
                    print(f"\n[!] Warning: failed to load with GPU. Falling back to {label}...")
                if i > 0 and candidate.context < plan.context:
                    print(
                        f"[!] Warning: requested ctx={plan.context} will NOT be met "
                        f"(using {candidate.context})."
                    )
                return
            except Exception as exc:  # try the next rung
                last_exc = exc

        raise RuntimeError(f"Failed to initialize model even on CPU: {last_exc}") from last_exc

    def _load(self, p: Plan):
        kv = _KV_TYPES.get(p.kv_cache)

        def _do(use_mlock: bool):
            return self._Llama(
                model_path=self._model_path,
                n_gpu_layers=p.gpu_layers,
                n_ctx=p.context,
                n_threads=p.cpu_threads,
                n_batch=p.n_batch,
                n_ubatch=min(p.n_batch, 512),
                flash_attn=p.flash_attn,
                offload_kqv=p.offload_kqv,
                type_k=kv,
                type_v=kv,
                use_mmap=True,
                use_mlock=use_mlock,
                verbose=False,
            )

        try:
            return _do(p.use_mlock)
        except Exception:
            if not p.use_mlock:
                raise
            return _do(False)

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any):
        """Run a chat completion, forwarding every supported OpenAI parameter.

        Only keys in ``_CHAT_PARAMS`` reach llama.cpp. ``None`` values are
        treated as "not set" so the backend keeps its own defaults for
        optional knobs such as ``seed`` or ``logit_bias``.
        """
        params = {k: v for k, v in kw.items() if k in _CHAT_PARAMS and v is not None}
        for key, default in _DEFAULT_SAMPLING.items():
            params.setdefault(key, default)
        return self.llm.create_chat_completion(messages=messages, stream=stream, **params)

    def tokenize(self, text: str | bytes) -> list[int]:
        """Tokenize with the loaded model's vocabulary.

        Callers must go through this instead of ``engine.llm.tokenize`` so the
        same code works against the server backend, which has no ``llm``.
        """
        data = text.encode() if isinstance(text, str) else text
        return self.llm.tokenize(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        llm = self.llm
        self.llm = None
        del llm
        gc.collect()

# gguf_analyzer.py
"""Inspect GGUF headers, tensor metadata, and per-layer size distribution.

The analyzer only reads the GGUF index. Tensor payloads stay on disk, so this
is safe to run on multi-gigabyte models and does not need llama.cpp.
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_QUANT_RE = re.compile(
    r"(IQ\d+_\w+|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d|NVFP4|MXFP4|TQ\d_0|BF16|F16|F32|Q\d+)",
    re.IGNORECASE,
)

_QUANT_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
    40: "NVFP4",
    41: "Q1_0",
    # Not present in gguf 0.19 (the reader rejects such files); kept so a raw
    # header dump can still label the id llama.cpp reserves for it.
    42: "Q2_0",
}

# Rough bits-per-weight ordering. Only used to flag quality-sensitive tensors
# quantized below Q4_K; it is a heuristic, not a quality guarantee.
_QUANT_RANK = {
    "F64": 64.0,
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_K": 8.0,
    "Q8_0": 8.0,
    "Q8_1": 8.0,
    "Q6_K": 6.0,
    "Q5_K": 5.0,
    "Q5_0": 5.0,
    "Q5_1": 5.0,
    "Q4_K": 4.5,
    "Q4_0": 4.0,
    "Q4_1": 4.0,
    "IQ4_NL": 4.0,
    "IQ4_XS": 4.0,
    "Q3_K": 3.0,
    "IQ3_XXS": 3.0,
    "IQ3_S": 3.0,
    "Q2_K": 2.0,
    "IQ2_XXS": 2.0,
    "IQ2_XS": 2.0,
    "IQ2_S": 2.0,
    "IQ1_S": 1.0,
    "IQ1_M": 1.0,
    "Q2_0": 2.0,
    "TQ2_0": 2.2,
    "TQ1_0": 1.7,
    "Q1_0": 1.0,
    # 4-bit float formats; ranked with the other 4-bit types, below Q4_K.
    "MXFP4": 4.0,
    "NVFP4": 4.0,
}

# Tensor names that imply a layer keeps a KV cache, and names that imply a
# recurrent/SSM layer with a fixed-size state instead. Hybrid stacks (KDA/BCSA,
# Mamba, RWKV) mix both; the planner only needs the split.
# `attn_k` also covers MLA/KDA-adjacent names: attn_k_b, attn_kv_a_mqa, attn_k_norm.
_KV_TENSOR_MARKERS = ("attn_k", "attn_v", "attn_qkv")
# `conv` also covers `conv1d` and the `shortconv.*` stacks LFM2.5 uses for its
# non-KV layers: those layers hold no K/V tensors at all, so without the marker
# they are skipped, `hybrid` never trips and the planner prices all layers as
# KV-bearing.
_RECURRENT_TENSOR_MARKERS = ("ssm_", "linear_attn", "kda", "mamba", "conv")

_CRITICAL_TENSORS = ("ffn_down", "attn_v")
_CRITICAL_THRESHOLD = "Q4_K"
_CRITICAL_RANK = _QUANT_RANK[_CRITICAL_THRESHOLD]

# Quant type ids the static table may lag behind: gguf grows a new type every
# few releases, and labelling an unknown id as TYPE_nn turns into a bogus
# quality warning because it ranks 0. Look the name up in gguf when possible.
_GGUF_QUANT_TYPES: dict[int, str] | None = None


class GGUFAnalysisError(RuntimeError):
    """Raised when a GGUF file cannot be inspected."""


def _gguf_quant_names() -> dict[int, str]:
    """Map id -> name from the optional `gguf` package, cached on first use."""
    global _GGUF_QUANT_TYPES
    if _GGUF_QUANT_TYPES is None:
        try:
            from gguf.constants import GGMLQuantizationType
        except Exception:  # package missing, or an older layout
            # Do not cache the failure: a later call (real gguf installed, or a
            # test that stubs the module) should still be able to resolve ids.
            return {}
        _GGUF_QUANT_TYPES = {int(t): t.name for t in GGMLQuantizationType}
    return _GGUF_QUANT_TYPES


def _quant_name(tensor_type: Any) -> str:
    try:
        value = int(tensor_type)
    except (TypeError, ValueError):
        return str(tensor_type)
    name = _QUANT_NAMES.get(value)
    if name:
        return name
    return _gguf_quant_names().get(value, f"TYPE_{value}")


def _quant_rank(name: str) -> float | None:
    """Rough bits-per-weight for a quant name; None when it is unknown.

    Unknown must not rank as 0: that would flag every future 4-bit type as
    "below Q4_K" and make the warning meaningless.
    """
    if name in _QUANT_RANK:
        return _QUANT_RANK[name]
    if name.startswith("IQ4"):
        return 4.0
    if name.startswith("IQ3"):
        return 3.0
    if name.startswith("IQ2"):
        return 2.0
    if name.startswith("IQ1"):
        return 1.0
    return None


def _classify_tensor(name: str) -> str:
    if name == "output.weight" or name.startswith("token_embd") or name == "token_types.weight":
        return "embedding"
    if "_norm" in name:
        return "norm"
    if "_exps" in name or "ffn_gate_inp" in name or "expert" in name:
        return "expert"
    if "attn_" in name:
        return "attention"
    if "ffn_" in name:
        return "ffn"
    return "other"


def _layer_index(name: str) -> int | None:
    match = re.match(r"blk\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _attention_profile(tensor_names: list[str], n_layers: int | None) -> dict[str, Any]:
    """Split layers into KV-cache layers and recurrent (state-only) layers.

    Detection is name-based because GGUF metadata does not describe per-layer
    attention type: KDA/Mamba layers carry `ssm_*` / `linear_attn` / `conv1d`
    tensors, attention layers carry `attn_k`/`attn_v`/`attn_qkv`. A layer that
    matches both counts as recurrent, since over-reserving KV is the safe side
    of a memory budget.
    """
    kv_layers: set[int] = set()
    recurrent_layers: set[int] = set()
    for name in tensor_names:
        layer = _layer_index(name)
        if layer is None:
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in _RECURRENT_TENSOR_MARKERS):
            recurrent_layers.add(layer)
        elif any(marker in lowered for marker in _KV_TENSOR_MARKERS):
            kv_layers.add(layer)

    recurrent_layers -= kv_layers
    total = int(n_layers or 0) or (max(kv_layers | recurrent_layers, default=-1) + 1)
    hybrid = bool(recurrent_layers) and bool(kv_layers)
    return {
        "hybrid_attention": hybrid,
        # Only shrink the budget when the split is actually understood; a
        # partial match on a dense model must not reduce the KV estimate.
        "kv_layers": len(kv_layers) if hybrid else total,
        "recurrent_layers": len(recurrent_layers),
    }


def _field_value(fields: dict, key: str) -> Any:
    field = fields.get(key)
    if field is None:
        return None
    try:
        value = field.contents()
    except Exception:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        item = value[0]
        if hasattr(item, "item"):
            item = item.item()
        return item
    return value


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field_first_int(fields: dict, key: str) -> int | None:
    """Return the representative element of an int-array field, or None.

    Per-layer arrays such as attention.head_count are uniform for the
    architectures we support, so the first entry describes the model. Hybrid
    stacks are the exception: lfm2moe writes ``attention.head_count_kv`` as
    0 on the conv-only layers and 8 on the attention ones, and taking the first
    entry reports zero KV heads, which makes the planner fall back to the full
    head count and price every layer as 4x its real KV. First non-zero entry
    wins; an all-zero array stays zero so the caller can fall back as before.
    """
    value = _field_value(fields, key)
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _as_int(item)
            if parsed:
                return parsed
        return _as_int(value[0]) if value else None
    return _as_int(value)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def analyze_gguf(path: str) -> dict[str, Any]:
    """Return metadata, tensor map, and size breakdown for a GGUF file.

    The return value is JSON-friendly except for integer layer keys, which are
    kept as integers for convenient sorting. It raises ``GGUFAnalysisError``
    with an actionable message when the file or the optional ``gguf`` package
    is unavailable.
    """
    model_path = Path(path).expanduser()
    if not model_path.is_file():
        raise GGUFAnalysisError(f"file not found: {model_path}")

    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise GGUFAnalysisError(
            'gguf is not installed. Install it with: python -m pip install "vipera-ai[gguf]"'
        ) from exc

    try:
        reader = GGUFReader(str(model_path))
    except Exception as exc:
        raise GGUFAnalysisError(f"failed to read {model_path}: {exc}") from exc

    fields = getattr(reader, "fields", {}) or {}
    architecture = _as_text(_field_value(fields, "general.architecture"))
    name = _as_text(_field_value(fields, "general.name"))

    n_layers = None
    context_length = None
    embedding_length = None
    n_heads = None
    n_kv_heads = None
    head_dim = None
    if architecture:
        n_layers = _as_int(_field_value(fields, f"{architecture}.block_count"))
        context_length = _as_int(_field_value(fields, f"{architecture}.context_length"))
        embedding_length = _as_int(_field_value(fields, f"{architecture}.embedding_length"))

        n_heads = _field_first_int(fields, f"{architecture}.attention.head_count")
        n_kv_heads = _field_first_int(fields, f"{architecture}.attention.head_count_kv")
        head_dim = _field_first_int(fields, f"{architecture}.attention.key_length")
        if head_dim is None and n_heads and embedding_length:
            head_dim = embedding_length // n_heads

    tensor_map: list[dict[str, Any]] = []
    layer_sizes: dict[int | None, int] = defaultdict(int)
    category_sizes: dict[str, int] = defaultdict(int)
    quant_sizes: dict[str, int] = defaultdict(int)
    quant_counts: dict[str, int] = defaultdict(int)
    warnings: list[dict[str, str]] = []
    max_layer = -1
    # MoE weights split in two classes because llama.cpp's --n-cpu-moe only
    # moves routed experts (*_exps). Shared experts (*_shexp) and the router
    # (ffn_gate_inp) run for every token and must stay on the GPU with the
    # attention block, so folding them into expert_bytes would over-offload.
    expert_bytes = 0
    shared_expert_bytes = 0
    expert_layer_bytes: dict[int, int] = defaultdict(int)

    for tensor in getattr(reader, "tensors", []):
        tensor_name = str(tensor.name)
        quant = _quant_name(tensor.tensor_type)
        category = _classify_tensor(tensor_name)
        layer = _layer_index(tensor_name)
        size = int(getattr(tensor, "n_bytes", 0) or 0)
        shape = [int(x) for x in getattr(tensor, "shape", [])]
        elements = int(getattr(tensor, "n_elements", 0) or 0)

        tensor_map.append(
            {
                "name": tensor_name,
                "shape": shape,
                "quant": quant,
                "category": category,
                "layer": layer,
                "n_elements": elements,
                "n_bytes": size,
            }
        )
        layer_sizes[layer] += size
        category_sizes[category] += size
        quant_sizes[quant] += size
        quant_counts[quant] += 1
        if "_exps" in tensor_name:
            expert_bytes += size
            if layer is not None:
                expert_layer_bytes[layer] += size
        elif "_shexp" in tensor_name:
            shared_expert_bytes += size
        if layer is not None:
            max_layer = max(max_layer, layer)

        rank = _quant_rank(quant)
        if (
            any(key in tensor_name for key in _CRITICAL_TENSORS)
            and rank is not None
            and rank < _CRITICAL_RANK
        ):
            warnings.append(
                {
                    "tensor": tensor_name,
                    "quant": quant,
                    "category": category,
                    "message": (
                        f"{tensor_name} is {quant}, below {_CRITICAL_THRESHOLD}; "
                        "this tensor is sensitive to quantization."
                    ),
                }
            )

    if n_layers is None:
        n_layers = max_layer + 1

    expert_layer_sizes = [int(expert_layer_bytes.get(i, 0)) for i in range(n_layers)]

    attention = _attention_profile([t["name"] for t in tensor_map], n_layers)

    filename_quant = None
    match = _QUANT_RE.search(model_path.name)
    if match:
        filename_quant = match.group(1).upper()

    dominant_quant = max(quant_sizes, key=lambda key: quant_sizes[key]) if quant_sizes else None

    return {
        "path": str(model_path),
        "metadata": {
            "architecture": architecture,
            "name": name,
            "n_layers": n_layers,
            "context_length": context_length,
            "embedding_length": embedding_length,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "n_params": _as_int(_field_value(fields, "general.parameter_count")),
            "quant": filename_quant or dominant_quant or "?",
            "dominant_quant": dominant_quant,
            "tensor_count": len(tensor_map),
            "file_size_bytes": os.path.getsize(model_path),
            # Routed experts are the only tensors --n-cpu-moe can move. The
            # per-layer list uses the model's layer indices so the planner can
            # fit the exact bytes of the layers that stay on the GPU.
            "expert_bytes": expert_bytes,
            "shared_expert_bytes": shared_expert_bytes,
            "expert_layer_bytes": expert_layer_sizes,
            "expert_layers": sum(1 for value in expert_layer_sizes if value > 0),
            **attention,
        },
        "tensor_map": tensor_map,
        "layer_distribution": dict(
            sorted(layer_sizes.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
        ),
        "category_sizes": dict(category_sizes),
        "quant_sizes": dict(quant_sizes),
        "quant_counts": dict(quant_counts),
        "warnings": warnings,
    }


# Backwards-compatible alias kept for callers that used the original name.
analyze_gguf_structure = analyze_gguf

# hardware.py
import contextlib
import platform
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_free_gb: float
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    is_apple_silicon: bool


def detect() -> HardwareInfo:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "psutil is required for hardware detection. Install it with: pip install psutil"
        ) from exc

    cores = psutil.cpu_count(logical=False) or 1
    threads = psutil.cpu_count(logical=True) or cores
    vm = psutil.virtual_memory()
    ram_total = vm.total / 1024**3
    ram_free = vm.available / 1024**3

    is_mac = platform.system() == "Darwin"
    is_apple_silicon = is_mac and platform.machine() == "arm64"

    gpu_name, vram_total, vram_free = "None", 0.0, 0.0

    if is_apple_silicon:
        gpu_name = "Apple Silicon (Unified Memory)"
        vram_total = ram_total
        vram_free = ram_free * 0.7
    else:
        with contextlib.suppress(Exception):
            import pynvml

            pynvml.nvmlInit()
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(h)
                gpu_name = name.decode() if isinstance(name, bytes) else name
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                vram_total = mem.total / 1024**3
                vram_free = mem.free / 1024**3
            finally:
                pynvml.nvmlShutdown()

    return HardwareInfo(
        cpu_cores=cores,
        cpu_threads=threads,
        ram_total_gb=ram_total,
        ram_free_gb=ram_free,
        gpu_name=gpu_name,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        is_apple_silicon=is_apple_silicon,
    )

# optimizer.py
"""Helpers for llama.cpp quantization, imatrix, and target-size planning.

The functions here build commands and make sizing decisions. They do not run
subprocesses; the CLI owns execution so it can stream progress and handle
interrupts.
"""

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_TOOL_NAMES = {
    "imatrix": "llama-imatrix",
    "quantize": "llama-quantize",
    "server": "llama-server",
}

_QUALITY_ORDER = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")

# Bytes-per-weight from gguf's block table: bytes * 8 / elements. Keep these
# exact (they drive --target-size math), not rounded to the nominal bit width.
_QUANT_BPW = {
    "NVFP4": 4.5,  # 36 B / 64 elems
    "MXFP4": 4.25,  # 17 B / 32 elems
    "TQ2_0": 2.0625,  # 66 B / 256 elems
    "TQ1_0": 1.6875,  # 54 B / 256 elems
    "Q1_0": 1.125,  # 18 B / 128 elems
    # Q2_0 is not in gguf 0.19 yet; this is the nominal 2-bit estimate.
    "Q2_0": 2.2,
    "Q8_0": 8.5,
    "Q6_K": 6.14,
    "Q5_K_M": 5.33,
    "Q5_K": 5.33,
    "Q4_K_M": 4.58,
    "Q4_K": 4.58,
    "Q3_K_M": 3.74,
    "Q3_K": 3.74,
    "Q2_K": 2.96,
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}

_COMMON_BIN_DIRS = (
    Path.home() / ".vipera" / "bin",
    Path.home() / "llama.cpp" / "build" / "bin",
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
)


class OptimizerError(RuntimeError):
    """Raised when an optimization command cannot be planned or executed."""


@dataclass(frozen=True)
class QuantTarget:
    quant: str
    estimated_bytes: int
    target_bpw: float


def normalize_quant(quant: str) -> str:
    key = quant.strip().upper().replace("-", "_")
    aliases = {"Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M", "Q3_K": "Q3_K_M"}
    return aliases.get(key, key)


def _quant_bpw(quant: str) -> float:
    key = normalize_quant(quant)
    if key in _QUANT_BPW:
        return _QUANT_BPW[key]
    if key.startswith("IQ2"):
        return 2.3
    if key.startswith("IQ3"):
        return 3.3
    if key.startswith("IQ4"):
        return 4.3
    if key.startswith("Q6"):
        return 6.14
    if key.startswith("Q5"):
        return 5.33
    if key.startswith("Q4"):
        return 4.58
    if key.startswith("Q3"):
        return 3.74
    if key.startswith("Q2"):
        return 2.96
    return 4.58


def parse_prune_layers(value: str | Sequence[int] | None) -> list[int]:
    if value is None or value == "":
        return []
    parts: Sequence[str | int] = value.split(",") if isinstance(value, str) else value
    layers: list[int] = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        try:
            layer = int(text)
        except ValueError as exc:
            raise OptimizerError(f"invalid layer number: {text}") from exc
        if layer < 0:
            raise OptimizerError(f"layer number must be non-negative: {layer}")
        if layer not in layers:
            layers.append(layer)
    return layers


def estimate_quant_for_target(
    current_size_bytes: int,
    current_quant: str,
    target_size_bytes: int,
) -> QuantTarget:
    """Pick the highest-quality known quant that should fit the target size.

    The estimate scales the current file size by bits-per-weight ratio. It is
    deliberately conservative and only covers common K-quants.
    """
    if current_size_bytes <= 0:
        raise OptimizerError("current model size must be greater than zero")
    if target_size_bytes <= 0:
        raise OptimizerError("target size must be greater than zero")

    current_bpw = _quant_bpw(current_quant)
    if target_size_bytes >= current_size_bytes:
        return QuantTarget(normalize_quant(current_quant), current_size_bytes, current_bpw)

    target_bpw = current_bpw * target_size_bytes / current_size_bytes
    for quant in _QUALITY_ORDER:
        bits = _quant_bpw(quant)
        if bits <= target_bpw:
            return QuantTarget(
                quant,
                int(current_size_bytes * bits / current_bpw),
                target_bpw,
            )

    return QuantTarget(
        "Q2_K",
        int(current_size_bytes * _quant_bpw("Q2_K") / current_bpw),
        target_bpw,
    )


def estimate_size_for_quant(
    current_size_bytes: int,
    current_quant: str,
    target_quant: str,
) -> int:
    if current_size_bytes <= 0:
        raise OptimizerError("current model size must be greater than zero")
    current_bpw = _quant_bpw(current_quant)
    return int(current_size_bytes * _quant_bpw(target_quant) / current_bpw)


def _is_runnable(path: Path) -> bool:
    if not path.is_file():
        return False
    if sys.platform == "win32":
        return path.suffix.lower() in (".exe", ".bat", ".cmd", "")
    return os.access(path, os.X_OK)


def find_llama_tool(tool: str, bin_dir: str | None = None) -> str:
    """Locate a llama.cpp tool by explicit directory, env var, PATH, or defaults."""
    primary = _TOOL_NAMES.get(tool, tool)
    names: list[str] = [primary]
    if tool not in names:
        names.append(tool)

    directories: list[Path] = []
    if bin_dir:
        directories.append(Path(bin_dir).expanduser())
    env_dir = os.environ.get("VIPERA_LLAMA_BIN")
    if env_dir:
        directories.append(Path(env_dir).expanduser())
    directories.extend(_COMMON_BIN_DIRS)

    for directory in directories:
        for name in names:
            for candidate in (directory / name, directory / f"{name}.exe"):
                if _is_runnable(candidate):
                    return str(candidate)

    for name in names:
        found = shutil.which(name)
        if found:
            return found

    raise OptimizerError(
        f"{primary} was not found. Build llama.cpp, add it to PATH, set "
        "VIPERA_LLAMA_BIN, or pass --bin-dir."
    )


def tool_supports_flag(binary: str, flag: str) -> bool:
    try:
        result = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in f"{result.stdout}\n{result.stderr}"


def build_imatrix_command(
    binary: str,
    model: str | Path,
    calibration: str | Path,
    output: str | Path,
    ctx_size: int = 512,
    threads: int | None = None,
) -> list[str]:
    command = [
        binary,
        "-m",
        str(model),
        "-f",
        str(calibration),
        "-o",
        str(output),
        "-c",
        str(ctx_size),
    ]
    if threads:
        command.extend(["-t", str(threads)])
    return command


def build_quantize_command(
    binary: str,
    model: str | Path,
    output: str | Path,
    quant: str,
    *,
    imatrix: str | Path | None = None,
    prune_layers: str | Sequence[int] | None = None,
    prior_weight: int = 1,
    allow_requantize: bool = False,
    supports_prior_weight: bool = False,
    output_tensor_type: str | None = None,
    token_embedding_type: str | None = None,
) -> list[str]:
    command = [binary]
    if allow_requantize:
        command.append("--allow-requantize")
    if imatrix:
        command.extend(["--imatrix", str(imatrix)])

    layers = parse_prune_layers(prune_layers)
    if layers:
        command.extend(["--prune-layers", ",".join(str(layer) for layer in layers)])

    if prior_weight != 1:
        if not supports_prior_weight:
            raise OptimizerError(
                "--prior-weight is not supported by this llama-quantize build; "
                "upgrade llama.cpp or omit the option."
            )
        command.extend(["--prior-weight", str(prior_weight)])

    # Mixed-precision: keep output layer and token embeddings at higher precision
    # while the rest of the model is quantized to `quant`.
    if output_tensor_type:
        command.extend(["--output-tensor-type", normalize_quant(output_tensor_type)])
    if token_embedding_type:
        command.extend(["--token-embedding-type", normalize_quant(token_embedding_type)])

    command.extend([str(model), str(output), normalize_quant(quant)])
    return command


def default_output_path(model: str | Path, quant: str) -> Path:
    path = Path(model).expanduser()
    suffix = normalize_quant(quant).lower()
    return path.parent / f"{path.stem}-vipera-{suffix}.gguf"


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OptimizerError(f"could not find a free output name for {path}")

# planner.py
from dataclasses import dataclass

from vipera.hardware import HardwareInfo


@dataclass
class Plan:
    gpu_layers: int
    cpu_threads: int
    context: int
    kv_cache: str
    flash_attn: bool
    n_batch: int
    offload_kqv: bool
    use_mlock: bool
    note: str
    # Rough host-RAM requirement: CPU-resident weights + CPU-resident KV plus
    # room for compute buffers. 0.0 means "not estimated" (callers skip the
    # preflight check), which keeps hand-built Plans in tests working.
    ram_estimate_gb: float = 0.0
    # KV cache size this plan expects, and how many layers actually keep KV.
    # 0.0 / 0 mean "not estimated" (hand-built Plans in tests stay valid).
    kv_cache_gb: float = 0.0
    kv_layers: int = 0
    # --n-cpu-moe value: the first N layers keep their routed experts on the
    # CPU. -1 means "no MoE plan"; 0 means every expert stays on the GPU.
    # The classic gpu_layers field stays untouched so an in-process fallback
    # still loads a split that fits.
    moe_cpu_layers: int = -1
    # llama.cpp ubatch for the server backend. 0 keeps the historical
    # min(n_batch, 512) behaviour.
    n_ubatch: int = 0
    n_layers: int = 0

    @property
    def requires_server(self) -> bool:
        """True when the plan needs llama-server flags the binding lacks."""
        return self.moe_cpu_layers >= 0


_VALID_PROFILES = ("auto", "cpu", "gpu", "lowmem")

# Room left on the card for the CUDA/driver context and llama.cpp's compute
# buffers. This used to be a flat 1.0 GB (driver) + 0.3 GB (compute), which on a
# 4 GB card gave away four layers: measured on a GTX 1650 with 3.64 GB free, a
# 14B/Q4 model at 8K offloads 14 layers (3.37 GB of weights+KV) and 15 fail, so
# the real reserve is ~0.3 GB. A flat 1.3 GB reserve measured 5.0 t/s where 14
# layers measured 5.96 t/s. Scale it with the card instead: small cards get the
# measured figure, big cards still keep ~1 GB for their larger compute buffers.
_VRAM_RESERVE_BASE_GB = 0.3
_VRAM_RESERVE_FRACTION = 0.07
# Extra reserve when the plan cannot use flash attention. Without FA llama.cpp
# allocates a full KQ matrix per batch instead of streaming it, which on the same
# 4 GB card cuts the offload from 14 layers to 10 (0.9 GB of weights).
_NO_FA_RESERVE_GB = 0.9

# Bytes per element for each supported kv_cache dtype. The values for the
# original five are the rounded ones the VRAM budget was measured against
# (11 layers load, 12 fail, on a 4 GB GTX 1650); the newer dtypes use their
# exact ggml block size so a q5_0 plan does not inherit a q4_0 estimate.
_KV_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0,
    "q5_0": 0.6875,  # 22 B / 32 elems
    "q5_1": 0.75,  # 24 B / 32 elems
    "q4_0": 0.5,
    "q4_1": 0.5,
    "iq4_nl": 0.5,
}

# llama.cpp can only create a context with these cache types when flash
# attention is on. f32/f16/bf16 do not need it, and the CPU flash-attention
# path is the slow one once the attention work grows with context, so it is
# requested only where it is required.
_FA_REQUIRED_KV = {"q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"}


def _kv_layer_count(meta: dict) -> int:
    """Number of layers that carry a KV cache.

    Dense models: every layer. Hybrid models report the attention subset in
    ``kv_layers``; anything missing or nonsensical falls back to the full stack,
    which is the conservative direction for a VRAM budget.
    """
    n_layers = int(meta.get("n_layers") or 0)
    kv_layers = int(meta.get("kv_layers") or 0)
    if kv_layers <= 0 or (n_layers > 0 and kv_layers > n_layers):
        return n_layers
    return kv_layers


def _estimate_kv_cache_gb(meta: dict, ctx: int, kv_cache: str) -> float:
    """Estimate KV-cache size in GB for a given context and cache dtype.

    Prefers exact per-model head counts when the metadata carries them. Falls
    back to a no-GQA worst-case estimate from embedding_length, then to the
    legacy 1.5 GB @ 8k heuristic.
    """
    # Hybrid stacks (KDA/BCSA/Mamba) keep a KV cache only on their attention
    # layers: the recurrent layers carry a fixed-size state instead. Using
    # n_layers here overestimates a 72-layer hybrid by roughly 4x.
    kv_layers = _kv_layer_count(meta)
    bytes_per_elem = _KV_BYTES_PER_ELEM.get(kv_cache, 2.0)

    if kv_layers > 0:
        n_heads = int(meta.get("n_heads") or 0)
        n_kv_heads = int(meta.get("n_kv_heads") or 0) or n_heads
        head_dim = int(meta.get("head_dim") or 0)
        embed = int(meta.get("embedding_length") or 0)

        if head_dim <= 0 and embed > 0 and n_heads > 0:
            head_dim = embed // n_heads

        if n_kv_heads > 0 and head_dim > 0:
            # K + V, kv-bearing layers x kv_heads x head_dim x ctx x bytes
            elems = 2 * kv_layers * n_kv_heads * head_dim * ctx
            return elems * bytes_per_elem / 1024**3

        if embed > 0:
            # No GQA info — assume n_kv_heads == n_heads (worst case)
            elems = 2 * kv_layers * embed * ctx
            return elems * bytes_per_elem / 1024**3

    # Last-resort heuristic (conservative for ~7-8B models), scaled by the KV
    # dtype: 1.5 GB @ 8k assumes f16. Without the scale factor a q4_0 plan
    # inherits the f16 number and under-offloads 1-2 layers.
    return (ctx / 8192.0) * 1.5 * (bytes_per_elem / 2.0)


# --- MoE placement ----------------------------------------------------------
#
# `--n-cpu-moe N` keeps the routed experts of the FIRST N layers on the CPU and
# leaves the rest on the GPU. This is the opposite direction from the Hugging
# Face guide, and was verified on a GTX 1650 by reading VRAM at different N:
# N=24 (all experts CPU) used 972 MiB, N=20 used 1844 MiB (the four experts
# layers 20-23, not 0-3), and N=12 used 3412 MiB (layers 12-23). The help text
# in llama.cpp is right; the guide is not.
#
# The budget separates the non-expert weights (attention, embeddings, shared
# experts, router) from the routed experts, because only the latter move with
# --n-cpu-moe. A shared expert or router must stay on the GPU with attention.
#
# Runtime overhead (CUDA context + compute buffers) measured on the reference
# box with LFM2.5-8B-A1B Q4_K_M, ctx 65536, q4_0 KV:
#   ub 512  -> 321 MiB
#   ub 896  -> 405 MiB
#   ub 1024 -> 432 MiB
# A linear fit (0.20 GB + 0.00022 GB per ubatch token) reproduces all three.
_MOE_UBATCH = 896
_MOE_OVERHEAD_BASE_GB = 0.20
_MOE_OVERHEAD_PER_UB_GB = 0.00022
# Driver/fragmentation headroom. The 0.3 GB dense reserve already covers the
# compute buffer, so charging it again here would drop one expert layer.
_MOE_VRAM_RESERVE_GB = 0.15


def _moe_cpu_layers(
    meta: dict,
    hw: HardwareInfo,
    n_layers: int,
    ctx: int,
    kv_cache: str,
    n_ubatch: int,
) -> int | None:
    """Pick the --n-cpu-moe value that fits, or None when MoE placement is unsafe.

    Greedily keeps the suffix of layers whose routed experts fit in VRAM.
    Because --n-cpu-moe counts from layer 0, the suffix is exactly what stays on
    the GPU. Requires the per-layer expert sizes from ``analyze_gguf``; without
    them the estimate would be a guess that can offload one layer too many and
    fail context creation.
    """
    expert_bytes = int(meta.get("expert_bytes") or 0)
    layer_experts = meta.get("expert_layer_bytes")
    if expert_bytes <= 0 or not isinstance(layer_experts, list) or len(layer_experts) != n_layers:
        return None
    sizes = [int(value or 0) for value in layer_experts]
    if sum(sizes) <= 0:
        return None

    size_bytes = int(meta.get("size_bytes") or 4_000_000_000)
    non_expert_gb = max(0.0, (size_bytes - expert_bytes) / 1024**3)
    kv_gb = _estimate_kv_cache_gb(meta, ctx, kv_cache)
    overhead_gb = _MOE_OVERHEAD_BASE_GB + _MOE_OVERHEAD_PER_UB_GB * n_ubatch
    budget_gb = (
        hw.vram_free_gb - _MOE_VRAM_RESERVE_GB - non_expert_gb - kv_gb - overhead_gb
    )
    if budget_gb <= 0:
        return None

    budget_bytes = int(budget_gb * 1024**3)
    used = 0
    gpu_layers = 0
    for size in reversed(sizes):
        if used + size > budget_bytes:
            break
        used += size
        gpu_layers += 1

    # At least one layer that actually carries routed experts must stay on the
    # GPU. "All experts on CPU" measured 40.8 t/s vs 56.7 for the MoE split on
    # the reference card, so the classic split is the better fallback.
    if not any(size > 0 for size in sizes[n_layers - gpu_layers :]):
        return None
    return n_layers - gpu_layers


def plan(
    meta: dict,
    hw: HardwareInfo,
    profile: str = "auto",
    ctx: int = 8192,
    kv_cache: str = "q8_0",
    n_batch: int = 512,
    n_ubatch: int = 0,
) -> Plan:
    if profile not in _VALID_PROFILES:
        raise ValueError(f"unknown profile '{profile}' (valid: {', '.join(_VALID_PROFILES)})")

    n_layers = int(meta.get("n_layers") or 32)
    kv_layers = _kv_layer_count(meta) or n_layers
    size_bytes = int(meta.get("size_bytes") or 4_000_000_000)

    # One thread per physical core. `cores - 1` used to leave a core idle: on a
    # 4C/8T box, threads=4 measured +19-21% decode over threads=3 (median-3,
    # both CPU-only and with 12 GPU layers), while threads=8 lost throughput to
    # SMT contention.
    cpu_threads = max(1, hw.cpu_cores)
    has_gpu = getattr(hw, "vram_total_gb", 0) > 0

    model_size_gb = size_bytes / 1024**3

    # Low-memory profile: never offload, small ctx, small batch.
    if profile == "lowmem":
        return Plan(
            gpu_layers=0,
            cpu_threads=cpu_threads,
            context=min(ctx, 2048),
            kv_cache="f16",
            flash_attn=False,
            n_batch=min(n_batch, 64),
            offload_kqv=False,
            use_mlock=False,
            note="CPU ONLY (Low Memory Profile)",
            kv_layers=kv_layers,
        )

    want_gpu = has_gpu and profile != "cpu"
    # Flash attention is required by quantized KV, and on the GPU it is also the
    # fast path. Decided before the VRAM budget because it changes how much
    # scratch space llama.cpp needs on the card.
    flash_attn_on_gpu = kv_cache in _FA_REQUIRED_KV or hw.vram_total_gb > 4
    reserve_gb = max(_VRAM_RESERVE_BASE_GB, min(1.0, _VRAM_RESERVE_FRACTION * hw.vram_total_gb))
    if want_gpu and not flash_attn_on_gpu:
        reserve_gb += _NO_FA_RESERVE_GB
    safe_vram = max(0.0, hw.vram_free_gb - reserve_gb)

    # llama.cpp allocates the KV cache per layer: only the layers offloaded to
    # the GPU take their slice of KV with them, the rest stays in system RAM.
    # Subtracting the FULL cache from VRAM before offloading anything made a
    # 14B/Q4 model on a 4 GB card look like it had 0.2 GB free (0 layers) even
    # though ~9 layers actually fit. Budget both weights and KV per layer.
    kv_cache_gb = _estimate_kv_cache_gb(meta, ctx, kv_cache)
    # Charge each KV-bearing layer, not each layer: on a hybrid stack most
    # layers hold a fixed recurrent state and never pay for context.
    per_layer_kv_gb = kv_cache_gb / kv_layers if kv_layers else 0.0

    # Charge every layer the mean weight size (size_bytes / n_layers). An
    # earlier version preferred meta["layer_bytes"] (the mean size of one
    # repeating blk.N block, which excludes token_embd/output.weight) on the
    # theory that those tensors never follow a layer to the GPU. Measured, that
    # is optimistic: llama.cpp's per-layer CUDA compute buffer grows with the
    # number of offloaded layers, and charging the embedding/output bytes to
    # each layer happens to cover it.
    #   Q4_K_M 9.00 GB, 8K q8_0: layer_bytes predicts 10+ layers too many (the
    #                            budget below lands on 13); measured ceiling 14.
    #                            With a 0.28 GB reserve it predicted 14 exactly,
    #                            so the base reserve keeps a one-layer margin.
    #   Q3_K_M 6.82 GB, 8K q8_0: layer_bytes predicts 20 against a measured 17.
    # Over-estimating is the expensive direction: a failed context creation
    # leaves its VRAM allocated, so the ladder can end up CPU-only.
    layer_gb = model_size_gb / n_layers if n_layers else 0.4
    layer_total_gb = layer_gb + per_layer_kv_gb
    gpu_layers = 0

    # profile=cpu forces no offload even when a GPU is present.
    if want_gpu and safe_vram > 0 and layer_total_gb > 0:
        gpu_layers = min(n_layers, int(safe_vram / layer_total_gb))

    # The classic split offloads whole layers, so a layer's experts travel with
    # it. MoE models do better by keeping every attention block on the GPU and
    # moving only routed experts, which is a flag the Python binding does not
    # expose. Only propose that plan when at least one classic layer fits (the
    # in-process fallback needs a valid gpu_layers) and at least one expert
    # layer fits.
    moe_cpu_layers = -1
    moe_ubatch = 0
    if want_gpu and gpu_layers > 0:
        candidate_ubatch = n_ubatch if n_ubatch > 0 else _MOE_UBATCH
        candidate = _moe_cpu_layers(meta, hw, n_layers, ctx, kv_cache, candidate_ubatch)
        if candidate is not None:
            moe_cpu_layers = candidate
            moe_ubatch = candidate_ubatch

    if gpu_layers == 0:
        offload_kqv = False
        n_batch = min(n_batch, 128)
        # An explicit quantized KV request survives the CPU-only path:
        # llama.cpp runs flash attention on the CPU backend too, and forcing
        # f16 here would make a 64K-context KV for a 14B model ~10 GiB, which
        # does not fit beside this gateway on a 15 GiB / no-swap box.
        # flash_attn is required for KV quantization, so enable it only for the
        # dtypes that need it (bf16/f16/f32 stay on the fast default path).
        flash_attn = kv_cache in _FA_REQUIRED_KV

        if profile == "gpu":
            note = "CPU ONLY (GPU profile requested, no usable GPU)"
        elif hw.ram_total_gb < 8:
            ctx = min(ctx, 2048)
            note = "CPU ONLY (Low RAM Safe Mode)"
        else:
            note = "CPU ONLY"
    else:
        # Quantized KV requires flash attention, regardless of VRAM size.
        flash_attn = flash_attn_on_gpu
        offload_kqv = True
        n_batch = min(n_batch, 512)
        note = "Full GPU" if gpu_layers == n_layers else f"Hybrid: {gpu_layers}/{n_layers} layers"
        if moe_cpu_layers >= 0:
            note = f"MoE: {n_layers - moe_cpu_layers}/{n_layers} expert layers on GPU"

    # mlock pins weights in RAM, so it is only safe when the machine can hold
    # the host-side weights AND the host-side KV cache with headroom. Checking
    # against model size alone ignores a 5 GB KV cache (Q8_0 at 64K), which
    # turns an mlock into an OOM on a 15 GB / no-swap box.
    if moe_cpu_layers >= 0:
        # A MoE plan offloads every layer for its dense tensors, so all KV
        # lives on the GPU; only the routed experts of the CPU prefix remain
        # in system RAM.
        layer_experts = meta.get("expert_layer_bytes") or []
        cpu_weights_gb = sum(int(value) for value in layer_experts[:moe_cpu_layers]) / 1024**3
        cpu_kv_gb = 0.0
    else:
        cpu_weights_gb = max(0.0, model_size_gb - layer_gb * gpu_layers)
        cpu_kv_gb = per_layer_kv_gb * max(0, kv_layers - gpu_layers)
    host_headroom_gb = hw.ram_free_gb - cpu_weights_gb - cpu_kv_gb
    # 3 GB of headroom leaves room for llama.cpp compute buffers; pinning with
    # less has OOM-killed this class of machine mid-load (no swap).
    use_mlock = host_headroom_gb > 3.0
    # llama.cpp needs more than weights+KV for compute buffers; 2 GB is a
    # conservative placeholder, calibrated against a 14B/Q8_0/64K load.
    ram_estimate_gb = cpu_weights_gb + cpu_kv_gb + 2.0

    return Plan(
        gpu_layers=gpu_layers,
        cpu_threads=cpu_threads,
        context=ctx,
        kv_cache=kv_cache,
        flash_attn=flash_attn,
        n_batch=n_batch,
        offload_kqv=offload_kqv,
        use_mlock=use_mlock,
        note=note,
        ram_estimate_gb=ram_estimate_gb,
        kv_cache_gb=kv_cache_gb,
        kv_layers=kv_layers,
        moe_cpu_layers=moe_cpu_layers,
        n_ubatch=moe_ubatch,
        n_layers=n_layers,
    )

# registry.py
import json
import os
from datetime import UTC, datetime

from vipera.config import REGISTRY_FILE, ensure_dirs, quarantine_corrupt


def _load() -> dict:
    ensure_dirs()
    if REGISTRY_FILE.exists():
        try:
            data = json.loads(REGISTRY_FILE.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            quarantine_corrupt(REGISTRY_FILE, exc)
        else:
            if isinstance(data, dict):
                return data
            quarantine_corrupt(REGISTRY_FILE, ValueError("not a JSON object"))
    return {}


def _save(data: dict) -> None:
    ensure_dirs()
    tmp = REGISTRY_FILE.with_suffix(REGISTRY_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, REGISTRY_FILE)


def add_model(name: str, path, meta: dict | None = None) -> dict:
    data = _load()
    data[name] = {
        "name": name,
        "path": str(path),
        "added": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "meta": meta or {},
    }
    _save(data)
    return data[name]


def list_models() -> list[dict]:
    return list(_load().values())


def get_model(name: str) -> dict | None:
    return _load().get(name)


def remove_model(name: str) -> bool:
    data = _load()
    if name in data:
        del data[name]
        _save(data)
        return True
    return False

# server.py
import asyncio
import contextlib
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from vipera import __version__

_MAX_TOKENS = 32_768
_MAX_MESSAGES = 200
_MAX_PROMPT_CHARS = 500_000

_SENTINEL = object()


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # Assistant tool-call turns legitimately have no text content.
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _require_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls:
            raise ValueError("content is required unless the message carries tool_calls")
        return self


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(..., min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    top_k: int | None = Field(None, ge=0)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    seed: int | None = Field(None, ge=0)
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(None, ge=0, le=20)
    response_format: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


def build_app(engine, model_name: str) -> FastAPI:
    app = FastAPI(title="Vipera Gateway", version=__version__)
    lock = asyncio.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "vipera"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        if not req.messages:
            raise HTTPException(400, "messages must not be empty")
        if req.model and req.model != model_name:
            raise HTTPException(404, f"model '{req.model}' not loaded (active: '{model_name}')")
        if len(req.messages) > _MAX_MESSAGES:
            raise HTTPException(400, f"too many messages (max {_MAX_MESSAGES})")
        if sum(len(m.content or "") for m in req.messages) > _MAX_PROMPT_CHARS:
            raise HTTPException(400, "prompt too large")
        if not 0 < req.max_tokens <= _MAX_TOKENS:
            raise HTTPException(400, f"max_tokens must be in [1, {_MAX_TOKENS}]")
        if not 0.0 <= req.temperature <= 2.0:
            raise HTTPException(400, "temperature must be in [0, 2]")
        if not 0.0 < req.top_p <= 1.0:
            raise HTTPException(400, "top_p must be in (0, 1]")

        # exclude_none keeps unset optional knobs out of the llama.cpp call so
        # the backend's own defaults apply (e.g. seed=None is not seed=0).
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        kwargs = req.model_dump(
            exclude={"model", "messages", "stream", "stream_options"},
            exclude_none=True,
        )
        # The in-process engine drops this (llama-cpp-python 0.3.x does not
        # report usage), but llama-server returns real usage when asked, which
        # is what turns the trailing choices:[] frame below into data.
        if req.stream and req.stream_options and req.stream_options.include_usage:
            kwargs["stream_options"] = {"include_usage": True}

        if req.stream:

            async def _stream():
                # The engine's stream is a SYNC generator. Iterating it inline
                # would block the event loop for the whole generation, so pull
                # each chunk in a worker thread instead.
                #
                # The lock is acquired explicitly (not via `async with`) so a
                # client disconnect cannot release it while a worker thread is
                # still inside next(): cancellation would otherwise let the
                # next request enter the engine concurrently with generation
                # that is still running.
                await lock.acquire()
                released = False
                iterator = None
                usage = None
                try:
                    try:
                        iterator = engine.chat(messages, stream=True, **kwargs)
                        while True:
                            task = asyncio.create_task(asyncio.to_thread(next, iterator, _SENTINEL))
                            try:
                                # shield: cancelling this request must not
                                # cancel the task that wraps the worker thread.
                                chunk = await asyncio.shield(task)
                            except asyncio.CancelledError:
                                # The worker is still inside next(); closing the
                                # generator now would raise "already executing"
                                # and release the lock too early. Wait for the
                                # thread, then close and release.
                                def _finish(_t, it=iterator):
                                    with contextlib.suppress(Exception):
                                        it.close()
                                    lock.release()

                                task.add_done_callback(_finish)
                                released = True
                                iterator = None
                                raise
                            if chunk is _SENTINEL:
                                break
                            if isinstance(chunk, dict) and chunk.get("usage"):
                                usage = chunk["usage"]
                            payload = json.dumps(chunk, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # engine.chat() itself may raise before the first
                        # chunk; report that as an SSE error event instead of
                        # aborting the response body mid-stream.
                        err = {"error": {"message": str(exc)}}
                        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    # OpenAI emits a trailing choices:[] frame when the client
                    # asks for usage. Only possible when the backend reported
                    # any; llama-cpp-python 0.3.x does not, so it is omitted.
                    if (
                        req.stream_options is not None
                        and req.stream_options.include_usage
                        and usage is not None
                    ):
                        payload = json.dumps({"choices": [], "usage": usage}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    if iterator is not None:
                        with contextlib.suppress(Exception):
                            iterator.close()
                    if not released:
                        lock.release()

            return StreamingResponse(_stream(), media_type="text/event-stream")

        # Same cancellation rule as the streaming path: do not hand the lock to
        # another request until the in-flight worker thread has returned.
        await lock.acquire()
        released = False
        try:
            task = asyncio.create_task(asyncio.to_thread(engine.chat, messages, False, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                task.add_done_callback(lambda _t: lock.release())
                released = True
                raise
        finally:
            if not released:
                lock.release()

    return app

# server_backend.py
"""llama-server backend.

Runs the `llama-server` binary as a child process and speaks the OpenAI HTTP
API to it. The public surface is the same as `vipera.engine.Engine`:
`.plan`, `.chat()`, `.tokenize()`, `.close()`.

Why this exists: the Python binding bundles its own llama.cpp and exposes only
part of it. A locally built `llama-server` can be newer (better CPU kernels,
new flags) and is the only way to use builds vipera does not ship. Selecting it
is opt-in through the `backend` config key; `inprocess` stays the default until
this path has been measured on real hardware.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

from vipera import config
from vipera.engine import _CHAT_PARAMS, _DEFAULT_SAMPLING, check_host_ram, load_attempts
from vipera.optimizer import OptimizerError, find_llama_tool
from vipera.planner import Plan

# Loading a 14B from a cold page cache on CPU takes minutes; the HTTP server
# binds first and answers 503 until the model is ready, so poll for a while.
_START_TIMEOUT_S = 900
_HEALTH_INTERVAL_S = 0.5
_STOP_TIMEOUT_S = 10.0

# Printed by llama.cpp builds without a GPU backend. The child keeps running
# but ignores --gpu-layers, so the plan must be corrected instead of reporting
# layers that are not on the GPU.
_GPU_IGNORED_MARKER = "--gpu-layers option will be ignored"

# Chat parameters llama-server understands. It ignores unknown JSON fields, but
# sending a curated set keeps a typo from becoming a 400 in the child.
_SERVER_CHAT_PARAMS = _CHAT_PARAMS - {"functions", "function_call", "grammar"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _supports_load_mode(binary: str) -> bool:
    """Whether this llama-server knows `--load-mode`.

    There is no `--mlock` on the server (that is a llama-cli flag); the mlock
    equivalent is a `--load-mode` value, and builds that predate it accept the
    flag only to exit with "invalid argument" after the weights are mapped.
    One `--help` run per spawn is cheaper than a failed child, and the caller
    may skip the probe entirely by passing the answer in.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--load-mode" in f"{probe.stdout}{probe.stderr}"


def supports_n_cpu_moe(binary: str) -> bool:
    """Whether this llama-server knows `-ncmoe` / `--n-cpu-moe`.

    A build without the flag exits after allocating the model, so the planner
    must know before choosing the server backend. Some builds print the short
    form, others only the long one; accept either.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = f"{probe.stdout}{probe.stderr}"
    return "-ncmoe" in help_text or "--n-cpu-moe" in help_text


def supports_spec_type(binary: str) -> bool:
    """Whether this llama-server knows `--spec-type` (the n-gram drafters).

    Older builds have neither the flag nor the n-gram drafters, and passing an
    unknown option kills the child, so the flag is only added after a probe.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--spec-type" in f"{probe.stdout}{probe.stderr}"


def _supports_fit(binary: str) -> bool:
    """Whether this llama-server knows `-fit` / `--fit`.

    Recent builds default `--fit on`, which adjusts unset arguments to leave
    `--fit-target` MiB free (1024 by default). That would override the
    planner's ngl/ncmoe placement, so Vipera turns it off — but only on builds
    that understand the flag.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = f"{probe.stdout}{probe.stderr}"
    return "-fit" in help_text or "--fit" in help_text


def server_command(
    binary: str,
    model_path: str,
    plan: Plan,
    port: int,
    *,
    load_mode_supported: bool | None = None,
    fit_supported: bool | None = None,
    spec_type: str | None = None,
    spec_type_supported: bool | None = None,
) -> list[str]:
    """Translate a Plan into llama-server flags.

    `--jinja` is required for tool calling and for models whose chat template
    is not one of the hard-coded ones; it is harmless for plain chat.

    `load_mode_supported`, `fit_supported` and `spec_type_supported` skip the
    `--help` probes; `None` probes the binary.

    `spec_type` is llama-server's own speculative decoding. Measured on the
    reference 4 GB card (Hermes-4-14B Q3_K_M, 17/40 layers): `ngram-map-k`
    decodes 8.54 t/s against 6.23 t/s with it off (+37%) when the prompt is a
    file the model then rewrites, and 6.33 against 6.40 (-1%, i.e. noise) on
    fresh prose — the n-gram drafter only pays where the output repeats the
    input, and costs a hash lookup where it does not.
    """
    # A MoE plan keeps every layer on the GPU for its dense tensors and moves
    # only routed experts with -ncmoe. That means -ngl is the full stack, not
    # the classic gpu_layers value (which exists for the in-process fallback).
    gpu_layers = (
        plan.n_layers if plan.moe_cpu_layers >= 0 and plan.n_layers > 0 else plan.gpu_layers
    )
    # -ub is the prefill lever (measured: 896 was the best all-round point on
    # the reference 4 GB card); -b must be at least as large as -ub.
    ubatch = plan.n_ubatch or min(plan.n_batch, 512)
    batch = max(plan.n_batch, ubatch)
    command = [
        binary,
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(plan.context),
        "-ngl",
        str(gpu_layers),
        "-t",
        str(plan.cpu_threads),
        "-b",
        str(batch),
        "-ub",
        str(ubatch),
        "-fa",
        "on" if plan.flash_attn else "off",
        "--jinja",
    ]
    if fit_supported is None:
        fit_supported = _supports_fit(binary)
    if fit_supported:
        # The planner owns placement; auto-fit would shrink it to leave 1 GiB
        # free and silently invalidate both the banner and the A/B numbers.
        command.extend(["-fit", "off"])
    if plan.moe_cpu_layers >= 0:
        command.extend(["-ncmoe", str(plan.moe_cpu_layers)])
    if spec_type and spec_type != "none":
        supported = (
            supports_spec_type(binary) if spec_type_supported is None else spec_type_supported
        )
        if supported:
            command.extend(["--spec-type", spec_type])
    if plan.kv_cache != "f16":
        command.extend(["-ctk", plan.kv_cache, "-ctv", plan.kv_cache])
    if plan.use_mlock:
        supported = (
            _supports_load_mode(binary) if load_mode_supported is None else load_mode_supported
        )
        if supported:
            command.extend(["-lm", "mlock"])
    return command


class ServerEngine:
    """Engine-compatible client for a `llama-server` child process."""

    def __init__(self, plan: Plan, model_path: str):
        check_host_ram(plan)

        self.plan = plan
        self._model_path = model_path
        self._closed = False
        self._proc: subprocess.Popen | None = None
        self._log: BinaryIO | None = None
        self._port: int | None = None
        self._log_path: Path | None = None

        try:
            self._binary = find_llama_tool("server")
        except OptimizerError as exc:
            raise RuntimeError(
                f"{exc} The 'server' backend needs the llama-server binary; "
                "the Python binding does not ship one."
            ) from exc

        # Probe once per engine: the load ladder may spawn the child several
        # times, and each extra --help run is wasted startup time.
        self._fit_supported = _supports_fit(self._binary)
        self._spec_type = config.load_config().get("spec_type", "none")
        self._spec_type_supported = supports_spec_type(self._binary)

        # Same staged fallback as the in-process engine: walk the offload down
        # before giving up the placement entirely, because CPU-only is the
        # slowest outcome and a one-layer-too-optimistic plan pays a few
        # seconds for the retry instead of the whole run.
        last_exc: Exception | None = None
        for i, (candidate, label) in enumerate(load_attempts(plan)):
            try:
                self._start(candidate)
                if i > 0:
                    print(
                        f"\n[!] Warning: llama-server failed to start with the planned "
                        f"split; serving {label}."
                    )
                break
            except RuntimeError as exc:
                last_exc = exc
        else:
            raise RuntimeError(
                f"llama-server failed to start on GPU and CPU: {last_exc}"
            ) from last_exc

        atexit.register(self.close)

    # -- lifecycle ---------------------------------------------------------

    def _start(self, plan: Plan) -> None:
        port = _free_port()
        log_path = config.LOGS_DIR / f"llama-server-{port}.log"
        command = server_command(
            self._binary,
            self._model_path,
            plan,
            port,
            fit_supported=self._fit_supported,
            spec_type=self._spec_type,
            spec_type_supported=self._spec_type_supported,
        )
        # The handle is handed to Popen and must outlive this function.
        handle = open(log_path, "wb")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"could not run {self._binary}: {exc}") from exc

        self._proc, self._log, self._port, self._log_path = proc, handle, port, log_path
        try:
            self._wait_ready(proc, port)
        except RuntimeError:
            self._stop_process()
            raise
        self.plan = self._adopt_placement(plan)

    def _wait_ready(self, proc: subprocess.Popen, port: int) -> None:
        deadline = time.monotonic() + _START_TIMEOUT_S
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if (rc := proc.poll()) is not None:
                raise RuntimeError(f"llama-server exited with code {rc}{self._tail()}")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(_HEALTH_INTERVAL_S)
        raise RuntimeError(
            f"llama-server did not become healthy within {_START_TIMEOUT_S}s{self._tail()}"
        )

    def _log_text(self, limit: int = 200_000) -> str:
        if self._log_path is None:
            return ""
        with contextlib.suppress(OSError):
            return self._log_path.read_text(errors="replace")[-limit:]
        return ""

    def _tail(self) -> str:
        text = self._log_text(1500)
        if text.strip():
            return f"\n--- llama-server log ({self._log_path}) ---\n{text}"
        return ""

    def _adopt_placement(self, plan: Plan) -> Plan:
        """Report what the child actually did, not what was asked.

        A CPU-only llama.cpp build starts normally and silently drops
        --gpu-layers, which would otherwise make the banner and `bench` claim
        layers that never left system RAM.
        """
        if plan.gpu_layers and _GPU_IGNORED_MARKER in self._log_text():
            print(
                "[!] Warning: llama-server was built without GPU support and "
                f"ignored --gpu-layers {plan.gpu_layers}; serving CPU-only."
            )
            return replace(
                plan,
                gpu_layers=0,
                offload_kqv=False,
                moe_cpu_layers=-1,
                n_ubatch=0,
                note="CPU ONLY (llama-server ignored --gpu-layers)",
            )
        return plan

    def _stop_process(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_process()

    @property
    def log_path(self) -> Path | None:
        """Where the child's stdout/stderr went; useful in bug reports."""
        return self._log_path

    # -- inference ---------------------------------------------------------

    @property
    def _base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("llama-server is not running")
        return f"http://127.0.0.1:{self._port}"

    def _post(self, path: str, body: dict):
        request = urllib.request.Request(
            self._base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=None)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"llama-server {path} returned {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llama-server is unreachable: {exc.reason}") from None

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any):
        params = {k: v for k, v in kw.items() if k in _SERVER_CHAT_PARAMS and v is not None}
        for key, default in _DEFAULT_SAMPLING.items():
            params.setdefault(key, default)
        body: dict[str, Any] = {"messages": messages, "stream": stream, **params}
        # Not part of the in-process engine's surface, but llama-server uses it
        # to put the exact token usage in the final SSE chunk; bench relies on
        # that for an exact generation count.
        if kw.get("stream_options") is not None:
            body["stream_options"] = kw["stream_options"]

        if not stream:
            with self._post("/v1/chat/completions", body) as resp:
                return json.loads(resp.read())

        # A streaming generator keeps the HTTP response open across yields, so
        # the caller can consume it exactly like llama.cpp's own generator.
        response = self._post("/v1/chat/completions", body)
        return self._iter_sse(response)

    @staticmethod
    def _iter_sse(response) -> Iterator[dict]:
        try:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                with contextlib.suppress(json.JSONDecodeError):
                    yield json.loads(payload)
        finally:
            with contextlib.suppress(OSError):
                response.close()

    def tokenize(self, text: str | bytes) -> list[int]:
        content = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
        with self._post("/tokenize", {"content": content}) as resp:
            return [int(tok) for tok in json.loads(resp.read()).get("tokens") or []]

# ui.py
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

C_PRIMARY = "#5B7C99"
C_ACCENT = "#8FB4CC"
C_HIGHLIGHT = "#B8C7D3"
C_TEXT = "bold #E4E8EC"
C_DIM = "#6B7A87"
C_BORDER = "#3E5568"

C_GREEN = "#7BA05B"
C_RED = "#C25B5B"
C_YELLOW = "#D4A574"
C_SUCCESS = C_GREEN
C_ERROR = C_RED
C_WARN = C_YELLOW


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


def print_banner(host: str = "localhost"):
    console.print()
    heartbeat_banner()

    rule = Text()
    rule.append("   " + "▬" * 74 + "\n", style=C_BORDER)
    rule.append("   ◆ ", style=C_ACCENT)
    rule.append("NODE   ", style=C_DIM)
    rule.append(host, style=C_TEXT)
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
    """Wrap text without destroying the indentation of code blocks."""
    out = []
    for line in text.split("\n"):
        if not line:
            out.append(" " * width)
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
        for w in wrapper.wrap(stripped) or [indent]:
            out.append(f"{w:<{width}}")
    return "\n".join(out)


_LIVE_UPDATE_INTERVAL = 0.03


def typewriter_chat_stream(stream_generator, title_text: str = " AI "):
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
    stat.append(f"{chunk_count} tokens  ·  {tps:.1f} tok/s  ·  {dt:.2f}s\n", style=C_DIM)
    console.print(stat)
    return full_response


def plain_chat_stream(stream_generator, title_text: str = " AI "):
    """Stream a chat reply as fast as the model produces tokens."""
    full = []
    start_time = time.time()
    chunk_count = 0

    with Live(console=console, refresh_per_second=20, transient=True) as live:
        for chunk in stream_generator:
            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
            except (KeyError, IndexError, TypeError):
                continue
            if not delta:
                continue
            chunk_count += 1
            full.append(delta)
            live.update(
                Panel(
                    Text("".join(full), style=C_TEXT),
                    title=Text(title_text, style=C_ACCENT),
                    title_align="left",
                    border_style=C_ACCENT,
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )

    text = "".join(full)
    console.print(
        Panel(
            Text(text, style=C_TEXT),
            title=Text(title_text, style=C_ACCENT),
            title_align="left",
            border_style=C_ACCENT,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    dt = time.time() - start_time
    tps = chunk_count / dt if dt > 0 else 0
    stat = Text("    ")
    stat.append("└─ ", style=C_BORDER)
    stat.append(f"{chunk_count} tokens  ·  {tps:.1f} tok/s  ·  {dt:.2f}s\n", style=C_DIM)
    console.print(stat)
    return text


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
```

---

# Đề xuất v2: 5 layer tối ưu cho Vipera

> ✅ **ĐÃ TRIỂN KHAI** (2026-10-21). Phần dưới là kế hoạch viết ngày 2026-10-20, giữ nguyên để đối chiếu; nó **không** còn mô tả hiện trạng. Hiện trạng trong `src/`: `_KV_BYTES_PER_ELEM` / `_KV_CACHE_TYPES` / `_KV_TYPES` đều **9/9** dtype llama.cpp hỗ trợ; fallback `_estimate_kv_cache_gb` đã scale theo dtype; backend `server` + `Engine.tokenize()` đã có (mặc định vẫn `inprocess`); `bench_cmd` dùng `engine.tokenize()`. Số đo nằm ở mục **Kết quả đo — 2026-10-21** cuối file.

## Kết quả đối chiếu ý kiến review

| # | Ý kiến review | Kết luận |
|---|---|---|
| 1 | L2 nói planner "chỉ tính được f16/q8_0/q4_0" | **Đúng — tôi sai, sửa.** `planner._KV_BYTES_PER_ELEM` (`planner.py:37`) có **5** loại: f32/f16/q8_0/q4_0/q4_1; `engine._KV_TYPES` (`engine.py:9`) và `config._KV_CACHE_TYPES` (`config.py:29`) khớp đúng 5 loại |
| 2 | 60 quant type của ik fork áp cho **weight**, không áp cho KV | **Đúng.** llama.cpp chỉ nhận 9 KV dtype (`common/arg.cpp:304-313`: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1). Planner đã phủ 5/9 → mở thêm tối đa 4 loại, lợi ích nhỏ. **L2 phải viết lại** |
| 3 | "−15% weight Hermes-4" là giả thuyết | **Đúng.** Chưa đo trên model này. Chuyển thành **tiêu chí nghiệm thu**: PPL/KL measured trên chính Hermes-4 trước khi chấp nhận con số |
| 4 | "vipera bench đã có sẵn" cho L5 | **Đúng.** `bench run` chỉ đo speed (TTFT/prefill/gen). Perplexity + KL per-tensor là **code mới** = phần lớn nhất của L5 |

> **Cập nhật 2026-10-21 — dòng #1 và #2 đã lỗi thời, đọc kèm ghi chú này.**
> Con số "5 dtype" là của bản code ngày 2026-10-20. Bản hiện tại phủ **9/9** dtype
> llama.cpp (`_KV_BYTES_PER_ELEM`, `_KV_CACHE_TYPES`, `_KV_TYPES` giống nhau, đều 9
> entry) và fallback `_estimate_kv_cache_gb` đã scale theo dtype. Vì vậy kết luận
> "mở thêm tối đa 4 loại / L2 phải viết lại" **không còn đúng**: phần L2 đã xong,
> việc còn lại chỉ là giữ test. Chỗ nào trong văn bản dưới còn nói "cần code" thì
> hiểu là *đã làm xong* — kiểm chứng bằng test, không bằng đoạn văn này.
| 5 | `bench_cmd._run_bench` gọi thẳng `engine.llm.tokenize` | **Đúng** (`bench_cmd.py:35`) → cần `Engine.tokenize()` wrapper cho backend llama-server |
| 6 | `serve_cmd.start` đọc `engine.plan.gpu_layers/kv_cache_gb/kv_layers` | **Không đúng.** `serve_cmd.py:152` tự gọi `plan(m["meta"], hw, ...)` từ planner — **không phụ thuộc backend**. Nơi duy nhất đọc `engine.plan` là **`bench_cmd.py:175,183`** (dự phòng `or max(engine.plan.gpu_layers, 1)`) → fix ở bench_cmd; panel serve chạy nguyên |
| 7 | server.py lock + `CancelledError` giả định sync generator | **Chấp nhận** — đọc tĩnh không chứng minh được; bắt buộc có test cancel-giữa-stream (test plan ở dưới) |
| 8 | "Repo chưa thấy tests/" | **Không đúng.** `tests/` tồn tại với **13 file** (test_planner.py, test_engine.py, test_serve_cmd.py, …), `pytest -q` = **167 passed**. (Có thể nhìn từ `fullcode.md` — file này chỉ dump `src/` theo thiết kế.) → **mở rộng** bộ test có sẵn, không tạo mới từ đầu |
| 9 | `vipera model quantize` đã ghi `*-vipera-<quant>.gguf` | **Đúng** (`optimizer.py:286`, `default_output_path`) → L1 dùng lại pipeline này để có rollback |
| 10 | Fallback cuối của planner bỏ qua `kv_cache` | **Đúng** (`planner.py:92-93`) → fix thuộc scope L2, kèm test |

## Bug nhận diện (thuộc scope L2)

`planner.py:92-93` — fallback cuối cùng không nhân theo bytes/elem:

```python
# hiện tại: q4_0 vẫn trả 1.5GB @ 8k (đúng ra ~0.375GB) → under-offload 1-2 layer
return (ctx / 8192.0) * 1.5
# đề xuất (bytes_per_elem đã có từ planner.py:71):
return (ctx / 8192.0) * 1.5 * (bytes_per_elem / 2.0)
```

## Các layer sau chỉnh

### L5 — Đo lường (tách L5a/L5b)
- **L5a (làm trước tiên)**: baseline speed + **perplexity trên model gốc** (trước khi đụng weight — không có mốc này thì L1 không chứng minh được gì). Đo bằng `llama-perplexity` (tools sẵn trong `~/dev/llama.cpp`, build ~1 phút) trên eval set nhỏ ~50 đoạn text; so **PPL tương đối giữa 2 bản cùng model** là đủ nhạy. Kết quả ghi thành script lặp lại được trong `tools/`, không đo tay.
- **L5b**: hạ tầng đo lặp lại được cho từng layer + KL per-tensor so bản f16 reference (code mới — phần lớn nhất của L5).

### L1 — File model (giả thuyết, có tiêu chí nghiệm thu)
- **L1a**: so PPL/byte giữa các bản quant tải lại được của Hermes-4 (Q4_K_M vs Q5_K_M vs Q4_K_S vs IQ4_XS…) bằng pipeline sẵn có (`vipera model quantize`, output `*-vipera-<quant>.gguf`, giữ file gốc → rollback = xóa file vipera). **Không viết pipeline mới.**
- **L1b (chỉ nếu L5a chứng minh mất chất lượng đáng kể)**: per-tensor search — nếu làm thì chọn `advanced-gguf-quantizer` (đang hoạt động 2026-08, per-tensor, có report); **bỏ** `llama-gguf-optimize` (stale 2025-01, deps nặng: optuna/h5py/astropy/datasets). YAGNI cho tới lúc đó.
- Con số −15% chỉ được đưa vào README khi **đo được** trên máy này.

### L3 — Backend llama-server (flag opt-in, độc lập L1)
- Thêm `"backend": "inprocess"` vào `DEFAULT_CONFIG` (`config.py:14`) + validate với `_BACKENDS = {"inprocess", "server"}`; **flip default sau khi L1+L2 validated** → có đường lui bằng config, không cần `git revert`.
- **Gap đã xác minh cần code**: (a) `Engine.tokenize()` wrapper (`bench_cmd.py:35`); (b) `bench_cmd.py:175,183` đọc `engine.plan` — backend server vẫn phải giữ plan trên Engine (plan chính là tập cờ sẽ truyền cho llama-server); (c) test cancel-giữa-stream cho `server.py` lock.
- **Không cần**: panel của `serve_cmd` (tự tính plan từ metadata).

### L2 — Planner (viết lại nội dung)
Không phải "thêm 60 quant type". Nội dung thật:
1. **Fix bug fallback** ở trên + test trong `tests/test_planner.py` (case GGUF thiếu `attention.head_count_kv`).
2. Mở KV dtype còn thiếu (bf16/iq4_nl/q5_0/q5_1) — chỉ 4 loại, lợi ích vừa phải, làm sau fix bug.
3. Ràng buộc theo backend đích (KV dtype llama-server hỗ trợ khác binding) — phụ thuộc L3 nên xếp sau.

### L4 — Speculative (tách L4a/L4b, cả hai đều không làm trong vòng này)
- **L4a — MTP**: llama.cpp **có** hỗ trợ (`llama-model.cpp:3087` `mtp_on_hybrid_qwen`, GLM5-Next, …). **Nhưng đã đo trên file thật**: Hermes-4 (443 tensors) và qwen15b (339 tensors) **không có tensor MTP/NextN/Medusa** → L4a chỉ khả thi nếu tải model có MTP head (Qwen3-Next, GLM-4.6). Điều kiện: đổi model.
- **L4b — DFlash**: bị chặn bởi VRAM (cần ≥6GiB, máy hiện có 4GB).

## Thứ tự (đã chốt theo review)
**L5a → L1a → L3 (flag opt-in, KHÔNG block vào kết quả L1) → L2 → (L1b/L4 nếu có điều kiện)**

## Test plan (mở rộng `tests/` có sẵn — 13 file, 167 test)
| File | Case mới |
|---|---|
| `tests/test_planner.py` | fallback fix (q4_0 @8k ≈ 0.375GB), case thiếu metadata head |
| `tests/test_engine.py` | `Engine.tokenize()` wrapper (mock backend) |
| `tests/test_server.py` | cancel-giữa-stream không release lock sớm |
| `tests/test_bench_cmd.py` | hiển thị plan với backend server |
| `tests/test_config.py` | `_coerce` cho `backend` mới |

## 4 câu hỏi — lựa chọn đề xuất của tôi
1. **L5a**: eval set nhỏ ~50 dòng + PPL qua `llama-perplexity` (script lặp lại được), **chưa** cần harness đầy đủ — dựng harness khi L5b biết rõ cần đo gì.
2. **L3**: flag trong **config** (`backend`, default `inprocess`) — `_coerce_strict` đã sẵn; env var sẽ tạo 2 nguồn sự thật.
3. **L1**: dùng pipeline `vipera model quantize/optimize` sẵn có (L1a); repo tích hợp **chưa** chọn — nếu L1b cần thì `advanced-gguf-quantizer`.
4. **L4**: tách L4a/L4b như trên — nhưng L4a ghi điều kiện "model có MTP head" (2 model hiện tại không có), L4b blocked. Kết luận: **không làm code L4 trong vòng này**.

---

# Kết quả đo — 2026-10-21

> Phần "Đề xuất v2" ở trên viết trước khi làm. Đây là số đo sau khi làm xong;
> chỗ nào lệch nhau thì **số đo thắng**.

## L1a v3 — imatrix đo từ nguồn Q8_0 (requantize thật)

**Ngày:** 2026-10-21 · **Trạng thái:** xong, 4/4 arm `rc=0` · **Kết quả:** Q4 trong nhiễu,
Q3 có hiệu ứng thật nhưng vẫn dưới ngưỡng gate đã chốt (0.3812 < 0.4).

---

### Vì sao phải chạy lại (khác L1a v2)

L1a v2 lượng tử hoá `Q4_K_M → Q4_K_M`. `llama-quantize` copy nguyên byte khi type
nguồn trùng type đích (`llama-quant.cpp`: `quantize = cur_type != new_type`), và
imatrix chỉ được tra **bên trong** nhánh lượng tử hoá thật. Nên arm Q4 của v2 là
no-op cấu trúc: `+0.0000` không phải "imatrix vô dụng ở Q4".

v3 sửa đúng chỗ đó: nguồn là **Q8_0** (15.7 GB, ~0.01% loss so với BF16) nên
`Q8_0 → Q4_K_M` và `Q8_0 → Q3_K_M` **bắt buộc** đi qua bộ lượng tử hoá, và imatrix
có đường tác động vào từng weight.

### Bằng chứng "imatrix có thật sự đi vào cuộc"

`tools/gguf_quant_audit.py` so từng tensor giữa hai output cùng nguồn:

| Cặp | Tensor đổi **type** | Tensor đổi **data** |
|---|---:|---:|
| `q8src-ctl-q4km` vs `q8src-imbf-q4km` | 0/443 | **280/443** |
| `q8src-ctl-q3km` vs `q8src-imbf-q3km` | 0/443 | **280/443** |
| (v2, để so) `ctl-hermes-q4km` vs `l1a-hermes-q4km` | 0/443 | 0/443 |

Cùng type (type do recipe quyết định), nhưng **280/443 tensor đổi giá trị** — đó
đúng là chỗ imatrix tác động. Ở v2 con số này là 0.

### Số đo

Điều kiện y hệt L5a/L1a v2: `llama-perplexity 0.7.0-dev (build 1, commit 797de63)`,
`ctx=512, ngl=0, chunks=24, n_seq=4, scored_tok=12264`, eval `wikitext_sample.txt`
(sha256 `2a8ef09474c9`), CPU-only, chạy `setsid` nên không chết theo session.

| Arm | PPL ± stderr | Δ so control | % | Đọc |
|---|---:|---:|---:|---|
| `Q4_K_M` không imatrix | 7.0056 ± 0.2256 | — | — | control |
| `Q4_K_M` + imatrix BF16 | **6.9383** ± 0.2238 | **−0.0673** | −0.96% | **0.3σ → nhiễu** |
| `Q3_K_M` không imatrix | 7.3617 ± 0.2407 | — | — | control |
| `Q3_K_M` + imatrix BF16 | **6.9805** ± 0.2253 | **−0.3812** | −5.18% | **1.6σ → thật, nhưng < ngưỡng 0.4** |

stderr là độ bất định của eval set (12,264 token), không phải nhiễu run-to-run:
run lặp lại cho **đúng từng số lẻ**. Vì vậy Δ nhỏ hơn 1σ thì không kết luận được
là có lợi trên văn bản khác; Δ = 1.6σ thì có lợi nhưng chưa đủ để tuyên bố.

### Đọc số

1. **Q4_K_M: imatrix gần như không đổi gì** (−0.96%, 0.3σ). Hợp lý — Q4_K_M đã
   mixed-precision, tensor nhạy cảm được đẩy lên type cao sẵn, còn ít chỗ để
   imatrix cải thiện.
2. **Q3_K_M: imatrix cải thiện 5.18%** — cùng chiều với v2 (−3.24%), khác nguồn
   đo và khác imatrix. Hai phép đo độc lập cùng dấu ⇒ hiệu ứng có thật; biên độ
   0.24–0.38 PPL.
3. **Kết quả thực dụng quan trọng nhất:**
   `Q3_K_M + imatrix` = **6.9805**, `Q4_K_M` (không imatrix) = **7.0056**: Δ = 0.025
   PPL = **0.11σ**, tức **tương đương trong sai số** — không phải "tốt hơn".
   Điều bảo vệ được là: chất lượng không giảm khi file **7.32 GB thay vì 9.00 GB
   (−1.68 GB, −18.7%)**, và perplexity chạy **nhanh hơn 27%** (23.6 vs 18.8 tok/s)
   trên cùng máy CPU-only. Đừng viết "Q3 tốt hơn Q4" ở README.
4. **Verdict theo gate đã chốt trước:** Q4 không kết luận được; Q3 = 0.3812 vẫn
   **dưới** ngưỡng 0.4 ⇒ **chưa mở L1b**. Gate đúng như đã viết — không sửa
   ngưỡng sau khi thấy số.
5. **Về việc sát ngưỡng:** 0.3812 chỉ cách 0.4 một chút, nên một eval set lớn hơn
   12k token rất có thể vượt ngưỡng. Quyết định: **giữ nguyên ngưỡng đã chốt
   trước khi thấy số và không mở L1b ở vòng này** — đổi ngưỡng sau khi nhìn kết
   quả thì gate không còn là gate. Muốn vượt thì chạy thêm eval set (ghi rõ là
   đo lại, không phải chỉnh ngưỡng).

### Hạn chế (phải nói kèm số)

- Một eval set duy nhất (WikiText, 12k token). Không đo trên văn bản khác, không
  đo KL per-tensor, không kiểm tra chất lượng hội thoại.
- Imatrix là bản có sẵn của bartowski build từ **BF16** của cùng model — không
  phải build tại máy này. Nó loại trừ được giả thuyết "imatrix từ Q4_K_M bị lệch"
  của v2, nhưng không chứng minh được "imatrix tự build cũng cho kết quả này".
- Q3_K_M ở 3-bit: 5% PPL là con số đẹp, nhưng 3-bit vẫn dễ hỏng ở suy luận
  nhiều bước; chưa đo task-level.

### Tái lập

```bash
bash tools/l1a2_run.sh          # resume: bỏ qua bước đã có, PPL đã có rc=0 thì skip
cat tools/l1a2_results/report_numbers.txt
python tools/gguf_quant_audit.py \
  --a ~/.vipera/models/q8src-ctl-q4km.gguf --b ~/.vipera/models/q8src-imbf-q4km.gguf
```

---

## L2/L3 — Các lớp tối ưu có thật sự làm gì? (số đo, không phải suy đoán)

**Ngày:** 2026-10-21 · Máy: 4C/8T, 15 GiB RAM, CPU-only · Model: `qwen2.5-0.5b-instruct-q4_k_m`
(24 layer, 14 head / 2 KV head, head_dim 64) · Prompt cố định ~470 token · Engine tạo mới
cho mỗi dòng nên `load s` là thời gian nạp thật.

Chạy bằng `tools/bench_layers.py`; dữ liệu thô trong `kv_backends.json`, `fa_cost.json`, `kv_16k.json`.

```bash
.venv/bin/python tools/bench_layers.py --model qwen2.5-0.5b-instruct-q4_k_m \
  --ctx 4096 --max-tokens 64 --backend inprocess server --kv f16 q8_0 q5_1 iq4_nl
```

### A. L2 — KV cache dtype × backend (ctx 4096, sinh 64 token)

| backend | kv | FA planner đặt | load s | TTFT ms | gen t/s | token sinh |
|---|---|---:|---:|---:|---:|---:|
| inprocess | f16 | off | 0.67 | 464 | **67.21** | 64 |
| inprocess | q8_0 | **on** | 0.22 | 597 | 64.97 | 64 |
| inprocess | q5_1 | **on** | 0.22 | 549 | 57.49 | 64 |
| inprocess | iq4_nl | **on** | 0.20 | 549 | 55.05 | 47 |
| server | f16 | off | 0.54 | 579 | 66.28 | 64 |
| server | q8_0 | **on** | 0.50 | 533 | 66.24 | 64 |
| server | q5_1 | **on** | 0.50 | 650 | 61.26 | 6 |
| server | iq4_nl | **on** | 0.50 | 687 | 55.93 | 64 |

Đọc:

- **Planner bật flash-attention đúng như thiết kế**: f16 → off, mọi dtype lượng tử → on.
  Không có dòng nào `eff=on` khi `kv=f16` với `--fa auto`.
- **q8_0 gần như miễn phí:** −3.3% gen t/s (67.2 → 65.0) để đổi **một nửa KV RAM**.
  Đây là mặc định đúng cho máy ít RAM.
- **q5_1 / iq4_nl đắt hơn:** −14% / −18% gen t/s. Ở 0.5B, KV chỉ ~50 MB nên tiết
  kiệm RAM không đáng giá trị tốc độ mất đi; chỉ nên dùng khi KV là thứ chặn ctx.
- **`token sinh` khác nhau là tín hiệu chất lượng, không phải lỗi:** iq4_nl dừng ở 47
  token, `server q5_1` dừng ở 6 token — KV 4–5 bit đổi phân bố xác suất đủ để model
  kết thúc sớm. q8_0 sinh đủ 64 như f16 ở cả hai backend.

### B. L2 — flash-attention trên CPU có thật sự chậm? (kv f16)

Giả định cũ trong `planner.py`/`OPTIMIZE-hermes4.md`: "FA + KV lượng tử trên CPU =
chậm kinh khủng" (đo trên Hermes-4 @65536: 0.1–0.2 tok/s). Đo lại bằng cách ép
`--fa on` với f16, hai độ dài sequence:

**Prompt ~470 token, ctx 4096:**

| backend | FA | TTFT ms | gen t/s |
|---|---|---:|---:|
| inprocess | auto (off) | 464 | 66.71 |
| inprocess | **on** | 425 | 66.09 |
| inprocess | off | 464 | 66.62 |
| server | auto (off) | 543 | 66.12 |
| server | **on** | 566 | 66.91 |
| server | off | 579 | 65.48 |

**Prompt ~3,760 token (repeat 8), ctx 8192:**

| FA | TTFT ms | gen t/s |
|---|---:|---:|
| auto (off) | 3549 | 58.58 |
| **on** | **3197** | 56.78 |
| off | 3260 | 60.31 |

Đọc: ở 0.5B và sequence ≤ ~4k, FA trên CPU **không đắt** — TTFT còn thấp hơn (3.20 s
vs 3.26 s), decode chậm 6%. Con số "50× chậm" của Hermes-4 là ở **ctx 65,536 với KV
lượng tử trên 14B**, một chế độ khác hẳn; phép đo này **không** bác bỏ nó.

Vì vậy: planner giữ nguyên hành vi bảo thủ (f16/bf16/f32 không bật FA) là hợp lý —
không có lợi ích đo được, và ở ctx lớn nó là đường đã biết rất chậm. Nhưng ghi chú
"FA luôn chậm" trong tài liệu là **quá rộng**: nó chỉ đúng ở ctx lớn / model lớn.

### C. L2 — KV RAM thật sự tiết kiệm bao nhiêu (tính bằng `planner._estimate_kv_cache_gb`)

Hermes-4-14B (40 layer, 8 KV head, head_dim 128):

| ctx | f16 | bf16 | q8_0 | q5_1 | q4_0 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.62 GB | 0.62 GB | 0.31 GB | 0.23 GB | 0.16 GB |
| 8,192 | 1.25 GB | 1.25 GB | 0.62 GB | 0.47 GB | 0.31 GB |
| 32,768 | 5.00 GB | 5.00 GB | 2.50 GB | 1.88 GB | 1.25 GB |

Đo thực tế ở ctx 16384 trên qwen0.5b không phân biệt được (KV chỉ ~100 MB f16 vs
~53 MB q8_0, nằm dưới nhiễu RSS), nên bảng trên là **số tính từ công thức đã có
test** (`tests/test_planner.py::test_kv_estimate_*`), không phải số đo RSS.

### D. L3 — backend `server` so với `inprocess`

| | inprocess | server |
|---|---|---|
| load (0.5B) | 0.20–0.67 s | 0.50 s (spawn + chờ `/health`) |
| gen t/s | 65–67 | 66–67 |
| streaming | generator trong process | SSE từ `llama-server` |
| cách ly | chung GIL/RAM với gateway | process riêng, chết không kéo gateway |

**Backend `server` không mất throughput** (66.2 vs 67.2 t/s, ~1.5% trong nhiễu) và
trả giá đúng 0.3 s khởi động trên model 0.5B. Lớp này hoạt động đúng như thiết kế:
đổi backend không đổi chất lượng sinh, chỉ đổi chỗ chạy.

### E. L1 — đổi weight Q4_K_M → Q3_K_M+imatrix trên Hermes-4 (14B), số đo thật

3 lần lặp mỗi model, cùng prompt (~470 token), ctx 4096, kv f16, 64 token sinh:

| | load s | TTFT s | decode t/s | PPL |
|---|---:|---:|---:|---:|
| `q8src-ctl-q4km` (9.00 GB) | 7.9 / 9.2 / 8.0 | 9.8 / 10.1 / 13.2 | 3.73 / 3.76 / 3.69 | 7.0056 |
| `q8src-imbf-q3km` (7.32 GB) | 6.1 / 6.0 / 5.9 | 8.0 / 7.4 / 7.4 | 3.85 / 3.89 / 3.74 | **6.9805** |
| chênh (median) | **−25%** | **−27%** | **+3%** | **−0.36%** |

Nghĩa là: bớt 18.7% dung lượng file, nạp nhanh hơn 25%, prefill nhanh hơn 27%,
decode gần như không đổi (+3%), chất lượng **không giảm** (PPL thấp hơn 0.36%).
Decode không nhanh hơn nhiều vì Q3_K_M dequant đắt hơn Q4_K_M — bớt byte đọc
nhưng thêm việc giải mã.

Ghi chú quan trọng: gọi bằng đường dẫn `.gguf` nên `meta` trống (`layers=0`), tức
cả hai model đi qua **nhánh fallback heuristic** của planner, không phải nhánh
per-model (`_kv_layer_count`/`head_dim` từ metadata). Vì cả hai đi cùng một nhánh
nên so sánh L1 vẫn công bằng — nhưng **không được** trích bảng E để nói "L2 KV
estimate chạy đúng": nhánh per-model không được chạm tới ở đây. Số L2 nằm ở
bảng C, và bảng C là số **tính** từ `planner._estimate_kv_cache_gb` (công thức đã
có test), **không phải RSS đo được**.

### Tóm lại

| Lớp | Có tối ưu thật không? | Bằng chứng |
|---|---|---|
| L2 KV dtype | **Có** — q8_0 nửa RAM, −3% tốc độ; q5_1/iq4_nl đắt hơn, chất lượng đổi | bảng A |
| L2 FA | **Rẻ ở sequence ≤4k** (+0%); chưa đo lại Hermes 64k nên không kết luận chung | bảng B |
| L2 KV estimate | **Có** — planner tính đúng theo dtype, không dùng heuristic 1.5 GB | bảng C + test |
| L3 backend | **Có** — cùng tốc độ, +0.3 s khởi động, cách ly process | bảng D |
| L1a imatrix | **Có ở Q3** (−5.18%), không kết luận được ở Q4 (−0.96%) | `../l1a2_results/REPORT.md` |
| L1 đổi weight | **−19% file, −25% load, −27% prefill, PPL không giảm** | bảng E |

#### Tổng % đạt được (Hermes-4, một máy CPU-only 15 GiB)

| Hạng mục | Trước | Sau (Q3+imatrix, kv q8_0) | Đổi |
|---|---:|---:|---:|
| Kích thước file | 9.00 GB | 7.32 GB | **−18.7%** |
| KV @8k (f16 → q8_0) | 1.25 GB | 0.62 GB | **−50%** |
| Tổng RAM @8k | 10.25 GB | 7.94 GB | **−22.5%** |
| Tổng RAM @32k | 14.00 GB | 9.82 GB | **−29.9%** |
| Thời gian nạp | 8.0 s | 6.0 s | **−25%** |
| Prefill (TTFT) | 10.1 s | 7.4 s | **−27%** |
| Decode | 3.73 t/s | 3.85 t/s | +3% |
| PPL (thấp = tốt) | 7.0056 | 6.9805 | **−0.36%** |

Hàng PPL: 7.0056 → 6.9805 là Δ = 0.025 = **0.11σ** — **tương đương trong sai số**,
không phải "Q3 tốt hơn Q4". Claim bảo vệ được là *"không giảm chất lượng"*.

Ba lớp còn lại (`server` backend, FA, dtype q5_1/iq4_nl) **không đem lại % nào** về
tốc độ hay dung lượng ở máy này — chúng đem lại cách ly process / khả năng chọn,
nên đừng tính vào con số "tối ưu".

### Hạn chế

- Bảng A–D: 0.5B, prompt ~470 token, 1 lần/dòng. Chênh lệch <5% là nhiễu; đừng
  đọc 67.21 vs 66.28 thành "inprocess nhanh hơn". Bảng E: Hermes-4 14B, 3 lần/model
  (median) nên chênh lệch 25–27% là thật, còn +3% decode thì vẫn trong nhiễu.
- CPU-only. Trên GPU, FA là đường nhanh **hơn** nhiều — kết luận B không áp dụng cho GPU.
- KV RAM ở bảng C là số **tính** từ `planner._estimate_kv_cache_gb`, không phải RSS
  đo được — chưa có phép đo RSS nào trong báo cáo này chứng minh con số đó.
- Bảng E chỉ đo nhánh fallback của planner (path `.gguf` → meta trống), không đo
  nhánh per-model. Xem ghi chú ở mục E.

---

## Dữ liệu thô

| Đường dẫn | Nội dung |
|---|---|
| `tools/l1a2_results/q4_ctl.json`, `q4_imbf.json`, `q3_ctl.json`, `q3_imbf.json` | PPL từng arm |
| `tools/l1a2_results/report_numbers.txt` | bảng số do `l1a2_report.py` sinh |
| `tools/layer_results/kv_backends.json` | KV dtype × backend |
| `tools/layer_results/fa_cost.json`, `fa_long.json` | chi phí flash-attention |
| `tools/layer_results/kv_16k.json` | ctx 16384, f16 vs q8_0 |
