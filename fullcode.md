# vipera — Full Source Code

```python
# __init__.py
__version__ = "1.0.0"

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

app.add_typer(model_cmd.app,  name="model",  help="Manage GGUF models")
app.add_typer(serve_cmd.app,  name="serve",  help="Start HTTP Gateway")
app.add_typer(chat_cmd.app,   name="chat",   help="Chat with AI")
app.add_typer(bench_cmd.app,  name="bench",  help="Benchmark performance")
app.add_typer(system_cmd.app, name="system", help="Diagnostics & Config")


@app.command("help")
def show_help():
    """Detailed usage guide."""
    guide = Text()

    guide.append("  ▸ QUICK START\n", style=C_ACCENT)
    guide.append("    • vipera                          Start gateway (default model)\n", style=C_TEXT)
    guide.append("    • vipera -s                       Start gateway with model picker\n", style=C_TEXT)
    guide.append("    • vipera model search             Search & download from HuggingFace\n", style=C_TEXT)
    guide.append("    • vipera chat repl                Chat in the terminal\n\n", style=C_TEXT)

    guide.append("  ▸ MODELS (vipera model ...)\n", style=C_ACCENT)
    guide.append("    • vipera model list               List installed models\n", style=C_TEXT)
    guide.append("    • vipera model import <path>      Import a local .gguf file\n", style=C_TEXT)
    guide.append("    • vipera model inspect <path>     Analyze GGUF layout & quants\n", style=C_TEXT)
    guide.append("    • vipera model imatrix <path>     Build an importance matrix\n", style=C_TEXT)
    guide.append("    • vipera model quantize <path>    Quantize with llama-quantize\n", style=C_TEXT)
    guide.append("    • vipera model optimize <path>    Inspect → imatrix → quantize\n", style=C_TEXT)
    guide.append("    • vipera model rm <name>          Remove a model\n\n", style=C_TEXT)

    guide.append("  ▸ INFERENCE (vipera chat / bench ...)\n", style=C_ACCENT)
    guide.append("    • vipera chat repl                Interactive chat\n", style=C_TEXT)
    guide.append("    • vipera chat oneshot \"<prompt>\"  One-shot query\n", style=C_TEXT)
    guide.append("    • vipera bench run                Benchmark model\n\n", style=C_TEXT)

    guide.append("  ▸ SYSTEM (vipera system ...)\n", style=C_ACCENT)
    guide.append("    • vipera system doctor            Hardware & library diagnostics\n", style=C_TEXT)
    guide.append("    • vipera system config            View/edit config\n\n", style=C_TEXT)

    guide.append("  💡 TIP: use `vipera -s` to switch the running model.\n", style=C_DIM)

    console.print(Panel(
        guide,
        title=Text(" VIPERA MANUAL ", style=C_ACCENT),
        border_style=C_ACCENT,
        box=box.ROUNDED,
        padding=(1, 2),
    ))


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
        console.print(Panel(
            Text("  [ NO MODELS INSTALLED ]\n\n"
                 "  Import a model first:\n"
                 "  ► vipera model import <path_to_gguf> --name my-model\n\n"
                 "  Or search Hugging Face:\n"
                 "  ► vipera model search", style=C_RED),
            title=Text(" WARNING ", style=C_RED),
            border_style=C_RED, box=box.ROUNDED,
        ))
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
from vipera.engine import Engine
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


def _run_bench(engine: Engine, prompt: str, max_tokens: int) -> dict:
    try:
        prompt_tok = len(engine.llm.tokenize(prompt.encode()))
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
        )
        for chunk in stream:
            usage = chunk.get("usage") if isinstance(chunk, dict) else None
            if usage:
                final_usage = usage
            try:
                delta = chunk["choices"][0]["delta"].get("content", "") if chunk.get("choices") else ""
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
    model: str = typer.Argument(None, help="Model name (default: current default model)"),
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
    ctx_list = [int(x) for x in ctx.split(",") if x.strip()]

    console.print(f"\n[{C_ACCENT}]▸ Benchmarking model:[/] [{C_TEXT}]{model}[/]")
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
            m["meta"], hw,
            ctx=c,
            kv_cache=cfg.get("kv_cache", "q8_0"),
            n_batch=cfg.get("n_batch", 512),
        )

        engine = None
        try:
            with console.status(f"[{C_ACCENT}] Loading model with ctx={c} (GPU: {p.gpu_layers} layers)…"):
                try:
                    engine = Engine(p, m["path"])
                except Exception as e:
                    console.print(f"[{C_RED}] ✗ Failed to load model with ctx={c}: {e}[/]")
                    continue

            animated_loading_bar(f"RUNNING CTX {c}", 0.5)
            result = _run_bench(engine, PROMPT, max_tokens)

            if "error" in result:
                console.print(f"[{C_RED}] ✗ Benchmark error at ctx={c}: {result['error']}[/]")
                continue

            layers_total = int(m["meta"].get("n_layers") or 0) or max(p.gpu_layers, 1)

            gen_tok_str = str(result["gen_tok"])
            if not result.get("token_count_is_exact"):
                gen_tok_str = f"~{result['gen_tok']}"

            t.add_row(
                str(c),
                f"{p.gpu_layers}/{layers_total}",
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
    plain_chat_stream,
    typewriter_chat_stream,
)

app = typer.Typer()


def _chat_kwargs(cfg: dict) -> dict:
    """Pull the sampling overrides the engine understands out of the config."""
    keys = ("temperature", "max_tokens", "top_p", "repeat_penalty")
    return {k: cfg[k] for k in keys if k in cfg and cfg[k] is not None}


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


def repl_loop(engine: Engine, typewriter: bool = False):
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
    model: str = typer.Option(None, "--model", "-m", help="Model alias"),
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
        repl_loop(engine, typewriter=typewriter)
    finally:
        engine.close()


@app.command("oneshot")
def oneshot(
    prompt: str = typer.Argument(..., help="Prompt to send"),
    model: str = typer.Option(None, "--model", "-m", help="Model alias"),
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
    }

    m = _QUANT_RE.search(name or path.name)
    if m:
        meta["quant"] = m.group(1).upper()

    try:
        from vipera.gguf_analyzer import analyze_gguf

        am = analyze_gguf(str(path))["metadata"]
        meta["n_layers"] = int(am.get("n_layers") or 0)
        meta["n_params"] = int(am.get("n_params") or 0)
        meta["context_length"] = int(am.get("context_length") or 0)
        meta["architecture"] = am.get("architecture")
        if meta["quant"] == "?" and am.get("dominant_quant"):
            meta["quant"] = am["dominant_quant"]
    except Exception:  # analyzer is best-effort
        pass

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
    console.print(Panel(
        "  Enter a keyword to search for GGUF models on Hugging Face.",
        title="[ SEARCH HUB ]",
        border_style=C_ACCENT,
        box=box.ROUNDED,
    ))
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
        console.print(f"\n[{C_ACCENT}] ▸ Just run [bold]vipera[/bold] and this model will be pre-selected.\n")
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
        console.print(
            f"[{C_YELLOW}] ⚠ Pruning layers can substantially reduce quality.[/]"
        )
    if not _run_tool(command, dry_run, cleanup=out):
        raise typer.Exit(1)
    if not dry_run:
        console.print(
            f"[{C_GREEN}] ✓ quantized model:[/] {out} "
            f"({_format_bytes(out.stat().st_size)})"
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
        supports_prior = prior_weight != 1 and tool_supports_flag(
            quantize_bin, "--prior-weight"
        )
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
        console.print(
            f"[{C_YELLOW}] ⚠ Pruning layers can substantially reduce quality.[/]"
        )
    console.print(
        f"[{C_YELLOW}] ⚠ Optimizing an already-quantized model requantizes it; "
        "quality can drop.[/]"
    )
    if not _run_tool(command, dry_run, cleanup=out):
        raise typer.Exit(1)
    if not dry_run:
        console.print(
            f"[{C_GREEN}] ✓ optimized model:[/] {out} "
            f"({_format_bytes(out.stat().st_size)})"
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
            f"[{i + 1:02d}]", m["name"], size_str,
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

    Path(m["path"]).unlink(missing_ok=True)

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
    console.print(f"  [{C_DIM}]To preserve quality, use mixed-precision for output & embeddings:[/]\n")

    console.print(f"  [{C_ACCENT}]Command to run:[/]")
    console.print(
        f"  vipera model quantize {path} \\\n"
        f"    --target-quant {current_quant} \\\n"
        f"    --output-tensor q6_k \\\n"
        f"    --token-embedding q6_k \\\n"
        f"    --imatrix <your_imatrix.gguf>\n"
    )

# commands/serve_cmd.py
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
    p = plan(
        m["meta"], hw, profile=profile, ctx=ctx,
        kv_cache=cfg.get("kv_cache", "q8_0"),
        n_batch=cfg.get("n_batch", 512),
    )

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
        from vipera.commands.chat_cmd import repl_loop

        repl_loop(engine)
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
    try:
        import llama_cpp  # noqa: F401
        row("llama-cpp-python", "installed")
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

    if llama_missing:
        console.print(
            f"[{C_YELLOW}] ⚠ llama-cpp-python is required to load models.[/]"
        )
        console.print(f"[{C_DIM}]   Install:[/] python -m pip install llama-cpp-python")
        console.print(f"[{C_DIM}]   Then restart the gateway.[/]\n")

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
    "profile": "auto",
    "ctx": 8192,
    "kv_cache": "q8_0",
    "flash_attn": True,
    "n_batch": 512,
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
    "show_bench_on_start": True,
}

_KV_CACHE_TYPES = {"f32", "f16", "q4_0", "q4_1", "q8_0"}

_TYPE_MAP = {
    "port": int, "ctx": int, "n_batch": int, "max_tokens": int,
    "temperature": float, "top_p": float, "repeat_penalty": float,
    "flash_attn": bool, "show_bench_on_start": bool,
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
    return cfg


def _coerce_strict(cfg: dict) -> dict:
    """Raise ValueError on any invalid typed value."""
    for k, cast in _TYPE_MAP.items():
        if k in cfg and cfg[k] is not None:
            try:
                if cast is bool:
                    cfg[k] = str(cfg[k]).lower() in ("1", "true", "yes", "on")
                else:
                    cfg[k] = cast(cfg[k])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{k}: invalid {cast.__name__} value '{cfg[k]}'") from exc
    if "kv_cache" in cfg and cfg["kv_cache"] not in _KV_CACHE_TYPES:
        valid = ", ".join(sorted(_KV_CACHE_TYPES))
        raise ValueError(f"kv_cache: must be one of {valid}")
    return cfg


def load_config() -> dict:
    ensure_dirs()
    if CONFIG_FILE.exists():
        with contextlib.suppress(Exception):
            return _coerce({**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text())})
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    ensure_dirs()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

# engine.py
import gc
from dataclasses import replace
from typing import Any

from vipera.planner import Plan

_KV_TYPES = {"f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q8_0": 8}


class Engine:
    def __init__(self, plan: Plan, model_path: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            ) from None

        self.plan = plan
        self._model_path = model_path
        self._Llama = Llama
        self._closed = False

        gpu_exc: Exception | None = None
        try:
            self.llm = self._load(plan)
            return
        except Exception as exc:  # need to try CPU fallback
            gpu_exc = exc

        print("\n[!] Warning: failed to load with GPU. Falling back to CPU...")
        safe_plan = replace(
            plan,
            gpu_layers=0,
            flash_attn=False,
            offload_kqv=False,
            kv_cache="f16",
            n_batch=128,
            context=min(plan.context, 2048),
        )

        try:
            self.llm = self._load(safe_plan)
            self.plan = safe_plan
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize model even on CPU: {exc}"
            ) from gpu_exc

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
        return self.llm.create_chat_completion(
            messages=messages,
            stream=stream,
            temperature=kw.get("temperature", 0.7),
            max_tokens=kw.get("max_tokens", 2048),
            top_p=kw.get("top_p", 0.95),
            repeat_penalty=kw.get("repeat_penalty", 1.1),
        )

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
    r"(IQ\d+_\w+|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d|BF16|F16|F32|Q\d+)",
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
}

_CRITICAL_TENSORS = ("ffn_down", "attn_v")
_CRITICAL_THRESHOLD = "Q4_K"


class GGUFAnalysisError(RuntimeError):
    """Raised when a GGUF file cannot be inspected."""


def _quant_name(tensor_type: Any) -> str:
    try:
        value = int(tensor_type)
    except (TypeError, ValueError):
        return str(tensor_type)
    return _QUANT_NAMES.get(value, f"TYPE_{value}")


def _quant_rank(name: str) -> float:
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
    return 0.0


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
    if architecture:
        n_layers = _as_int(_field_value(fields, f"{architecture}.block_count"))
        context_length = _as_int(_field_value(fields, f"{architecture}.context_length"))
        embedding_length = _as_int(_field_value(fields, f"{architecture}.embedding_length"))

    tensor_map: list[dict[str, Any]] = []
    layer_sizes: dict[int | None, int] = defaultdict(int)
    category_sizes: dict[str, int] = defaultdict(int)
    quant_sizes: dict[str, int] = defaultdict(int)
    quant_counts: dict[str, int] = defaultdict(int)
    warnings: list[dict[str, str]] = []
    max_layer = -1

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
        if layer is not None:
            max_layer = max(max_layer, layer)

        if any(key in tensor_name for key in _CRITICAL_TENSORS) and _quant_rank(
            quant
        ) < _quant_rank(_CRITICAL_THRESHOLD):
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
            "n_params": _as_int(_field_value(fields, "general.parameter_count")),
            "quant": filename_quant or dominant_quant or "?",
            "dominant_quant": dominant_quant,
            "tensor_count": len(tensor_map),
            "file_size_bytes": os.path.getsize(model_path),
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
            "psutil is required for hardware detection. "
            "Install it with: pip install psutil"
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
        cpu_cores=cores, cpu_threads=threads,
        ram_total_gb=ram_total, ram_free_gb=ram_free,
        gpu_name=gpu_name,
        vram_total_gb=vram_total, vram_free_gb=vram_free,
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
}

_QUALITY_ORDER = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")

_QUANT_BPW = {
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


def plan(
    meta: dict,
    hw: HardwareInfo,
    profile: str = "auto",
    ctx: int = 8192,
    kv_cache: str = "q8_0",
    n_batch: int = 512,
) -> Plan:
    n_layers = int(meta.get("n_layers") or 32)
    size_bytes = int(meta.get("size_bytes") or 4_000_000_000)

    cpu_threads = max(1, hw.cpu_cores - 1)
    has_gpu = getattr(hw, "vram_total_gb", 0) > 0

    model_size_gb = size_bytes / 1024**3
    use_mlock = (hw.ram_free_gb - model_size_gb) > 2.0

    safe_vram = max(0.0, hw.vram_free_gb - 1.0)  # reserve 1GB for OS/driver

    # KV cache estimate: roughly 1.5x the context size in bytes for a 7-8B model.
    # This is conservative — assumes uniform layer sizes and a 128-byte KV per token.
    kv_cache_gb = (ctx / 8192.0) * 1.5

    usable_vram = max(0.0, safe_vram - kv_cache_gb)

    layer_gb = (size_bytes / 1024**3) / n_layers if n_layers else 0.4
    gpu_layers = 0

    if has_gpu and usable_vram > 0 and layer_gb > 0:
        gpu_layers = min(n_layers, int(usable_vram / layer_gb))

    if gpu_layers == 0:
        flash_attn = False
        offload_kqv = False
        n_batch = min(n_batch, 128)
        kv_cache = "f16"

        if hw.ram_total_gb < 8:
            ctx = min(ctx, 2048)
            note = "CPU ONLY (Low RAM Safe Mode)"
        else:
            note = "CPU ONLY"
    else:
        flash_attn = has_gpu and hw.vram_total_gb > 4
        offload_kqv = True
        n_batch = min(n_batch, 512)
        note = "Full GPU" if gpu_layers == n_layers else f"Hybrid: {gpu_layers}/{n_layers} layers"

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
    )

# registry.py
import contextlib
import json
import os
from datetime import UTC, datetime

from vipera.config import REGISTRY_FILE, ensure_dirs


def _load() -> dict:
    ensure_dirs()
    if REGISTRY_FILE.exists():
        with contextlib.suppress(Exception):
            return json.loads(REGISTRY_FILE.read_text())
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
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

_MAX_TOKENS = 32_768
_MAX_MESSAGES = 200
_MAX_PROMPT_CHARS = 500_000


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(..., min_length=1)
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    repeat_penalty: float = 1.1


def build_app(engine, model_name: str) -> FastAPI:
    app = FastAPI(title="Vipera Gateway", version="1.0.0")
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
            raise HTTPException(
                404, f"model '{req.model}' not loaded (active: '{model_name}')"
            )
        if len(req.messages) > _MAX_MESSAGES:
            raise HTTPException(400, f"too many messages (max {_MAX_MESSAGES})")
        if sum(len(m.content) for m in req.messages) > _MAX_PROMPT_CHARS:
            raise HTTPException(400, "prompt too large")
        if not 0 < req.max_tokens <= _MAX_TOKENS:
            raise HTTPException(400, f"max_tokens must be in [1, {_MAX_TOKENS}]")
        if not 0.0 <= req.temperature <= 2.0:
            raise HTTPException(400, "temperature must be in [0, 2]")
        if not 0.0 < req.top_p <= 1.0:
            raise HTTPException(400, "top_p must be in (0, 1]")

        messages = [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in req.messages]
        kwargs = dict(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            repeat_penalty=req.repeat_penalty,
        )

        if req.stream:
            async def _stream():
                # asyncio.Lock (unlike the old threading.Lock + threadpool acquire)
                # is released correctly if the client disconnects mid-stream,
                # which used to leak the lock and deadlock the gateway.
                async with lock:
                    try:
                        for chunk in engine.chat(messages, stream=True, **kwargs):
                            payload = json.dumps(chunk, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    except Exception as exc:
                        err = {"error": {"message": str(exc)}}
                        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        async with lock:
            return await run_in_threadpool(engine.chat, messages, stream=False, **kwargs)

    return app

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
    out = []
    for line in text.split("\n"):
        if not line:
            out.append(" " * width)
        else:
            for w in textwrap.wrap(line, width=width) or [""]:
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
    stat.append(f"{chunk_count} chunks  ·  {tps:.1f} c/s  ·  {dt:.2f}s\n", style=C_DIM)
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
    stat.append(f"{chunk_count} chunks  ·  {tps:.1f} c/s  ·  {dt:.2f}s\n", style=C_DIM)
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
