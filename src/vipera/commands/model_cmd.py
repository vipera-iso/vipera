import re
import shlex
import shutil
import subprocess
from pathlib import Path

import typer
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from rich import box
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

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
    C_TEXT,
    C_YELLOW,
    animated_loading_bar,
    console,
    note,
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
        console.print(f"file not found: {src_path}")
        raise typer.Exit(1)

    base_alias = (name or src_path.stem).removesuffix(".gguf")
    alias = _unique_alias(base_alias)
    dst = MODELS_DIR / f"{alias}.gguf"

    if src_path.resolve() != dst.resolve():
        with console.status(f"[{C_ACCENT}] Copying {src_path.name}…"):
            shutil.copy2(src_path, dst)

    animated_loading_bar("importing gguf", 1.0)
    meta = _read_meta(dst, src_path.name)
    add_model(alias, dst, meta)

    cfg = load_config()
    cfg["default_model"] = alias
    save_config(cfg)

    param_str = f"{meta.get('n_params', 0) / 1e9:.2f}B · " if meta.get("n_params") else ""
    console.print(f"imported {alias} and set it as default")
    console.print(
        f"    [{C_DIM}]└─ {param_str}{meta['size_bytes'] / 1e9:.2f} GB  ·  "
        f"{meta.get('n_layers', 0)} layers  ·  {meta['quant']}[/]\n"
    )


@app.command("search")
def search_model():
    """Search and download GGUF models from Hugging Face."""
    ensure_dirs()
    note("search Hugging Face for GGUFs. a keyword is enough, e.g. `qwen2.5 7b`.")
    query = Prompt.ask(f"[{C_ACCENT}]   keyword[/]")
    if not query:
        return

    with console.status(f"[{C_ACCENT}] Searching for '{query}'…"):
        try:
            models = list(api.list_models(search=query, filter="gguf", limit=10))
        except Exception as e:
            console.print(f"could not reach Hugging Face: {e}")
            raise typer.Exit(1) from e

    if not models:
        note("nothing matched that keyword. try a shorter one.", tone="warn")
        raise typer.Exit(0)

    t = Table(box=box.SIMPLE_HEAD, title=f"results for {query}", title_style=C_DIM, title_justify="left")
    t.add_column("id", style=C_DIM, justify="center")
    t.add_column("repo id", style=C_TEXT)
    t.add_column("downloads", style=C_ACCENT, justify="right")
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
        console.print("no .gguf files in that repository")
        raise typer.Exit(1)

    ft = Table(box=box.SIMPLE_HEAD, title=f"files in {selected_repo}", title_style=C_DIM, title_justify="left")
    ft.add_column("id", style=C_DIM, justify="center")
    ft.add_column("file name", style=C_TEXT)
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
        console.print("download complete")
        animated_loading_bar("finalising", 0.5)

        base_alias = Path(downloaded_path).stem
        alias = _unique_alias(base_alias)
        meta = _read_meta(Path(downloaded_path))
        add_model(alias, downloaded_path, meta)

        cfg = load_config()
        cfg["default_model"] = alias
        save_config(cfg)

        param_str = f"{meta.get('n_params', 0) / 1e9:.2f}B · " if meta.get("n_params") else ""
        console.print(f"registered {alias} and set it as default")
        console.print(
            f"    [{C_DIM}]└─ {param_str}{meta['size_bytes'] / 1e9:.2f} GB  ·  {meta['quant']}[/]"
        )
        console.print(
            f"\n[{C_ACCENT}] ▸ Just run [bold]vipera[/bold] and this model will be pre-selected.\n"
        )
    except KeyboardInterrupt:
        console.print(f"\n[{C_YELLOW}] ⚠ Download cancelled.[/]")
    except Exception as e:
        console.print(f"download failed: {e}")


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
        console.print(f"{e}")
        raise typer.Exit(1) from e

    meta = result["metadata"]
    summary = Table(box=box.SIMPLE_HEAD, title="gguf anatomy", title_style=C_DIM, title_justify="left")
    summary.add_column("property", style=C_DIM)
    summary.add_column("value", style=C_TEXT)
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

    categories = Table(box=box.SIMPLE_HEAD, title="size by category", title_style=C_DIM, title_justify="left")
    categories.add_column("category", style=C_TEXT)
    categories.add_column("tensors", style=C_DIM, justify="right")
    categories.add_column("size", style=C_ACCENT, justify="right")
    categories.add_column("share", style=C_DIM, justify="right")
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

    quant_table = Table(box=box.SIMPLE_HEAD, title="quantization mix", title_style=C_DIM, title_justify="left")
    quant_table.add_column("quant", style=C_ACCENT)
    quant_table.add_column("tensors", style=C_DIM, justify="right")
    quant_table.add_column("size", style=C_TEXT, justify="right")
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
            lines.append(f"{warning['tensor']}: {warning['quant']}")
        if len(warnings) > 10:
            lines.append(f"and {len(warnings) - 10} more")
        console.print()
        note("\n".join(lines), tone="warn")

    if tensors:
        tensor_table = Table(
            box=box.SIMPLE_HEAD, title="tensors", title_style=C_DIM, title_justify="left"
        )
        tensor_table.add_column("layer", style=C_DIM, justify="right")
        tensor_table.add_column("name", style=C_TEXT)
        tensor_table.add_column("quant", style=C_ACCENT)
        tensor_table.add_column("shape", style=C_DIM, justify="right")
        tensor_table.add_column("size", style=C_DIM, justify="right")
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
        console.print(f"{label} not found: {path}")
        raise typer.Exit(1)
    return path


def _prepare_output(value: str | None, fallback: Path) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.exists():
            console.print(f"output already exists: {path}")
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
        console.print(f"{e}")
        ok = False
    except subprocess.CalledProcessError as e:
        console.print(f"llama.cpp exited with code {e.returncode}")
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
        console.print("--ctx must be greater than zero")
        raise typer.Exit(1)

    try:
        binary = find_llama_tool("imatrix", bin_dir)
        default_out = model.with_suffix(".imatrix.gguf")
        out = _prepare_output(output, default_out)
        command = build_imatrix_command(binary, model, calibration, out, ctx, threads)
    except OptimizerError as e:
        console.print(f"{e}")
        raise typer.Exit(1) from e

    if not _run_tool(command, dry_run, cleanup=out):
        raise typer.Exit(1)
    if not dry_run:
        console.print(f"imatrix written to {out}")


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
        console.print("--prior-weight must be at least 1")
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
        console.print(f"{e}")
        raise typer.Exit(1) from e

    if prune_layers:
        console.print(f"[{C_YELLOW}] ⚠ Pruning layers can substantially reduce quality.[/]")
    if not _run_tool(command, dry_run, cleanup=out):
        raise typer.Exit(1)
    if not dry_run:
        console.print(
            f"quantized {out} ({_format_bytes(out.stat().st_size)})"
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
        console.print("provide --target-size or --target-quant")
        raise typer.Exit(1)
    if target_size is not None and target_size <= 0:
        console.print("--target-size must be greater than zero")
        raise typer.Exit(1)
    if prior_weight < 1:
        console.print("--prior-weight must be at least 1")
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
        console.print(f"{e}")
        raise typer.Exit(1) from e

    try:
        quantize_bin = find_llama_tool("quantize", bin_dir)
        supports_prior = prior_weight != 1 and tool_supports_flag(quantize_bin, "--prior-weight")
    except OptimizerError as e:
        console.print(f"{e}")
        raise typer.Exit(1) from e

    stem = model.stem
    out = _prepare_output(output, default_output_path(model, quant))
    imatrix_out = model.with_name(f"{stem}.imatrix.gguf")
    if imatrix_out.exists():
        imatrix_out = unique_output_path(imatrix_out)

    plan = Table(box=box.SIMPLE_HEAD, title="optimization plan", title_style=C_DIM, title_justify="left")
    plan.add_column("property", style=C_DIM)
    plan.add_column("value", style=C_TEXT)
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
            console.print(f"{e}")
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
        console.print(f"{e}")
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
            f"optimized {out} ({_format_bytes(out.stat().st_size)})"
        )


@app.command("list")
def list_cmd():
    """List all installed models."""
    models = list_models()
    if not models:
        console.print(f"[{C_DIM}]No models. Run `vipera model import` or `vipera model search`.[/]")
        return

    t = Table(box=box.SIMPLE_HEAD, title="local models", title_style=C_DIM, title_justify="left")
    t.add_column("id", style=C_DIM, justify="center")
    t.add_column("alias", style=C_TEXT)
    t.add_column("size", style=C_DIM)
    t.add_column("quant", style=C_ACCENT)
    t.add_column("layers", style=C_DIM, justify="right")
    t.add_column("ctx", style=C_DIM, justify="right")

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
        console.print(f"there is no model called {name}. `vipera model list` shows them.")
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

    console.print(f"removed {name}")


@app.command("recommend")
def recommend_quantization(
    path: str = typer.Argument(..., help="Path to a .gguf file"),
):
    """Analyze a model and recommend mixed-precision quantization settings."""
    try:
        result = analyze_gguf(path)
    except GGUFAnalysisError as e:
        console.print(f"{e}")
        raise typer.Exit(1) from e

    meta = result["metadata"]
    current_quant = meta.get("quant", "Unknown")
    warnings = result["warnings"]

    console.print(f"\n[{C_ACCENT}]▸ Optimization Recommendations for {path}[/]\n")

    if not warnings:
        console.print("this model has no low-precision tensors worth fixing")
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
