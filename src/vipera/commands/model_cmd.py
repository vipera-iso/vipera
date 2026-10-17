import contextlib
import re
import shutil
from pathlib import Path

import typer
from rich import box
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from vipera.config import MODELS_DIR, ensure_dirs, load_config, save_config
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


def _read_meta(path: Path) -> dict:
    meta = {
        "size_bytes": path.stat().st_size,
        "n_layers": 0,
        "n_params": 0,
        "context_length": 0,
        "quant": "?",
        "architecture": None,
    }

    m = _QUANT_RE.search(path.name)
    if m:
        meta["quant"] = m.group(1).upper()

    with contextlib.suppress(Exception):
        from gguf import GGUFReader

        r = GGUFReader(str(path))
        fields = getattr(r, "fields", {})

        def _val(key: str):
            f = fields.get(key)
            if not f:
                return None
            try:
                v = f.contents()
                return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
            except Exception:
                return None

        arch = _val("general.architecture")
        if arch:
            meta["architecture"] = str(arch)
            nl = _val(f"{arch}.block_count")
            if nl:
                meta["n_layers"] = int(nl)
            ctx = _val(f"{arch}.context_length")
            if ctx:
                meta["context_length"] = int(ctx)
        np_ = _val("general.parameter_count")
        if np_:
            meta["n_params"] = int(np_)

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
    meta = _read_meta(dst)
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
            raise typer.Exit(1)

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
        except Exception:
            raise typer.Exit(1)

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
    Path(m["path"]).unlink(missing_ok=True)
    remove_model(name)

    cfg = load_config()
    if cfg.get("default_model") == name:
        cfg["default_model"] = None
        save_config(cfg)

    console.print(f"[{C_GREEN}] ✓ Removed: {name}[/]")
