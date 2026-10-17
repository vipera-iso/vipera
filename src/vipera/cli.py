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

    print_banner()

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
