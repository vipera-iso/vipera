import typer

from vipera import __version__
from vipera.commands import bench_cmd, chat_cmd, model_cmd, serve_cmd, system_cmd
from vipera.config import load_config
from vipera.registry import list_models
from vipera.ui import (
    catalog,
    console,
    note,
    pick_model,
    print_banner,
    section,
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
    section("start")
    catalog(
        [
            ("vipera", "start the gateway with the default model"),
            ("vipera -s", "start the gateway and pick a model first"),
            ("vipera model search", "search Hugging Face and download a GGUF"),
            ("vipera chat repl", "chat against a local model"),
        ]
    )

    section("models")
    catalog(
        [
            ("vipera model list", "list installed models"),
            ("vipera model import <path>", "import a local .gguf"),
            ("vipera model inspect <path>", "read the GGUF layout and quants"),
            ("vipera model imatrix <path>", "build an importance matrix"),
            ("vipera model quantize <path>", "quantize with llama-quantize"),
            ("vipera model optimize <path>", "inspect, then imatrix, then quantize"),
            ("vipera model rm <name>", "remove a model"),
        ]
    )

    section("inference")
    catalog(
        [
            ("vipera chat repl", "interactive chat"),
            ('vipera chat oneshot "<prompt>"', "one question, then exit"),
            ("vipera bench run", "measure prompt and generation throughput"),
        ]
    )

    section("system")
    catalog(
        [
            ("vipera system doctor", "hardware, runtime, and library check"),
            ("vipera system config", "read or change config.json keys"),
        ]
    )

    console.print()
    note("use `vipera -s` to switch the model the gateway starts with.")


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
        note(
            "no models installed yet.\n"
            "  vipera model search            download one from Hugging Face\n"
            "  vipera model import <path>     import a local .gguf",
            tone="error",
        )
        raise typer.Exit(1)

    cfg = load_config()
    model = cfg.get("default_model")

    if select or not model:
        if len(installed) == 1 and not select:
            model = installed[0]["name"]
            console.print(f"   only one model is installed, using {model}")
            console.print()
        else:
            model = pick_model("select a model to start")
    else:
        console.print(f"   starting the gateway with {model}")
        console.print("   use `vipera -s` to choose another model")
        console.print()

    ctx.invoke(serve_cmd.start, model=model)
