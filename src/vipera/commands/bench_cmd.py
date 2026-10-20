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

            # Report the plan the engine actually loaded: a GPU load can fail
            # and fall back to CPU, which would otherwise still print the
            # optimistic planned layer count.
            layers_total = int(m["meta"].get("n_layers") or 0) or max(engine.plan.gpu_layers, 1)

            gen_tok_str = str(result["gen_tok"])
            if not result.get("token_count_is_exact"):
                gen_tok_str = f"~{result['gen_tok']}"

            t.add_row(
                str(c),
                f"{engine.plan.gpu_layers}/{layers_total}",
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
