import time

import typer
from rich import box
from rich.table import Table

from vipera.backends.protocol import ChatEngine
from vipera.backends.registry import create_engine
from vipera.config import load_config
from vipera.hardware import detect
from vipera.optimize.tuning import TuningStore
from vipera.planner import plan
from vipera.registry import get_model
from vipera.ui import (
    C_ACCENT,
    C_DIM,
    C_GREEN,
    C_TEXT,
    animated_loading_bar,
    console,
    ledger,
    note,
    pick_model,
    section,
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
                delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
            except (KeyError, IndexError, TypeError):
                delta = {}
            # A reasoning model streams its chain of thought in
            # `reasoning_content` before (or instead of) `content`. Both are
            # output tokens: the first one is the real ttft, and chunk counting
            # has to include them or a reasoning-only reply reports 0 tokens.
            content = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if content or thought:
                chunk_count += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
    except Exception as e:
        return {"error": str(e), "prompt_tok": prompt_tok, "gen_tok": chunk_count}

    dt = time.perf_counter() - t0
    ttft = (first_token_at - t0) if first_token_at else dt
    gen_time = dt - ttft

    # Prefer the model's own token counts when the backend reports them;
    # llama.cpp only does so when asked via stream_options, which this build
    # may not support, so tokenize() and the streamed-chunk count remain the
    # fallbacks. The server's prompt count is exact: counting the raw prompt
    # text under-reports by the chat template's tokens.
    gen_tok = chunk_count
    token_count_is_exact = False
    if final_usage:
        if final_usage.get("prompt_tokens"):
            prompt_tok = int(final_usage["prompt_tokens"])
        if final_usage.get("completion_tokens"):
            gen_tok = int(final_usage["completion_tokens"])
            token_count_is_exact = True

    prompt_tps = prompt_tok / ttft if ttft > 0 and prompt_tok > 0 else 0.0

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
        model = pick_model("select a model to benchmark")

    m = get_model(model)
    if not m:
        note(f"no model called {model}. run `vipera model list`.", tone="error")
        raise typer.Exit(1)

    hw = detect()
    try:
        ctx_list = [int(x) for x in ctx.split(",") if x.strip()]
    except ValueError as exc:
        note(f"--ctx takes comma-separated integers (512,8192,65536); got {ctx!r}", tone="error")
        raise typer.Exit(1) from exc
    if not ctx_list or any(c <= 0 for c in ctx_list):
        note(f"--ctx values must be positive integers; got {ctx!r}", tone="error")
        raise typer.Exit(1)

    backend = cfg.get("backend", "inprocess")
    # The in-process engine cannot do speculative decoding (llama-cpp-python's
    # prompt-lookup draft measured -29%), so the setting only exists here as
    # context for reading the numbers: the server backend applies it itself.
    spec = cfg.get("spec_type", "none") if backend == "server" else "off"
    console.print()
    ledger(
        [
            ("model", model),
            ("backend", f"{backend}, spec {spec}"),
            (
                "machine",
                f"{hw.cpu_cores} cores, {hw.ram_free_gb:.1f} GB RAM free, {hw.gpu_name}",
            ),
        ]
    )

    section("results")
    t = Table(box=box.SIMPLE_HEAD, padding=(0, 2), pad_edge=False, show_edge=False)
    t.add_column("ctx", style=C_TEXT, justify="right")
    t.add_column("layers", style=C_DIM, justify="right")
    t.add_column("prompt", style=C_DIM, justify="right")
    t.add_column("ttft", style=C_DIM, justify="right")
    t.add_column("prefill t/s", style=C_GREEN, justify="right")
    t.add_column("gen", style=C_DIM, justify="right")
    t.add_column("gen t/s", style=C_GREEN, justify="right")
    t.add_column("total", style=C_TEXT, justify="right")

    # Remember the fastest split this run loads at, so the next `serve`/`chat`
    # starts from a measurement instead of the VRAM formula.
    tuning = TuningStore() if cfg.get("tuning", True) else None
    stored = False

    for c in ctx_list:
        p = plan(
            m.meta,
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
                    engine = create_engine(p, m.path, backend)
                except Exception as e:
                    note(f"ctx {c} did not load: {e}", tone="error")
                    continue

            animated_loading_bar(f"ctx {c}", 0.5)
            result = _run_bench(engine, PROMPT, max_tokens)

            if "error" in result:
                note(f"ctx {c} failed during the run: {result['error']}", tone="error")
                continue

            if tuning is not None:
                # engine.plan, not the requested plan: a failed GPU load falls
                # back a rung, and that fallback is the split that ran.
                stored |= tuning.record(
                    model,
                    hw,
                    gpu_layers=engine.plan.gpu_layers,
                    ctx=engine.plan.context,
                    gen_tps=result["gen_tps"],
                )

            # Report the plan the engine actually loaded: a GPU load can fail
            # and fall back to CPU, which would otherwise still print the
            # optimistic planned layer count.
            layers_total = int(m.meta.get("n_layers") or 0) or max(engine.plan.gpu_layers, 1)

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
    note(
        "ttft is time to first token, and prefill t/s is prompt tokens over ttft.\n"
        "gen t/s is measured after the first token. a `~` on gen means the backend\n"
        "reported no usage and the count is a streamed-chunk approximation."
    )
    if stored and tuning is not None:
        note(f"saved the fastest split to {tuning.path}; serve/chat will use it.", tone="ok")
