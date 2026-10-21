#!/usr/bin/env python3
"""Measure what the optimization layers actually change.

Two things are easy to claim and hard to see in a unit test:

1. KV cache dtype and flash attention. Quantized KV needs flash attention, but
   on CPU the flash-attention path is much slower than the default one, so the
   planner only enables it for dtypes that require it. This script runs the same
   model/context across cache types and an FA override, so the cost of asking
   for the slow path is a number instead of an assumption.
2. The `inprocess` vs `server` backend. Same plan, same model, same prompt, one
   child process versus the Python binding.

Every row is a fresh engine, so load time is included and shown separately.
`--profile cpu` is forced: mixing a CUDA build (in-process wheel) with a CPU
build (local llama-server) would compare hardware, not backends.

Usage:
    .venv/bin/python tools/bench_layers.py \
        --model qwen2.5-0.5b-instruct-q4_k_m --ctx 4096 --max-tokens 64 \
        --backend inprocess server --kv f16 q8_0 q5_1 iq4_nl --fa auto on
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from vipera.commands.bench_cmd import _run_bench
from vipera.engine import create_engine
from vipera.hardware import detect
from vipera.planner import plan
from vipera.registry import get_model

PROMPT = (
    "Explain, step by step, how a transformer language model turns a prompt "
    "into tokens. Cover tokenization, embeddings, attention, the feed-forward "
    "layers, the output projection and sampling. Keep the explanation concrete "
    "and avoid marketing language. "
) * 3


def _resolve(model: str) -> tuple[str, dict]:
    entry = get_model(model)
    if entry:
        return entry["path"], entry.get("meta", {})
    path = Path(model).expanduser()
    if not path.is_file():
        sys.exit(f"model not found: {model}")
    return str(path), {"n_layers": 0, "size_bytes": path.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry alias or a .gguf path")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--n-batch", type=int, default=256)
    ap.add_argument("--backend", nargs="+", default=["inprocess", "server"])
    ap.add_argument(
        "--profile",
        default="cpu",
        choices=["cpu", "auto", "gpu", "lowmem"],
        help="Planner profile; cpu makes backends comparable, auto measures the GPU split",
    )
    ap.add_argument("--kv", nargs="+", default=["f16", "q8_0"])
    ap.add_argument(
        "--fa",
        nargs="+",
        default=["auto"],
        choices=["auto", "on", "off"],
        help="auto = whatever the planner decides; on/off = override for the test",
    )
    ap.add_argument(
        "--prompt-repeat",
        type=int,
        default=3,
        help="Times the prompt is repeated; raise it to measure long sequences",
    )
    ap.add_argument("--json", help="Also write the rows here")
    args = ap.parse_args()

    model_path, meta = _resolve(args.model)
    hw = detect()
    rows = []

    print(f"model : {model_path}")
    print(f"meta  : layers={meta.get('n_layers')} quant={meta.get('quant')}")
    print(f"hw    : {hw.cpu_cores}C/{hw.cpu_threads}T  {hw.ram_free_gb:.1f} GB free")
    print(
        f"ctx   : {args.ctx}  max_tokens: {args.max_tokens}  n_batch: {args.n_batch}  "
        f"profile: {args.profile}  backends: {', '.join(args.backend)}"
    )
    print()

    for backend in args.backend:
        for kv in args.kv:
            for fa in args.fa:
                p = plan(
                    meta,
                    hw,
                    profile=args.profile,
                    ctx=args.ctx,
                    kv_cache=kv,
                    n_batch=args.n_batch,
                )
                if fa != "auto":
                    p = replace(p, flash_attn=fa == "on")

                row = {
                    "backend": backend,
                    "kv": kv,
                    "fa_requested": fa,
                    "fa_effective": p.flash_attn,
                    "load_s": None,
                    "gen_tps": None,
                    "ttft_ms": None,
                    "prompt_tok": None,
                    "error": None,
                }
                engine = None
                try:
                    t0 = time.perf_counter()
                    engine = create_engine(p, model_path, backend)
                    row["load_s"] = round(time.perf_counter() - t0, 2)
                    row["fa_effective"] = engine.plan.flash_attn
                    row["gpu_layers"] = engine.plan.gpu_layers
                    result = _run_bench(engine, PROMPT * args.prompt_repeat, args.max_tokens)
                    if "error" in result:
                        row["error"] = result["error"]
                    else:
                        row.update(
                            gen_tps=round(result["gen_tps"], 2),
                            ttft_ms=round(result["ttft_ms"]),
                            prompt_tok=result["prompt_tok"],
                            gen_tok=result["gen_tok"],
                            exact_tokens=result["token_count_is_exact"],
                        )
                except Exception as exc:  # a bad KV type should be one failed row
                    row["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    if engine is not None:
                        engine.close()
                rows.append(row)

    print(
        f"{'backend':<10}{'kv':<8}{'fa':<7}{'eff':<5}{'ngl':>4}{'load s':>7}"
        f"{'TTFT ms':>9}{'gen t/s':>9}{'tok':>5}  status"
    )
    for r in rows:
        status = "ok" if r["error"] is None else r["error"][:60]
        gen = "-" if r["gen_tps"] is None else f"{r['gen_tps']:.2f}"
        tok = "-" if r.get("gen_tok") is None else str(r["gen_tok"])
        ngl = r.get("gpu_layers", "-")
        print(
            f"{r['backend']:<10}{r['kv']:<8}{r['fa_requested']:<7}"
            f"{'on' if r['fa_effective'] else 'off':<5}{ngl:>4}"
            f"{(r['load_s'] or 0):>7.2f}{(r['ttft_ms'] or 0):>9}"
            f"{gen:>9}{tok:>5}  {status}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
