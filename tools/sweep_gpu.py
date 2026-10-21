#!/usr/bin/env python3
"""Sweep `-ngl` on a real GPU and report what each layer count actually buys.

`tools/bench_layers.py` forces `--profile cpu` on purpose so that two backends
can be compared without comparing hardware instead. That leaves the GPU split
unmeasured, which is the layer with the most room on this box: the planner
picks a layer count from a VRAM formula, and the engine silently falls back to
CPU-only when the guess is too high.

This tool closes that gap. For every requested layer count it:

1. refuses to run when the loaded `llama-cpp-python` has no GPU backend — a
   CPU-only binding reports every row as "CPU", which is how a CUDA-less
   virtualenv ended up being benchmarked as if it had a card;
2. builds a Plan, asks the Engine for exactly that `-ngl`, and reports when the
   engine fell back (so a failed row is visible, not silently CPU-only);
3. samples VRAM and RSS while the model is loaded and generating, because the
   number that decides the next `-ngl` is peak VRAM, not the layer count.

Run it in the virtualenv that has the CUDA build:

    ~/.venvs/vipera/bin/python tools/sweep_gpu.py --model ... --ngl 0 4 8 10 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import psutil

from vipera.engine import Engine
from vipera.hardware import detect
from vipera.planner import plan
from vipera.registry import get_model

PROMPT = (
    "Explain, step by step, how a transformer language model turns a prompt "
    "into tokens. Cover tokenization, embeddings, attention, the feed-forward "
    "layers, the output projection and sampling. Keep the explanation concrete "
    "and avoid marketing language. "
) * 3


class _VramSampler(threading.Thread):
    """Peak VRAM/RSS while the engine is busy. Polling is cheap; a missed
    sample under-reports the peak, which only makes the verdict conservative."""

    def __init__(self, interval_s: float = 0.3):
        super().__init__(daemon=True)
        self._interval_s = interval_s
        # Named `_done`: `Thread._stop` is a method, and shadowing it breaks
        # `join()`.
        self._done = threading.Event()
        self.peak_vram_mib = 0
        self.peak_rss_mib = 0

    def run(self) -> None:
        proc = psutil.Process()
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.peak_vram_mib = max(self.peak_vram_mib, int(out.stdout.strip() or 0))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.peak_rss_mib = max(self.peak_rss_mib, proc.memory_info().rss / 1024**2)
            self._done.wait(self._interval_s)

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=5)


def _resolve(model: str) -> tuple[str, dict]:
    entry = get_model(model)
    if entry:
        return entry["path"], entry.get("meta", {})
    path = Path(model).expanduser()
    if not path.is_file():
        sys.exit(f"model not found: {model}")
    return str(path), {"n_layers": 40, "size_bytes": path.stat().st_size}


def _run(engine: Engine, max_tokens: int) -> dict:
    """Same measurement shape as `vipera bench run`: TTFT, then decode."""
    stream = engine.chat(
        [{"role": "user", "content": PROMPT}],
        stream=True,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    first_at = None
    count = 0
    t0 = time.perf_counter()
    for _chunk in stream:
        if first_at is None:
            first_at = time.perf_counter()
        count += 1
    done = time.perf_counter()
    decode_s = done - first_at if first_at else 0.0
    decode_tok = max(0, count - 1)
    return {
        "tokens": count,
        "ttft_s": (first_at - t0) if first_at else None,
        "decode_tps": (decode_tok / decode_s) if decode_s > 0 and decode_tok else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry alias or a .gguf path")
    ap.add_argument("--ngl", nargs="+", type=int, default=[0, 4, 8, 10, 12, 14, 16])
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--kv", default="f16", help="kv_cache dtype, e.g. f16 or q8_0")
    ap.add_argument("--threads", type=int, default=None, help="Override plan.cpu_threads")
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--json", help="Also write the rows here")
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run even when the binding has no GPU backend (every row is CPU-only)",
    )
    args = ap.parse_args()

    try:
        from llama_cpp import llama_supports_gpu_offload
    except ImportError:
        sys.exit("llama-cpp-python is not installed")
    if not llama_supports_gpu_offload() and not args.allow_cpu:
        sys.exit(
            "this llama-cpp-python build has no GPU backend: every row would be CPU-only. "
            "Run it in the venv that has CUDA, or pass --allow-cpu to measure the CPU fallback."
        )

    model_path, meta = _resolve(args.model)
    hw = detect()
    print(f"model : {model_path}")
    print(f"gpu   : {hw.gpu_name} ({hw.vram_free_gb:.2f} GB free of {hw.vram_total_gb:.2f})")
    print(f"ctx   : {args.ctx}  kv: {args.kv}  threads: {args.threads or 'planner'}  max_tokens: {args.max_tokens}")
    print()

    header = (
        f"{'ngl':>4} {'actual':>8} {'load_s':>8} {'TTFT_s':>8} {'decode_t/s':>11} "
        f"{'VRAM_MiB':>9} {'RSS_MiB':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for ngl in args.ngl:
        p = plan(meta, hw, profile="auto", ctx=args.ctx, kv_cache=args.kv, n_batch=args.n_batch)
        if args.threads:
            p = replace(p, cpu_threads=args.threads)
        p = replace(p, gpu_layers=ngl)

        sampler = _VramSampler()
        sampler.start()
        t0 = time.perf_counter()
        engine = None
        row = {
            "ngl_requested": ngl,
            "ctx": args.ctx,
            "kv": args.kv,
            "threads": p.cpu_threads,
            "error": None,
        }
        try:
            engine = Engine(p, model_path)
            load_s = time.perf_counter() - t0
            row["ngl_actual"] = engine.plan.gpu_layers
            row["load_s"] = round(load_s, 2)
            row.update(_run(engine, args.max_tokens))
        except Exception as exc:
            row["ngl_actual"] = engine.plan.gpu_layers if engine else None
            row["load_s"] = round(time.perf_counter() - t0, 2)
            row["error"] = f"{type(exc).__name__}: {exc}".split("\n")[0][:90]
        finally:
            if engine is not None:
                engine.close()
            sampler.stop()
            # Let the driver release the VRAM before the next row.
            time.sleep(2)

        row["vram_peak_mib"] = sampler.peak_vram_mib
        row["rss_peak_mib"] = round(sampler.peak_rss_mib)
        rows.append(row)

        if row.get("error"):
            print(f"{ngl:>4} {'FAIL':>8} {row['load_s']:>8.1f} {'-':>8} {row['error']}")
        else:
            actual = row["ngl_actual"]
            note = "" if actual == ngl else f"  (engine fell back to {actual})"
            print(
                f"{ngl:>4} {actual!s:>8} {row['load_s']:>8.1f} "
                f"{(row.get('ttft_s') or 0):>8.2f} {(row.get('decode_tps') or 0):>11.2f} "
                f"{row['vram_peak_mib']:>9} {row['rss_peak_mib']:>9}{note}"
            )

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
