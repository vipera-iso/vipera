#!/usr/bin/env python3
"""Measure the pure-Python loops in vipera that scale with input size.

Each entry is a per-item cost, which is what decides whether a language change
would pay: a loop costing 0.1 ms per item over 3 items is noise, and the same
0.1 ms over 150,000 items is the whole startup.

Usage:
    python3 tools/loop_survey.py            # median-3, all entries
    python3 tools/loop_survey.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

RESULTS: list[dict] = []


def bench(name: str, items: int, fn, *, repeat: int = 3) -> None:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(samples)
    RESULTS.append(
        {
            "loop": name,
            "items": items,
            "ms": round(ms, 2),
            "ms_per_item": round(ms / items * 1000.0, 3) if items else None,
            "us_per_item": round(ms / items * 1000.0, 2) if items else None,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    from vipera.gguf_analyzer import analyze_gguf
    from vipera.hardware import detect
    from vipera.optimize.prefix import prefix_key, reusable_prefix
    from vipera.optimize.tuning import apply_tuning
    from vipera.planner import plan
    from vipera.registry import get_model, list_models

    models_dir = Path.home() / ".vipera" / "models"
    model_path = Path(args.model) if args.model else None
    if model_path is None:
        candidates = sorted(models_dir.glob("*.gguf"), key=lambda p: -p.stat().st_size)
        model_path = candidates[0] if candidates else None

    if model_path and model_path.is_file():
        os.environ.pop("VIPERA_GGUF_READER", None)
        analysis = analyze_gguf(str(model_path))
        tensors = analysis["metadata"]["tensor_count"]
        bench(f"analyze_gguf index ({model_path.name})", tensors, lambda: analyze_gguf(str(model_path)))
        meta = analysis["metadata"]
    else:
        analysis = None
        meta = None

    hw = detect()
    bench("hardware.detect", 1, detect)

    aliases = [m.alias for m in list_models()]
    bench("registry.list_models", len(aliases) or 1, list_models)

    if meta is not None:
        def _plan():
            plan(
                meta,
                hw,
                profile="auto",
                ctx=2048,
                kv_cache="q8_0",
                n_batch=512,
                n_ubatch=0,
                parallel=1,
            )

        items = max(1, meta.get("n_layers") or 1)
        bench(f"planner.plan ({meta.get('architecture')}, {items} layers)", items, _plan)
        p = plan(meta, hw, profile="auto", ctx=2048, kv_cache="q8_0", n_batch=512,
                 n_ubatch=0, parallel=1)
        info = get_model(aliases[0]) if aliases else None
        if info is not None:
            bench("optimize.apply_tuning", 1, lambda: apply_tuning(p, info.alias, hw))

    for count in (20, 200, 2000):
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " * 40}
            for i in range(count)
        ]
        bench(f"optimize.prefix_key ({count} msgs)", count, lambda m=messages: prefix_key(m))
        bench(f"optimize.reusable_prefix ({count} msgs)", count, lambda m=messages: reusable_prefix(m))

    if args.json:
        print(json.dumps(RESULTS, indent=2))
        return 0

    print(f"{'loop':<52} {'items':>8} {'ms':>9} {'us/item':>9}")
    for row in RESULTS:
        per = f"{row['us_per_item']:.2f}" if row["us_per_item"] is not None else "-"
        print(f"{row['loop']:<52} {row['items']:>8} {row['ms']:>9.2f} {per:>9}")
    print("\nA loop is worth porting only when total ms is large AND us/item is flat:")
    print("I/O or C calls per item do not speed up with a different interpreter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
