#!/usr/bin/env python3
"""Stage-by-stage measurement of the vipera request pipeline.

Every stage reports: measured cost, what it is bound by, and the physical floor
we could reach if the code were free. That separation is the point -- a stage
sitting at 90% of its roofline cannot be improved by rewriting it in Rust, and a
stage sitting at 5% of its floor is a real target.

The GPU stages (prefill, decode) are not run here: they need a served model.
Their numbers are read from `tools/stage_gpu.json` when present, which the
llama-bench run writes, and the roofline is computed from the model size and the
memory bandwidth of the card.

Usage:
    python3 tools/pipeline_audit.py --model qwen2.5-1.5b-instruct-q4_k_m
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_JSON = ROOT / "tools" / "stage_gpu.json"


def median_ms(fn, *, repeat: int = 5) -> float:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def subprocess_ms(args: list[str], *, repeat: int = 5) -> float:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def read_gpu_numbers() -> dict:
    if not GPU_JSON.exists():
        return {}
    try:
        return json.loads(GPU_JSON.read_text())
    except (OSError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-1.5b-instruct-q4_k_m")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from vipera.config import load_config
    from vipera.gguf_analyzer import analyze_gguf
    from vipera.hardware import detect
    from vipera.planner import plan
    from vipera.registry import get_model

    info = get_model(args.model)
    if info is None:
        print(f"model not found: {args.model}")
        return 1

    rows: list[dict] = []

    def add(stage: str, ms: float, bound: str, floor_ms: float | None, note: str = "") -> None:
        rows.append(
            {
                "stage": stage,
                "ms": round(ms, 2),
                "bound": bound,
                "floor_ms": None if floor_ms is None else round(floor_ms, 2),
                "headroom": (
                    None if floor_ms is None or floor_ms <= 0 else round(ms / floor_ms, 2)
                ),
                "note": note,
            }
        )

    py = sys.executable
    add(
        "process start (bare python)",
        subprocess_ms([py, "-c", "pass"]),
        "interpreter",
        None,
        "floor for anything the CLI does",
    )
    add(
        "CLI: import + parse",
        subprocess_ms([py, "-m", "vipera", "serve", "--help"]),
        "imports (typer, rich)",
        subprocess_ms([py, "-c", "pass"]),
        "above the bare interpreter; typer+rich are ~26+16 ms of it",
    )
    cfg = median_ms(load_config)
    add("config load", cfg, "disk", None)
    add("registry resolve", median_ms(lambda: get_model(args.model)), "stat", None)
    add(
        "GGUF index parse",
        median_ms(lambda: analyze_gguf(info.path)),
        "python/rust parse",
        None,
        "student: was 4138 ms through the gguf package",
    )
    add(
        "hardware probe", median_ms(detect), "NVML driver", None, "~7 ms is the first device handle"
    )
    hw = detect()
    p = plan(
        info.meta,
        hw,
        profile="auto",
        ctx=2048,
        kv_cache="q8_0",
        n_batch=512,
        n_ubatch=0,
        parallel=1,
    )
    add(
        "plan",
        median_ms(
            lambda: plan(
                info.meta,
                hw,
                profile="auto",
                ctx=2048,
                kv_cache="q8_0",
                n_batch=512,
                n_ubatch=0,
                parallel=1,
            )
        ),
        "pure math",
        None,
    )

    gpu = read_gpu_numbers()
    # The achievable number beats the spec sheet: a card never streams at its
    # theoretical figure, and the gap between the two is not ours to close.
    bandwidth = float(gpu.get("bandwidth_achievable_gb_s") or gpu.get("bandwidth_gb_s") or 0.0)
    model_gb = info.meta.get("size_bytes", 0) / 1e9
    if gpu.get("decode_t_s") and bandwidth and model_gb:
        floor = bandwidth / model_gb
        add(
            "decode (tokens/s)",
            gpu["decode_t_s"],
            "memory bandwidth (measured read)",
            floor,
            f"{gpu['decode_t_s']:.1f} t/s vs roofline {floor:.1f} t/s",
        )
    if gpu.get("prefill_t_s") and gpu.get("flops_tflops") and model_gb:
        # Prefill is compute bound: 2 * params FLOPs per token.
        params = info.meta.get("n_params") or gpu.get("n_params") or 0
        if params:
            floor = gpu["flops_tflops"] * 1e12 / (2 * params)
            add(
                "prefill (tokens/s)",
                gpu["prefill_t_s"],
                "compute (FLOPs)",
                floor,
                f"{gpu['prefill_t_s']:.1f} t/s vs roofline {floor:.1f} t/s",
            )
    if gpu.get("gateway_overhead_pct") is not None:
        add(
            "gateway overhead (% of wall)",
            gpu["gateway_overhead_pct"],
            "socket relay",
            gpu.get("gateway_floor_pct"),
            "passthrough floor is the same bytes through one more process",
        )

    model_bytes = info.meta.get("size_bytes", 0)
    add(
        "model load (GB/s from disk)",
        0.0,
        "disk + VRAM copy",
        None,
        f"{model_bytes / 1e9:.2f} GB to move" if model_bytes else "",
    )

    if args.json:
        print(json.dumps({"rows": rows, "plan_ctx": p.context}, indent=2))
        return 0

    print(f"{'stage':<30} {'ms':>10} {'bound by':<22} {'headroom':>9}")
    for row in rows:
        head = "-" if row["headroom"] is None else f"{row['headroom']:.2f}x"
        print(f"{row['stage']:<30} {row['ms']:>10.2f} {row['bound']:<22} {head:>9}")
        if row["note"]:
            print(f"{'':<30} {row['note']}")
    print("\nheadroom = measured / physical floor. > 2x is a real target;")
    print("<= 1.2x means the code is already at the wall and only the hardware moves it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
