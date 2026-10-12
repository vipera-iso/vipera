#!/usr/bin/env python3
"""Collapse N bench reports of the same configuration into one median report.

`python3 -m bench.bench` measures ONE run. One run is not a result: this laptop
thermally throttles, and the repo's own hardware probe repeated twice already
shows 6.6% spread on read bandwidth (bench/results/hwprobe-run1 vs run2). So the
measurement step of the loop is: run --repeat times, take the median, and report
the spread so a later comparison can tell a real change from machine noise.

It also refuses to mix reports that are not comparable (different model or
different llama-bench command) — merging them would manufacture a result.

    python3 skills/expert-io-loop/scripts/aggregate.py run1.json run2.json run3.json
    python3 skills/expert-io-loop/scripts/aggregate.py --json run*.json > median.json

Exit codes: 0 aggregated · 1 at least one input was not ok · 2 bad usage or
incomparable inputs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Default noise band, justified by bench/results/hwprobe-run{1,2}.json: the same
# probe run twice on this machine differed by 6.6% on read bandwidth, 1.9% on
# copy, <1% on GEMV. Anything under ~7% on this box is machine jitter.
DEFAULT_NOISE = 0.07


def load_report(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: not a readable JSON report ({e})") from e
    if not isinstance(report, dict) or "results" not in report:
        raise ValueError(f"{path}: not a bench report (no 'results')")
    return report


def check_comparable(reports: list[dict], paths: list[str]) -> str | None:
    """Reports may only be merged if they measured the same model and command."""
    models = {str(r.get("model_filename")) for r in reports}
    if len(models) > 1:
        return "different model_filename: " + ", ".join(sorted(models))
    commands = {tuple(r.get("command") or []) for r in reports}
    if len(commands) > 1:
        detail = []
        for path, report in zip(paths, reports):
            detail.append(f"{path}: {' '.join(report.get('command') or [])}")
        return "different llama-bench command:\n  " + "\n  ".join(detail)
    return None


def median(values: list[float | int | None]) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return float(statistics.median(clean))


def spread_pct(values: list[float | int | None]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if len(clean) < 2:
        return None
    mid = statistics.median(clean)
    if mid == 0:
        return None
    return (max(clean) - min(clean)) / mid * 100.0


def aggregate(reports: list[dict], paths: list[str], noise: float) -> dict:
    labels: list[str] = []
    for report in reports:
        for row in report.get("results", []):
            label = str(row.get("test") or "unknown")
            if label not in labels:
                labels.append(label)

    tests = []
    worst = 0.0
    for label in labels:
        rows = [
            row for report in reports
            for row in report.get("results", [])
            if str(row.get("test") or "unknown") == label
        ]
        avg = [row.get("avg_ts") for row in rows]
        dev = [row.get("stddev_ts") for row in rows]
        spread = spread_pct(avg)
        if spread is not None:
            worst = max(worst, spread)
        tests.append({
            "test": label,
            "runs": len(rows),
            "avg_ts_median": median(avg),
            "avg_ts_min": min((v for v in avg if isinstance(v, (int, float))), default=None),
            "avg_ts_max": max((v for v in avg if isinstance(v, (int, float))), default=None),
            "spread_pct": round(spread, 2) if spread is not None else None,
            "stddev_ts_median": median(dev),
        })

    return {
        "ok": all(r.get("ok") for r in reports),
        "runs": len(reports),
        "sources": list(paths),
        "model_filename": reports[0].get("model_filename"),
        "command": reports[0].get("command"),
        "noise_fraction": noise,
        "tests": tests,
        "max_spread_pct": round(worst, 2),
        "within_noise": worst <= noise * 100.0,
        "bytes_read_per_token_median": median([r.get("bytes_read_per_token") for r in reports]),
        "read_bytes_median": median([(r.get("io") or {}).get("read_bytes") for r in reports]),
        "peak_rss_kb_median": median([r.get("peak_rss_kb") for r in reports]),
        "peak_gpu_mib_median": median([r.get("peak_gpu_mib") for r in reports]),
        "elapsed_s_median": median([r.get("elapsed_s") for r in reports]),
    }


def format_table(agg: dict) -> str:
    lines = [
        f"runs         {agg['runs']}  (all ok: {agg['ok']})",
        f"model        {agg.get('model_filename')}",
        f"spread       max {agg['max_spread_pct']}%  within noise "
        f"({agg['noise_fraction'] * 100:.0f}%): {agg['within_noise']}",
    ]
    bpt = agg.get("bytes_read_per_token_median")
    if bpt is not None:
        lines.append(f"read/token   {bpt / (1024 * 1024):.3f} MiB  (median)")
    if agg.get("peak_rss_kb_median") is not None:
        lines.append(f"peak RSS     {agg['peak_rss_kb_median'] / 1024:.1f} MiB (median)")
    if agg.get("peak_gpu_mib_median") is not None:
        lines.append(f"peak GPU     {agg['peak_gpu_mib_median']} MiB — CPU-only build, not a VRAM figure")
    for t in agg["tests"]:
        ts = t["avg_ts_median"]
        ts_text = f"{ts:.2f} t/s" if isinstance(ts, float) else "n/a"
        spread = f"{t['spread_pct']}%" if t["spread_pct"] is not None else "n/a"
        lines.append(f"  {t['test']:<12} {ts_text:>10}  min {t['avg_ts_min']}  max {t['avg_ts_max']}  spread {spread}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aggregate", description=__doc__)
    p.add_argument("reports", nargs="+", help="bench JSON reports from the same configuration")
    p.add_argument("--noise", type=float, default=DEFAULT_NOISE,
                   help=f"noise band as a fraction (default {DEFAULT_NOISE}, from hwprobe repeat spread)")
    p.add_argument("--json", action="store_true", help="print the aggregate as JSON")
    args = p.parse_args(argv)

    if len(args.reports) < 2:
        p.error("need at least two reports to aggregate (one run cannot show its own noise)")

    paths = [str(Path(r)) for r in args.reports]
    try:
        reports = [load_report(r) for r in args.reports]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    problem = check_comparable(reports, paths)
    if problem:
        print(f"error: reports are not comparable: {problem}", file=sys.stderr)
        return 2

    agg = aggregate(reports, paths, args.noise)
    if args.json:
        print(json.dumps(agg, indent=2))
    else:
        print(format_table(agg))
    return 0 if agg["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
