#!/usr/bin/env python3
"""Compare a candidate measurement against a baseline and return a verdict.

This is the re-measure step of the loop. It answers one question only: is the
difference bigger than machine noise, and does it survive the quality gate?

Inputs may be a raw `python3 -m bench.bench --json` report or the output of
`aggregate.py` (the median of --repeat runs; prefer that one). Perplexity comes
from `python3 -m bench.perplexity --json`, because a speedup bought with quality
is not a win (docs/OPTIMIZER.md §6: ppl rise ≤ 0.3 or the transform stays off).

    python3 skills/expert-io-loop/scripts/compare.py base.json cand.json
    python3 skills/expert-io-loop/scripts/compare.py base.json cand.json \
        --ppl-base 8.6426 --ppl-cand 8.7011 --json

Exit codes ARE the verdict: 0 improved · 1 inconclusive · 2 regressed ·
3 usage, parse or comparability error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Same default as aggregate.py, sourced from bench/results/hwprobe-run{1,2}.json
# (6.6% spread on read bandwidth between two identical probe runs on this box).
DEFAULT_NOISE = 0.07
# docs/OPTIMIZER.md §6 quality gate.
DEFAULT_MAX_PPL_RISE = 0.3

EXIT_IMPROVED, EXIT_INCONCLUSIVE, EXIT_REGRESSED, EXIT_USAGE = 0, 1, 2, 3


def load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: not readable JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def normalise(report: dict) -> dict:
    """Accept a raw bench report or an aggregate; return one shape."""
    if "tests" in report:  # aggregate.py output
        tests = {
            str(t.get("test") or "unknown"): t.get("avg_ts_median")
            for t in report.get("tests", [])
        }
        return {
            "shape": "aggregate",
            "ok": bool(report.get("ok")),
            "tests": tests,
            "bytes_read_per_token": report.get("bytes_read_per_token_median"),
            "peak_rss_kb": report.get("peak_rss_kb_median"),
            "model_filename": report.get("model_filename"),
            "command": report.get("command"),
            "runs": report.get("runs"),
        }
    if "results" in report:  # bench/bench.py output
        return {
            "shape": "bench",
            "ok": bool(report.get("ok", True)),
            "tests": {
                str(r.get("test") or "unknown"): r.get("avg_ts")
                for r in report.get("results", [])
            },
            "bytes_read_per_token": report.get("bytes_read_per_token"),
            "peak_rss_kb": report.get("peak_rss_kb"),
            "model_filename": report.get("model_filename"),
            "command": report.get("command"),
            "runs": 1,
        }
    raise ValueError("neither a bench report ('results') nor an aggregate ('tests')")


def delta_pct(base: float | None, cand: float | None) -> float | None:
    """Signed delta in percent, positive = candidate is bigger."""
    if not isinstance(base, (int, float)) or not isinstance(cand, (int, float)):
        return None
    if base == 0:
        return None
    return (cand - base) / base * 100.0


def compare(base: dict, cand: dict, *, noise: float, test: str | None,
            primary: str, ppl_base: float | None, ppl_cand: float | None,
            max_ppl_rise: float) -> dict:
    band = noise * 100.0
    notes: list[str] = []

    labels = sorted(set(base["tests"]) & set(cand["tests"]))
    if test:
        if test not in labels:
            return {"verdict": "inconclusive", "reason": f"test {test!r} not in both reports",
                    "available": labels}
        labels = [test]
    if not labels:
        return {"verdict": "inconclusive", "reason": "no matching test labels between the two reports"}

    if base.get("model_filename") != cand.get("model_filename"):
        notes.append(f"DIFFERENT MODEL: {base.get('model_filename')} vs {cand.get('model_filename')}")
    if (base.get("command") or []) != (cand.get("command") or []):
        notes.append("commands differ — expected if the change WAS a runtime flag; "
                     "otherwise you changed two things")

    per_test = []
    deltas = []
    for label in labels:
        d = delta_pct(base["tests"].get(label), cand["tests"].get(label))
        if d is not None:
            deltas.append(d)
        per_test.append({
            "test": label,
            "base_avg_ts": base["tests"].get(label),
            "cand_avg_ts": cand["tests"].get(label),
            "delta_pct": round(d, 2) if d is not None else None,
        })
    # One headline number: the median across tests, so a prefill/decode pair
    # cannot be cherry-picked in either direction.
    primary_delta = sorted(deltas)[len(deltas) // 2] if deltas else None

    io_delta = delta_pct(base.get("bytes_read_per_token"), cand.get("bytes_read_per_token"))
    rss_delta = delta_pct(base.get("peak_rss_kb"), cand.get("peak_rss_kb"))

    # `avg_ts` wants to go up; `read_per_token` wants to go down.
    higher_is_better = primary == "avg_ts"
    signal = primary_delta if higher_is_better else io_delta
    if signal is None:
        notes.append(f"primary metric {primary} unavailable — falling back to the other one")
        signal = io_delta if higher_is_better else primary_delta
        higher_is_better = not higher_is_better
    if signal is None:
        return {"verdict": "inconclusive", "reason": "neither avg_ts nor read/token present",
                "notes": notes}

    favors = signal if higher_is_better else -signal
    if favors > band:
        verdict = "improved"
    elif favors < -band:
        verdict = "regressed"
    else:
        verdict = "inconclusive"

    # Secondary metrics may not give back what the primary gained.
    if verdict == "improved" and io_delta is not None and io_delta > band:
        verdict = "regressed"
        notes.append(f"read/token rose {io_delta:.1f}% > noise {band:.0f}% — the I/O got worse")
    if verdict == "improved" and rss_delta is not None and rss_delta > band:
        verdict = "regressed"
        notes.append(f"peak RSS rose {rss_delta:.1f}% > noise {band:.0f}% — it no longer fits")

    quality = None
    if ppl_base is not None and ppl_cand is not None:
        rise = delta_pct(ppl_base, ppl_cand)
        quality = {
            "ppl_base": ppl_base,
            "ppl_cand": ppl_cand,
            "rise_pct": round(rise, 2) if rise is not None else None,
            "max_ppl_rise": max_ppl_rise,
            "absolute_rise": round(ppl_cand - ppl_base, 4),
        }
        if ppl_cand - ppl_base > max_ppl_rise:
            verdict = "regressed"
            notes.append(f"quality gate failed: ppl {ppl_base} -> {ppl_cand} "
                         f"(rise > {max_ppl_rise})")
    elif verdict == "improved":
        notes.append("no perplexity supplied — speed claim is UNVALIDATED")

    return {
        "verdict": verdict,
        "band_pct": round(band, 2),
        "primary": primary,
        "primary_delta_pct": round(primary_delta, 2) if primary_delta is not None else None,
        "signal_delta_pct": round(signal, 2),
        "favorable": favors > band,
        "tests": per_test,
        "read_per_token_delta_pct": round(io_delta, 2) if io_delta is not None else None,
        "peak_rss_delta_pct": round(rss_delta, 2) if rss_delta is not None else None,
        "quality": quality,
        "runs": {"baseline": base.get("runs"), "candidate": cand.get("runs")},
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compare", description=__doc__)
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--noise", type=float, default=DEFAULT_NOISE,
                   help=f"noise band as a fraction (default {DEFAULT_NOISE})")
    p.add_argument("--test", help="compare only this test label (e.g. tg128)")
    p.add_argument("--primary", choices=["avg_ts", "read_per_token"], default="avg_ts",
                   help="metric that decides the verdict (default avg_ts)")
    p.add_argument("--ppl-base", type=float, help="perplexity of the baseline")
    p.add_argument("--ppl-cand", type=float, help="perplexity of the candidate")
    p.add_argument("--max-ppl-rise", type=float, default=DEFAULT_MAX_PPL_RISE,
                   help=f"quality gate, absolute ppl rise (default {DEFAULT_MAX_PPL_RISE})")
    p.add_argument("--strict-command", action="store_true",
                   help="treat a differing llama-bench command as a comparability error")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        base = normalise(load(args.baseline))
        cand = normalise(load(args.candidate))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE

    if not base["ok"] or not cand["ok"]:
        print("error: one of the reports is a failed run (ok=false)", file=sys.stderr)
        return EXIT_USAGE
    if base.get("model_filename") != cand.get("model_filename"):
        print(f"error: different models: {base.get('model_filename')} vs "
              f"{cand.get('model_filename')}", file=sys.stderr)
        return EXIT_USAGE
    if args.strict_command and (base.get("command") or []) != (cand.get("command") or []):
        print("error: commands differ and --strict-command was given", file=sys.stderr)
        return EXIT_USAGE

    result = compare(base, cand, noise=args.noise, test=args.test, primary=args.primary,
                     ppl_base=args.ppl_base, ppl_cand=args.ppl_cand,
                     max_ppl_rise=args.max_ppl_rise)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"verdict      {result['verdict'].upper()}  (noise band {result.get('band_pct')}%)")
        for t in result.get("tests", []):
            base_ts, cand_ts = t.get("base_avg_ts"), t.get("cand_avg_ts")
            base_text = f"{base_ts:.2f}" if isinstance(base_ts, (int, float)) else "n/a"
            cand_text = f"{cand_ts:.2f}" if isinstance(cand_ts, (int, float)) else "n/a"
            print(f"  {t['test']:<12} {base_text} -> {cand_text} t/s ({t['delta_pct']}%)")
        if result.get("read_per_token_delta_pct") is not None:
            print(f"read/token   {result['read_per_token_delta_pct']}%")
        if result.get("peak_rss_delta_pct") is not None:
            print(f"peak RSS     {result['peak_rss_delta_pct']}%")
        if result.get("quality"):
            q = result["quality"]
            print(f"ppl          {q['ppl_base']} -> {q['ppl_cand']} "
                  f"(gate: rise <= {q['max_ppl_rise']})")
        for note in result.get("notes", []):
            print(f"note         {note}")
        if result.get("reason"):
            print(f"reason       {result['reason']}")

    return {"improved": EXIT_IMPROVED, "inconclusive": EXIT_INCONCLUSIVE,
            "regressed": EXIT_REGRESSED}.get(result["verdict"], EXIT_USAGE)


if __name__ == "__main__":
    raise SystemExit(main())
