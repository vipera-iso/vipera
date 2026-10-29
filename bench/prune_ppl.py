"""Measure what expert pruning costs in perplexity (S10), instead of inferring it.

`vipera prune` reports the **routing mass** a keep-set retains, and every document
that quotes it has to add the same disclaimer: mass is a necessary condition, not
a quality result. This harness removes the disclaimer's excuse. It runs
`llama-perplexity` on the *same weights* with the router restricted to a keep-set
(`bench/llama-moe-prune.patch`: a pruned expert's probability becomes exactly 0,
so the surviving weights renormalise exactly as they do in a model whose slabs
were deleted), and reports measured PPL next to the mass the plan predicted.

Two arms per size, which is what makes the result a test and not a demonstration:

* **trace-frequency** — the keep-set `vipera prune` chooses, ranked by how often
  the captured trace routed to each expert;
* **uniform** — the same number of experts per layer, drawn at random. Its
  expected retained mass is exactly `kept / n_experts`, i.e. the number the
  "pruning is cheap" story assumes you get for free.

The proxy under test is "quality follows retained mass". It is fitted on the arms
as `log(ppl_ratio) = b * (1 - mass)` and then checked **leave-one-out**: each arm
is predicted from the others, in standard errors of its own measurement. A proxy
that cannot predict an arm it did not see is not a proxy, which is why the
criterion is stated before the numbers are read.

    python3 -m bench.prune_ppl \\
        --bin ~/dev/llama.cpp/build/bin/llama-perplexity \\
        --model models/olmoe/olmoe-1b-7b-0924-q4_k_m.gguf \\
        --trace bench/results/olmoe-multitopic-545tok.jsonl \\
        --file ~/dev/llama.cpp/wikitext-2-raw/wiki.test.raw \\
        --arch arch/olmoe-1b-7b.toml --chunks 20 \\
        --fractions 0.125,0.25,0.5 --json bench/results/prune-ppl.json

Exit codes: 0 every arm measured, 1 a run failed or --json says so, 2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

from bench.perplexity import PRUNE_MODES, run_perplexity
from vipera.arch import Architecture
from vipera.prune import (
    keep_sets,
    plan_at_fraction,
    uniform_keep_sets,
    write_mask,
)
from vipera.trace import load_trace
from vipera.trace import validate as validate_trace


def arms(routing: list[list[list[int]]], *, n_experts: int, fractions: list[float],
         seed: int = 0) -> list[dict]:
    """One frequency arm and one uniform arm per keep-fraction, same size each."""
    out: list[dict] = []
    for fraction in fractions:
        plan = plan_at_fraction(routing, n_experts=n_experts, fraction=fraction)
        keep_per_layer = len(plan.layers[0].keep)
        out.append({
            "arm": f"freq@{fraction:g}",
            "policy": "trace-frequency",
            "fraction": fraction,
            "keep_per_layer": keep_per_layer,
            # measured on the trace: how much of the routing the keep-set covers
            "kept_mass": plan.mean_kept_mass,
            "keep": keep_sets(plan),
        })
        out.append({
            "arm": f"uniform@{fraction:g}",
            "policy": "uniform",
            "fraction": fraction,
            "keep_per_layer": keep_per_layer,
            # a random keep-set of this size retains this share in expectation,
            # which is exactly the number assumed by "prune to k% and keep going"
            "kept_mass": keep_per_layer / n_experts,
            "keep": uniform_keep_sets(n_layers=len(plan.layers), n_experts=n_experts,
                                      keep_per_layer=keep_per_layer, seed=seed),
        })
    return out


def fit_proxy(rows: list[dict]) -> dict:
    """Fit `log(ppl_ratio) = b * (1 - mass)` and score it leave-one-out.

    One parameter, forced through the baseline (mass 1 -> PPL ratio 1): that is
    the shape the "quality follows retained mass" claim has, so the fit measures
    the claim rather than a curve with enough freedom to absorb its failures.
    """
    usable = [r for r in rows if r.get("ppl_ratio") is not None]
    if len(usable) < 2:
        return {"ok": False, "error": "need at least two measured arms to fit"}

    def slope(sample: list[dict]) -> float:
        num = sum((1.0 - r["kept_mass"]) * math.log(r["ppl_ratio"]) for r in sample)
        den = sum((1.0 - r["kept_mass"]) ** 2 for r in sample)
        return num / den if den > 0 else 0.0

    b = slope(usable)
    loo: list[dict] = []
    for i, row in enumerate(usable):
        train = usable[:i] + usable[i + 1:]
        if not train:
            continue
        bi = slope(train)
        predicted = math.exp(bi * (1.0 - row["kept_mass"]))
        error = row["ppl_ratio"] - predicted
        # the arm's own standard error, as a ratio, is what the prediction has to
        # beat - comparing against the run-to-run drift instead would let a proxy
        # look good by being vague
        se_ratio = (row.get("ppl_stderr") or 0.0) / row["ppl"] if row["ppl"] else 0.0
        loo.append({
            "arm": row["arm"],
            "predicted_ratio": round(predicted, 4),
            "measured_ratio": round(row["ppl_ratio"], 4),
            "error_ratio": round(error, 4),
            "error_pct": round(100.0 * error, 2),
            "sigmas": round(abs(error) / (row["ppl_stderr"] / row["ppl"]), 2)
            if row.get("ppl_stderr") and row["ppl"] else None,
            "within_2_sigma": bool(se_ratio > 0
                                   and abs(error) / (row["ppl_stderr"] / row["ppl"]) <= 2.0),
        })

    worst = max(loo, key=lambda r: abs(r["error_ratio"])) if loo else None
    return {
        "ok": all(r["within_2_sigma"] for r in loo) if loo else False,
        "form": "log(ppl_ratio) = b * (1 - kept_mass)",
        "b": round(b, 4),
        "arms": loo,
        "worst_arm": worst["arm"] if worst else None,
        "worst_error_pct": worst["error_pct"] if worst else None,
        "criterion": "every leave-one-out prediction within 2 standard errors",
    }


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, so "mass orders quality" can be checked without a fit."""
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        for rank, i in enumerate(order, 1):
            out[i] = float(rank)
        return out

    if len(xs) < 2:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def run_sweep(binary: str, model: str, *, corpus: str, routing, n_experts: int,
              fractions: list[float], chunks: int | None, context: int,
              threads: int | None, seed: int = 0, timeout: float | None = None,
              workdir: str | None = None, mask_mode: str = "renorm") -> dict:
    """Baseline plus every arm, all on the same corpus/context/chunk count.

    `mask_mode` is what the mask does to the surviving weights: `renorm` (a pruned
    model), or `keep-scale` (selection blocked, weights left at their full-model
    values) - the one-parameter stand-in for healing.
    """
    baseline = run_perplexity(binary, model, corpus=corpus, chunks=chunks,
                              context=context, threads=threads, ngl=0, timeout=timeout)
    report: dict = {"baseline": baseline, "context": context, "chunks": chunks,
                    "corpus": os.path.basename(corpus), "mask_mode": mask_mode,
                    "arms": []}
    if not baseline.get("ok"):
        report["ok"] = False
        return report

    tmp = workdir or tempfile.mkdtemp(prefix="vipera-prune-ppl-")
    os.makedirs(tmp, exist_ok=True)
    rows: list[dict] = []
    for arm in arms(routing, n_experts=n_experts, fractions=fractions, seed=seed):
        path = os.path.join(tmp, arm["arm"].replace("@", "-") + ".txt")
        write_mask(path, arm["keep"])
        run = run_perplexity(binary, model, corpus=corpus, chunks=chunks,
                             context=context, threads=threads, ngl=0,
                             timeout=timeout, prune_mask=path, mask_mode=mask_mode)
        row = {
            "arm": arm["arm"],
            "policy": arm["policy"],
            "fraction": arm["fraction"],
            "keep_per_layer": arm["keep_per_layer"],
            "kept_mass": round(arm["kept_mass"], 4),
            "mask": os.path.basename(path),
            "mask_mode": mask_mode,
            "ok": bool(run.get("ok")),
            "ppl": run.get("ppl"),
            "ppl_stderr": run.get("ppl_stderr"),
            "elapsed_s": run.get("elapsed_s"),
        }
        if run.get("ok") and baseline["ppl"]:
            row["ppl_ratio"] = run["ppl"] / baseline["ppl"]
            row["ppl_delta_pct"] = round(100.0 * (run["ppl"] / baseline["ppl"] - 1.0), 2)
            # sigmas against the baseline's own standard error, quadrature summed
            se = math.sqrt((run["ppl_stderr"] or 0.0) ** 2
                           + (baseline["ppl_stderr"] or 0.0) ** 2)
            row["sigmas"] = round(abs(run["ppl"] - baseline["ppl"]) / se, 1) if se else None
        elif not run.get("ok"):
            row["error"] = run.get("error")
        rows.append(row)

    report["arms"] = rows
    measured = [r for r in rows if r.get("ppl_ratio") is not None]
    report["fit"] = fit_proxy(measured)
    if len(measured) >= 2:
        report["spearman_mass_vs_ppl"] = round(
            spearman([r["kept_mass"] for r in measured], [r["ppl"] for r in measured]), 3)
    report["ok"] = all(r["ok"] for r in rows) and bool(rows)
    return report


def format_report(report: dict) -> str:
    lines: list[str] = []
    baseline = report.get("baseline", {})
    if not baseline.get("ok"):
        return f"baseline FAILED: {baseline.get('error') or baseline.get('exit_code')}"
    lines.append(f"corpus        {report['corpus']}  ({report['chunks']} chunks, "
                 f"ctx {report['context']})")
    lines.append(f"baseline      PPL {baseline['ppl']:.4f} +/- {baseline['ppl_stderr']:.5f}")
    if report.get("mask_mode"):
        lines.append(f"mask mode     {report['mask_mode']}")
    lines.append("")
    lines.append(f"{'arm':<16}{'kept':>6}{'mass':>8}{'PPL':>12}{'vs base':>10}{'sigma':>8}")
    for row in report["arms"]:
        if not row["ok"]:
            lines.append(f"{row['arm']:<16}{row['keep_per_layer']:>6}"
                         f"{row['kept_mass']*100:>7.1f}%   FAILED  {row.get('error')}")
            continue
        lines.append(f"{row['arm']:<16}{row['keep_per_layer']:>6}"
                     f"{row['kept_mass']*100:>7.1f}%{row['ppl']:>12.4f}"
                     f"{row['ppl_delta_pct']:>+9.1f}%{row['sigmas'] if row['sigmas'] is not None else '-':>8}")
    fit = report.get("fit", {})
    if fit.get("ok") is not None:
        lines.append("")
        lines.append(f"proxy         {fit.get('form')}   b = {fit.get('b')}")
        for row in fit.get("arms", []):
            lines.append(f"  {row['arm']:<16}predicted {row['predicted_ratio']:.4f}  "
                         f"measured {row['measured_ratio']:.4f}  "
                         f"error {row['error_pct']:+.1f}%  {row['sigmas']} sigma  "
                         f"{'ok' if row['within_2_sigma'] else 'MISS'}")
        lines.append(f"verdict       {'mass predicts the measured PPL' if fit.get('ok') else 'mass does NOT predict the measured PPL'}"
                     f"  ({fit.get('criterion')})")
        if report.get("spearman_mass_vs_ppl") is not None:
            lines.append(f"rank          mass vs PPL spearman = {report['spearman_mass_vs_ppl']:+.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prune_ppl", description=__doc__)
    p.add_argument("--bin", default="llama-perplexity")
    p.add_argument("--model", required=True)
    p.add_argument("--trace", required=True, help="captured routing trace (JSONL)")
    p.add_argument("--file", required=True, dest="corpus", help="text corpus")
    p.add_argument("--arch", default="arch/olmoe-1b-7b.toml",
                   help="architecture profile for the expert count")
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--fractions", default="0.125,0.25,0.5",
                   help="comma-separated keep-fractions (share of the pool per layer)")
    p.add_argument("--seed", type=int, default=0, help="seed for the uniform arm")
    p.add_argument("--mask-mode", choices=list(PRUNE_MODES), default="renorm",
                   help="renorm: the survivors renormalise (a pruned model). "
                        "keep-scale: block selection only, keep the full-model weights "
                        "- the cheap stand-in for healing")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--workdir", default=None, help="where the mask files are written")
    p.add_argument("--json", dest="json_path")
    args = p.parse_args(argv)

    try:
        fractions = [float(f) for f in args.fractions.split(",") if f.strip()]
    except ValueError:
        print("--fractions must be a comma-separated list of numbers", file=sys.stderr)
        return 2
    if not fractions or not all(0.0 < f <= 1.0 for f in fractions):
        print("--fractions values must be in (0, 1]", file=sys.stderr)
        return 2

    arch = Architecture.from_toml(args.arch)
    routing = load_trace(args.trace)
    validate_trace(routing, n_layers=arch.moe_layers, n_experts=arch.n_experts,
                   top_k=arch.top_k)

    report = run_sweep(args.bin, args.model, corpus=args.corpus, routing=routing,
                       n_experts=arch.n_experts, fractions=fractions, chunks=args.chunks,
                       context=args.context, threads=args.threads, seed=args.seed,
                       timeout=args.timeout, workdir=args.workdir,
                       mask_mode=args.mask_mode)
    print(format_report(report))

    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"report written to {args.json_path}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
