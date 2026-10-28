"""Diagnostic: read a MoE routing trace and report whether adaptive
per-expert bit allocation has room to help.

This is a bit-budget simulation, NOT a measurement. Savings here are an
upper bound; bits saved != perplexity preserved. See docs/REVIEW.md.

stdlib only by design (vipera adds no deps for diagnostics).

Usage:
    python3 bench/analyze_trace.py bench/results/olmoe.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys

# Candidate key names, tried in order. Read bench/capture_trace.py if none
# of these match; do not guess silently.
LAYER_KEYS = ("layer", "layer_id", "block_idx", "block_index")
WEIGHT_KEYS = ("topk_weights", "weights", "router_weights")
ID_KEYS = ("topk_ids", "expert_ids", "indices", "expert_indices")

BASE_BITS = 4
GAMMAS = (0.25, 0.5, 0.75, 1.0)
MASS_TARGET = 0.95


def _first(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def inspect(path: str) -> dict | None:
    """Print the first non-empty line so the caller can verify keys."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    print("--- SAMPLE TRACE LINE ---")
                    print(json.dumps(sample, indent=2))
                    print("--- KEYS ---")
                    print(list(sample.keys()))
                    print()
                    return sample
    except FileNotFoundError:
        print(f"[!] file not found: {path}")
    return None


def load(path: str) -> dict:
    """Return {layer_id: {'tokens': [pairs, ...], 'all_have_ids': bool}}."""
    per_layer: dict = {}
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            layer = _first(d, LAYER_KEYS)
            if layer is None:
                layer = 0

            weights = _first(d, WEIGHT_KEYS)
            if not weights:
                continue

            ids = _first(d, ID_KEYS)
            has_ids = bool(ids) and len(ids) == len(weights)

            if layer not in per_layer:
                per_layer[layer] = {"tokens": [], "all_have_ids": True}

            if has_ids:
                pairs = list(zip(ids, weights))
            else:
                per_layer[layer]["all_have_ids"] = False
                pairs = list(enumerate(weights))

            per_layer[layer]["tokens"].append(pairs)
            total += 1

    print(f"[*] loaded {total} tokens across {len(per_layer)} layers")
    return per_layer


def layer_sort_key(item):
    k = item[0]
    if isinstance(k, int):
        return (0, k)
    if str(k).isdigit():
        return (0, int(k))
    return (1, str(k))


def global_prune(tokens: list, k: int) -> tuple[int, bool]:
    """Experts needed to retain MASS_TARGET of routing mass across the layer.

    Uses expert identity, not rank. Returns (k_global, ids_complete).
    """
    mass: dict = {}
    for pairs in tokens:
        s = sum(w for _, w in pairs)
        if s <= 0:
            continue
        for exp_id, w in pairs:
            mass[exp_id] = mass.get(exp_id, 0.0) + (w / s)

    if not mass:
        return k, False

    total = sum(mass.values())
    ordered = sorted(mass.values(), reverse=True)

    cum = 0.0
    k_global = 0
    for m in ordered:
        cum += m / total
        k_global += 1
        if cum >= MASS_TARGET:
            break
    return k_global, True


def adaptive_savings(tokens: list, gamma: float) -> list[float]:
    """Per-token savings for head-preserving gamma scaling."""
    out = []
    baseline = len(tokens[0]) * BASE_BITS
    for pairs in tokens:
        ws = [w for _, w in pairs]
        s = sum(ws)
        if s <= 0:
            continue
        wn = [w / s for w in ws]
        w_max = max(wn) or 1.0
        cost = 0
        for x in wn:
            raw = ((x / w_max) ** gamma) * BASE_BITS
            bits = int(raw + 0.5)  # half-up, avoids banker's rounding
            bits = max(1, min(BASE_BITS, bits))
            cost += bits
        out.append(1 - cost / baseline)
    return out


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    i = int(len(sorted_vals) * p)
    i = max(0, min(len(sorted_vals) - 1, i))
    return sorted_vals[i]


def report(per_layer: dict) -> None:
    print("=" * 72)
    print(" BIT-BUDGET SIMULATION (UPPER BOUND, NOT A MEASUREMENT)")
    print("=" * 72)

    for layer, data in sorted(per_layer.items(), key=layer_sort_key):
        tokens = data["tokens"]
        if not tokens:
            continue
        k = len(tokens[0])
        ids_ok = data["all_have_ids"]

        print(f"\n[Layer {layer}] top-k={k} tokens={len(tokens)} "
              f"expert_ids={'yes' if ids_ok else 'no'}")

        # --- prune baseline ---
        if ids_ok:
            k_global, _ = global_prune(tokens, k)
            if k_global >= k:
                print(f"  prune: {k_global} experts needed for "
                      f"{MASS_TARGET:.0%} mass (>= top-k)")
                print(f"         per-token load UNCHANGED; helps file size "
                      f"and cache locality, NOT bytes/token")
            else:
                saved = 1 - k_global / k
                print(f"  prune: {k_global}/{k} experts -> "
                      f"per-token savings {saved:.1%}")
        else:
            print("  prune: SKIPPED (trace lacks expert ids; "
                  "rank-frequency would be wrong)")

        # --- adaptive sweep ---
        best_med = 0.0
        for g in GAMMAS:
            vals = sorted(adaptive_savings(tokens, g))
            if not vals:
                continue
            med = statistics.median(vals)
            p10 = pct(vals, 0.10)
            p90 = pct(vals, 0.90)
            spread = p90 - p10
            tag = "spread>30% (layout hard to fix)" if spread > 0.30 \
                  else "stable"
            best_med = max(best_med, med)
            print(f"  gamma={g:<5} median={med:6.1%}  "
                  f"P10={p10:6.1%}  P90={p90:6.1%}  "
                  f"spread={spread:6.1%}  {tag}")

    print("\n" + "=" * 72)
    print(" RUNTIME REALITY CHECK")
    print("=" * 72)
    print(" - prune: executable on stock llama.cpp once a gguf writer exists")
    print(" - adaptive per-expert bits: NOT executable on stock llama.cpp;")
    print("   requires a custom runtime. Numbers above are an upper bound.")
    print(" - bits saved != perplexity preserved. Prune drops tail mass;")
    print("   adaptive keeps it at 1-bit (noise). Only perplexity decides.")
    print("=" * 72)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "trace.jsonl"
    print(f"[*] reading {path}")
    if inspect(path) is None:
        return 1
    per_layer = load(path)
    if not per_layer:
        print("[!] no usable tokens; check keys against the sample above")
        return 1
    report(per_layer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
