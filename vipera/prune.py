"""Trace-driven expert pruning: choose *which* experts to keep, not just how many.

`Optimizations.prune_experts` shrinks the expert pool by a fraction and then the
scheduler treats the surviving pool as if it were free. That hides two very
different worlds:

* keep a **random** 5% of experts and you retain ~5% of the routing mass — the
  model is badly damaged;
* keep the **hottest** 5% (as a real trace reveals) and you can retain most of
  the mass.

So "pruning to 5% makes the pool fit the cache" is only true if the keep-set is
chosen by mass. This module measures that gap on a captured trace: for each
layer, rank experts by selection frequency, keep the smallest prefix reaching a
target mass, and report what uniform pruning would have kept at the same size.

The number this produces is a **necessary condition**, not a quality result:
retained routing mass bounds how much of the model's behaviour survives, but
actual quality (perplexity / task accuracy) needs the pruned weights to be built
and run (S9 + S10).
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class LayerKeep:
    index: int
    distinct: int
    keep: list[int]
    kept_mass: float


@dataclass
class PrunePlan:
    target_mass: float
    n_experts: int
    layers: list[LayerKeep] = field(default_factory=list)
    # Set when the keep-set was sized by *count* instead of by mass
    # (`plan_at_fraction`), which is the arm a perplexity sweep compares.
    fraction: float | None = None

    @property
    def total_keep(self) -> int:
        return sum(len(l.keep) for l in self.layers)

    @property
    def total_experts(self) -> int:
        return len(self.layers) * self.n_experts

    @property
    def keep_fraction(self) -> float:
        return self.total_keep / self.total_experts if self.total_experts else 0.0

    @property
    def mean_kept_mass(self) -> float:
        if not self.layers:
            return 0.0
        return sum(l.kept_mass for l in self.layers) / len(self.layers)

    def as_dict(self) -> dict:
        out = {
            "target_mass": self.target_mass,
            "n_experts": self.n_experts,
            "keep_fraction": round(self.keep_fraction, 4),
            "total_keep": self.total_keep,
            "total_experts": self.total_experts,
            "mean_kept_mass": round(self.mean_kept_mass, 4),
            "layers": [
                {
                    "index": l.index,
                    "distinct": l.distinct,
                    "keep": len(l.keep),
                    "kept_mass": round(l.kept_mass, 4),
                }
                for l in self.layers
            ],
        }
        if self.fraction is not None:
            out["fraction"] = self.fraction
        return out


def _layer_counts(routing: list[list[list[int]]], n_experts: int) -> list[list[int]]:
    layers = len(routing[0])
    counts = [[0] * n_experts for _ in range(layers)]
    for token in routing:
        for li, experts in enumerate(token):
            for e in experts:
                counts[li][e] += 1
    return counts


def plan_from_trace(routing: list[list[list[int]]], *, n_experts: int,
                    target_mass: float = 0.95) -> PrunePlan:
    """Smallest per-layer expert set that retains `target_mass` of routing mass."""
    if not 0.0 < target_mass <= 1.0:
        raise ValueError(f"target_mass must be in (0, 1], got {target_mass}")
    counts = _layer_counts(routing, n_experts)
    layers: list[LayerKeep] = []
    for li, freq in enumerate(counts):
        total = sum(freq) or 1
        ranked = sorted(range(n_experts), key=lambda e: freq[e], reverse=True)
        keep: list[int] = []
        acc = 0
        for e in ranked:
            if freq[e] == 0:
                break
            keep.append(e)
            acc += freq[e]
            if acc / total >= target_mass:
                break
        layers.append(LayerKeep(index=li, distinct=sum(1 for c in freq if c > 0),
                                keep=keep, kept_mass=acc / total))
    return PrunePlan(target_mass=target_mass, n_experts=n_experts, layers=layers)


def plan_at_fraction(routing: list[list[list[int]]], *, n_experts: int,
                     fraction: float) -> PrunePlan:
    """Keep a fixed *share* of each layer's experts, ranked by trace frequency.

    `plan_from_trace` sizes the keep-set by the mass it has to retain; this sizes it
    by count, which is the arm a quality sweep needs: a frequency keep-set and a
    random one of the same size are what separate "pruning is free" from "pruning
    is free only if you choose the experts by the trace".
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    keep = max(1, int(round(n_experts * fraction)))
    counts = _layer_counts(routing, n_experts)
    layers: list[LayerKeep] = []
    for li, freq in enumerate(counts):
        total = sum(freq) or 1
        ranked = sorted(range(n_experts), key=lambda e: (-freq[e], e))
        chosen = ranked[:keep]
        layers.append(LayerKeep(index=li, distinct=sum(1 for c in freq if c > 0),
                                keep=chosen,
                                kept_mass=sum(freq[e] for e in chosen) / total))
    return PrunePlan(target_mass=0.0, n_experts=n_experts, layers=layers, fraction=fraction)


def keep_sets(plan: PrunePlan) -> dict[int, list[int]]:
    """The plan's per-layer keep-sets, keyed by block index."""
    return {l.index: list(l.keep) for l in plan.layers}


def uniform_keep_sets(*, n_layers: int, n_experts: int, keep_per_layer: int,
                      seed: int = 0) -> dict[int, list[int]]:
    """The control arm: the same number of experts per layer, chosen without a trace.

    A seeded sample is the honest stand-in for uniform pruning: it keeps the size
    identical and destroys any trace-derived structure, so a measured quality gap
    against `keep_sets` is what the trace is actually worth. Sorted, because the
    mask file is a set and its order must not depend on the seed's luck twice.
    """
    if not 0 < keep_per_layer <= n_experts:
        raise ValueError(
            f"keep_per_layer must be in (0, {n_experts}], got {keep_per_layer}"
        )
    rng = random.Random(seed)
    return {li: sorted(rng.sample(range(n_experts), keep_per_layer))
            for li in range(n_layers)}


#: First field is the block index, the rest are the expert ids to keep. Read by
#: `bench/llama-moe-prune.patch` through LLAMA_MOE_PRUNE.
MASK_ENV = "LLAMA_MOE_PRUNE"


def mask_text(keep: Mapping[int, Sequence[int]]) -> str:
    """Render keep-sets in the mask format the patched llama.cpp reads.

    Keeping the format this small is deliberate: the reader is ~90 lines of C in a
    patch, and every field it has to parse is a place a sweep can fail silently.
    """
    if not keep:
        raise ValueError("a mask needs at least one layer; empty would mask nothing")
    lines = ["# vipera expert mask: <layer> <expert ids to keep>"]
    for layer in sorted(keep):
        ids = list(keep[layer])
        if not ids:
            raise ValueError(f"layer {layer} keeps no experts")
        if len(set(ids)) != len(ids):
            raise ValueError(f"layer {layer} repeats an expert id: {ids}")
        if min(ids) < 0:
            raise ValueError(f"layer {layer} has a negative expert id: {ids}")
        lines.append(f"{layer} " + ",".join(str(e) for e in sorted(ids)))
    return "\n".join(lines) + "\n"


def write_mask(path: str, keep: Mapping[int, Sequence[int]]) -> int:
    """Write `keep` as a mask file; return the number of layers written."""
    text = mask_text(keep)
    with open(path, "w") as f:
        f.write(text)
    return len(keep)


def mass_at_fraction(routing: list[list[list[int]]], *, n_experts: int,
                     fraction: float) -> float:
    """Mean per-layer routing mass retained by the best `fraction` of experts.

    The comparison point is a **uniform** keep-set, which retains ≈ `fraction`:
    the returned value is what choosing by mass buys instead.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    keep = max(1, int(round(n_experts * fraction)))
    counts = _layer_counts(routing, n_experts)
    masses = []
    for freq in counts:
        total = sum(freq) or 1
        ranked = sorted(freq, reverse=True)
        masses.append(sum(ranked[:keep]) / total)
    return sum(masses) / len(masses) if masses else 0.0


def mass_curve(routing: list[list[list[int]]], *, n_experts: int,
               fractions: list[float]) -> list[dict]:
    """Mass retained per fraction, next to what a uniform keep-set would retain."""
    return [
        {
            "fraction": f,
            "kept_mass": round(mass_at_fraction(routing, n_experts=n_experts, fraction=f), 4),
            "uniform_kept_mass": round(f, 4),
        }
        for f in fractions
    ]


# --- P4: gated expert pruning with per-layer rollback (docs/PROPOSALS.md §5) ---
#
# `plan_from_trace` answers "which experts hold the mass". It still *applies* that
# keep-set unconditionally. P4 adds a gate: a layer is pruned only if the keep-set
# it needs is affordable (mass gate) and, when a drift proxy has been measured, the
# layer's output drift is within budget. Any layer failing a gate keeps ALL of its
# experts, so the failure is local instead of silently damaging the whole model.
#
# vipera never runs a model, so the drift proxy is an *input* (per-layer numbers
# from a probe), not something computed here; the mass gate needs only the trace.


@dataclass
class LayerGate:
    index: int
    keep: list[int]
    kept_mass: float
    # Experts the mass gate *wanted* to keep, before any rollback.
    candidate_keep: int
    pruned: bool
    reason: str
    drift: float | None = None


@dataclass
class GatedPlan:
    target_mass: float
    n_experts: int
    max_keep_fraction: float
    delta: float | None = None
    layers: list[LayerGate] = field(default_factory=list)

    @property
    def pruned_layers(self) -> int:
        return sum(1 for l in self.layers if l.pruned)

    @property
    def rolled_back_layers(self) -> int:
        return len(self.layers) - self.pruned_layers

    @property
    def total_keep(self) -> int:
        return sum(len(l.keep) for l in self.layers)

    @property
    def total_experts(self) -> int:
        return len(self.layers) * self.n_experts

    @property
    def keep_fraction(self) -> float:
        return self.total_keep / self.total_experts if self.total_experts else 0.0

    @property
    def mean_kept_mass(self) -> float:
        if not self.layers:
            return 0.0
        return sum(l.kept_mass for l in self.layers) / len(self.layers)

    def as_dict(self) -> dict:
        reasons: dict[str, int] = {}
        for l in self.layers:
            reasons[l.reason] = reasons.get(l.reason, 0) + 1
        return {
            "algorithm": "gated-expert-pruning",
            "target_mass": self.target_mass,
            "n_experts": self.n_experts,
            "max_keep_fraction": self.max_keep_fraction,
            "delta": self.delta,
            "keep_fraction": round(self.keep_fraction, 4),
            "total_keep": self.total_keep,
            "total_experts": self.total_experts,
            "mean_kept_mass": round(self.mean_kept_mass, 4),
            "pruned_layers": self.pruned_layers,
            "rolled_back_layers": self.rolled_back_layers,
            "reasons": reasons,
            "layers": [
                {
                    "index": l.index,
                    "keep": len(l.keep),
                    "candidate_keep": l.candidate_keep,
                    "kept_mass": round(l.kept_mass, 4),
                    "pruned": l.pruned,
                    "reason": l.reason,
                    "drift": l.drift,
                }
                for l in self.layers
            ],
        }


def _candidate_keep(freq: list[int], target_mass: float) -> list[int]:
    """Smallest prefix of the mass-ranked experts that reaches `target_mass`."""
    total = sum(freq) or 1
    ranked = sorted(range(len(freq)), key=lambda e: freq[e], reverse=True)
    keep: list[int] = []
    acc = 0
    for e in ranked:
        if freq[e] == 0:
            break
        keep.append(e)
        acc += freq[e]
        if acc / total >= target_mass:
            break
    return keep


def _drift_at(drift, index: int) -> float | None:
    if drift is None:
        return None
    if isinstance(drift, dict):
        return float(drift.get(index, 0.0))
    return float(drift[index])


def gated_prune(routing: list[list[list[int]]], *, n_experts: int,
                target_mass: float = 0.95, max_keep_fraction: float = 1.0,
                delta: float | None = None, drift=None) -> GatedPlan:
    """P4 (Gated Expert Pruning, GEP): prune per layer only where a gate passes.

    Gates, evaluated in order per layer:
      * nothing to prune — the mass keep-set is the whole layer;
      * mass unreachable — the keep-set exceeds `max_keep_fraction`;
      * drift too high   — a measured per-layer drift proxy exceeds `delta`.
    A layer failing any gate is rolled back (keeps all experts). `drift` is a
    per-layer mapping or sequence of measured drift values; `delta` without drift
    data is an error, because the drift gate cannot be evaluated.
    """
    if not 0.0 < target_mass <= 1.0:
        raise ValueError(f"target_mass must be in (0, 1], got {target_mass}")
    if not 0.0 < max_keep_fraction <= 1.0:
        raise ValueError(f"max_keep_fraction must be in (0, 1], got {max_keep_fraction}")
    if delta is not None and drift is None:
        raise ValueError("delta needs drift data: the drift gate cannot be evaluated")

    counts = _layer_counts(routing, n_experts)
    layers: list[LayerGate] = []
    for li, freq in enumerate(counts):
        total = sum(freq)
        d = _drift_at(drift, li)
        if total == 0:
            layers.append(LayerGate(li, list(range(n_experts)), 1.0,
                                    n_experts, False, "no_routing", d))
            continue

        cand = _candidate_keep(freq, target_mass)
        cand_fraction = len(cand) / n_experts
        cand_mass = sum(freq[e] for e in cand) / total

        if len(cand) >= n_experts:
            pruned, reason = False, "nothing_to_prune"
        elif cand_fraction > max_keep_fraction:
            pruned, reason = False, "mass_unreachable"
        elif delta is not None and d is not None and d > delta:
            pruned, reason = False, "drift"
        else:
            pruned, reason = True, "pruned"

        if pruned:
            keep, kept_mass = cand, cand_mass
        else:
            keep, kept_mass = list(range(n_experts)), 1.0
        layers.append(LayerGate(li, keep, kept_mass, len(cand), pruned, reason, d))

    return GatedPlan(target_mass=target_mass, n_experts=n_experts,
                     max_keep_fraction=max_keep_fraction, delta=delta, layers=layers)


def dumps(plan: PrunePlan) -> str:
    return json.dumps(plan.as_dict(), indent=2) + "\n"


def gated_dumps(plan: GatedPlan) -> str:
    return json.dumps(plan.as_dict(), indent=2) + "\n"
