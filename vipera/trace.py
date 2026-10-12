"""Routing traces: capture format, loader, validation and statistics.

The single biggest weakness of the scheduler is that expert routing is *modelled*
(a Zipf skew), not measured. Every cache/hit-rate claim in `docs/PROPOSALS.md`
carries that caveat. This module is the enabling piece: it defines a small trace
format, loads and validates it, and reports the statistics that decide whether a
cache policy can work at all (skew, per-layer entropy, static-cache ceiling).

Format: JSON Lines, one record per generated token.

    {"t": 0, "layers": [[3,1,7,2,5,9], [8,0,4,4,...], ...]}

`layers[l]` is the list of expert ids selected at layer `l` for that token. The
loader returns the structure `schedule.simulate(routing=...)` expects:
`routing[token][layer][slot] = expert_id`.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field


class TraceError(ValueError):
    """Raised when a trace is malformed or incompatible with an architecture."""


@dataclass
class LayerStats:
    index: int
    mean_entropy: float
    top_fraction_coverage: float


@dataclass
class TraceStats:
    tokens: int
    layers: int
    n_experts: int
    top_k: int
    # Normalised Shannon entropy of the expert distribution, averaged over
    # layers. 1.0 = uniform (no cache can help), 0.0 = one expert used.
    mean_entropy: float
    # Max hit rate of a *static* cache holding `cache_fraction` of experts per
    # layer (the best any non-adaptive policy could do on this trace).
    static_cache_fraction: float
    static_cache_hit_ceiling: float
    per_layer: list[LayerStats] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "layers": self.layers,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "mean_entropy": round(self.mean_entropy, 4),
            "static_cache_fraction": self.static_cache_fraction,
            "static_cache_hit_ceiling": round(self.static_cache_hit_ceiling, 4),
            "per_layer": [
                {"index": s.index,
                 "mean_entropy": round(s.mean_entropy, 4),
                 "top_fraction_coverage": round(s.top_fraction_coverage, 4)}
                for s in self.per_layer
            ],
        }


def load_trace(path: str, *, limit: int | None = None) -> list[list[list[int]]]:
    """Read a JSONL routing trace into the scheduler's `routing` structure."""
    routing: list[list[list[int]]] = []
    try:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise TraceError(f"{path}:{lineno}: invalid JSON: {e}") from e
                layers = rec.get("layers")
                if not isinstance(layers, list) or not layers:
                    raise TraceError(f"{path}:{lineno}: missing non-empty 'layers'")
                token: list[list[int]] = []
                for l, experts in enumerate(layers):
                    if not isinstance(experts, list) or not experts:
                        raise TraceError(f"{path}:{lineno}: layer {l} has no experts")
                    token.append([int(e) for e in experts])
                routing.append(token)
                if limit is not None and len(routing) >= limit:
                    break
    except OSError as e:
        raise TraceError(f"cannot read trace {path}: {e}") from e
    if not routing:
        raise TraceError(f"{path}: trace is empty")
    return routing


def validate(routing: list[list[list[int]]], *, n_layers: int, n_experts: int,
             top_k: int | None = None) -> None:
    """Raise TraceError if a trace is inconsistent with an architecture."""
    if not routing:
        raise TraceError("trace is empty")
    for ti, token in enumerate(routing):
        if len(token) != n_layers:
            raise TraceError(
                f"token {ti}: trace has {len(token)} layers, architecture has {n_layers}"
            )
        for l, experts in enumerate(token):
            if top_k is not None and len(experts) != top_k:
                raise TraceError(
                    f"token {ti} layer {l}: {len(experts)} experts, architecture top_k={top_k}"
                )
            for e in experts:
                if not (0 <= e < n_experts):
                    raise TraceError(
                        f"token {ti} layer {l}: expert id {e} outside [0,{n_experts})"
                    )


def _layer_frequency(token_layers: list[list[int]], n_experts: int) -> list[int]:
    freq = [0] * n_experts
    for e in token_layers:
        freq[e] += 1
    return freq


def stats(routing: list[list[list[int]]], *, n_experts: int,
          cache_fraction: float = 0.03) -> TraceStats:
    """Compute skew/entropy and the static-cache hit ceiling for a trace."""
    tokens = len(routing)
    layers = len(routing[0])
    top_k = len(routing[0][0])
    keep = max(1, int(round(n_experts * cache_fraction)))

    per_layer: list[LayerStats] = []
    entropies: list[float] = []
    coverages: list[float] = []

    for l in range(layers):
        freq = [0] * n_experts
        for t in range(tokens):
            for e in routing[t][l]:
                freq[e] += 1
        total = sum(freq) or 1
        probs = [c / total for c in freq if c > 0]
        if len(probs) > 1:
            h = -sum(p * math.log2(p) for p in probs) / math.log2(n_experts)
        else:
            h = 0.0
        ranked = sorted(freq, reverse=True)
        coverage = sum(ranked[:keep]) / total
        entropies.append(h)
        coverages.append(coverage)
        per_layer.append(LayerStats(index=l, mean_entropy=h, top_fraction_coverage=coverage))

    return TraceStats(
        tokens=tokens,
        layers=layers,
        n_experts=n_experts,
        top_k=top_k,
        mean_entropy=sum(entropies) / len(entropies),
        static_cache_fraction=cache_fraction,
        static_cache_hit_ceiling=sum(coverages) / len(coverages),
        per_layer=per_layer,
    )


def global_cache_ceiling(
    routing: list[list[list[int]]], slots: int, *, warmup: int = 0
) -> float:
    """Best hit rate a *static* cache of `slots` keys could achieve.

    Counts ``(layer, expert)`` frequency over ``routing[warmup:]`` and returns the
    mass fraction of the `slots` most frequent pairs. Unlike `stats()`'s ceiling
    (which forces an equal, per-layer share), this pools all layers into one
    global cache — the tightest upper bound for a policy whose slots can hold any
    layer's expert, and therefore the honest S2 baseline.
    """
    counts: dict[tuple[int, int], int] = {}
    total = 0
    for token in routing[warmup:]:
        for layer, experts in enumerate(token):
            for e in experts:
                key = (layer, e)
                counts[key] = counts.get(key, 0) + 1
                total += 1
    if total == 0 or slots <= 0:
        return 0.0
    top = sorted(counts.values(), reverse=True)[:slots]
    return sum(top) / total


def synth_trace(tokens: int, *, n_layers: int, n_experts: int, top_k: int,
                skew: float = 1.0, seed: int = 0) -> list[list[list[int]]]:
    """Generate a Zipf-skewed trace, for tests and for `--trace` smoke runs."""
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** skew) for i in range(n_experts)] if skew > 0 else [1.0] * n_experts
    out: list[list[list[int]]] = []
    for _ in range(tokens):
        token: list[list[int]] = []
        for _ in range(n_layers):
            idx = list(range(n_experts))
            chosen: list[int] = []
            for _ in range(top_k):
                total = sum(weights[i] for i in idx)
                roll = rng.random() * total
                acc = 0.0
                pick = idx[-1]
                for i in idx:
                    acc += weights[i]
                    if roll <= acc:
                        pick = i
                        break
                chosen.append(pick)
                idx.remove(pick)
            token.append(chosen)
        out.append(token)
    return out


def synth_shift_trace(tokens: int, *, n_layers: int, n_experts: int, top_k: int,
                      skew: float = 1.0, seed: int = 0, split: float = 0.5,
                      shift: int | None = None) -> list[list[list[int]]]:
    """A non-stationary trace: the hot set moves partway through.

    The first ``split`` fraction is a Zipf trace; the rest is another Zipf trace
    with expert ids rotated by ``shift`` (default: half the experts), so the
    experts that were hot early are cold later. This is the input that separates
    an aging cache (which can follow the shift) from a plain LFU (which cannot).
    """
    if shift is None:
        shift = n_experts // 2
    first = synth_trace(int(tokens * split), n_layers=n_layers, n_experts=n_experts,
                        top_k=top_k, skew=skew, seed=seed)
    second = synth_trace(tokens - int(tokens * split), n_layers=n_layers,
                         n_experts=n_experts, top_k=top_k, skew=skew, seed=seed + 1)
    moved = [
        [[(e + shift) % n_experts for e in layer] for layer in token]
        for token in second
    ]
    return first + moved


def save_trace(path: str, routing: list[list[list[int]]]) -> None:
    """Write a trace in the JSONL format `load_trace` reads (round-trippable)."""
    with open(path, "w") as f:
        for t, token in enumerate(routing):
            f.write(json.dumps({"t": t, "layers": token}) + "\n")


def dumps(s: TraceStats) -> str:
    return json.dumps(s.as_dict(), indent=2) + "\n"
