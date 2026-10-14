"""One registry for the offline optimization algorithms.

Every entry here runs without a model, a GPU or a network, and every entry names
the paper it comes from. That is the point of the registry: the literature pack is
a folder of PDFs, and a folder of PDFs is not a system. This maps each algorithm
that survived review onto a function that can be executed and checked.

`run` is deliberately a plain ``fn(**kwargs) -> dict``: the CLI prints the dict,
the bench harness writes it to JSON, and the tests assert on it. There is no
second code path where a number could be produced differently.

Status is not decoration — it is the repo's confidence convention:

``[exact]``     the property is a theorem, asserted in the tests
``[simulated]`` a real algorithm over a constructed input, so the number
                describes the algorithm and not the machine
``[measured]``  run against an artifact in this repo (see `bench/algo_measure.py`
                for the two that reach real weights and a real trace)

A status is a claim about *where the input came from*, and it is not a claim that
an entry is fully measured. `quant-gptq` is `[measured]` because it reads real
Q4_K weights out of `models/Qwen3-0.6B-Q4_K_M.gguf`; the Hessian it needs comes
from no capture that exists here, so it stays synthetic and the summary says so.
Reading `measured` as "everything here is real" would be the same over-claim this
status column exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Algorithm:
    name: str
    kind: str          # kv | quant | select | spec
    source: str        # the paper this came from, in the pack
    status: str        # exact | simulated | measured
    summary: str
    run: Callable[..., dict]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "status": self.status,
            "summary": self.summary,
        }


def _rng_matrix(rows: int, cols: int, seed: int, scale: float = 1.0) -> list[list[float]]:
    """A deterministic pseudo-random matrix. Same shape of data as real weights."""
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    out: list[list[float]] = []
    for _ in range(rows):
        row: list[float] = []
        for _ in range(cols):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            u1 = ((state >> 11) + 1) / float(1 << 53)
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            u2 = ((state >> 11) + 1) / float(1 << 53)
            # Box-Muller: a weight matrix is roughly normal, not uniform, and the
            # quantizer's behaviour depends on the tails.
            import math

            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            row.append(z * scale)
        out.append(row)
    return out


def _synthetic_calibration(cols: int, seed: int) -> list[list[float]]:
    """A positive-definite Hessian-ish matrix, as `E[x xᵀ]` would be.

    ``H = A Aᵀ / n + eps I`` is positive definite by construction, so the inverse
    exists and GPTQ's propagation is well defined. An indefinite H would make the
    comparison meaningless while still returning numbers.
    """
    n = 64
    A = _rng_matrix(n, cols, seed)
    H = [[0.0] * cols for _ in range(cols)]
    for i in range(cols):
        for j in range(cols):
            H[i][j] = sum(A[r][i] * A[r][j] for r in range(n)) / n
    # A few directions of the synthetic calibration are nearly unexcited; the ridge
    # keeps them invertible, which is what `damp` exists for.
    for i in range(cols):
        H[i][i] += 1e-3
    return H


def _run_kv_retention(*, steps: int = 256, budget: int = 32, sinks: int = 4, seed: int = 1) -> dict:
    from .kv import compare, synth_attention, synth_surprise

    rows = synth_attention(steps, seed=seed, sinks=sinks)
    surprise = synth_surprise(rows, seed=seed)
    results = compare(rows, budget=budget, sinks=sinks, surprise=surprise)
    by_mass = sorted(results, key=lambda r: r.retained_mass, reverse=True)
    by_acc = sorted(results, key=lambda r: r.retained_accumulated, reverse=True)
    return {
        "stream": {"steps": steps, "budget": budget, "sinks": sinks, "seed": seed},
        "policies": [r.as_dict() for r in results],
        # The two rankings disagree, and that is the result: H2O wins the objective
        # it was written to maximize and loses the per-step one. Printing only the
        # first would be scoring H2O against itself.
        "best_by_retained_mass": by_mass[0].policy,
        "best_by_retained_accumulated": by_acc[0].policy,
        "rankings_agree": by_mass[0].policy == by_acc[0].policy,
        "newest_pin_events": {r.policy: r.newest_would_drop for r in results},
    }


def _run_quant(*, rows: int = 8, cols: int = 32, bits: int = 4, group: int | None = None,
               seed: int = 3) -> dict:
    from .quant import compare

    W = _rng_matrix(rows, cols, seed)
    H = _synthetic_calibration(cols, seed + 1)
    out = compare(W, H, bits=bits, group=group)
    out["hessian_improvement"] = (
        out["rtn_hessian"] / out["gptq_hessian"] if out["gptq_hessian"] > 0 else float("inf")
    )
    return out


def _select_from_trace(trace_path: str, k: int) -> dict:
    """Expert selection on a captured routing trace, and what the theory is worth there.

    The objective a trace actually defines is **modular**: keeping expert `e` serves
    its own selection count and nothing overlaps, so `f(S) = Σ_{e∈S} freq(e)` and the
    optimum is simply the top-k by frequency. That is exactly the rule
    `vipera/prune.py` already uses.

    The point of running greedy on it is to *show* that: greedy, lazy greedy and the
    frequency prefix all return the same set, and the exact optimum computed by
    sorting confirms no loss. The (1−1/e) guarantee is then vacuous here — it bounds a
    ratio that is 1.0 — and saying so is more useful than a bound presented as if it
    were doing work. The submodular machinery earns its place on the synthetic
    covering instance below, where the objective is genuinely non-modular.
    """
    from ..trace import load_trace
    from .select import coverage_objective, greedy, lazy_greedy

    routing = load_trace(trace_path)
    if not routing:
        raise ValueError(f"trace {trace_path} is empty")
    n_layers = len(routing[0])
    # The trace does not carry the expert count, so infer it from the largest id
    # seen. Guessing a round number instead would silently change what `k` means.
    n_experts = max(e for token in routing for layer in token for e in layer) + 1
    kk = max(1, min(k, n_experts))

    rows: list[dict] = []
    coincident = 0
    for layer in range(n_layers):
        freq = [0.0] * n_experts
        for token in routing:
            for e in token[layer]:
                freq[e] += 1.0
        # One element per expert and a weight of that expert's selection count is
        # the modular objective a trace defines: no two items share a covered
        # element, so there is nothing for submodularity to bite on.
        f = coverage_objective([[e] for e in range(n_experts)], freq)
        g = greedy(f, range(n_experts), kk)
        lz = lazy_greedy(f, range(n_experts), kk)
        prefix = _top_by(freq, kk)
        exact = sum(sorted(freq, reverse=True)[:kk])
        same = sorted(g.chosen) == sorted(prefix)
        coincident += 1 if same else 0
        rows.append({
            "layer": layer,
            "k": kk,
            "greedy_mass": g.value,
            "lazy_matches_greedy": lz.chosen == g.chosen,
            "prefix_matches_greedy": same,
            "exact_optimum_mass": exact,
            "ratio_to_optimum": (g.value / exact) if exact > 0 else 1.0,
        })

    return {
        "trace": trace_path,
        "tokens": len(routing),
        "layers": n_layers,
        "experts": n_experts,
        "top_k": kk,
        "total_selections": int(sum(r["greedy_mass"] for r in rows)),
        "objective": "modular (sum of per-expert selection counts)",
        "submodular_bound_applies": False,
        "layers_where_prefix_equals_greedy": f"{coincident}/{n_layers}",
        "min_ratio_to_optimum": min(r["ratio_to_optimum"] for r in rows) if rows else 1.0,
        "per_layer": rows,
    }


def _top_by(scores: list[float], k: int) -> list[int]:
    """Indices of the `k` largest scores, ties broken by index for determinism."""
    return sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k])


def _run_select(*, items: int = 24, universe: int = 64, k: int = 6, seed: int = 5,
                trace: str | None = None) -> dict:
    from .select import brute_force, coverage_objective, greedy, lazy_greedy

    if trace:
        per_trace = _select_from_trace(trace, k)
        return {"trace_mode": per_trace, "synthetic_instance": _synthetic_select(items, universe, k, seed)}

    return _synthetic_select(items, universe, k, seed)


def _synthetic_select(items: int, universe: int, k: int, seed: int) -> dict:
    from .select import brute_force, coverage_objective, greedy, lazy_greedy

    # Each item covers a handful of universe elements; weights make some elements
    # worth much more than others, which is what creates a non-trivial optimum.
    covers: list[list[int]] = []
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    for _ in range(items):
        pick: list[int] = []
        for _ in range(4):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            pick.append((state >> 33) % universe)
        covers.append(sorted(set(pick)))
    weights = [1.0 + 9.0 * ((e * 2654435761) % 97) / 97.0 for e in range(universe)]
    f = coverage_objective(covers, weights)

    g = greedy(f, range(items), k)
    lz = lazy_greedy(f, range(items), k)
    opt_set, opt_v = brute_force(f, range(items), k)
    return {
        "instance": {"items": items, "universe": universe, "k": k, "seed": seed},
        "greedy": g.as_dict(),
        "lazy": lz.as_dict(),
        "optimal": {"chosen": opt_set, "value": opt_v},
        "lazy_matches_greedy": lz.chosen == g.chosen and abs(lz.value - g.value) < 1e-12,
        "guarantee_ratio": (g.value / opt_v) if opt_v > 0 else 1.0,
        # The (1 - 1/e) bound is the theorem; this is what the instance achieved.
        "guarantee_bound": 1.0 - 1.0 / 2.718281828459045,
    }


def _run_spec(*, k_max: int = 8, vocab: int = 16, gap: float = 0.35, seed: int = 7,
              cost_ratio: float = 0.15) -> dict:
    from .spec import expected_accepted, measure_accepted, overlap, sweep_draft_length

    def position(i: int) -> tuple[list[float], list[float]]:
        # The draft is a noisier version of the target, and the gap widens with
        # position -- what a real draft model does as it extrapolates further.
        g = gap * (1.0 + 0.25 * i)
        p = _softmax_row(vocab, seed + i)
        q = [x * (1.0 - g) + g / vocab for x in p]
        s = sum(q)
        return p, [x / s for x in q]

    sweep = sweep_draft_length(position, max_k=k_max, cost_ratio=cost_ratio)
    pairs = [position(i) for i in range(k_max)]
    t = [p for p, _ in pairs]
    d = [q for _, q in pairs]
    return {
        "profile": {"k_max": k_max, "vocab": vocab, "gap": gap, "seed": seed,
                    "cost_ratio": cost_ratio},
        "overlaps": [overlap(p, q) for p, q in pairs],
        "closed_form_accepted": expected_accepted(t, d),
        "monte_carlo_accepted": measure_accepted(t, d, trials=20000, seed=seed),
        "sweep": sweep.as_dict(),
    }


def _softmax_row(vocab: int, seed: int) -> list[float]:
    """A deterministic distribution that is peaked but not degenerate."""
    import math

    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    logits: list[float] = []
    for _ in range(vocab):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        logits.append(((state >> 33) % 1000) / 100.0)
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [x / s for x in ex]


REGISTRY: tuple[Algorithm, ...] = (
    Algorithm(
        name="kv-retention",
        kind="kv",
        source="2309.17453 (StreamingLLM sink+window), 2306.14048 (H2O)",
        status="simulated",
        summary=(
            "Retention policies (recency, sink-window, H2O heavy-hitter, surprise) "
            "over an explicit attention stream, scored by retained attention mass"
        ),
        run=_run_kv_retention,
    ),
    Algorithm(
        name="quant-gptq",
        kind="quant",
        source="2210.17323 (GPTQ)",
        status="measured",
        summary=(
            "Hessian-weighted error diffusion vs round-to-nearest at an identical "
            "bit grid, reported in both the Frobenius and the Hessian norm; on real "
            "Q4_K weights from models/Qwen3-0.6B-Q4_K_M.gguf the Hessian norm "
            "improves 2.09x while the Frobenius norm gets 1.55x worse (the Hessian "
            "itself is synthetic -- no activation capture exists in this repo)"
        ),
        run=_run_quant,
    ),
    Algorithm(
        name="select-submodular",
        kind="select",
        source="krause-golovin14 (Submodular Function Maximization)",
        status="measured",
        summary=(
            "Lazy greedy (CELF) maximization of a monotone submodular coverage "
            "objective, with the (1-1/e) guarantee checked against brute force; on "
            "the real 511-token OLMoE trace the objective turns out to be modular, "
            "so greedy ties the exact optimum and the (1-1/e) bound is vacuous there "
        ),
        run=_run_select,
    ),
    Algorithm(
        name="spec-draft-length",
        kind="spec",
        source="2302.01318 / 2211.17192 (speculative decoding)",
        status="exact",
        summary=(
            "Closed-form expected accepted length for draft length k, cross-checked "
            "against a Monte-Carlo of the actual accept/reject/residual sampler"
        ),
        run=_run_spec,
    ),
)

BY_NAME = {a.name: a for a in REGISTRY}


def list_algorithms() -> list[Algorithm]:
    return list(REGISTRY)


def get(name: str) -> Algorithm:
    try:
        return BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(BY_NAME))
        raise KeyError(f"unknown algorithm {name!r}; known: {known}") from None


def run(name: str, **kwargs) -> dict:
    """Execute one registered algorithm and return its JSON-able result."""
    return get(name).run(**kwargs)
