"""Submodular maximization under a cardinality constraint, and why it applies here.

Krause & Golovin's survey (`book-krause-golovin14-submodular-max.pdf`) is in the
pack for one reason: **selecting which experts to keep is a submodular
maximization problem**, and that is the only class of selection problem in this
project with a *provable* approximation guarantee.

A set function `f` is submodular when adding an element helps less the more you
already have (diminishing returns). For the instance that matters here — cover as
much routing mass as possible with `k` experts — that holds, and it is monotone.
Under a cardinality constraint the plain greedy algorithm then guarantees

    f(S_greedy) >= (1 - 1/e) * f(S_optimal) ≈ 0.632 * OPT

and no polynomial algorithm does better unless P = NP. The survey's second
contribution is the one that makes it usable at scale: **lazy greedy** (CELF)
caches marginal gains in a max-heap and re-checks only the top, which returns the
*identical* set in far fewer evaluations. That identity is asserted in the tests,
not assumed.

`vipera/prune.py` already chooses a keep-set, by ranking per-layer frequency and
taking the shortest prefix that reaches a target mass. This module provides the
general machinery so that choice can be stated as the optimization problem it is,
and so a second objective (coverage across *sets*, not a prefix) is available.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

# A set function over a ground set of indices.
SetFunction = Callable[[frozenset[int]], float]


@dataclass
class GreedyResult:
    """A chosen set plus the evaluation count that produced it."""

    chosen: list[int]
    value: float
    evaluations: int
    # "greedy" or "lazy" — both must agree on `chosen` and `value`.
    method: str

    def as_dict(self) -> dict:
        return {
            "chosen": list(self.chosen),
            "value": self.value,
            "evaluations": self.evaluations,
            "method": self.method,
        }


def greedy(
    f: SetFunction, ground: Sequence[int], k: int
) -> GreedyResult:
    """Textbook greedy: repeatedly add the element with the largest marginal gain.

    O(k * |ground|) evaluations of `f`. Returns an empty set for `k <= 0` rather
    than raising, so a sweep can pass `k=0` to mean "keep nothing".
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    chosen: list[int] = []
    current = frozenset()
    value = f(current)
    evaluations = 1
    remaining = list(ground)
    while len(chosen) < k and remaining:
        best_gain = None
        best = None
        current_value = value
        for e in remaining:
            v = f(current | {e})
            evaluations += 1
            gain = v - current_value
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best = e
        # A zero-gain step means the function has saturated; stopping keeps the
        # result identical while avoiding k - |S| pointless rounds.
        if best is None or (best_gain is not None and best_gain <= 0.0):
            break
        chosen.append(best)
        current = current | {best}
        value = f(current)
        remaining.remove(best)
    return GreedyResult(chosen, value, evaluations, "greedy")


def lazy_greedy(f: SetFunction, ground: Sequence[int], k: int) -> GreedyResult:
    """CELF lazy greedy — the same set, fewer evaluations.

    Marginals are non-increasing under submodularity, so a stale upper bound is
    still a valid upper bound: pop the best bound, recompute it, and if it is
    still the best, it is optimal to take. Pushed back with `-gain` because
    `heapq` is a min-heap.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    chosen: list[int] = []
    current = frozenset()
    value = f(current)
    evaluations = 1
    if k == 0 or not ground:
        return GreedyResult(chosen, value, evaluations, "lazy")

    heap: list[tuple[float, int]] = []
    for e in ground:
        v = f(current | {e})
        evaluations += 1
        heapq.heappush(heap, (-(v - value), e))

    while heap and len(chosen) < k:
        neg_bound, e = heapq.heappop(heap)
        if e in current:
            continue
        v = f(current | {e})
        evaluations += 1
        gain = v - value
        # The staleness check must come BEFORE the zero-gain stop. A first version
        # of this function stopped on `gain <= 0` first, which is only sound when
        # the popped element also holds the largest cached bound. It does not have
        # to: a stale bound of 5 can recompute to 0 while another element still
        # carries a genuine bound of 4, and stopping there silently returns a set
        # strictly smaller than k with a worse value than plain greedy.
        if heap and -heap[0][0] > gain:
            # The cached bound is stale; the true gain may still win later.
            heapq.heappush(heap, (-gain, e))
            continue
        if gain <= 0.0:
            # Nothing left can help: this element is the largest bound *and* the
            # largest bound is non-positive, so every true gain is zero too.
            break
        chosen.append(e)
        current = current | {e}
        value = v
    return GreedyResult(chosen, value, evaluations, "lazy")


def brute_force(f: SetFunction, ground: Sequence[int], k: int) -> tuple[list[int], float]:
    """Exhaustive optimum, for testing the guarantee. Exponential — small inputs only."""
    from itertools import combinations

    best: list[int] = []
    best_v = f(frozenset())
    n = len(ground)
    kk = min(k, n)
    for size in range(1, kk + 1):
        for combo in combinations(ground, size):
            v = f(frozenset(combo))
            if v > best_v or (v == best_v and list(combo) < best):
                best_v = v
                best = list(combo)
    return best, best_v


def coverage_objective(
    sets: Iterable[Sequence[int]], weights: Sequence[float]
) -> SetFunction:
    """Weighted max-coverage: a monotone submodular function, and the useful one here.

    `sets[i]` is the set of elements that item `i` covers; `weights[e]` is the
    value of **covering element `e`**. For expert selection, item `i` is an expert
    and `sets[i]` is the set of tokens that route to it (or, in a per-layer view,
    the tokens it serves), with one weight per token — then `f(S)` is exactly the
    routing mass the kept experts still serve, the quantity `vipera/prune.py`
    reports as `kept_mass`.

    Note that `weights` is indexed by *element*, not by item, so its length is the
    size of the covered universe and is unrelated to `len(sets)`. An element index
    out of range of `weights` raises IndexError at evaluation time rather than
    being silently treated as zero.

    Returns a callable so it can be handed to `greedy`, `lazy_greedy` or
    `brute_force` unchanged.
    """
    sets_list = [tuple(s) for s in sets]
    w = list(weights)
    if not w:
        raise ValueError("weights is empty; there is no element to value")

    def f(chosen: frozenset[int]) -> float:
        covered: set[int] = set()
        for i in chosen:
            covered.update(sets_list[i])
        return sum(weights[e] for e in covered)

    return f
