"""Speculative decoding: what a draft of length `k` is actually worth.

Two independent results from the pack live here, and they answer the two
questions the repo's `--draft-tokens` knob raises but never checks.

**1. The acceptance rule is exact, and that is testable.**
`2302.01318` / `2211.17192`: accept a drafted token with probability
``min(1, p(x)/q(x))`` and, on rejection, resample from the normalised residual
``max(0, p − q)``. The samples then follow the *target* distribution exactly —
not approximately. `residual_sample` implements it, and the test asserts the
empirical distribution converges to `p` rather than merely "looks reasonable".

**2. Expected accepted length has a closed form.**
For per-position distributions ``p_i`` (target) and ``q_i`` (draft), define
``a_i = Σ_x min(p_i(x), q_i(x))`` — the total-variation overlap of the two, which
is the probability that position `i` is accepted *given every position before it
was*. Then

    E[accepted] = Σ_{i=1..k} Π_{j<i} a_j

so each additional drafted token is worth less than the last, and the value of
raising `k` is bounded by `1/(1 − a)`. `expected_accepted` computes the formula;
`measure_accepted` Monte-Carlos the sampler; the test asserts they agree, which
is what makes the formula trustworthy on a stream where Monte-Carlo is too slow.

This is the number that decides whether `--draft-tokens 3` or `6` is right, and it
is offline: no draft model and no target model are involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Distribution = list[float]


class _Rng:
    """A tiny deterministic generator.

    Deliberately not `random`: the tests assert on tolerance windows around
    empirical frequencies, and a shared global generator makes a failure
    unreproducible the moment another test draws from it first.
    """

    def __init__(self, seed: int = 0):
        self._state = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def next_float(self) -> float:
        """Uniform in [0, 1)."""
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self._state >> 11) / float(1 << 53)

    def choice(self, probs: Distribution) -> int:
        """Sample an index from `probs` by inverse CDF."""
        u = self.next_float()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if u < acc:
                return i
        return len(probs) - 1


def _validate_pair(p: Distribution, q: Distribution) -> None:
    if not p or not q:
        raise ValueError("distributions must be non-empty")
    if len(p) != len(q):
        raise ValueError(f"p has {len(p)} entries, q has {len(q)}; same vocabulary required")
    if any(x < 0 for x in p) or any(x < 0 for x in q):
        raise ValueError("probabilities must be non-negative")
    if not math.isclose(math.fsum(p), 1.0, abs_tol=1e-6):
        raise ValueError(f"p sums to {math.fsum(p)}, not 1")
    if not math.isclose(math.fsum(q), 1.0, abs_tol=1e-6):
        raise ValueError(f"q sums to {math.fsum(q)}, not 1")


def accept_prob(p_x: float, q_x: float) -> float:
    """``min(1, p(x)/q(x))`` — the acceptance probability for one drafted token.

    A drafted token the target likes *more* than the draft does is always kept;
    otherwise it survives with the ratio of the two. ``q_x == 0`` means the draft
    could not have proposed `x`, so it is rejected.
    """
    if q_x < 0 or p_x < 0:
        raise ValueError("probabilities must be non-negative")
    if q_x == 0:
        return 0.0
    return min(1.0, p_x / q_x)


def residual(p: Distribution, q: Distribution) -> Distribution:
    """``p(x) − min(p(x), q(x))`` renormalised: what to sample after a rejection.

    Sums to 1 by construction (the residual mass equals ``1 − a``). Raises when
    the residual is empty, which happens only when `q` is a point mass on a value
    the draft already proposed — then there is nothing left to correct with and
    the caller must fall back to the target itself.
    """
    _validate_pair(p, q)
    r = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    total = math.fsum(r)
    if total <= 0:
        raise ValueError(
            "empty residual: q has no mass anywhere p does not, so rejection "
            "cannot be corrected (p == q on every value)"
        )
    return [x / total for x in r]


def overlap(p: Distribution, q: Distribution) -> float:
    """``a = Σ_x min(p(x), q(x))`` — the chance position i is accepted."""
    _validate_pair(p, q)
    return math.fsum(min(pi, qi) for pi, qi in zip(p, q))


def expected_accepted(
    target: list[Distribution], draft: list[Distribution]
) -> float:
    """Closed form ``Σ_{i=1..k} Π_{j=1..i} a_j`` over per-position distributions.

    Reads off the tail-sum of the survival probabilities: ``P(accepted >= i)`` is
    the product of the first `i` overlaps, so the expectation is the sum of those
    products. Multiplying *before* accumulating is what makes the first term
    ``a_1`` rather than 1 — a version that accumulates first answers a different
    question and is off by exactly one term at each end.

    `target[i]` / `draft[i]` describe the same position `i`, so the two lists must
    be the same length — that length *is* `k`, the draft length.
    """
    if len(target) != len(draft):
        raise ValueError(
            f"target has {len(target)} positions, draft has {len(draft)}"
        )
    if not target:
        return 0.0
    total = 0.0
    survive = 1.0
    for p_i, q_i in zip(target, draft):
        survive *= overlap(p_i, q_i)
        total += survive
        if survive <= 0.0:
            break
    return total


def measure_accepted(
    target: list[Distribution],
    draft: list[Distribution],
    *,
    trials: int = 20000,
    seed: int = 0,
) -> float:
    """Monte-Carlo the sampler, for checking the closed form.

    The key detail, and the one a first version of this function got wrong: the
    draft proposes a token **sampled from `q_i`**, not `argmax q_i`. The closed
    form's ``a_i = Σ_x min(p_i(x), q_i(x))`` is the acceptance probability under
    ``x ~ q_i``; drafting the argmax instead measures a different scheme whose
    acceptance is ``min(1, p_i(x*) / q_i(x*))`` and can be 1 at every position,
    so a Monte-Carlo built that way reports `k` accepted and appears to contradict
    a formula that is in fact correct. Both schemes exist in the literature; this
    function implements the one the formula describes, and says so.
    """
    if len(target) != len(draft):
        raise ValueError("target and draft must describe the same positions")
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    rng = _Rng(seed)
    k = len(target)
    total = 0
    for _ in range(trials):
        accepted = 0
        for i in range(k):
            p_i, q_i = target[i], draft[i]
            x = rng.choice(q_i)                       # x ~ q_i
            if rng.next_float() < accept_prob(p_i[x], q_i[x]):
                accepted += 1
            else:
                break
        total += accepted
    return total / trials


def expected_tokens_per_step(
    target: list[Distribution], draft: list[Distribution]
) -> float:
    """Accepted tokens **plus one**: the step always emits at least one token."""
    return 1.0 + expected_accepted(target, draft)


def speedup(
    target: list[Distribution], draft: list[Distribution], *, cost_ratio: float = 0.15
) -> float:
    """Tokens per unit work against plain decoding, for one draft length `k = len(draft)`.

    `cost_ratio` is the cost of drafting **one** token as a fraction of one target
    forward pass (a draft model of a tenth the size is roughly `0.1`). Drafting `k`
    candidates costs ``k * cost_ratio`` and verifying them costs one target pass, so

        speedup(k) = (1 + E[accepted]) / (1 + k * cost_ratio)

    The `k *` is the whole point: a *constant* denominator would make the ratio
    increase monotonically in `k` and the cost term could never change the optimum,
    so `--draft-tokens` would have no reason to stop. With drafting priced per token
    the optimum is finite, `speedup(0) == 1.0` exactly (no draft, one token per
    step), and more drafting stops paying once the next candidate is unlikely to be
    accepted.
    """
    if cost_ratio < 0:
        raise ValueError(f"cost_ratio must be >= 0, got {cost_ratio}")
    k = len(draft)
    return expected_tokens_per_step(target, draft) / (1.0 + k * cost_ratio)


@dataclass
class DraftSweep:
    """The whole `k` sweep, so the optimum is read off one object."""

    rows: list[dict]

    def best_k(self) -> int:
        return max(self.rows, key=lambda r: r["speedup"])["k"] if self.rows else 0

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "best_k": self.best_k(),
        }


def sweep_draft_length(
    make_position: "callable",
    *,
    max_k: int,
    cost_ratio: float = 0.15,
) -> DraftSweep:
    """Evaluate every draft length up to `max_k` for one accept/reject profile.

    `make_position(i)` returns ``(p_i, q_i)`` for position `i`. Building the
    prefix once and slicing it is what makes the sweep cheap: the per-position
    overlap `a_i` does not depend on `k`.
    """
    if max_k < 0:
        raise ValueError(f"max_k must be >= 0, got {max_k}")
    if cost_ratio < 0:
        raise ValueError(f"cost_ratio must be >= 0, got {cost_ratio}")
    pairs = [make_position(i) for i in range(max_k)]
    rows: list[dict] = []
    for k in range(max_k + 1):
        t = [p for p, _ in pairs[:k]]
        d = [q for _, q in pairs[:k]]
        acc = expected_accepted(t, d)
        rows.append({
            "k": k,
            "expected_accepted": acc,
            "tokens_per_step": 1.0 + acc,
            "draft_cost": k * cost_ratio,
            "speedup": (1.0 + acc) / (1.0 + k * cost_ratio),
            "cumulative_accept": math.prod(overlap(*pairs[i]) for i in range(k)) if k else 1.0,
        })
    return DraftSweep(rows)
