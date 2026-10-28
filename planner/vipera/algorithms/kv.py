"""KV retention policies, as offline reference implementations.

llama.cpp implements exactly one of these — StreamingLLM's sink + sliding window
(`--context-shift --keep N`, `2309.17453`) — and the local patch
`bench/llama-ppl-stream.patch` adds a second family that keys on the *streaming*
generation signal (called `surprise` here). H2O (`2306.14048`) is a third, and it
is **not** implemented in the patch because it needs the attention weight each
cached token received, and llama.cpp exposes no such number.

This module is the reference implementation of all four over an explicit
attention stream. It never runs a model: it answers "what would H2O buy?" on a
stream whose answer can be computed, and it puts the policies on one code path so
they can be compared without four builds.

**The two metrics are the point.** A retention policy is scored against an
attention stream, but "retained attention" is ambiguous, and the ambiguity is
exactly where a comparison goes wrong:

* ``retained_mass`` — the share of **this step's** attention row held by the
  survivors. This is what the *next* query needs, and it is what a sliding window
  is built to serve.
* ``retained_accumulated`` — the share of the **accumulated** score held by the
  survivors. This is the objective H2O's Algorithm 1 maximizes, so H2O wins it by
  construction; a simulation that reported only this metric would be measuring the
  algorithm against its own objective and calling it a result.

The two disagree, and the repo has already measured which one tracks quality. On
the same 49921 tokens and 131 context shifts (`bench/results/ppl-policy-summary.txt`):

    recency   PPL 21.4478   <- keeps the recent row, ignores accumulation
    surprise  PPL 31.9334   <- scores on an accumulated generation signal

The accumulated-signal policy lost by 48%. Treat ``retained_mass`` as the proxy
worth ranking on, and read ``retained_accumulated`` as "the objective this policy
was written to maximize", never as a quality prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The four policies this module implements. Order is by how much information each
# one uses, not by expected quality — the ranking depends on the stream.
POLICIES = ("recency", "sink-window", "h2o", "surprise")

# StreamingLLM's own count, and the same floor `vipera.longcontext` uses: with
# zero sinks the model degrades sharply, which is why llama.cpp's `--keep 0`
# default is not usable as-is. H2O deliberately does **not** use this: the paper
# has no forced sink set, and its own pinned-front variant is what the bench patch
# already covers.
DEFAULT_SINKS = 4


@dataclass
class RetentionResult:
    """What one policy retained over one attention stream."""

    policy: str
    budget: int
    sinks: int
    steps: int
    # Mean share of each step's attention row held by the survivors.
    retained_mass: float
    # Mean share of the accumulated score held by the survivors, at each step.
    retained_accumulated: float
    # Positions dropped across the whole run.
    evictions: int
    # The newest position must be pinned for llama.cpp (the next batch requires
    # KV position Y = X + 1). This counts the steps where the policy's *own*
    # choice would have dropped the newest, before the pin is applied.
    newest_would_drop: int

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "budget": self.budget,
            "sinks": self.sinks,
            "steps": self.steps,
            "retained_mass": round(self.retained_mass, 6),
            "retained_accumulated": round(self.retained_accumulated, 6),
            "evictions": self.evictions,
            "newest_would_drop": self.newest_would_drop,
        }


def _top_positions(scores: list[float], count: int) -> list[int]:
    """The `count` highest-scoring positions, ties broken toward recency."""
    if count <= 0:
        return []
    if count >= len(scores):
        return list(range(len(scores)))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i), reverse=True)
    return sorted(order[:count])


def _keep_set(
    policy: str,
    *,
    step: int,
    budget: int,
    sinks: int,
    attention_score: list[float],
    surprise_score: list[float],
) -> tuple[list[int], bool]:
    """The positions retained once `step` is decoded, and whether the newest was pinned.

    Returns `(keep, newest_pinned)`. The second value is True when the policy's own
    choice dropped position `step` and the llama.cpp adjacency rule forced it back
    in — the count of those is reported rather than hidden, because a policy that
    needs the rule applied is a different policy from the one in the paper.
    """
    live = step + 1
    keep_n = min(budget, live)

    if policy == "recency":
        keep = set(range(live - keep_n, live))

    elif policy == "sink-window":
        # Sinks are capped at a quarter of the budget so a small budget still
        # leaves somewhere to attend to, exactly as `longcontext` clamps them.
        sink_n = min(sinks, max(1, keep_n // 4))
        window = keep_n - sink_n
        tail = set(range(live - window, live)) if window else set()
        keep = set(range(sink_n)) | tail

    elif policy == "h2o":
        # H2O Algorithm 1: S_i = S_{i-1} ∪ {i} then evict the element whose removal
        # leaves the largest remaining score — i.e. evict the *minimum* accumulated
        # score. No sink set, no recency window: the accumulated score is the whole
        # decision, which is why the paper calls it a dynamic submodular
        # maximization of F_score.
        keep = set(_top_positions(attention_score[:live], keep_n))

    elif policy == "surprise":
        # Same shape of decision as h2o, a different signal: the NLL the model
        # assigned to each token rather than the attention it received.
        keep = set(_top_positions(surprise_score[:live], keep_n))

    else:
        raise ValueError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")

    pinned = step not in keep
    if pinned:
        # Drop the lowest-scoring survivor to stay inside the budget.
        victims = sorted(
            (p for p in keep if p != step),
            key=lambda i: (
                attention_score[i] if policy != "surprise" else surprise_score[i],
                i,
            ),
        )
        if victims:
            keep.discard(victims[0])
        keep.add(step)
    return sorted(keep), pinned


def simulate(
    policy: str,
    attention: list[list[float]],
    *,
    budget: int,
    sinks: int = DEFAULT_SINKS,
    surprise: list[float] | None = None,
) -> RetentionResult:
    """Run `policy` over an explicit attention stream and report what it kept.

    `attention[t]` holds the attention weight the query at step `t` gives to every
    position `0..t`, so `len(attention[t]) == t + 1`. This is the **full-attention
    oracle**: the weights the model gives when nothing has been evicted. Scoring
    retention against it is the standard offline comparison of eviction policies,
    and it is also why the result is a proxy — a real cache decides on partial
    information and then attends to whatever survived.

    `surprise[t]`, when given, is the negative log-likelihood the model assigned to
    the token emitted at step `t`. Only the `surprise` policy reads it.

    Raises ValueError unless `budget >= 1` and every row is one longer than the last
    (a malformed stream would silently shift every index).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    for t, row in enumerate(attention):
        if len(row) != t + 1:
            raise ValueError(
                f"attention[{t}] has {len(row)} entries, expected {t + 1}: the "
                "stream must be lower-triangular (step t attends to 0..t)"
            )
    if surprise is not None and len(surprise) != len(attention):
        raise ValueError(
            f"surprise has {len(surprise)} entries, attention has {len(attention)}"
        )

    steps = len(attention)
    if steps == 0:
        return RetentionResult(policy, budget, sinks, 0, 0.0, 0.0, 0, 0)

    attention_score = [0.0] * steps
    surprise_score = [0.0] * steps
    mass_total = 0.0
    mass_kept = 0.0
    acc_total = 0.0
    acc_kept = 0.0
    evictions = 0
    newest_drops = 0

    for t in range(steps):
        row = attention[t]
        row_sum = math.fsum(row)
        for i, w in enumerate(row):
            attention_score[i] += w
        if surprise is not None:
            surprise_score[t] = surprise[t]

        keep, pinned = _keep_set(
            policy,
            step=t,
            budget=budget,
            sinks=sinks,
            attention_score=attention_score,
            surprise_score=surprise_score,
        )
        if pinned:
            newest_drops += 1
        evictions += (t + 1) - len(keep)

        if row_sum > 0:
            mass_total += row_sum
            mass_kept += math.fsum(row[i] for i in keep if i <= t)

        # The accumulated objective, on the scores as they stand after this step.
        live_scores = attention_score[: t + 1]
        total_now = math.fsum(live_scores)
        if total_now > 0:
            acc_total += total_now
            acc_kept += math.fsum(live_scores[i] for i in keep if i <= t)

    return RetentionResult(
        policy=policy,
        budget=budget,
        sinks=sinks,
        steps=steps,
        retained_mass=mass_kept / mass_total if mass_total > 0 else 0.0,
        retained_accumulated=acc_kept / acc_total if acc_total > 0 else 0.0,
        evictions=evictions,
        newest_would_drop=newest_drops,
    )


def compare(
    attention: list[list[float]],
    *,
    budget: int,
    sinks: int = DEFAULT_SINKS,
    surprise: list[float] | None = None,
) -> list[RetentionResult]:
    """Every policy over the same stream, so the ranking is on one footing."""
    return [
        simulate(p, attention, budget=budget, sinks=sinks, surprise=surprise)
        for p in POLICIES
    ]


def synth_attention(
    steps: int,
    *,
    n_positions: int | None = None,
    seed: int = 0,
    sinks: int = 4,
    decay: float = 0.85,
    heavy_hitters: int = 6,
    sink_share: float = 0.35,
    heavy_share: float = 0.25,
    recency_share: float = 0.35,
) -> list[list[float]]:
    """A synthetic lower-triangular attention stream with the four real effects.

    Each is there because it changes which policy wins, and leaving one out is how
    a comparison silently becomes about the generator instead of the policies:

    * `sinks` positions at the front take a fixed share (attention sinks — why
      StreamingLLM pins the first tokens and llama.cpp has `--keep`);
    * a **decaying** recency share, so the last few positions dominate (locality);
    * `heavy_hitters` positions drawn from anywhere in the sequence take a share
      that does not decay with age — the tokens only an accumulated-score rule can
      find and a sliding window always drops;
    * a small uniform tail over everything still live.

    Deterministic given `seed`, so a test asserts on exact numbers rather than on a
    tolerance around noise. Pass `heavy_share=0` for the pure streaming regime.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    n = n_positions if n_positions is not None else steps
    if n < 1:
        raise ValueError(f"n_positions must be >= 1, got {n}")
    total = sink_share + heavy_share + recency_share
    if total <= 0:
        raise ValueError("at least one of the shares must be positive")
    if min(sink_share, heavy_share, recency_share) < 0:
        raise ValueError("shares must be non-negative")
    tail_share = max(0.0, 1.0 - total)

    # A small deterministic generator: os.urandom would make the numbers
    # unreproducible and `random` is slow for the volume a sweep produces.
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    hitters: set[int] = set()
    guard = 0
    while len(hitters) < min(heavy_hitters, n) and guard < 10000:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        hitters.add((state >> 33) % n)
        guard += 1

    rows: list[list[float]] = []
    for t in range(steps):
        live = min(t + 1, n)
        row = [0.0] * (t + 1)
        sink_each = sink_share / max(1, sinks) if sinks else 0.0
        heavy_each = heavy_share / max(1, len(hitters)) if hitters else 0.0
        for i in range(live):
            w = 0.0
            if i < sinks:
                w += sink_each
            if i in hitters:
                w += heavy_each
            if i not in hitters and i >= sinks:
                w += recency_share * (decay ** (t - i))
            if w == 0.0:
                w = tail_share / max(1, live)
            row[i] += w
        if tail_share > 0 and live:
            tail_each = tail_share / live
            for i in range(live):
                row[i] += tail_each * (0.5 + ((i * 2654435761) % 1000) / 2000.0)
        # `row[t]` must be strictly positive: a zero would make the newest position
        # unattendable and the mass ratio undefined at that step.
        row[t] += 1e-6
        rows.append(row)
    return rows


def synth_surprise(attention: list[list[float]], *, seed: int = 0) -> list[float]:
    """A deterministic surprise stream, anti-correlated with received attention.

    Deliberately adversarial: the highest-surprise tokens are the ones the model
    attends to least. A stream built the other way would make the `surprise` policy
    look like `h2o` and hide that they are different signals.
    """
    state = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)
    out: list[float] = []
    for row in attention:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        jitter = 0.5 + (state >> 11) / (1 << 53)
        mass = math.fsum(row) or 1.0
        share = max(row) / mass
        out.append((1.0 - share) * jitter * 4.0)
    return out
