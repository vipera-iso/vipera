"""Read a routing trace and answer: can an expert pool of this size be armed?

Not a benchmark - a budget check. It reports three numbers the plan depends on:

* how many expert tensors a single decode step touches (top_k x MoE layers),
* how many bytes that working set is on the target model,
* the hit rate an LRU pool of a given size would reach on this trace.

The hit rate is what decides whether a pool helps: a pool smaller than the
per-step working set evicts the expert it will need next layer, and llama.cpp
pays the copy twice (see `_MOE_CACHE_MIN_SPARE_GB` in the planner).

    python3 bench/trace_pool_budget.py trace.jsonl --expert-bytes 1696481280
"""

from __future__ import annotations

import argparse
import sys

from vipera.trace import load_trace, stats


def working_set_per_step(routing) -> int:
    """Expert tensors one decode step reads: top_k for every layer."""
    return len(routing[0]) * len(routing[0][0])


def lru_hit_rate(routing, capacity: int) -> float:
    """Fraction of (layer, expert) requests a global LRU pool of `capacity` serves."""
    if capacity <= 0:
        return 0.0
    cache: list[tuple[int, int]] = []
    hits = misses = 0
    for token in routing:
        for layer, experts in enumerate(token):
            for e in experts:
                key = (layer, e)
                if key in cache:
                    hits += 1
                    cache.remove(key)
                    cache.append(key)
                else:
                    misses += 1
                    if len(cache) >= capacity:
                        cache.pop(0)
                    cache.append(key)
    return hits / (hits + misses) if hits + misses else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument(
        "--expert-bytes", type=int, required=True, help="bytes of one MoE layer's experts"
    )
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument(
        "--pool-mib", type=int, nargs="*", default=[128, 256, 512, 832, 1024, 2048, 4096]
    )
    args = ap.parse_args(argv)

    routing = load_trace(args.trace)
    per_expert = args.expert_bytes / args.n_experts
    ws = working_set_per_step(routing)
    s = stats(routing, n_experts=args.n_experts)

    print(f"trace          {len(routing)} tokens, {s.layers} layers, top_k {s.top_k}")
    print(f"expert tensor  {per_expert / 1024**2:.2f} MiB")
    print(f"step working set {ws} tensors = {ws * per_expert / 1024**3:.2f} GB")
    print(f"mean entropy   {s.mean_entropy:.3f} (1.0 = uniform)")
    print()
    print(f"{'pool':>8}  {'tensors':>7}  {'hit rate':>8}")
    for mib in args.pool_mib:
        cap = int(mib * 1024**2 // per_expert)
        print(f"{mib:>6} MiB  {cap:>7}  {lru_hit_rate(routing, cap):>7.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
