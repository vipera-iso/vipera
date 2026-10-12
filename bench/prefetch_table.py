"""Turn a routing trace into the expert prefetch table the llama.cpp fork patch reads.

`bench/llama-moe-prefetch.patch` teaches llama.cpp's MoE expert cache to upload the
experts a layer is *likely* to need while the previous layer computes. llama.cpp
cannot work that out on its own: the router only reveals a layer's experts when that
layer is about to run, so the stock cache is filled strictly on demand and every
upload is exposed. The prediction has to come from a routing trace, and this script
is the bridge from vipera's traces to that patch.

Output format, one line per layer (read by `llama_moe_prefetch_table`):

    # comment
    <layer index>  <expert>,<expert>,...      experts most likely first

Usage:

    python3 -m bench.prefetch_table bench/results/olmoe-1b-7b-511tok.jsonl -o /tmp/pf.txt
    LLAMA_MOE_PREFETCH=/tmp/pf.txt llama-cli -m model.gguf \\
        -ot exps=CPU --moe-cache-mib 1024 -p "hello"

The candidates are vipera's own answer to "which experts hold this layer's routing
mass" (`vipera.prune.plan_from_trace`), so the table and the pruning advice stay the
same measurement. `--top` caps each layer, which is what the cache budget usually
forces: with a small `--moe-cache-mib` there are fewer slots than candidates.
"""

from __future__ import annotations

import argparse
import sys

from vipera.prune import plan_from_trace
from vipera.trace import load_trace


def n_experts_in(routing: list[list[list[int]]]) -> int:
    """Number of experts the trace refers to (one past the largest id seen)."""
    largest = -1
    for token in routing:
        for layer in token:
            for e in layer:
                if e > largest:
                    largest = e
    return largest + 1


def build_table(routing: list[list[list[int]]], *, target_mass: float = 0.95,
                top: int = 0) -> list[list[int]]:
    """Per-layer expert candidates, most likely first.

    `target_mass` is how much of each layer's routing mass the candidates must
    cover; `top` (0 = no cap) truncates each layer afterwards. Returns one entry
    per layer, in layer order.
    """
    if not routing:
        return []
    n_experts = n_experts_in(routing)
    if n_experts <= 0:
        return []
    plan = plan_from_trace(routing, n_experts=n_experts, target_mass=target_mass)
    return [l.keep[:top] if top > 0 else list(l.keep) for l in plan.layers]


def format_table(table: list[list[int]], *, source: str = "", tokens: int = 0,
                 target_mass: float = 0.95) -> str:
    """Render the table in the format `llama_moe_prefetch_table` parses."""
    head = ["# llama.cpp MoE prefetch table (bench/llama-moe-prefetch.patch)"]
    if source:
        head.append(f"# source: {source}")
    head.append(f"# {tokens} tokens, target routing mass {target_mass}")
    lines = head
    for il, keep in enumerate(table):
        if not keep:
            continue
        lines.append(f"{il}\t{','.join(str(e) for e in keep)}")
    return "\n".join(lines) + "\n"


def table_stats(table: list[list[int]]) -> dict:
    """What the table costs and covers, so a run can be compared with the printout."""
    n_layers = len(table)
    total = sum(len(k) for k in table)
    return {
        "layers": n_layers,
        "experts": total,
        "mean_per_layer": round(total / n_layers, 2) if n_layers else 0.0,
        "max_per_layer": max((len(k) for k in table), default=0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", help="routing trace (.jsonl) to learn the table from")
    ap.add_argument("-o", "--out", default="-", help="output path, '-' for stdout")
    ap.add_argument("--target-mass", type=float, default=0.95,
                    help="fraction of each layer's routing mass to cover (default 0.95)")
    ap.add_argument("--top", type=int, default=0,
                    help="cap experts per layer, 0 = keep the whole keep-set")
    ap.add_argument("--limit", type=int, default=None, help="only read the first N tokens")
    args = ap.parse_args(argv)

    routing = load_trace(args.trace, limit=args.limit)
    table = build_table(routing, target_mass=args.target_mass, top=args.top)
    text = format_table(table, source=args.trace, tokens=len(routing),
                        target_mass=args.target_mass)

    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"{args.out}: {table_stats(table)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
