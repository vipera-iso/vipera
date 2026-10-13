"""Turn a routing trace into the expert prefetch table the llama.cpp patch reads.

Thin CLI over `vipera.prefetch` — the table logic moved into the package so
`vipera run --trace` can build a table without importing `bench`.

    python3 -m bench.prefetch_table bench/results/olmoe-1b-7b-511tok.jsonl -o /tmp/pf.txt
    LLAMA_MOE_PREFETCH=/tmp/pf.txt llama-cli -m model.gguf \\
        -ot exps=CPU --moe-cache-mib 1024 -p "hello"
"""

from __future__ import annotations

import argparse
import sys

from vipera.prefetch import (
    build_table,
    format_table,
    n_experts_in,
    table_stats,
)
from vipera.trace import load_trace

__all__ = ["build_table", "format_table", "n_experts_in", "table_stats", "main"]


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
