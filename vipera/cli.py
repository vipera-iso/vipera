"""vipera command line.

    python -m vipera analyze <model.gguf> [--json]
    python -m vipera plan    <model.gguf> --target hw/gtx1650.toml [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .analyze import analyze, to_dict
from .gguf import GGUFError, read_gguf
from .manifest import Target, dumps, plan

MIB = 1024 * 1024


def _fmt(n: int) -> str:
    if n >= 1024 * MIB:
        return f"{n / (1024 * MIB):.2f} GiB"
    return f"{n / MIB:.1f} MiB"


def cmd_analyze(args: argparse.Namespace) -> int:
    gguf = read_gguf(args.model)
    inv = analyze(gguf)
    if args.json:
        json.dump(to_dict(inv), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"file          {args.model}")
    print(f"format        GGUF v{gguf.version}  ({_fmt(gguf.file_size)} on disk)")
    print(f"architecture  {inv.architecture}")
    print(f"layers        {inv.n_layers}")
    print(f"experts       {inv.n_experts}")
    print(f"tensors       {len(inv.tensors)}")
    print()
    print("bytes")
    print(f"  resident (attn/norm)   {_fmt(inv.resident_bytes)}")
    print(f"  router                 {_fmt(inv.router_bytes)}")
    print(f"  shared experts         {_fmt(inv.shared_expert_bytes)}")
    print(f"  other (embd/output)    {_fmt(inv.other_bytes)}")
    print(f"  -> always resident     {_fmt(inv.always_resident_bytes)}")
    print(f"  routable experts       {_fmt(inv.expert_bytes)}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    gguf = read_gguf(args.model)
    inv = analyze(gguf)
    target = Target.from_toml(args.target)
    result = plan(inv, target)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.model))[0]
        path = os.path.join(args.out, f"{base}.vipera.json")
        with open(path, "w") as f:
            f.write(dumps(result))
        print(f"manifest written to {path}")

    print(json.dumps(result.manifest["placement"], indent=2))
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0 if result.runnable else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vipera", description=__doc__)
    p.add_argument("--version", action="version", version=f"vipera {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="inventory a GGUF model")
    a.add_argument("model")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    pl = sub.add_parser("plan", help="plan weight placement for a hardware target")
    pl.add_argument("model")
    pl.add_argument("--target", required=True, help="hardware profile (TOML)")
    pl.add_argument("--out", help="directory to write <model>.vipera.json into")
    pl.set_defaults(func=cmd_plan)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GGUFError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
