"""vipera command line.

    python -m vipera analyze  <model.gguf> [--json]
    python -m vipera plan     <model.gguf> --target hw/gtx1650.toml [--out DIR]
    python -m vipera arch     arch/glm-5.3-flash.toml
    python -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .analyze import analyze, to_dict
from .arch import Architecture
from .gguf import GGUFError, read_gguf
from .manifest import Target, dumps, plan
from .schedule import Quant, dumps as schedule_dumps, simulate

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


def cmd_arch(args: argparse.Namespace) -> int:
    arch = Architecture.from_toml(args.arch)
    kinds = arch.layer_kinds()
    print(f"architecture  {arch.name}")
    print(f"layers        {arch.n_layers}  ({kinds.count('kda')} KDA + {kinds.count('mla')} MLA)")
    print(f"experts       {arch.n_experts}  top-{arch.top_k}")
    print(f"hidden        {arch.hidden}   moe_inter {arch.moe_inter}")
    print()
    print("parameters")
    for name, n in arch.params_summary().items():
        print(f"  {name:<16} {n / 1e9:8.3f} B")
    total = arch.total_params(include_vision=False)
    active = arch.active_params()
    print(f"  {'TOTAL':<16} {total / 1e9:8.3f} B")
    print(f"  {'ACTIVE/token':<16} {active / 1e9:8.3f} B")
    print(f"  {'ratio':<16} {total / active:8.2f}")
    if arch.stated_total and arch.stated_active:
        cal = arch.calibrate(arch.stated_total, arch.stated_active)
        print()
        print("calibration vs published numbers")
        print(f"  total   derived {cal['total'] / 1e9:.1f}B vs stated {cal['total_stated'] / 1e9:.0f}B"
              f"  ({cal['total_error'] * 100:+.1f}%)")
        print(f"  active  derived {cal['active'] / 1e9:.2f}B vs stated {cal['active_stated'] / 1e9:.0f}B"
              f"  ({cal['active_error'] * 100:+.1f}%)")
    return 0


_QUANTS = {"fp8": Quant.fp8, "q4-experts": Quant.q4_experts, "q4": Quant.q4_all}


def cmd_schedule(args: argparse.Namespace) -> int:
    arch = Architecture.from_toml(args.arch)
    target = Target.from_toml(args.target)
    result = simulate(
        arch, target,
        quant=_QUANTS[args.quant](),
        context=args.context,
        tokens=args.tokens,
        prefetch=not args.no_prefetch,
    )
    r = result.manifest["residency"]
    p = result.manifest["performance"]
    kv = result.manifest["context"]
    print(f"architecture  {arch.name}  ({arch.n_layers} layers, {arch.n_experts} experts top-{arch.top_k})")
    print(f"target        {target.name}")
    print(f"quant         {args.quant}")
    print()
    print("residency")
    print(f"  always resident      {_fmt(r['resident_bytes'])}")
    print(f"    -> VRAM            {_fmt(r['resident_vram_bytes'])}")
    print(f"    -> RAM             {_fmt(r['resident_ram_bytes'])}")
    print(f"    -> disk (stream)   {_fmt(r['resident_spill_disk_bytes'])}")
    print(f"  expert cache         {_fmt(r['expert_cache_bytes'])}  ({r['expert_cache_slots']} slots)")
    print(f"  expert pool          {_fmt(r['expert_pool_bytes'])}")
    print(f"  experts on disk      {_fmt(r['expert_spill_disk_bytes'])}")
    print(f"  KV @ {kv['tokens']} ctx       {_fmt(kv['kv_bytes'])}")
    print()
    print("performance (modelled)")
    print(f"  cache hit rate       {result.hit_rate * 100:.1f}%  ({result.manifest['expert_cache']['routing_source']})")
    print(f"  bytes/token          {_fmt(result.bytes_per_token)}")
    print(f"  disk bytes/token     {_fmt(result.disk_bytes_per_token)}")
    print(f"  ms/token             {result.ms_per_token:.1f}")
    print(f"  tokens/s             {result.tok_s:.3f}   (bound: {p['bound']})")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{arch.name}.vipera.json")
        with open(path, "w") as f:
            f.write(schedule_dumps(result))
        print(f"\nmanifest written to {path}")
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


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

    ar = sub.add_parser("arch", help="report a model architecture and its calibration")
    ar.add_argument("arch", help="architecture profile (TOML)")
    ar.set_defaults(func=cmd_arch)

    sc = sub.add_parser("schedule", help="plan a streaming schedule and model its speed")
    sc.add_argument("arch", help="architecture profile (TOML)")
    sc.add_argument("--target", required=True, help="hardware profile (TOML)")
    sc.add_argument("--quant", choices=sorted(_QUANTS), default="fp8")
    sc.add_argument("--context", type=int, default=2048)
    sc.add_argument("--tokens", type=int, default=8, help="decode steps to simulate")
    sc.add_argument("--no-prefetch", action="store_true", help="serialise I/O and compute")
    sc.add_argument("--out", help="directory to write <arch>.vipera.json into")
    sc.set_defaults(func=cmd_schedule)
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
