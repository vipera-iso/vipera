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
from .schedule import (
    Optimizations,
    Quant,
    dumps as schedule_dumps,
    plan_residency,
    simulate,
)
from .prune import (
    dumps as prune_dumps,
    gated_dumps,
    gated_prune,
    mass_curve,
    plan_from_trace,
)
from .scoring import local_llm_16gb_score
from .trace import (
    TraceError,
    dumps as trace_dumps,
    global_cache_ceiling,
    load_trace,
    stats as trace_stats,
    validate as validate_trace,
)

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
    opt = Optimizations(
        prune_experts=args.prune_experts,
        keep_layers=args.keep_layers,
        kv_bits=args.kv_bits,
        draft_tokens=args.draft_tokens,
    )
    routing = None
    # With a trace, use all of it unless the caller asked for a shorter run —
    # silently capping a 100k-token trace at the modelled default of 8 steps is
    # what made round 1's hit-rate numbers a 10-token sample.
    tokens = args.tokens
    if args.trace:
        limit = None if tokens is None else tokens + 2
        routing = load_trace(args.trace, limit=limit)
        validate_trace(routing, n_layers=arch.n_layers, n_experts=arch.n_experts,
                       top_k=arch.top_k)
    result = simulate(
        arch, target,
        quant=_QUANTS[args.quant](),
        optimizations=opt,
        context=args.context,
        tokens=tokens if tokens is not None else 8,
        prefetch=not args.no_prefetch,
        cache_policy=args.cache_policy,
        cache_alloc=args.cache_alloc,
        routing=routing,
        eval_tail=args.eval_tail,
    )
    r = result.manifest["residency"]
    p = result.manifest["performance"]
    kv = result.manifest["context"]
    n_layers = result.manifest["architecture"]["n_layers"]
    n_experts = result.manifest["architecture"]["n_experts"]
    print(f"architecture  {arch.name}  ({n_layers} layers, {n_experts} experts top-{arch.top_k})")
    print(f"target        {target.name}")
    print(f"quant         {args.quant}")
    print(f"transforms    prune_experts={opt.prune_experts} keep_layers={opt.keep_layers} "
          f"kv_bits={opt.kv_bits} draft_tokens={opt.draft_tokens}")
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
    ec = result.manifest["expert_cache"]
    print(f"  cache hit rate       {result.hit_rate * 100:.1f}%  ({ec['routing_source']})")
    if ec.get("alloc") == "per-layer":
        print(f"  allocation           per-layer water-filling ({ec['reallocations']} reallocations)")
    # The static ceiling is computed on this very routing at this very cache
    # size, so the ratio can never be a mismatched-baseline artefact again.
    print(f"  static ceiling       {ec['static_cache_ceiling'] * 100:.1f}%  "
          f"({ec['slots']} slots = {ec['per_layer_fraction'] * 100:.2f}% of experts/layer)")
    if ec["ceiling_ratio"] is not None:
        print(f"  policy / ceiling     {ec['ceiling_ratio']:.2f}x")
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


def cmd_trace(args: argparse.Namespace) -> int:
    arch = Architecture.from_toml(args.arch)
    routing = load_trace(args.trace)
    validate_trace(routing, n_layers=arch.n_layers, n_experts=arch.n_experts, top_k=arch.top_k)

    # Cache size: derived from the *same* arithmetic `schedule` uses when a
    # target is given, so the ceiling here is at the size the simulation will
    # actually have. Without it we fall back to a manual fraction.
    slots: int | None = None
    if args.target:
        target = Target.from_toml(args.target)
        plan = plan_residency(arch, target, quant=_QUANTS[args.quant]())
        fraction = plan.per_layer_fraction
        slots = plan.slots
        size_source = f"matched to {target.name} / {args.quant} ({slots} slots)"
    else:
        fraction = args.cache_fraction if args.cache_fraction is not None else 0.03
        size_source = f"manual fraction {fraction}"

    s = trace_stats(routing, n_experts=arch.n_experts, cache_fraction=fraction)
    if args.json:
        out = s.as_dict()
        out["cache_size_source"] = size_source
        if slots is not None:
            out["slots"] = slots
            out["global_static_cache_hit_ceiling"] = round(
                global_cache_ceiling(routing, slots), 4)
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print(f"trace         {args.trace}")
    print(f"architecture  {arch.name}")
    print(f"tokens        {s.tokens}   layers {s.layers}   top_k {s.top_k}")
    print(f"mean entropy  {s.mean_entropy:.3f}  (1.0 = uniform, no cache can help)")
    print(f"cache size    {size_source}")
    print(f"  per-layer ceiling  {s.static_cache_fraction * 100:.2f}% of experts/layer "
          f"-> {s.static_cache_hit_ceiling * 100:.1f}%")
    if slots is not None:
        g = global_cache_ceiling(routing, slots)
        print(f"  global ceiling     {slots} pooled slots -> {g * 100:.1f}%  "
              "(the size-matched bound a streamed cache compares against)")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    """How much of the routing mass survives a given expert sparsity?"""
    arch = Architecture.from_toml(args.arch)
    routing = load_trace(args.trace)
    validate_trace(routing, n_layers=arch.n_layers, n_experts=arch.n_experts,
                   top_k=arch.top_k)
    plan = plan_from_trace(routing, n_experts=arch.n_experts,
                           target_mass=args.target_mass)
    if args.json:
        out = plan.as_dict()
        out["mass_curve"] = mass_curve(
            routing, n_experts=arch.n_experts,
            fractions=[0.05, 0.10, 0.25, 0.50, 0.75],
        )
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"trace         {args.trace}")
    print(f"architecture  {arch.name}  ({arch.n_experts} experts/layer)")
    print(f"target mass   {plan.target_mass:.0%}")
    print()
    print(f"mass-based keep-set: {plan.total_keep} of {plan.total_experts} experts "
          f"({plan.keep_fraction * 100:.2f}% of the pool), "
          f"retaining {plan.mean_kept_mass * 100:.1f}% of routing mass")
    print()
    print(f"{'sparsity':>9}{'mass kept':>12}{'if uniform':>12}{'gain':>9}")
    for row in mass_curve(routing, n_experts=arch.n_experts,
                          fractions=[0.05, 0.10, 0.25, 0.50, 0.75]):
        f = row["fraction"]
        print(f"{f * 100:>8.0f}%{row['kept_mass'] * 100:>11.1f}%"
              f"{row['uniform_kept_mass'] * 100:>11.1f}%"
              f"{(row['kept_mass'] - row['uniform_kept_mass']) * 100:>+8.1f}")
    print()
    print("retained routing mass is a NECESSARY condition, not a quality result: "
          "building and running pruned weights is S9+S10")
    return 0


def cmd_prune_gated(args: argparse.Namespace) -> int:
    """P4: prune experts only where the gate passes, roll back the layers that don't."""
    arch = Architecture.from_toml(args.arch)
    routing = load_trace(args.trace)
    validate_trace(routing, n_layers=arch.n_layers, n_experts=arch.n_experts,
                   top_k=arch.top_k)

    drift = None
    if args.drift:
        with open(args.drift) as f:
            drift = {int(k): float(v) for k, v in json.load(f).items()}

    plan = gated_prune(routing, n_experts=arch.n_experts, target_mass=args.target_mass,
                       max_keep_fraction=args.max_keep_fraction,
                       delta=args.delta, drift=drift)
    if args.json:
        json.dump(plan.as_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"trace         {args.trace}")
    print(f"architecture  {arch.name}  ({arch.n_experts} experts/layer)")
    print(f"target mass   {plan.target_mass:.0%}")
    print(f"max keep      {plan.max_keep_fraction:.0%} of experts/layer")
    if plan.delta is not None:
        print(f"drift budget  {plan.delta}")
    print()
    print(f"pruned {plan.pruned_layers} of {len(plan.layers)} layers "
          f"({plan.keep_fraction * 100:.2f}% of the pool kept), "
          f"retaining {plan.mean_kept_mass * 100:.1f}% of routing mass")
    print(f"rolled back {plan.rolled_back_layers} layers")
    reasons = plan.as_dict()["reasons"]
    print(f"reasons: {', '.join(f'{k}={v}' for k, v in sorted(reasons.items()))}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run a fixed set of layer-level strategies and print them side by side."""
    arch = Architecture.from_toml(args.arch)
    target = Target.from_toml(args.target)

    strategies = [
        ("baseline fp8", Quant.fp8(), Optimizations()),
        ("q4 experts", Quant.q4_experts(), Optimizations()),
        ("q4 experts + KV 8-bit", Quant.q4_experts(), Optimizations(kv_bits=8)),
        ("+ prune 25% experts", Quant.q4_experts(), Optimizations(prune_experts=0.25, kv_bits=8)),
        ("+ prune 5% experts", Quant.q4_experts(), Optimizations(prune_experts=0.05, kv_bits=8)),
        ("+ prune 5% + draft x3", Quant.q4_experts(),
         Optimizations(prune_experts=0.05, kv_bits=8, draft_tokens=3.0)),
        ("+ drop 20 layers too", Quant.q4_experts(),
         Optimizations(prune_experts=0.05, kv_bits=8, draft_tokens=3.0, keep_layers=25)),
    ]

    header = (f"{'strategy':<22}{'hit':>7}{'disk GB/tok':>13}{'bytes GB/tok':>14}"
              f"{'tok/s':>10}  bound")
    print(f"architecture  {arch.name}   target  {target.name}   ctx {args.context}")
    print(header)
    print("-" * len(header))
    for name, quant, opt in strategies:
        r = simulate(arch, target, quant=quant, optimizations=opt, context=args.context)
        p = r.manifest["performance"]
        print(f"{name:<22}{r.hit_rate * 100:>6.1f}%{r.disk_bytes_per_token / 1e9:>13.2f}"
              f"{r.bytes_per_token / 1e9:>14.2f}{r.tok_s:>10.2f}  {p['bound']}")
    print()
    print("all values MODELLED on the target machine's bandwidths; not a real run")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score a config with the LOCAL-LLM-16GB formula.

    S = (Q/100) * (T/10) * (16/M) * (65/P) * (32k/C)

    T comes from the simulation; Q (quality %) and P (watts) are inputs because
    vipera does not model accuracy or power. M is the modelled peak RAM (resident
    RAM + KV cache); C is the context that actually fit.
    """
    arch = Architecture.from_toml(args.arch)
    target = Target.from_toml(args.target)
    result = simulate(
        arch, target, quant=_QUANTS[args.quant](), context=args.context, tokens=args.tokens
    )
    r = result.manifest["residency"]
    ram_gb = (r["resident_ram_bytes"] + result.manifest["context"]["kv_bytes"]) / 1e9
    c = args.context
    since = local_llm_16gb_score(
        quality_pct=args.quality, tok_s=result.tok_s, ram_gb=ram_gb,
        power_w=args.power, context=c,
    )
    print(f"architecture  {arch.name}")
    print(f"target        {target.name}")
    print(f"quant         {args.quant}")
    print()
    print(f"Q (quality)   {args.quality}%")
    print(f"T (tokens/s)  {result.tok_s:.3f}")
    print(f"M (peak RAM)  {ram_gb:.2f} GB")
    print(f"P (power)     {args.power} W")
    print(f"C (context)   {c} tokens")
    print(f"S             {since:.4f}   ({'PASS' if since >= 1.0 else 'FAIL'})")
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
    sc.add_argument("--tokens", type=int, default=None,
                    help="decode steps to simulate; with --trace, defaults to the whole trace")
    sc.add_argument("--no-prefetch", action="store_true", help="serialise I/O and compute")
    sc.add_argument("--prune-experts", type=float, default=1.0,
                    help="fraction of experts kept per layer (1.0 = none pruned)")
    sc.add_argument("--keep-layers", type=int, default=0, help="drop layers from the end (0 = all)")
    sc.add_argument("--kv-bits", type=int, default=16, help="KV cache precision (16/8/4)")
    sc.add_argument("--draft-tokens", type=float, default=1.0,
                    help="accepted tokens per target forward pass (MTP/speculative)")
    sc.add_argument("--cache-policy", choices=["lru", "lfu", "hot"], default="lru",
                    help="expert cache eviction policy (hot = admission + pinned hot set)")
    sc.add_argument("--cache-alloc", choices=["global", "per-layer"], default="global",
                    help="slot-budget allocation: one global pool, or per-layer water-filling")
    sc.add_argument("--trace", help="routing trace (JSONL) to drive the simulation")
    sc.add_argument("--eval-tail", type=float, default=0.0,
                    help="measure only the last fraction of the trace (post-shift window)")

    tr = sub.add_parser("trace", help="inspect a routing trace against an architecture")
    tr.add_argument("trace")
    tr.add_argument("--arch", required=True, help="architecture profile (TOML)")
    tr.add_argument("--target", help="hardware profile; derive the cache size from it "
                                      "(matched to `schedule`) instead of --cache-fraction")
    tr.add_argument("--quant", choices=sorted(_QUANTS), default="q4-experts",
                    help="quantization used when deriving the cache size from --target")
    tr.add_argument("--cache-fraction", type=float, default=None,
                    help="static cache size as a fraction of experts per layer "
                         "(default 0.03 when no --target is given)")
    tr.add_argument("--json", action="store_true")
    tr.set_defaults(func=cmd_trace)
    sc.add_argument("--out", help="directory to write <arch>.vipera.json into")
    sc.set_defaults(func=cmd_schedule)

    sc2 = sub.add_parser("score", help="score a config with the LOCAL-LLM-16GB formula")
    sc2.add_argument("arch", help="architecture profile (TOML)")
    sc2.add_argument("--target", required=True, help="hardware profile (TOML)")
    sc2.add_argument("--quant", choices=sorted(_QUANTS), default="q4-experts")
    sc2.add_argument("--context", type=int, default=32768)
    sc2.add_argument("--tokens", type=int, default=8)
    sc2.add_argument("--quality", type=float, default=99.0, help="quality %% of FP16 (input)")
    sc2.add_argument("--power", type=float, default=65.0, help="average watts (input)")
    sc2.set_defaults(func=cmd_score)

    pr = sub.add_parser("prune", help="choose which experts to keep from a routing trace")
    pr.add_argument("trace")
    pr.add_argument("--arch", required=True, help="architecture profile (TOML)")
    pr.add_argument("--target-mass", type=float, default=0.95,
                    help="routing mass the keep-set must retain (default 0.95)")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_prune)

    pg = sub.add_parser("prune-gated",
                        help="prune only where a gate passes, else roll back the layer (P4)")
    pg.add_argument("trace")
    pg.add_argument("--arch", required=True, help="architecture profile (TOML)")
    pg.add_argument("--target-mass", type=float, default=0.95,
                    help="routing mass the keep-set must retain (default 0.95)")
    pg.add_argument("--max-keep-fraction", type=float, default=1.0,
                    help="reject a layer whose keep-set exceeds this fraction (default 1.0)")
    pg.add_argument("--delta", type=float, default=None,
                    help="max drift proxy a layer may exceed before rollback")
    pg.add_argument("--drift", help="JSON file mapping layer index -> measured drift proxy")
    pg.add_argument("--json", action="store_true")
    pg.set_defaults(func=cmd_prune_gated)

    sw = sub.add_parser("sweep", help="compare layer-level optimizations on the same model+machine")
    sw.add_argument("arch", help="architecture profile (TOML)")
    sw.add_argument("--target", required=True, help="hardware profile (TOML)")
    sw.add_argument("--context", type=int, default=2048)
    sw.set_defaults(func=cmd_sweep)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (GGUFError, TraceError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
