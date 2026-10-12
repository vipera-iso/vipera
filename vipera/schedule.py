"""Bandwidth- and cache-aware per-layer streaming scheduler.

This is stage 2's core algorithm. Given an `Architecture` (what the model *is*)
and a `Target` (what the machine *has*), it decides for every layer where its
weights live — VRAM, RAM, or disk — and simulates decode steps to answer the one
question that decides feasibility on weak hardware:

    how many bytes must be read per token, and from which memory tier?

The simulator is explicit about its assumptions:

* Decode is memory-bandwidth-bound: time is ``bytes_read / tier_bandwidth``,
  plus a compute floor of ``2 * active_params / cpu_gflops``.
* Expert routing is *modelled* (a Zipf skew over expert ids), not measured.
  A real routing trace can be supplied; until then the hit rate is an estimate
  and the manifest says so in ``expert_cache.routing_source``.
* ``prefetch=True`` assumes a dedicated I/O thread hides disk reads of layer n
  behind compute of layer n-1, so the per-token cost is ``max(io, compute)``
  instead of ``io + compute``.

Nothing here loads a model or touches tensor data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .arch import Architecture
from .manifest import Target

GIB = 1024 ** 3
MB = 1024 * 1024


@dataclass(frozen=True)
class Quant:
    """Bytes per parameter, per component. Defaults mimic the FP8 checkpoint."""

    expert_bytes: float = 1.0
    attention_bytes: float = 1.0
    other_bytes: float = 1.0

    @classmethod
    def fp8(cls) -> "Quant":
        return cls(1.0, 1.0, 1.0)

    @classmethod
    def q4_experts(cls) -> "Quant":
        """4-bit routed experts, FP8 for everything else (the practical target)."""
        return cls(0.5, 1.0, 1.0)

    @classmethod
    def q4_all(cls) -> "Quant":
        return cls(0.5, 0.5, 0.5)

    def as_dict(self) -> dict:
        return {
            "expert_bytes_per_param": self.expert_bytes,
            "attention_bytes_per_param": self.attention_bytes,
            "other_bytes_per_param": self.other_bytes,
        }


@dataclass(frozen=True)
class Optimizations:
    """Layer-level transforms applied before scheduling (docs/OPTIMIZER.md).

    Every field is a *lever the planner models*, not something vipera performs on
    tensors yet. They exist so the effect of each technique can be compared on
    one model and one machine, instead of argued in the abstract.

    * `prune_experts`  keep only this fraction of experts per layer, by routing
      frequency. Shrinks the pooled size so more of it fits the cache (expert
      pruning has a quality cost and needs healing; assume it is measured).
    * `keep_layers`    drop layers from the end (0 = keep all). Layer pruning.
    * `kv_bits`        KV cache precision (16, 8, 4). KV compression.
    * `draft_tokens`   accepted tokens per target forward pass (MTP/speculative
      decoding). 1.0 = off; 3.0 means one pass yields three tokens.
    """

    prune_experts: float = 1.0
    keep_layers: int = 0
    kv_bits: int = 16
    draft_tokens: float = 1.0

    def as_dict(self) -> dict:
        return {
            "prune_experts": self.prune_experts,
            "keep_layers": self.keep_layers,
            "kv_bits": self.kv_bits,
            "draft_tokens": self.draft_tokens,
        }


@dataclass
class LayerBytes:
    index: int
    kind: str
    attention: int
    router: int
    shared: int
    mhc: int

    @property
    def resident(self) -> int:
        return self.attention + self.router + self.shared + self.mhc


@dataclass
class ModelBytes:
    layers: list[LayerBytes]
    expert_each: int
    n_experts: int
    embeddings: int
    mtp: int
    vision: int

    @property
    def resident_total(self) -> int:
        """Weights touched on every token: attention + router + shared + embd."""
        return sum(l.resident for l in self.layers) + self.embeddings

    @property
    def expert_pool(self) -> int:
        return self.expert_each * len(self.layers) * self.n_experts


def model_bytes(
    arch: Architecture, quant: Quant, *, n_layers: int | None = None, n_experts: int | None = None
) -> ModelBytes:
    """Turn parameter counts into byte sizes for one copy of the model."""
    kinds = arch.layer_kinds()
    if n_layers is not None:
        kinds = kinds[:n_layers]
    layers: list[LayerBytes] = []
    for i, kind in enumerate(kinds):
        layers.append(
            LayerBytes(
                index=i,
                kind=kind,
                attention=int(arch.attn_params(kind) * quant.attention_bytes),
                router=int(arch.router_params() * quant.other_bytes),
                shared=int(arch.shared_expert_params() * quant.expert_bytes),
                mhc=int(arch.mhc_params() * quant.other_bytes),
            )
        )
    return ModelBytes(
        layers=layers,
        expert_each=int(arch.expert_params_each * quant.expert_bytes),
        n_experts=n_experts if n_experts is not None else arch.n_experts,
        embeddings=int(arch.vocab * arch.hidden * quant.other_bytes),
        mtp=int(arch.mtp_params() * quant.expert_bytes),
        vision=int(arch.vision_params * quant.other_bytes),
    )


@dataclass
class LayerStep:
    index: int
    kind: str
    read_bytes: int
    disk_bytes: int
    hits: int
    misses: int
    time_ms: float

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


@dataclass
class SimResult:
    manifest: dict
    warnings: list[str] = field(default_factory=list)
    layers: list[LayerStep] = field(default_factory=list)
    steps: int = 0
    # Raw numbers kept alongside the manifest so tests (and callers) can assert
    # on them without re-deriving from rounded JSON.
    tok_s: float = 0.0
    ms_per_token: float = 0.0
    hit_rate: float = 0.0
    bytes_per_token: int = 0
    disk_bytes_per_token: int = 0

    @property
    def feasible(self) -> bool:
        """A model can only run if its always-resident weights fit VRAM+RAM."""
        return self.manifest["residency"]["resident_spill_disk_bytes"] == 0


def _zipf_sample(rng: random.Random, n: int, k: int, skew: float) -> list[int]:
    """Pick k distinct ids in [0, n) with a Zipf(skew) popularity skew."""
    if skew <= 0:
        return rng.sample(range(n), k)
    weights = [1.0 / ((i + 1) ** skew) for i in range(n)]
    chosen: list[int] = []
    idx = list(range(n))
    for _ in range(k):
        total = sum(weights[i] for i in idx)
        roll = rng.random() * total
        acc = 0.0
        picked = idx[-1]
        for i in idx:
            acc += weights[i]
            if roll <= acc:
                picked = i
                break
        chosen.append(picked)
        idx.remove(picked)
    return chosen


def _routing(
    n_layers: int, n_experts: int, top_k: int, tokens: int, skew: float, seed: int
) -> list[list[list[int]]]:
    """routing[token][layer] = list of selected expert ids."""
    rng = random.Random(seed)
    return [
        [_zipf_sample(rng, n_experts, top_k, skew) for _ in range(n_layers)]
        for _ in range(tokens)
    ]


def simulate(
    arch: Architecture,
    target: Target,
    *,
    quant: Quant | None = None,
    optimizations: Optimizations | None = None,
    context: int = 2048,
    tokens: int = 8,
    warmup: int = 2,
    prefetch: bool = True,
    seed: int = 0,
    skew: float = 1.0,
    routing: list[list[list[int]]] | None = None,
) -> SimResult:
    """Plan residency and simulate `tokens` decode steps.

    `routing`, when supplied, is ``routing[token][layer][slot] = expert_id`` and
    replaces the modelled Zipf routing with a real trace.
    """
    quant = quant or Quant.fp8()
    opt = optimizations or Optimizations()
    warnings: list[str] = []
    n_layers = opt.keep_layers or arch.n_layers
    n_experts = max(arch.top_k, int(round(arch.n_experts * opt.prune_experts)))
    mb = model_bytes(arch, quant, n_layers=n_layers, n_experts=n_experts)
    kinds = arch.layer_kinds()[:n_layers]

    # --- residency: always-needed weights first, leftover becomes expert cache
    resident_total = mb.resident_total
    resident_vram = min(resident_total, target.usable_vram)
    resident_ram = min(resident_total - resident_vram, target.usable_ram)
    resident_spill = resident_total - resident_vram - resident_ram

    cache_vram = max(0, target.usable_vram - resident_vram)
    cache_ram = max(0, target.usable_ram - resident_ram)
    cache_bytes = cache_vram + cache_ram
    slots = cache_bytes // mb.expert_each if mb.expert_each else 0

    total_experts = n_layers * n_experts
    expert_pool = mb.expert_each * total_experts
    expert_spill = max(0, expert_pool - cache_bytes)

    if resident_spill > 0:
        warnings.append(
            f"always-resident weights ({resident_total / GIB:.1f} GiB) exceed VRAM+RAM "
            f"({(target.usable_vram + target.usable_ram) / GIB:.1f} GiB); "
            f"{resident_spill / GIB:.1f} GiB of attention/shared weights would stream "
            "from disk every token"
        )
    if resident_spill == 0 and cache_bytes < expert_pool:
        warnings.append(
            f"{expert_spill / GIB:.1f} GiB of experts stay on disk; only "
            f"{slots} of {total_experts} experts fit the cache"
        )

    kv_bytes = arch.kv_bytes(context, bytes_per_param=opt.kv_bits / 8)

    def bw(tier: str) -> float:
        return {
            "vram": target.vram_bw_gbs,
            "ram": target.ram_bw_gbs,
            "disk": target.disk_bw_gbs,
        }[tier]

    if routing is None:
        routing = _routing(n_layers, n_experts, arch.top_k, tokens + warmup, skew, seed)
        routing_source = f"modelled_zipf(skew={skew},seed={seed})"
    else:
        routing_source = "trace"

    # LRU expert cache over (layer, expert). Misses are read from disk and then
    # admitted if a slot is free.
    cache: dict[tuple[int, int], str] = {}
    order: list[tuple[int, int]] = []

    # Per-layer accumulators, averaged over the measured (post-warmup) tokens.
    acc = [
        {"read": 0, "disk": 0, "hits": 0, "misses": 0, "time_s": 0.0}
        for _ in range(n_layers)
    ]
    # Active parameters scale with the layers kept; expert pruning does not change
    # how many are active per token, only how much of the pool stays on disk.
    active_eff = arch.active_params() * n_layers / arch.n_layers
    n = max(1, len(routing) - warmup)
    token_times: list[float] = []
    compute_times: list[float] = []
    raw_hits = 0
    raw_misses = 0

    for ti, tok in enumerate(routing):
        measured_this = ti >= warmup
        compute_s = 2.0 * active_eff / (target.cpu_gflops * 1e9)
        io_s = 0.0
        for i, kind in enumerate(kinds):
            lb = mb.layers[i]
            read = lb.resident
            disk_read = 0
            hits = 0
            misses = 0
            for e in tok[i]:
                key = (i, e)
                if key in cache:
                    hits += 1
                    read += mb.expert_each
                else:
                    misses += 1
                    disk_read += mb.expert_each
                    read += mb.expert_each
                    if slots > 0:
                        if len(order) >= slots:
                            cache.pop(order.pop(0), None)
                        cache[key] = "vram" if cache_vram >= cache_ram else "ram"
                        order.append(key)
            secs = read / (bw("ram") * 1e9) + disk_read / (bw("disk") * 1e9)
            io_s += secs
            if measured_this:
                acc[i]["read"] += read
                acc[i]["disk"] += disk_read
                acc[i]["hits"] += hits
                acc[i]["misses"] += misses
                acc[i]["time_s"] += secs
                raw_hits += hits
                raw_misses += misses

        # output projection streams the embedding matrix once per token
        io_s += mb.embeddings / (bw("ram") * 1e9)

        if measured_this:
            token_times.append(max(io_s, compute_s) if prefetch else io_s + compute_s)
            compute_times.append(compute_s)

    per_token_s = sum(token_times) / max(1, len(token_times))
    compute_ms = (sum(compute_times) / max(1, len(compute_times))) * 1000.0
    forward_ms = per_token_s * 1000.0
    # One forward pass yields `draft_tokens` accepted tokens with speculative/MTP.
    ms_per_token = forward_ms / opt.draft_tokens if opt.draft_tokens else forward_ms
    tok_s = 1000.0 / ms_per_token if ms_per_token else 0.0

    steps = [
        LayerStep(
            index=i,
            kind=kinds[i],
            read_bytes=acc[i]["read"] // n,
            disk_bytes=acc[i]["disk"] // n,
            hits=acc[i]["hits"] // n,
            misses=acc[i]["misses"] // n,
            time_ms=(acc[i]["time_s"] / n) * 1000.0,
        )
        for i in range(n_layers)
    ]
    total_read = sum(s.read_bytes for s in steps)
    total_disk = sum(s.disk_bytes for s in steps)
    hit_rate = raw_hits / (raw_hits + raw_misses) if (raw_hits + raw_misses) else 0.0

    manifest = {
        "vipera_version": 1,
        "kind": "schedule",
        "architecture": {
            "name": arch.name,
            "n_layers": n_layers,
            "n_experts": n_experts,
            "top_k": arch.top_k,
            "kda_layers": kinds.count("kda"),
            "mla_layers": kinds.count("mla"),
            "total_params": arch.total_params(include_vision=False),
            "active_params": int(active_eff),
        },
        "optimizations": opt.as_dict(),
        "target": {
            "name": target.name,
            "vram_mb": target.vram_bytes // MB,
            "ram_mb": target.ram_bytes // MB,
            "disk": target.disk,
            "vram_bw_gbs": target.vram_bw_gbs,
            "ram_bw_gbs": target.ram_bw_gbs,
            "disk_bw_gbs": target.disk_bw_gbs,
            "cpu_gflops": target.cpu_gflops,
        },
        "quant": quant.as_dict(),
        "residency": {
            "resident_bytes": resident_total,
            "resident_vram_bytes": resident_vram,
            "resident_ram_bytes": resident_ram,
            "resident_spill_disk_bytes": resident_spill,
            "expert_cache_bytes": cache_bytes,
            "expert_cache_slots": slots,
            "expert_pool_bytes": expert_pool,
            "expert_spill_disk_bytes": expert_spill,
        },
        "context": {"tokens": context, "kv_bytes": kv_bytes},
        "expert_cache": {
            "policy": "lru",
            "routing_source": routing_source,
            "hit_rate": round(hit_rate, 4),
        },
        "performance": {
            "prefetch": prefetch,
            "forward_ms": round(forward_ms, 3),
            "ms_per_token": round(ms_per_token, 3),
            "tokens_per_second": round(tok_s, 4),
            "bytes_per_token": total_read,
            "disk_bytes_per_token": total_disk,
            "compute_ms_per_token": round(compute_ms, 3),
            "bound": "io" if forward_ms - compute_ms > compute_ms else "compute",
        },
        "layers": [
            {
                "index": s.index,
                "kind": s.kind,
                "read_bytes_per_token": s.read_bytes,
                "disk_bytes_per_token": s.disk_bytes,
                "hit_rate": round(s.hit_rate, 4),
                "time_ms": round(s.time_ms, 3),
            }
            for s in steps
        ],
    }
    return SimResult(
        manifest=manifest,
        warnings=warnings,
        layers=steps,
        steps=len(token_times),
        tok_s=tok_s,
        ms_per_token=ms_per_token,
        hit_rate=hit_rate,
        bytes_per_token=total_read,
        disk_bytes_per_token=total_disk,
    )


def dumps(result: SimResult) -> str:
    import json

    return json.dumps(result.manifest, indent=2) + "\n"
