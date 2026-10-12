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
from .trace import global_cache_ceiling

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
    # Dense FFN weights: always needed, so they belong to the resident set.
    ffn: int = 0

    @property
    def resident(self) -> int:
        return self.attention + self.router + self.shared + self.mhc + self.ffn


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
    dense = arch.dense
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
                # Dense: the FFN is one always-needed block, not a routable pool.
                ffn=int(arch.expert_params_each * quant.expert_bytes) if dense else 0,
            )
        )
    return ModelBytes(
        layers=layers,
        expert_each=0 if dense else int(arch.expert_params_each * quant.expert_bytes),
        n_experts=1 if dense else (n_experts if n_experts is not None else arch.n_experts),
        embeddings=int(arch.vocab * arch.hidden * quant.other_bytes),
        mtp=int(arch.mtp_params() * quant.expert_bytes),
        vision=int(arch.vision_params * quant.other_bytes),
    )


@dataclass
class Plan:
    """Residency arithmetic shared by `simulate` and the CLI.

    It lives in one function so the cache size a simulation uses can never drift
    from the size the `trace` command assumes when it computes the static
    ceiling. That drift is exactly what produced the wrong hit-rate comparison
    in round 1 (achieved hit rate at a 6.09% cache vs a ceiling computed at 3%).
    """

    n_layers: int
    n_experts: int
    kinds: list[str]
    mb: ModelBytes
    resident_total: int
    resident_vram: int
    resident_ram: int
    resident_spill: int
    cache_vram: int
    cache_ram: int
    cache_bytes: int
    slots: int
    f_vram: float
    f_ram: float
    f_disk: float
    expert_pool: int
    expert_spill: int
    total_experts: int
    warnings: list[str] = field(default_factory=list)

    @property
    def expert_each(self) -> int:
        return self.mb.expert_each

    @property
    def per_layer_fraction(self) -> float:
        """Cache size as a fraction of *one layer's* experts.

        The pool is global (a slot can hold any layer's expert), so the average
        split across layers is ``slots / (n_layers * n_experts)``. This is the
        number the trace's per-layer static ceiling must use to be comparable.
        """
        return self.slots / self.total_experts if self.total_experts else 0.0


def plan_residency(
    arch: Architecture,
    target: Target,
    *,
    quant: "Quant | None" = None,
    optimizations: "Optimizations | None" = None,
) -> Plan:
    """Decide what is always resident, what becomes expert cache, what spills."""
    quant = quant or Quant.fp8()
    opt = optimizations or Optimizations()
    n_layers = opt.keep_layers or arch.n_layers
    n_experts = max(arch.top_k, int(round(arch.n_experts * opt.prune_experts)))
    mb = model_bytes(arch, quant, n_layers=n_layers, n_experts=n_experts)
    kinds = arch.layer_kinds()[:n_layers]

    # always-needed weights first; whatever is left over becomes the expert cache
    resident_total = mb.resident_total
    resident_vram = min(resident_total, target.usable_vram)
    resident_ram = min(resident_total - resident_vram, target.usable_ram)
    resident_spill = resident_total - resident_vram - resident_ram

    cache_vram = max(0, target.usable_vram - resident_vram)
    cache_ram = max(0, target.usable_ram - resident_ram)
    cache_bytes = cache_vram + cache_ram
    slots = cache_bytes // mb.expert_each if mb.expert_each else 0

    f_vram = resident_vram / resident_total if resident_total else 0.0
    f_ram = resident_ram / resident_total if resident_total else 0.0
    f_disk = resident_spill / resident_total if resident_total else 0.0

    total_experts = n_layers * n_experts
    expert_pool = mb.expert_each * total_experts
    expert_spill = max(0, expert_pool - cache_bytes)

    warnings: list[str] = []
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

    return Plan(
        n_layers=n_layers,
        n_experts=n_experts,
        kinds=kinds,
        mb=mb,
        resident_total=resident_total,
        resident_vram=resident_vram,
        resident_ram=resident_ram,
        resident_spill=resident_spill,
        cache_vram=cache_vram,
        cache_ram=cache_ram,
        cache_bytes=cache_bytes,
        slots=slots,
        f_vram=f_vram,
        f_ram=f_ram,
        f_disk=f_disk,
        expert_pool=expert_pool,
        expert_spill=expert_spill,
        total_experts=total_experts,
        warnings=warnings,
    )


class ExpertCache:
    """Online expert cache over ``(layer, expert)`` keys.

    Eviction policy:
    * ``lru``  evict the least-recently used key (a baseline).
    * ``lfu``  evict the least-frequently used key. Every miss is admitted, so a
      burst of one-off experts can flush the hot set — the round-1 policy.
    * ``hot``  LFU with an **admission filter** and a **pinned hot set** (S2).
      A new key is admitted only if it has been seen more often than the key it
      would evict, so one-off experts bypass the cache instead of polluting it.
      The most-frequent keys are pinned against eviction, and counters are aged
      (halved) periodically so the hot set can follow a drifting router.

    Allocation of the slot budget (S3):
    * ``global``    one pool of `capacity` keys shared by every layer (default).
    * ``per-layer`` each layer gets its own quota, re-solved periodically by
      **marginal water-filling** over the observed per-layer frequencies: the
      next slot always goes to the layer whose next-most-popular expert carries
      the most probability mass. When layers differ in how concentrated their
      routing is, this beats splitting the budget evenly; when they are
      identical it converges to the even split and costs nothing.
    """

    def __init__(
        self,
        capacity: int,
        policy: str = "lru",
        *,
        pin_fraction: float = 0.5,
        aging_interval: int | None = None,
        n_layers: int | None = None,
        alloc: str = "global",
        alloc_interval: int | None = None,
    ) -> None:
        if policy not in ("lru", "lfu", "hot"):
            raise ValueError(f"unknown cache_policy {policy!r}")
        if alloc not in ("global", "per-layer"):
            raise ValueError(f"unknown cache_alloc {alloc!r}")
        if alloc == "per-layer" and not n_layers:
            raise ValueError("per-layer allocation needs n_layers")
        self.capacity = max(0, capacity)
        self.policy = policy
        self.pin_fraction = pin_fraction
        # Age after ~8 turns of the cache. Measured (round 2, ROADMAP §7): 2x is
        # too aggressive (stationary ratio 0.80) and no aging at all cannot follow
        # a routing shift (shift ratio 0.70). 8x is the balance that clears 0.9x on
        # both a stationary and a shifted trace; see the sweep in the ROADMAP.
        self.aging_interval = aging_interval or max(1, 8 * self.capacity)
        self.n_layers = n_layers
        self.alloc = alloc
        self.alloc_interval = alloc_interval or max(1, 4 * self.capacity)
        self._keys: dict[tuple[int, int], None] = {}
        self._order: list[tuple[int, int]] = []
        self._freq: dict[tuple[int, int], int] = {}
        self._layer_freq: dict[int, dict[int, int]] = {}
        self._count: dict[int, int] = {}
        self._cap: dict[int, int] = {}
        self._pins: set[tuple[int, int]] = set()
        self._accesses = 0
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.reallocations = 0

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    # -- internals ---------------------------------------------------------

    def _touch(self, key: tuple[int, int]) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _insert(self, key: tuple[int, int]) -> None:
        self._keys[key] = None
        self._order.append(key)
        self._count[key[0]] = self._count.get(key[0], 0) + 1

    def _remove(self, key: tuple[int, int]) -> None:
        self._keys.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        self._pins.discard(key)
        self._count[key[0]] = self._count.get(key[0], 0) - 1

    def _default_cap(self) -> int:
        return max(1, self.capacity // max(1, self.n_layers or 1))

    def _victim(self, layer: int | None = None) -> tuple[int, int] | None:
        if self.alloc == "per-layer" and layer is not None:
            pool = [k for k in self._keys if k[0] == layer]
            candidates = [k for k in pool if k not in self._pins] or pool
        else:
            candidates = [k for k in self._keys if k not in self._pins] or list(self._keys)
        if not candidates:
            return None
        if self.policy == "lru":
            for k in self._order:
                if k in candidates:
                    return k
            return candidates[0]
        return min(candidates, key=lambda k: self._freq.get(k, 0))

    def _age(self) -> None:
        for k in list(self._freq):
            self._freq[k] //= 2
        for layer in self._layer_freq:
            for e in list(self._layer_freq[layer]):
                self._layer_freq[layer][e] //= 2
        keep = max(1, int(self.capacity * self.pin_fraction))
        hot = sorted(self._keys, key=lambda k: self._freq.get(k, 0), reverse=True)[:keep]
        self._pins = set(hot)

    def _reallocate(self) -> None:
        """Greedy marginal water-filling of `capacity` slots across layers.

        The marginal value of one more slot in layer l is the probability mass of
        layer l's next-most-popular expert; giving each slot to the current argmax
        solves the separable allocation exactly (all marginals are non-increasing).
        """
        ranked = {l: sorted(f.values(), reverse=True) for l, f in self._layer_freq.items() if f}
        total = sum(sum(r) for r in ranked.values())
        if not ranked or total <= 0 or self.capacity <= 0:
            return
        assigned = dict.fromkeys(ranked, 0)
        layers = list(ranked)
        for _ in range(self.capacity):
            best, best_mass = None, -1.0
            for l in layers:
                k = assigned[l]
                r = ranked[l]
                mass = (r[k] / total) if k < len(r) else 0.0
                if mass > best_mass:
                    best_mass, best = mass, l
            assigned[best] += 1
        self._cap = assigned
        self.reallocations += 1
        # Trim any layer now over its new quota.
        for l in layers:
            while self._count.get(l, 0) > assigned[l]:
                victim = self._victim(l)
                if victim is None:
                    break
                self._remove(victim)

    def _admit(self, key: tuple[int, int]) -> None:
        if self.alloc == "per-layer":
            layer = key[0]
            self._cap.setdefault(layer, self._default_cap())
            if self._count.get(layer, 0) < self._cap[layer]:
                self._insert(key)
                return
            victim = self._victim(layer)
        else:
            if len(self._keys) < self.capacity:
                self._insert(key)
                return
            victim = self._victim()
        if victim is None:
            self._insert(key)
            return
        if self.policy == "hot" and self._freq[key] <= self._freq.get(victim, 0):
            # Admission filter: a key no more popular than its would-be victim
            # is not worth a slot. It bypasses the cache and must be re-read.
            self.bypasses += 1
            return
        self._remove(victim)
        self._insert(key)

    def access(self, key: tuple[int, int]) -> bool:
        """Record one reference; return True on a cache hit."""
        self._accesses += 1
        self._freq[key] = self._freq.get(key, 0) + 1
        lf = self._layer_freq.setdefault(key[0], {})
        lf[key[1]] = lf.get(key[1], 0) + 1
        if key in self._keys:
            self.hits += 1
            self._touch(key)
            self._maybe()
            return True
        self.misses += 1
        if self.capacity > 0:
            self._admit(key)
        self._maybe()
        return False

    def _maybe(self) -> None:
        if self.policy == "hot" and self._accesses % self.aging_interval == 0:
            self._age()
        if self.alloc == "per-layer" and self._accesses % self.alloc_interval == 0:
            self._reallocate()


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
    cache_policy: str = "lru",
    cache_alloc: str = "global",
    routing: list[list[list[int]]] | None = None,
    eval_tail: float = 0.0,
) -> SimResult:
    """Plan residency and simulate `tokens` decode steps.

    `routing`, when supplied, is ``routing[token][layer][slot] = expert_id`` and
    replaces the modelled Zipf routing with a real trace.

    `eval_tail` > 0 restricts the reported hit rate (and its ceiling) to the last
    fraction of the trace — the window that isolates how fast a policy adapts to
    a routing shift, rather than averaging the pre-shift period in.
    """
    quant = quant or Quant.fp8()
    opt = optimizations or Optimizations()
    plan = plan_residency(arch, target, quant=quant, optimizations=opt)
    warnings: list[str] = list(plan.warnings)
    n_layers = plan.n_layers
    n_experts = plan.n_experts
    kinds = plan.kinds
    mb = plan.mb
    resident_total = plan.resident_total
    resident_vram = plan.resident_vram
    resident_ram = plan.resident_ram
    resident_spill = plan.resident_spill
    cache_vram = plan.cache_vram
    cache_ram = plan.cache_ram
    cache_bytes = plan.cache_bytes
    slots = plan.slots
    f_vram = plan.f_vram
    f_ram = plan.f_ram
    f_disk = plan.f_disk
    total_experts = plan.total_experts
    expert_pool = plan.expert_pool
    expert_spill = plan.expert_spill

    kv_bytes = arch.kv_bytes(context, bytes_per_param=opt.kv_bits / 8)

    def bw(tier: str) -> float:
        return {
            "vram": target.vram_bw_gbs,
            "ram": target.ram_bw_gbs,
            "disk": target.disk_bw_gbs,
        }[tier]

    def tier_seconds(nbytes: int) -> float:
        """Seconds to read `nbytes` when they are spread across the tiers."""
        return nbytes * (f_vram / bw("vram") + f_ram / bw("ram") + f_disk / bw("disk")) / 1e9

    if routing is None:
        routing = _routing(n_layers, n_experts, arch.top_k, tokens + warmup, skew, seed)
        routing_source = f"modelled_zipf(skew={skew},seed={seed})"
    else:
        routing_source = "trace"

    # Expert cache over (layer, expert). `hot` adds an admission filter and a
    # pinned hot set on top of LFU (see ExpertCache).
    cache = ExpertCache(slots, cache_policy, n_layers=n_layers, alloc=cache_alloc)

    # Per-layer accumulators, averaged over the measured (post-warmup) tokens.
    acc = [
        {"read": 0, "disk": 0, "hits": 0, "misses": 0, "time_s": 0.0}
        for _ in range(n_layers)
    ]
    # Active parameters scale with the layers kept; expert pruning does not change
    # how many are active per token, only how much of the pool stays on disk.
    active_eff = arch.active_params() * n_layers / arch.n_layers
    # First token of the measured window. `eval_tail` trims the front so a policy
    # that only adapts slowly is not rescued by the tokens before the shift.
    eval_start = warmup
    if eval_tail > 0:
        eval_start = max(warmup, int(len(routing) * (1 - eval_tail)))
    n = max(1, len(routing) - eval_start)
    token_times: list[float] = []
    compute_times: list[float] = []
    raw_hits = 0
    raw_misses = 0

    for ti, tok in enumerate(routing):
        measured_this = ti >= eval_start
        compute_s = 2.0 * active_eff / (target.cpu_gflops * 1e9)
        io_s = 0.0
        for i, kind in enumerate(kinds):
            lb = mb.layers[i]
            expert_cached = 0
            expert_disk = 0
            hits = 0
            misses = 0
            for e in tok[i]:
                if cache.access((i, e)):
                    hits += 1
                    expert_cached += mb.expert_each
                else:
                    misses += 1
                    expert_disk += mb.expert_each
            read = lb.resident + expert_cached + expert_disk
            disk_read = int(lb.resident * f_disk) + expert_disk
            secs = (
                tier_seconds(lb.resident)
                + expert_cached / (bw("ram") * 1e9)
                + expert_disk / (bw("disk") * 1e9)
            )
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
        io_s += tier_seconds(mb.embeddings)

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
    # The oracle a *static* cache of the same size would reach on this very
    # routing, so `schedule` can never again compare a hit rate to a ceiling at a
    # different cache size. Computed over the measured (post-warmup) tokens only.
    static_ceiling = global_cache_ceiling(routing, slots, warmup=eval_start)

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
            "policy": cache_policy,
            "alloc": cache_alloc,
            "reallocations": cache.reallocations,
            "routing_source": routing_source,
            "hit_rate": round(hit_rate, 4),
            "slots": slots,
            "per_layer_fraction": round(plan.per_layer_fraction, 4),
            "static_cache_ceiling": round(static_ceiling, 4),
            "ceiling_ratio": round(hit_rate / static_ceiling, 4) if static_ceiling else None,
            "eval_window": {"start": eval_start, "end": len(routing), "tail": eval_tail},
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
