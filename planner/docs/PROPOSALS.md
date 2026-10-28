# PROPOSALS — new algorithms for local LLM inference

> Role: research proposals for running large LLMs on constrained personal
> hardware. The survey and per-technique evaluation live in
> [`RESEARCH.md`](RESEARCH.md); this document goes one step further and proposes
> **new or improved algorithms**, each with pseudocode, complexity, an evaluation
> method, and risks.
>
> **Honesty rules.** Numbers marked *(published)* are other people's and are not
> reproduced here. Numbers marked **vipera** are produced by this repo and are
> reproducible with the command shown. Anything not labelled is an estimate and is
> called out as such.
>
> **Target context** (the instances already modelled here): a 4 GB-VRAM GTX 1650
> laptop and a 16 GB unified-memory laptop, NVMe ~3 GB/s, wall power caps as low
> as 65 W, models from 15B MoE to 321B-A18B and dense 70B.

---

## 1. Where existing methods fall short

From `RESEARCH.md` §7, the decisive metric is **bytes read per token**, and the
open gaps are:

1. **Cache policy is treated as a constant.** Systems use LRU, or a hand-tuned
   "≈2× active experts" rule *(published)*. Nobody schedules the *policy* per
   layer or per skew.
2. **Quantization granularity ignores the scheduler.** Bit allocation is chosen
   once per tensor class, not against the *cost of the tier the tensor will be
   read from*.
3. **Offloading optimizes latency only.** The one published energy result says SSD
   offload costs **5–12× more energy per token than DRAM** *(published)* — fatal
   under a watt cap — yet latency-only schedulers still prefer SSD.
4. **Pruning is unverified.** Expert/layer pruning is proposed with a "heal it"
   step and evaluated after the fact; there is no cheap gate that decides *before*
   committing.

The four proposals below target exactly these gaps.

---

## 2. Proposal P1 — Skew-adaptive, admission-controlled expert cache (SAC)

**Status: implemented and measured in this repo.**

### Core idea
Replace the single fixed eviction policy with two mechanisms chosen per layer by
observed routing statistics:

* **Eviction policy by measured skew.** If a layer's expert frequency is heavy-
  tailed, LFU retains the hot set far better than LRU, which thrashes because the
  access pattern is cyclic (every token revisits every layer). vipera already
  showed this failure mode on the dense/round-robin case.
* **Admission control.** Do not admit an expert seen once ("one-hit wonder") into
  a precious slot; keep a small probation queue instead. This is standard CDN
  practice, absent from MoE caches.

### Pseudocode
```
PER LAYER l: estimate skew s_l from a routing sample
  if s_l is heavy-tailed:  policy = FREQUENCY
  else:                    policy = RECENCY

on access(l, e):
  hit[l][e] += 1
  if e in cache:            serve(cache[l][e])
  elif hit[l][e] >= 2:      # admission control
      if full(l):  evict_min_freq(l)
      admit(l, e, tier=FASTEST_FREE)
  else:
      serve(DISK); probation[l].add(e)

evict_min_freq(l): remove argmin_{e in cache[l]} hit[l][e]
```

### Complexity
* Time: eviction is `O(|cache_l|)` per miss with a linear min-scan. With a
  heap keyed by `hit`, `O(log |cache_l|)`; memory `O(|cache_l|)`.
* Extra memory: one counter per cached expert — negligible versus the expert
  bytes it protects.
* Bandwidth: never increases bytes/token; it only decides *which* bytes come from
  disk.

### Evaluation method
Run `vipera schedule … --cache-policy lru|lfu` and compare hit rate, disk
bytes/token and tok/s on the same routing model; then repeat on a **captured
routing trace** (see §4) and on WikiText-2 perplexity with the real runtime.

### Measured result (vipera, reproducible)
```
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml \
    --quant q4-experts --cache-policy lru    # hit 21.5%  0.544 tok/s
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml \
    --quant q4-experts --cache-policy lfu    # hit 32.6%  0.620 tok/s
```

| policy | hit rate | disk bytes/token | tok/s |
|---|---|---|---|
| LRU | 21.5% | 3.11 GiB | 0.544 |
| **LFU (P1 core)** | **32.6%** | **2.66 GiB** | **0.620** |

**+11.1 points** of hit rate `[modelled]`, from a policy change alone, with **no quality cost**.
The **+14% throughput** figure this section previously led with is **not a measurement** — it is
modelled over simulated Zipf routing, and the one real measurement of the same machinery on a
RAM-resident model is **0.92–1.00×** (`docs/FINDINGS.md` §Round 4). See §6 for why that caveat
decides the result rather than qualifying it.

### Risks & mitigations
* **Stale hot set** under distribution shift (LFU never forgets) → add **aging**
  (halve counters every N tokens) or a 2-queue design. *Not yet implemented.*
* **Synthetic-skew optimism** → must be re-run on a real routing trace.

---

## 3. Proposal P2 — Tier-aware bit water-filling (TAW)

### Core idea
Choose quantization *per component* to minimize bytes read per token subject to a
quality budget `ΔQ ≤ ε`, where each component's byte savings are weighted by **how
often it is read** and **from how slow a tier**. The novelty is making the bit
assignment a function of the scheduler's placement, not of the tensor class alone.

### Formulation
Let components `i` have sensitivity `σᵢ` (quality lost per bit removed) and a
read frequency `fᵢ` (≈1 for resident weights, `hit`-weighted for experts). For
expert `e` we lose sensitivity but the *effective* saving is discounted by its
cache hit rate, because a cached expert's bytes never touch the slow tier:

```
minimize   Σᵢ  fᵢ · bytesᵢ(bᵢ)
subject to Σᵢ  σᵢ · (Bᵢ − bᵢ)  ≤  ε        (quality budget)
           bᵢ ∈ {2,3,4,5,6,8,16}
```

Lagrangian relaxation: assign bits in decreasing order of
`(bytes saved) / (σᵢ · bits removed)` — a greedy knapsack. Experts whose hit rate
is high get *fewer* bits (they are cheap to read anyway); always-resident weights
that stream from disk get *more*.

### Complexity
`O(n log n)` over components to sort by the ratio; memory `O(n)`. Trivially
parallel; no model run required for the plan.

### Evaluation method
Plan with TAW, then measure WikiText-2 perplexity (quality) and tok/s against a
uniform-Q4 baseline at **matched model size**, on the same machine. Track the
Pareto front (ppl vs tok/s).

### Predicted result (estimate, not measured)
**+10–25% tok/s at equal perplexity** on a disk-bound instance, larger where the
disk dominates. No size reduction — bytes are *reallocated*, not removed.

### Risks & mitigations
* `σᵢ` is unknown and non-additive (quality loss is not a sum over components) →
  treat P2 as a *heuristic ordering*, validate with a real perplexity sweep before
  trusting it. This is the weakest assumption in the document.

---

## 4. Proposal P3 — Energy-aware tiering (EAT)

### Core idea
Rank placements by **energy per byte read**, not only latency, because SSD
offloading costs **5–12× more energy per token than DRAM** *(published)*. With a
watt cap, a scheduler that only minimizes time can *fail a hard constraint*.

### Formulation
Choose the resident/cached set `C` to minimize a weighted objective under both a
time budget and a power budget:

```
minimize   α · T(C)  +  β · E(C)
subject to P(C) ≤ P_max ,   Σ_{e∈C} S_e ≤ M_fast
T(C) = Σ_cached  S_e/BW_fast_bandwidth  +  Σ_disk  S_e/BW_disk
E(C) = Σ_disk  S_e · joules_per_byte(tier)      # DRAM ≪ SSD
```

The solution prefers **DRAM** offloading over SSD even when SSD is faster to hide,
unless the time term dominates.

### Complexity
Same knapsack shape as P2: `O(n log n)` greedy on `S_e · (β·joules − α/bandwidth)`.

### Evaluation method
Measure **actual wall power with a wattmeter** (required, not estimated) while
running the same config with SSD-heavy vs DRAM-heavy placement; report
J/token and time. vipera cannot do this yet — the energy model is the gap.

### Predicted result (estimate)
**20–40% lower J/token** at **≤5% latency cost** on SSD-bound cases; zero effect
where everything already fits DRAM. The value is in not *violating* a watt cap.

### Risks & mitigations
* Published energy ratios may not hold for this SSD/CPU → measure, don't trust.
* Idle/platform power often dominates → report the delta, not the absolute.

---

## 5. Proposal P4 — Gated expert pruning with rollback (GEP)

### Core idea
Prune experts *only* when a cheap proxy proves the loss is small; otherwise roll
back per layer. This turns pruning from an unverified hope into a guarded step.

### Formulation
For layer `l`, prune to keep-set `K_l`. Accept only if both hold:

```
routing_mass(K_l)  =  Σ_{e∈K_l} freq_e / Σ_e freq_e   ≥  1 − τ
drift_proxy(K_l)   =  || f(x; K_l) − f(x; full) ||  / || f(x; full) ||  ≤  δ
```

on a small held-out prompt set. Otherwise keep the layer unpruned.

### Complexity
Routing mass is free (a histogram). The drift proxy needs a few forward passes:
`O(#probe_tokens · L · active_params)` per candidate keep-set — the cheap part is
that it is *per layer*, so rollback is local.

### Evaluation method
Compare against unguarded pruning at equal sparsity on MMLU/GSM8K/HumanEval plus
perplexity; the gate should reduce the *variance* of quality loss across layers.

### Predicted result (estimate)
Recovers most of the quality lost by unguarded pruning at moderate sparsity
(vipera showed 5% experts → 95% hit rate; the question P4 answers is *which* 5%).

### Risks & mitigations
* Proxy may not correlate with downstream quality → calibrate `δ` against a real
  benchmark once, then trust it only within that regime.

---

## 6. Experiment design (applies to P1–P4)

**Datasets / workloads**
* Quality: WikiText-2 (perplexity), MMLU, GSM8K, HumanEval.
* Routing: a **captured expert-routing trace** of ≥10k tokens across ≥3 domains
  (the current simulator uses synthetic Zipf and must be replaced).
* Power: a wattmeter trace at ≥1 Hz.

**Metrics**
* tok/s (prefill + decode, ≥3 runs, median + spread)
* bytes/token and **disk bytes/token** (`/proc/<pid>/io`)
* expert cache hit rate
* peak RSS (`VmHWM`), peak VRAM (`nvidia-smi`)
* J/token (wall power × time)
* quality deltas vs FP16

**Baselines**
* `llama.cpp` default placement, LRU cache, uniform `Q4_K_M`
* each proposal against the *specific* baseline it claims to beat (P1 vs LRU,
  P2 vs uniform Q4, P3 vs latency-only tiering, P4 vs unguarded pruning)

**Procedure & noise control**
* fixed seed, pinned CPU cores, warm cache, same build/commit
* interleave A/B runs to cancel thermal drift; discard the first run
* report median and interquartile range, never a single run
* state the machine, model hash, and context length with every number

**Pass criteria**
* P1: hit rate ≥ +5 points over LRU **and** no perplexity regression.
* P2: ≥ +10% tok/s at **equal perplexity** and equal model size.
* P3: ≥ 20% lower J/token at ≤ 5% latency cost, wattmeter-measured.
* P4: quality loss variance reduced ≥ 50% vs unguarded pruning at equal sparsity.

---

## 7. Self-critique (as a hostile reviewer)

* **P1 is under-tested.** The +11-point gain is on *synthetic Zipf* routing, and
  LFU has no aging — on a domain shift it can pin a stale hot set and lose to LRU.
  The honest claim is narrower than it looks: *on skewed, stationary routing*, LFU
  beats LRU on this model.
* **P2 rests on an unknown function.** `σᵢ` (quality per bit) is not separable and
  the greedy ordering assumes it is. Without a measured quality-vs-bits curve per
  component, P2 is a heuristic wearing a Lagrangian's clothes.
* **P3's energy numbers are borrowed.** The 5–12× SSD figure is published on other
  hardware; idle power may dominate. It may be the most *important* proposal and
  the least *evidenced*.
* **P4 doubles the runtime cost** of planning (forward passes per candidate) for a
  proxy that is only as good as its calibration.
* **Metric-gaming risk.** Every proposal optimizes bytes/token; a config that
  drops quality to shrink bytes can "win" the throughput table while failing the
  quality gate. Any evaluation that does not jointly report quality is invalid.
* **Reproducibility.** Only P1 has a one-line reproduction. P2–P4 have no
  implementation yet, so their "feasibility" is asserted, not shown.

---

## 8. Comparison

| Proposal | Gap it closes | Bytes/token | Quality risk | Energy | Implemented? | Evidence |
|---|---|---|---|---|---|---|
| **P1 SAC** (skew-adaptive cache) | fixed cache policy | ↓ (hit ↑) | none | n/a | **yes** | hit rate **+11 pts** `[modelled]`; the `+14% tok/s` once claimed here is modelled over synthetic Zipf and the real measurement is **0.92–1.00×** |
| **P2 TAW** (tier-aware bits) | scheduler-blind quantization | ↓ (realloc) | medium | n/a | no | estimate |
| **P3 EAT** (energy-aware tiering) | latency-only offload | ~ | none | **↓ 20–40%** | no | published energy ratio |
| **P4 GEP** (gated pruning) | unverified pruning | ↓↓ (MoE) | **reduced** | n/a | no | estimate |

---

## 9. Next research directions

1. **Capture a real routing trace** and re-run P1; the synthetic-skew caveat is
   the single biggest blocker to trusting any cache claim here.
2. **Per-layer cache allocation** (how many of the slots each layer gets) — a
   global solver like MoEpic *(published)* but driven by measured per-layer skew.
3. **Energy model in vipera**, so P3 becomes schedulable and testable.
4. **Distribution-shift-aware caching** (aging / 2Q) as an extension of P1.
5. **A joint scheduler** that co-optimizes quant bits, placement and cache under
   one bytes/token + quality + energy objective — the union of P1–P3.

## Assumptions & limitations

* The simulator models decode as bandwidth-bound with a flat compute floor; GPU
  compute and kernel efficiency are **not** modelled.
* Routing is modelled, not traced (except when a trace is supplied).
* Energy and quality are **not** modelled at all today — P2–P4 depend on inputs
  vipera cannot yet produce.
* Published figures in `RESEARCH.md` and here are not reproduced.
