# algorithms-digest.md — Every optimization algorithm behind running an LLM on weak hardware

> This document gathers **all the context in one place**: the target machine, the target model, the
> problem, and every algorithm vipera uses to make an LLM run. The sources are the real code in
> `vipera/` and `bench/`, plus `docs/ALGORITHMS.md`, `docs/PROPOSALS.md`, `docs/OPTIMIZER.md` and
> `docs/ROADMAP.md`.
>
> **Confidence convention** (the same one the rest of the repo uses):
> **measured** = a number taken on real hardware with a reproducible command · **modelled** = a number
> vipera computes, not from a real run · **proposed** = not implemented yet · **published** = someone
> else's number, not re-verified here.

---

## 0. Context

### 0.1 Target machine (measured 2026-10-12)

| Component | Specification | Why it matters |
|---|---|---|
| CPU | Intel i5-11300H, **4 cores / 8 threads**, up to 4.4 GHz | MoE compute falls back to the CPU, which is very slow |
| RAM | **15 GiB**, ~10 GiB available, **no swap** | the model must be ≤ ~11 GiB to fit in RAM entirely |
| Discrete GPU | NVIDIA **GTX 1650 Max-Q, 4 GB VRAM**, compute 7.5 (Turing) | holds attention + KV + a few experts, **not the whole model** |
| Integrated GPU | Intel Iris Xe (Tiger Lake) | no unified API alongside NVIDIA ⇒ dropped |
| Disk | WD SN530 NVMe, **DRAM-less** | sequential reads are fine, random/burst reads are poor |
| OS | Arch Linux, no sudo, Python 3.14 | installs have to be root-free |

Constants in `hw/gtx1650.toml`:

| Constant | Value | Status |
|---|---|---|
| `ram_bw_gbs` | 44.0 | **measured** (40–46 GB/s, `bench/hwprobe.c`) |
| `disk_bw_gbs` | 8.0 | **measured** (7.75–9.1 GB/s, O_DIRECT) |
| `cpu_gflops` | 110.0 | **measured** (GEMV ALU-bound, 8 threads) |
| `vram_bw_gbs` | 192.0 | **datasheet, unverified** |
| `ram_headroom_mb` | 5120 | **measured** (what the OS and the runtime actually occupy) |
| `vram_headroom_mb` | 512 | reserved for KV + activations |

### 0.2 Target models

| Phase | Model | Total / active | Q4_K_M | Fits 11 GiB RAM? |
|---|---|---|---|---|
| P0–P2 | **DeepSeek-V2-Lite** | 15.7B / 2.4B | ~9.5–10.4 GB | ✅ |
| stretch | Qwen3-30B-A3B | 30.5B / 3.3B | ~18.6 GB | ❌ (needs streaming) |

### 0.3 The problem and the single principle

```
token cost = (bytes read from VRAM/RAM)  +  (bytes read from disk)
             └── fast, constant ──┘        └── SLOW — this is what to optimize ──┘
```

`tokens/s ≈ bandwidth / (bytes of weights that must be read per token)`.

Three ceilings on this machine (measured 2026-10-13) — the lowest one binds:

| Ceiling | Computation | Result |
|---|---|---|
| Arithmetic (ALU) | 2.4B active × 2 = 4.8 GFLOP/token ÷ 110 GFLOP/s | **~23 tok/s** ← the real bound |
| RAM bandwidth | 1.2 GB/token ÷ 44 GB/s | ~37 tok/s |
| Disk read on a cache miss | 1.2 GB ÷ 8 GB/s | **~6.7 tok/s** |

⇒ The gap between "fully resident in RAM" (~20 tok/s) and "spilling to disk" (~6.7 tok/s) is **~3×**.
Design conclusion: **raising the expert cache hit rate is goal number one**; "just stream faster"
does not close that gap.

### 0.4 The two modes of vipera

| Mode | Commands | Nature |
|---|---|---|
| Modelled | `analyze`, `plan`, `schedule`, `trace`, `prune`, `sweep`, `score` | arithmetic over tensor sizes plus a decode simulation. **No model is loaded.** |
| Real | `run` | reads the manifest → translates it into llama.cpp flags → launches the real binary → **measures** throughput/RSS/disk |

---

## 1. Pipeline map

```
 raw GGUF / safetensors
        │
        ▼
  (1) analyze   → Inventory: split bytes into resident / router / shared-expert / other / expert
        │
        ▼
  (2) plan      → *.vipera.json: where each group lives (VRAM / RAM / disk spill)
        │
        ▼
  (3) schedule  → models bytes/token + tok/s, runs the cache policy & allocation
        │
        ├── trace   → routing entropy, static-cache ceiling at the same size
        ├── prune   → expert keep-set by routing mass (P4 has a gate + rollback)
        └── prefetch → predicted expert table for the llama.cpp patch
        │
        ▼
  (4) run       → applies the manifest to llama.cpp and actually RUNS it, measuring the result
```

---

## 2. Algorithm A — Tensor classification (`vipera/analyze.py`)

Every tensor is placed in one of 5 groups, decided by its name:

| Group | Recognised by | Present on every token? |
|---|---|---|
| resident | the remaining `blk.N.*` (attention, norm) | yes |
| router | `ffn_gate_inp.weight` | yes |
| shared expert | `*_shexp*` | yes |
| other | `token_embd`, `output`, `output_norm` | yes |
| routable expert | `*_exps*` (and not `_shexp`) | **no** — only when the router picks it |

```
always_resident_bytes = resident + router + shared_expert + other
expert_bytes          = sum of all *_exps tensors
n_experts             = mode of shape[2] over the 3-D tensors
n_layers              = metadata block_count, fallback = max blk index + 1
```

This is what turns "model optimization" into a problem **purely about byte sizes** — no tensor data
needs to be read (parsing the header + descriptors of a 9.65 GiB file takes ~0.24–0.37 s).

---

## 3. Algorithm B — Residency planning, greedy "always-needed first" (`vipera/schedule.py:227`)

```
1. resident_total = sum of the always-resident bytes
2. resident_vram = min(resident_total, usable_vram)
   resident_ram  = min(resident_total − resident_vram, usable_ram)
   resident_spill = resident_total − resident_vram − resident_ram     → warn if > 0
3. cache_bytes = (usable_vram − resident_vram) + (usable_ram − resident_ram)
4. slots = cache_bytes // expert_each              ← how many experts fit in the cache
5. expert_pool  = expert_each × (moe_layers × n_experts)
   expert_spill = max(0, expert_pool − cache_bytes)
```

Complexity is `O(#tensor)`. Properties:

- **Size-only** — it cannot rank which expert is "hot" until a real trace exists; the manifest says so
  explicitly in `placement.expert_order_pending_trace`.
- In `vipera/manifest.py` this variant emits the `*.vipera.json`. It **cannot** enforce what a real
  runtime does (llama.cpp, for instance, keeps the input embedding on the CPU), so the documentation
  states that limit.
- `ram_headroom_mb` must be a **measured** number, not a round one: an older value of 1536 MiB made
  the planner call a 15.9 GiB model "fits entirely" while the machine did not have the RAM.

---

## 4. Algorithm C — Expert cache: `lru` / `lfu` / `hot` (`vipera/schedule.py:308`)

The cache key is the pair `(layer, expert)`. Three policies:

| policy | picks as victim | note |
|---|---|---|
| `lru` | the least recently used key | baseline; this is what an MoE router touches cyclically, so it thrashes easily |
| `lfu` | the key with the lowest frequency | **every miss is admitted** → one-off bursts shove the hot set aside |
| `hot` | LFU + **admission filter** + **pinned hot set** + **aging** | P1 (SAC) — the best default |

### 4.1 `hot` is three mechanisms

**(a) Admission filter** (`_admit`):
```
if the cache is full and freq[new_key] <= freq[victim]:  → BYPASS (takes no slot)
otherwise:                                               → evict the victim, admit the key
```
"One-hit wonder" experts are blocked and counted separately in `bypasses`. This is standard CDN
practice, absent from ordinary MoE caches.

**(b) Pinned hot set** (`_age`): pins the `pin_fraction × capacity` highest-frequency experts
(**50% by default**) so they cannot be evicted, even when the cache is full.

**(c) Aging**: every `aging_interval = 8 × capacity` accesses it **halves every frequency counter**,
then re-pins the top K. The value **8** is **measured**, not picked at random:

| aging | ratio on stationary routing | ratio after a routing shift |
|---|---|---|
| 2× | 0.80 (too aggressive) | — |
| no aging | — | 0.70 (does not keep up) |
| **8×** | **> 0.9** | **> 0.9** |

### 4.2 Measured (modelled over Zipf)

```
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --cache-policy lru
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --cache-policy lfu
```

| policy | hit rate | disk bytes/token | tok/s |
|---|---|---|---|
| LRU | 21.5% | 3.11 GiB | 0.544 |
| **LFU (P1)** | **32.6%** | **2.66 GiB** | **0.620** |

→ **+11.1 points of hit rate, +14% throughput, no loss of quality.**

On the **real OLMoE trace** (511 tokens, 64 experts, top-8), against the ceiling at the same size:

| cache | ceiling | LRU | LFU | `hot` |
|---|---|---|---|---|
| 10% | 42.7% | 0.0% | **0.97×** | 0.93× |
| 25% | 71.1% | 0.75× | **0.97×** | 0.93× |
| 50% | 94.1% | 0.99× | 0.99× | 0.98× |

**Three important corrections the real trace bought:**
1. on **stationary** routing, **LFU beats `hot`** (aging only pays when the distribution *shifts*);
2. **per-layer allocation does not help** (§8 of ROADMAP);
3. pruning **is not free** (§8 below).

---

## 5. Algorithm D — Per-layer slot allocation: marginal water-filling (`_reallocate`)

```
marginal value of 1 slot at layer l = routing probability of l's "next hottest" expert
repeat capacity times:
    give the slot to the layer with the largest marginal value (argmax)
```

Because every marginal value is **non-increasing** in the number of slots already granted, this
greedy argmax yields the **provably optimal** solution of the separable knapsack problem. It is
recomputed every `alloc_interval = 4 × capacity`. When the layers are alike it degenerates to an even
split ⇒ it costs nothing.

**Measured: per-layer does NOT help** on real data ⇒ the default is `global` (`--cache-alloc global`).
After each reallocation, any layer over its new quota has its victims trimmed until it fits.

---

## 6. Algorithm E — Trace, entropy, and the control "ceiling" (`vipera/trace.py`)

### 6.1 Trace format
JSONL, one line per token: `{"t": <step>, "layers": [[e,...], ...]}` — each entry is one **MoE block**
(the leading dense block has no router, so it does not appear).

`load_trace` / `validate(routing, n_layers=arch.moe_layers, n_experts, top_k)`:
- `moe_layers = n_layers − leading_dense_layers` — this is exactly why the real model (DeepSeek-V2-Lite
  has 27 blocks, 1 leading dense block) yields a trace with 26 layers; it is the thing that used to
  block every trace command.

### 6.2 Routing skew — entropy
```
H = entropy of a layer's expert-choice distribution
H ≈ 1.0  → routing is uniform, NO cache can help
```

### 6.3 Static cache ceiling — against self-deception
`global_cache_ceiling(routing, slots)` = a **static oracle**: the highest hit rate a cache of
**exactly `slots`** can reach on that same routing (pick the most popular experts, keep them forever).

Every policy hit rate is reported as the ratio `policy / ceiling` **at the same cache size**. The
ceiling used to be computed at a different size, which made a correct policy look better than it was.

---

## 7. Algorithm F — Prefetch driven by routing prediction

### 7.1 Why prediction is possible
MoE is **layer-local**: the experts of layer n+1 depend on layer n's hidden state. The router only
reveals its choice when that layer is about to run ⇒ stock llama.cpp **loads on demand**, so every
upload is exposed (latency cannot be hidden).

### 7.2 The prediction table (`vipera/prefetch.py`)
The table is a **per-layer mass keep-set** taken from the trace — **the same measurement pruning
uses**, so the prune advice and the prefetch table can never contradict each other.

```
# comment
<layer>\t<expert>,<expert>,...     most likely experts first
```
`build_table(routing, target_mass=0.95, top=N)`; `top` truncates each layer (a small cache forces
truncation).

### 7.3 The llama.cpp patch (`bench/llama-moe-prefetch.patch`) — **no fork of the core**
The patch only adds:
- `moe_cache_lru::reserve(il, e)` — takes a slot **without** uploading (pulled off the LRU head as if
  it were a miss),
- table loading from `LLAMA_MOE_PREFETCH`, with `LLAMA_MOE_PREFETCH_DEPTH` (how many layers ahead) and
  `_TOP` (per-layer cap) controlling it,
- while the current layer computes, `tensor_set_async` the predicted experts of the next layer **on
  the same device group**,
- scoring itself: `placed` / `useful` (predicted and picked by the router before eviction) /
  `extra upload`.

vipera **does not patch llama.cpp further**: `vipera run --trace` learns the table, writes it to a
file, then arms the patch through the environment. If the binary lacks the patch the runner **says so**
(`has no MoE prefetch support`) instead of letting a run look like "prefetch did nothing".

### 7.4 Measured (a 199-token trace captured from DeepSeek-V2-Lite itself)

| configuration | gen t/s | the patch's verdict |
|---|---|---|
| Vulkan + prefetch (512 MiB cache) | 1.30 | 474/5440 (**8.7%**), +28.8 GiB uploaded |
| CUDA + prefetch (1024 MiB cache) | 2.00 | 935/8320 (**11.2%**), +44.0 GiB uploaded |

**Honest conclusion:** prefetch **makes things slower** here. The trace is too short ⇒ the table is
wrong. On OLMoE, a 0.95-mass table needs **541 experts / 16 layers** (33.8 of 64 on average) — i.e.
routing is concentrated enough to help a cache that **already holds about half the experts**, and too
flat to help a much smaller one.

---

## 8. Algorithm G — Expert pruning

### 8.1 Pruning by mass (`prune.py:91`)
```
for each layer:
  count freq[e] over the trace
  sort experts by freq descending
  take the shortest prefix such that Σfreq / Σall ≥ target_mass   (0.95 by default)
```
Greedy, `O(E log E)` per layer, needs only a histogram — free.

**Measured** on DeepSeek-V2-Lite, with the 199-token trace captured in the same session:
```
python3 -m vipera prune /tmp/ds_trace.jsonl --arch arch/deepseek-v2-lite.toml --target-mass 0.95
→ mass-based keep-set: 1325 of 1664 experts (79.63% of the pool), retaining 95.3% of routing mass
```
(On a different trace `report.md` records 1292/1664 = 77.6% — same conclusion, different trace.)

Sparsity table for this trace:

| sparsity | mass retained | if spread evenly | gain |
|---|---|---|---|
| 5% | 14.9% | 5.0% | +9.9 |
| 10% | 25.0% | 10.0% | +15.0 |
| 25% | 49.5% | 25.0% | +24.5 |
| 50% | 76.5% | 50.0% | +26.5 |
| 75% | 93.0% | 75.0% | +18.0 |

⇒ **~78–80% of the experts must be kept to hold 95% of the routing mass.** The memory saving is
smaller than it looks — pruning **is not free**.

### 8.2 P4 — Gated pruning with rollback (`prune.py:261`)
The gates are checked in order, each layer independently; a layer that fails a gate **keeps every one
of its experts** (a local failure, never a silent corruption of the whole model):

```
no_routing       → the layer has no routing signal
nothing_to_prune → the keep-set is already the whole layer
mass_unreachable → keep-set size > max_keep_fraction
drift            → the layer's measured drift > delta
else             → prune
```

The canonical gate formulas (`docs/PROPOSALS.md` §5):
```
routing_mass(K) = Σ_{e∈K} freq_e / Σ_e freq_e ≥ 1 − τ
drift_proxy(K)  = ‖f(x;K) − f(x;full)‖ / ‖f(x;full)‖ ≤ δ
```
Note: vipera **does not run the model**, so `drift` is an **input** from a probe, not something it
computes. A `delta` without drift data is an **error**, because the drift gate cannot be evaluated.

---

## 9. Algorithm H — The decode cost model (`simulate`, `schedule.py:608`)

For each token, walk every layer:

```
read       = resident + (expert hit) × expert_each + (expert miss) × expert_each
disk_read  = resident × f_disk + (expert miss) × expert_each
secs       = tier_seconds(resident)                      # split by f_vram / f_ram / f_disk
           + cached / bw(ram) + missed / bw(disk)

tier_seconds(n) = n × ( f_vram/bw(vram) + f_ram/bw(ram) + f_disk/bw(disk) ) / 1e9
```

Add the **embeddings**, streamed once per token. Compute floor:

```
compute_s = 2 × active_params / cpu_gflops        # 2 FLOP per parameter
```

Two ways of combining I/O with compute:

| mode | formula | meaning |
|---|---|---|
| with prefetch (`prefetch=True`) | `time = max(io, compute)` | I/O is **hidden** behind compute |
| without prefetch | `time = io + compute` | sequential |

Finally: `ms_per_token = forward_ms / draft_tokens`, `tok_s = 1000 / ms_per_token`.

`--eval-tail <f>` measures the hit rate over only the **last fraction f** of the trace (the window
after a shift), so a slowly adapting policy cannot be "rescued" by the earlier tokens.

**These are MODELLED numbers** — the scheduler always prints an "all values MODELLED" line.

---

## 10. The knobs at this layer

| Knob | Flag | Effect |
|---|---|---|
| Quantization | `--quant fp8\|q4\|q4-experts` | `q4-experts` = 4-bit **experts only**, keeping attention/norms more accurate |
| KV cache | `--kv-bits 16\|8\|4` | KV at `q8_0` is nearly free in quality |
| Pruning | `--prune-experts <p>` | keep fraction `p` of the experts per layer |
| Layer dropping | `--keep-layers N` | drop layers from the end |
| Speculative/MTP | `--draft-tokens γ` | one forward pass produces γ tokens |
| Cache policy | `--cache-policy lru\|lfu\|hot` | victim selection |
| Allocation | `--cache-alloc global\|per-layer` | water-filling |
| Prefetch | `--no-prefetch` | turn it off to compare against the sequential case |

`vipera sweep` runs a fixed set of configurations through exactly this objective function and prints
them side by side.

---

## 11. File-layer optimization (T1–T6) — **core positioning, not yet implemented**

`docs/OPTIMIZER.md`: a runtime only pushes you up to the I/O ceiling; it **cannot change** the fact
that hot experts are scattered across the original file.

| Transform | Input | Output | Benefit on weak hardware |
|---|---|---|---|
| **T1** reorder hot/cold | routing trace | a re-laid-out file | hot experts become **contiguous** ⇒ one sequential read instead of N scattered ones |
| **T2** residency manifest | model + machine | `*.vipera.json` | the runtime knows the VRAM/RAM/disk split instead of guessing `-ngl` |
| **T3** block-wise quant | safetensors | 4-bit GGUF | independent blocks ⇒ `mmap` one block at a time |
| **T4** prune/merge experts | usage frequency | fewer experts | smaller ⇒ fits in RAM |
| **T5** KV placement hints | config | manifest | know where KV lives and at how many bits (the machine has **no swap**) |
| **T6** chunk alignment | the model file | page/disk aligned | avoids over-reading at page edges (DRAM-less disk) |

The binding rule: every transform must **prove itself** with before/after measurements. A transform
that does not win on the target machine is **off by default** and recorded as a failure.

---

## 12. Proposals P2 / P3 — **not yet implemented**

### P2 — Tier-aware bit water-filling (TAW)
```
min  Σᵢ fᵢ · bytesᵢ(bᵢ)
s.t. Σᵢ σᵢ · (Bᵢ − bᵢ) ≤ ε          (quality budget)
     bᵢ ∈ {2,3,4,5,6,8,16}
```
Assign bits in decreasing order of `(bytes saved) / (σᵢ · bits dropped)` — a greedy knapsack,
`O(n log n)`. The idea: a high-hit-rate expert needs **fewer bits** (it is cheap to read anyway),
while always-resident weights streamed from disk need **more bits**.
*The repo's own criticism:* `σᵢ` (the quality lost per bit) is not separable, so P2 is a heuristic
wearing Lagrangian clothes. Estimated at +10–25% tok/s at equal perplexity.

### P3 — Energy-aware tiering (EAT)
```
min  α·T(C) + β·E(C)     s.t. P(C) ≤ P_max , Σ_{e∈C} S_e ≤ M_fast
E(C) = Σ_disk S_e · joules_per_byte(tier)      # DRAM ≪ SSD
```
SSD offload costs **5–12× the energy per token of DRAM** (**published**, not re-verified). Adds an
**energy gate** so energy is not traded for speed when a power cap applies. This may be the
**most important but least evidenced** proposal, because it rests on other people's numbers.

---

## 13. Applying the manifest to a runtime — `vipera run` (implemented)

The mapping is small and inspectable:

| Manifest field | llama.cpp flag |
|---|---|
| every expert in RAM | `--n-cpu-moe <n_layers>` |
| `expert_cache_vram_bytes` | `--moe-cache-mib <MiB>` |
| `resident_vram_bytes` (attention/norm) | `-ngl <n_layers>` |
| `kv_cache.type_k` / `type_v` | `-ctk` / `-ctv` |

Two things the runner **will not pretend**:
1. a build with **no GPU backend** ⇒ the VRAM half cannot be honoured → warns and runs `-ngl 0`;
2. llama.cpp has **no** per-expert disk streaming ⇒ a plan with spilled experts is reported as
   **unmappable** (it falls back to mmap paging, a cost the modelled hit rate does not describe).

The runner also:
- `--compare`: runs **plain llama.cpp defaults** and the manifest-driven command back to back,
  printing delta + speedup;
- `--trace` / `--prefetch-table`: learns the prefetch table and arms the patch through the environment;
- adds `-v` automatically when arming prefetch, because cache/prefetch statistics only log at INFO;
- detects a GPU backend by scanning the `libggml-*.so` files next to the binary;
- detects the prefetch patch by scanning **the shared libraries too** (the patch lives in
  `libllama.so`, not in the `llama-cli` stub).

---

## 14. Real measurements (same configuration: ctx 512, 32 tokens, 8 threads, 9.65 GiB model, experts on CPU)

| build | gen t/s | prompt t/s | peak RSS | disk read | wall |
|---|---|---|---|---|---|
| **CPU-only** | **12.80** | 9.6 | 12369 MiB | 2886.5 MiB | 7.9 s |
| CUDA | 6.90 | 10.2 | 10337 MiB | 404.0 MiB | 8.2 s |
| Vulkan | 3.40 | 5.0 | 10110 MiB | 4196.7 MiB | 16.9 s |

Honest notes:
- **CPU-only wins** — because the model does not fit in 4 GB of VRAM, the experts live on the CPU and
  the GPU only accelerates attention while paying PCIe for every expert. CUDA is ~2× Vulkan and still
  below CPU.
- The manifest's `2789 MiB` cache **does not fit the real VRAM** (Vulkan fails to allocate, CUDA fails
  to create a context) ⇒ it has to be lowered to 1024 through the `--` pass-through.
- `disk read` is noisy with the page cache; `prompt t/s` is noisy because the prompt is short.
- `--compare` on the CPU build: default 12.20 → manifest (`-ctk/-ctv q8_0`) 14.90 t/s = **1.22×**.

---

## 15. Interface

### 15.1 CLI
```
vipera analyze  <model.gguf> [--json]
vipera plan     <model.gguf> --target hw/gtx1650.toml [--out DIR]
vipera arch     <arch.toml>
vipera schedule <arch.toml> --target <hw.toml> [--quant ...] [--cache-policy lru|lfu|hot]
                [--cache-alloc global|per-layer] [--trace T] [--eval-tail F] [--no-prefetch]
vipera trace    <trace.jsonl> --arch <arch.toml> [--target <hw.toml>] [--json]
vipera prune    <trace.jsonl> --arch <arch.toml> --target-mass 0.95 [--json]
vipera prune-gated <trace.jsonl> --arch <arch.toml> [--max-keep-fraction F] [--delta D] [--drift J]
vipera sweep    <arch.toml> --target <hw.toml>
vipera score    <arch.toml> --target <hw.toml>
vipera run      <model.gguf> --target <hw.toml> | --manifest M.vipera.json
                [--compare] [--trace T] [--prefetch-table P] [--dry-run] [--json R]
                [--binary BIN] [-- <llama.cpp flags>]
```

### 15.2 Manifest `*.vipera.json`
```jsonc
{
  "vipera_version": 1,
  "source":  { "architecture": "deepseek2", "n_layers": 27, "n_experts": 64 },
  "target":  { "name": "gtx1650-laptop", "usable_vram_mb": 3584, "usable_ram_mb": 13824 },
  "placement": {
    "resident_vram_bytes": 833058816,
    "expert_cache_vram_bytes": 2925037568,
    "expert_ram_bytes": 6602323968,
    "expert_spill_disk_bytes": 0,
    "expert_cache_fraction": 0.307,
    "expert_order": "uniform",
    "expert_order_pending_trace": true      // no trace yet ⇒ experts cannot be ranked hot yet
  },
  "kv_cache": { "type_k": "q8_0", "type_v": "q8_0" }
}
```

### 15.3 Trace JSONL
```
{"t": 0, "layers": [[20,5,42,22,46,45], [7,59,42,61,60,35], ...]}
```
Each entry = one **MoE block** (the leading dense block has no router).

### 15.4 Prefetch table
```
# llama.cpp MoE prefetch table
<layer>\t<expert>,<expert>,...
```

---

## 16. Status & limits

| Algorithm | Status | Evidence |
|---|---|---|
| A. Tensor classification | ✅ | `analyze` |
| B. Residency planning (greedy) | ✅ | `plan`, `schedule` |
| C. Expert cache LRU/LFU/`hot` | ✅ | +11 points of hit rate, +14% tok/s |
| D. Per-layer water-filling | ✅ | **does not help** → default `global` |
| E. Trace + entropy + ceiling | ✅ | `global_cache_ceiling` |
| F. Trace-driven prefetch | ✅ (via patch, env) | 8.7–11.2% useful |
| G. Mass pruning + P4 gated | ✅ | 79.6% of the pool keeps 95.3% of the mass |
| H. Decode cost model | ✅ (modelled) | `vipera sweep` |
| T1–T6 (file layer) | ❌ positioned, not coded | `docs/OPTIMIZER.md` |
| P2 tier-aware bits | ❌ proposal | estimate |
| P3 energy-aware tiering | ❌ proposal | published numbers |
| CPU–GPU co-exec, KV offload | ❌ not yet | — |

**Limits that must always be stated alongside:**
1. Every `schedule` number is **modelled**; the scheduler does not model GPU compute or kernel
   efficiency.
2. Default routing is **modelled Zipf**, replaced by a real trace only when `--trace` is passed.
3. Energy and quality are **not** modelled ⇒ P2/P3 depend on inputs vipera cannot produce yet.
4. `vram_bw_gbs = 192` is a **datasheet** figure, not measured (a root-free CUDA toolkit and a
   `libggml-cuda` build now exist, but no VRAM bandwidth probe has been run).
5. The GPU measurements are **one run per configuration**, not averaged; `disk read` is noisy with the
   page cache.
6. `sweep` and `prune` do **not** demonstrate quality — they only count routing mass; any quality claim
   needs perplexity/benchmark measurements (not done yet).
