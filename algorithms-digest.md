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

### 4.2 Modelled over synthetic Zipf routing — **not measured**

```
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --cache-policy lru
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --cache-policy lfu
```

| policy | hit rate | disk bytes/token | tok/s |
|---|---|---|---|
| LRU | 21.5% | 3.11 GiB | 0.544 |
| **LFU (P1)** | **32.6%** | **2.66 GiB** | **0.620** |

→ **+11.1 points of hit rate, +14% modelled throughput**, on a *synthetic* Zipf routing
distribution. `[modelled]`

> **These are not throughput measurements and were being read as if they were.** `vipera schedule`
> reports a model, and the routing here is modelled Zipf rather than a captured trace. The one
> real measurement of the same machinery — the manifest, which carries the cache placement —
> moves generation throughput by **0.92–1.00×** on the target machine (`docs/FINDINGS.md` §Round 4),
> and cutting disk reads by 26% moved it by −4%. Hit rate is real and is worth points; **the
> conversion from hit rate to tokens per second does not survive contact with a RAM-resident
> model**, because disk is ≤0.2% of decode there and there is nothing for the saved reads to buy.

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
- `--compare` on the CPU build: default 12.20 → manifest (`-ctk/-ctv q8_0`) 14.90 t/s = **1.22×**
  **[superseded — re-measured 2026-10-13, see below]**
- **Re-measured `--compare`, same build, DeepSeek-V2-Lite Q4_K_M, `-t 4`, ctx 512, 32 tokens, 2
  reps each** `[measured]` — the 1.22× above **does not reproduce**, on either footing:

  | baseline | manifest | speedup | notes |
  |---|---|---|---|
  | 22.80 tok/s (`-lm mlock`) | 22.50 | **0.99×** | clean page cache, no paging at all |
  | 22.60 tok/s (`-lm mlock`) | 22.50 | **1.00×** | second repetition |
  | 13.50 tok/s (`mmap`, default) | 12.90 | **0.96×** | default path, page cache thrashing |

  So the manifest is worth **nothing to −4%** whether the baseline is clean or thrashing. Two
  things fall out of this, both `[measured]`:
  - **The old 1.22× cannot be attributed to KV quantization at ctx 512.** q8_0 vs f16 is worth
    ~0 at this context (16.41 vs 16.51 tok/s in `bench/step1_attrib.py`), and the manifest maps
    on a CPU-only build to exactly `-ngl 0 -ctk q8_0 -ctv q8_0` and nothing else
    (`vipera run … --dry-run`). Whatever produced 14.90 needs re-isolating before it is claimed.
  - **Reducing disk read does not improve throughput on a RAM-resident model.** On the `mmap`
    run the manifest cut disk read by 3918 MiB (−26%) and still scored 0.96×. This is the
    direct, non-inferential form of the STEP 1 finding that decode touches 0.4 MiB/token.
  - The manifest's value is confined to placement across VRAM/RAM/disk, and on a CPU-only build
    with the model resident in RAM that placement has nothing to act on. Its case has to be made
    on a model that spills, or at long context — see `plantodo.md`.
- **Three sequential `--compare` runs at ctx 2048 are valid** (exit 0; 1.04×, 0.81×, 1.12×).
  An earlier pair at that ctx was SIGKILL'd, and the cause was **running two benchmarks
  concurrently** on a 15 GiB machine with no swap — each side of `--compare` loads the model, so
  two comparisons at once ask for ~20 GiB. The lesson is not about ctx: an OOM kill surfaces as
  `n/a` with **exit 0**, so a killed run is indistinguishable from a missing number unless the
  error line is read. **Never run these benchmarks in parallel here.**
- The 0.81–1.12× spread at ctx 2048 is `mmap` thrash variance, not a ctx effect: the same
  settings at ctx 512 under `-lm mlock` held 0.96–1.00×.

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
| C. Expert cache LRU/LFU/`hot` | ✅ hit rate · **❌ throughput** | +11 pts hit `[modelled]`; the **+14% tok/s** previously claimed here is modelled over synthetic Zipf and is contradicted by the one real measurement: **0.92–1.00×** |
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
4. **A hit-rate gain does not convert to throughput on a RAM-resident model.** This was measured,
   not assumed: hit rate rose 11.1 points and disk reads fell 26%, and generation throughput moved
   by **−4%** (`docs/FINDINGS.md` §Round 4). The reason is in the attribution — disk is **≤0.2% of
   decode** on this machine, so reads that never happen were never on the critical path. The hit
   rate is real and worth points; **the conversion factor from hit rate to tok/s is what failed**,
   and any figure that applies one should say which regime it assumes.
4. `vram_bw_gbs = 192` is a **datasheet** figure, not measured (a root-free CUDA toolkit and a
   `libggml-cuda` build now exist, but no VRAM bandwidth probe has been run).
5. The GPU measurements are **one run per configuration**, not averaged; `disk read` is noisy with the
   page cache.
6. `sweep` and `prune` do **not** demonstrate quality — they only count routing mass; any quality claim
   needs perplexity/benchmark measurements (not done yet).

---

## 17. The two literature packs, cross-checked against this repo

`docs/reference/literature/` (19 papers) and `docs/reference/literature/llm-optimization/`
(24 papers) each carry a synthesis that grades every algorithm on whether *vipera can implement
it*. This section is the half those documents do not do: for each of the **43 papers** it names the
**actual code surface** the algorithm would attach to — read out of the repository, not inferred
from the grade — or says plainly that no such surface exists yet.

Grades are the letters the two syntheses already use: **A** fits the existing surface now · **B** the
planner skeleton fits but a missing input (Hessian / F-norm / KL / a trained head) blocks it · **C**
blocked on the GGUF tensor writer (S9) · **D** needs activation sparsity the target model
(DeepSeek-V2-Lite, SwiGLU, non-ReLU) does not have · **E** out of scope. **Every “repo surface” cell
is a `file:line` into the tree at `1414346`, verified by `grep`/`code_search` on 2026-10-14.** The
tests that pin those surfaces are green: `227 passed` (`./.venv/bin/python -m pytest`).

The surface anchors referenced repeatedly below:

| Surface | Where |
|---|---|
| Tensor classification | `vipera/analyze.py:18` `is_expert_tensor` |
| Residency planner (greedy) | `vipera/manifest.py:89` `plan` |
| Decode cost model | `vipera/schedule.py:608` `simulate` |
| Expert cache (lru / lfu / hot) | `vipera/schedule.py:308` `ExpertCache` (`_admit:473`, `_age:431`, `_reallocate:441`) |
| Routing trace + ceiling | `vipera/trace.py:70` `load_trace`, `:171` `global_cache_ceiling` |
| Mass pruning + P4 gate | `vipera/prune.py:91` `plan_from_trace`, `:261` `gated_prune` |
| Prefetch table → llama.cpp | `vipera/prefetch.py:37` `build_table`; `bench/llama-moe-prefetch.patch`; `vipera/run.py:107` `supports_prefetch` |
| Manifest → runtime flags | `vipera/run.py:133` `plan_flags` |
| Quant / KV / prune knobs | `vipera/cli.py:147` `_QUANTS`; `:567` `--kv-bits`; `:570` `--cache-policy`; `:574` `--trace` |
| GGUF reader (header + descriptors only) | `vipera/gguf.py:309` `read_gguf` |
| GGUF writer + expert reorder/prune | `vipera/gguf_write.py` `rewrite_experts`, `expert_order`; `vipera/cli.py` `rewrite` |
| Long-context KV budget + sink policy | `vipera/longcontext.py` `plan_long_context`; `vipera run --stream` |
| Launch policy (load mode + threads) | `vipera/launch.py` `choose_policy` |
| Perplexity, incl. streaming | `bench/perplexity.py` `compare_stream`; `bench/llama-ppl-stream.patch` |
| Activation-energy probe | `bench/sparsity.py` (imatrix → channel stats) |
| Architecture model | `vipera/arch.py:94` `Architecture` |

**One field is a label, not a signal.** `placement.expert_order` and
`placement.expert_order_pending_trace` are written at `vipera/manifest.py:169-170`, and
`expert_order` appears **nowhere else under `vipera/`** except its own module comment
(`grep -rn expert_order`, ignoring the stale `.bridge` worktree) and two assertions in
`tests/test_manifest.py:70-71`. `vipera/run.py:133 plan_flags` never reads it. So Neuralink’s
“give the dead field a consumer” is a genuine gap, not a wire that already exists. It still has
no consumer in `plan_flags`; the layout that consumes a trace today is written straight to the
file by `vipera rewrite --from-trace` instead.

**The writer gap this digest opened with is closed.** `vipera/gguf.py` reads descriptors only,
and the note here used to end at “nothing in `vipera/` writes tensor data” — which is why every
format-level paper (C) was blocked by *one* missing writer. `vipera/gguf_write.py` is that
writer, and because it edits already-quantized expert slabs byte-wise, one module covers both
the reorder (`2410.19274`) and the prune path rather than one writer per quant format. What it
does **not** yet do is run on a real multi-gigabyte model: the byte-exactness is established on
fixtures (`docs/REWRITE.md` §C), so treat the format-level grades below as *unblocked*, not
*delivered*.

### 17.1 Pack 19 — `docs/reference/literature/`

| # | arXiv | Algorithm | Repo surface (verified) | Grade |
|---|---|---|---|---|
| 1 | 2312.11514 | LLM in a Flash — windowing + row-column bundling | byte-level cache only (`schedule.py:308`); no windowing, no ReLU-sparsity path | D |
| 2 | 2310.16795 | QMoE — sub-1-bit format + decode kernel | `gguf.py:309 read_gguf` is read-only; no writer in `vipera/` | C |
| 3 | 2410.19274 | Neuralink — co-activation → flash layout | target field exists but dead (`manifest.py:169-170`); no reader in `run.py`/`schedule.py` | A (gap) |
| 4 | 2310.02410 | MoQE — experts tolerate 2–3 bit | modelled quant only: `schedule.py:38 Quant`, `cli.py:147 _QUANTS["q4-experts"]` | B |
| 5 | 2310.19102 | Atom — W4A4 mixed group quant | needs a kernel + activation path; `vipera/` has neither | E |
| 6 | 2505.05799 | MxMoE — ILP bit allocation under budget | planner skeleton could live in `schedule.py`; per-block sensitivity input absent | B |
| 7 | 2603.19172 | DyMoE — depth-aware bit schedule + prefetch | prefetch half present (`prefetch.py:37`); bit schedule / importance gate absent | B/D |
| 8 | 2509.02512 | MoPEQ — k-means by Hessian sensitivity | no weight access (vipera never opens tensor data) | B |
| 9 | 2303.06865 | FlexGen — LP placement + zig-zag schedule | greedy, not LP: `manifest.py:89 plan`, `schedule.py:227 plan_residency` | A (planner) |
| 10 | 2310.07240 | CacheGen — KV as an adaptive bitstream | KV is one uniform bit-width: `cli.py:567 --kv-bits`, manifest `kv_cache` | E |
| 11 | 2406.02500 | Holistic MoE — Expert→Layer→Block drop | `cli.py:566 --keep-layers`, `prune.py:91`; the drop pieces exist, the recovery finetune does not | B |
| 12 | 2410.06270 | MC — LP alloc + quant-loss signal | allocation surface = the planner; the ϵᵢⱼ loss needs forward passes | B |
| 13 | 2606.17118 | MODE — expert-level ILP | same planner gap; modality / key-token sensitivity needs calibration | B |
| 14 | 1911.03852 | HAWQ-V2 — bit alloc as rate–distortion | knapsack framing fits `schedule.py`; σᵢ needs weights | B |
| 15 | 2412.14219 | Survey — the 3-tier map | reference only; maps onto `docs/RESEARCH.md`, `docs/ALGORITHMS.md` | E (ref) |
| 16 | 2502.05370 | FineMoE — iteration-level patterns + semantic hints | trace already per-token (`trace.py:70`); cache/prefetch `schedule.py:308`, `prefetch.py:37` | A |
| 17 | 2601.14277 | Which Quantization — Q4_K_S / Q5_0 frontier | default quant knob `cli.py:147`; evidence only, no code path | A (evidence) |
| 18 | 2510.26692 | Kimi Linear — KDA attention | `arch.py:94 Architecture` models the shape; no attention runtime | E |
| 19 | 2504.08378 | ActiveFlow — cross-layer preloading | `prefetch.py:37 build_table` is routing-based, not activation-based; no activation sparsity | D |

### 17.2 Pack 24 — `docs/reference/literature/llm-optimization/`

| # | arXiv | Algorithm | Repo surface (verified) | Grade |
|---|---|---|---|---|
| 1 | 2312.12456 | PowerInfer — neuron ILP + predictors + neuron-aware ops | expert-level only (`prune.py:91`); no neuron/activation code in `vipera/` | A (gap) |
| 2 | 2406.11674 | Endor — bitmap sparse format (Alg. 1/2) | file layer vipera owns (T1 precedent) but reader-only: `gguf.py:309` | A (layer) |
| 3 | 2306.14048 | H2O — greedy KV eviction from attention | vipera models KV bytes, never attention: `cli.py:567 --kv-bits` only | A |
| 4 | 2309.17453 | StreamingLLM — sink tokens + rolling window | same: no KV runtime; `--kv-bits` shrinks, it does not evict | A |
| 5 | 2306.00978 | AWQ — α grid-search scaling | no quantiser/writer; only modelled quant (`schedule.py:38`) | B |
| 6 | 2301.00774 | SparseGPT — one-shot weight prune | prunes **experts by routing mass** (`prune.py:91`), not weights by Hessian | B |
| 7 | 2411.01433 | HOBBIT — T1/T2 dual precision + LHU/FLD cache | cache policies are lru/lfu/hot (`schedule.py:308`); no dual precision | B |
| 8 | 2410.17954 | ExpertFlow — routing predictor + token scheduler | `prefetch.py:37` is a mass keep-set, not a predictor; no batching/scheduler | B |
| 9 | 2305.09781 | SpecInfer — token-tree speculative verify | `cli.py:568 --draft-tokens` is a model multiplier; no draft model is run | B |
| 10 | 2401.15077 | EAGLE — feature-level draft head | same: `--draft-tokens`, no trained head | B |
| 11 | 2210.17323 | GPTQ — PTQ + Cholesky reformulation | modelled bits only; no calibration/writer | B |
| 12 | 2305.14314 | QLoRA — NF4 + double quantization | training out of scope; modelled quant only | B |
| 13 | 2106.09685 | LoRA — low-rank adapters | no training code in the repo | B |
| 14 | 2309.06180 | vLLM / PagedAttention — paged KV + block table | no KV allocator; `cli.py:567 --kv-bits` only | C |
| 15 | 2403.02310 | Sarathi-Serve — chunked prefill, stall-free batching | single-sequence simulator (`schedule.py:608`); no scheduler | C |
| 16 | 2312.07104 | SGLang / RadixAttention — prefix KV reuse | no multi-request server / prefix sharing | C |
| 17 | 2311.18677 | Splitwise — phase split + placement | no cluster; nothing in `vipera/` | C |
| 18 | 2401.09670 | DistServe — per-phase parallelism search | no cluster; nothing in `vipera/` | C |
| 19 | 2211.10438 | SmoothQuant — per-channel smoothing (W8A8) | `_QUANTS` (`cli.py:147`) has no W8A8 / INT8 activation path | C |
| 20 | 2205.14135 | FlashAttention — IO-aware tiling | lives in ggml/llama.cpp, which `vipera/run.py` drives as a binary | C |
| 21 | 2407.08608 | FlashAttention-3 — warp-spec + FP8 | Hopper-only; vipera ships no kernel | D |
| 22 | 2312.00752 | Mamba — selective SSM | `arch.py:94` models shapes only; cannot apply to a Transformer GGUF | D |
| 23 | 1910.02054 | ZeRO — training DP partitioning | training is out of scope; no optimizer | C |
| 24 | 2101.06840 | ZeRO-Offload — offload training state/compute | same: no training state to offload | C |

### 17.3 What the code says that the grades do not

Reading the tree sharpens three of the syntheses’ claims:

1. **The one dead label is smaller than it looks.** Neuralink’s gap is *only* the runtime consumer:
   the ordering itself is computable today from `trace.py:70 load_trace`, and `prefetch.py:37`
   already proves the pipeline (trace → `prune.plan_from_trace` → a file) works. What is missing is
   one line in `run.py:133` that reads it — everything else is built.
2. **The C-grade papers share exactly one missing module.** Every “blocked on the S9 GGUF tensor
   writer” row (QMoE, Endor, SparseGPT’s output, AWQ/GPTQ/SmoothQuant) hits the same wall: there is
   no writer anywhere in `vipera/`. The reader (`gguf.py:309`) and the descriptor parser
   (`gguf.py:273 validate_extents`) are done; the write half is the single unbuilt transform.
3. **The cache papers (H2O, StreamingLLM, HOBBIT, FineMoE) are runtime patches, not vipera code.**
   `ExpertCache` (`schedule.py:308`) is a **simulation** over a trace; it is never in the decode
   path. So these items can be *modelled* here but only *realised* in a llama.cpp patch, which is
   the same split `prefetch.py` + `bench/llama-moe-prefetch.patch` already demonstrates.
