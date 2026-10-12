# ROADMAP — vipera decomposed into solvable sub-problems

> A research-engineering decomposition: what the project must solve, in what
> order, with measurable success criteria. Plus the log of the first autonomous
> loop round (spec → code → test → review → integrate → reflect).
>
> Status labels: **done**, **in progress**, **blocked**, **open**.

---

## 1. Sub-problems

| # | Sub-problem | Research question | Input → Output | Depends on | Difficulty | Priority | Success criterion |
|---|---|---|---|---|---|---|---|
| **S1** | **Routing trace capture & loader** | Is expert routing skewed and locally consistent enough for a cache to help? | JSONL trace → `routing` structure + skew/entropy/ceiling stats | — | 2 | **5** | **done** — loader+stats+validator, 95 tests, CLI `vipera trace` |
| **S2** | **Cache policy: pinned hot-set + admission control** | Can a policy reach the static-cache ceiling? | trace + cache size → hit rate | S1 | 3 | 5 | **done** — `hot` reaches 0.92× on a post-shift trace (LFU 0.71×) and 0.94× stationary; every `schedule` run now prints the ceiling at the *matched* size |
| **S3** | **Per-layer cache allocation** | How many slots should each layer get? | per-layer frequency → slot budget | S1 | 4 | 4 | **done (negative)** — water-filling by per-layer marginal mass does **not** beat the global pool (±1 pt on real and synthetic traces); a global LFU already realises the same greedy marginal allocation implicitly |
| **S4** | **Tier-aware bit allocation (TAW)** | Which component should absorb the next lower bit? | sensitivity + tier cost → per-component bits | S10 | 5 | 4 | +10% tok/s at equal perplexity |
| **S5** | **Energy model + energy-aware tiering (EAT)** | When does SSD offload violate a watt cap? | tier table → J/token | S7 | 3 | 4 | modelled J/token within 20% of wattmeter |
| **S6** | **Gated expert pruning with rollback (GEP)** | Which experts are safe to delete? | routing mass + drift proxy → keep-set | S1, S10 | 4 | 4 | quality-loss variance halved at equal sparsity |
| **S7** | **On-hardware benchmark harness** | What does the target machine actually do? | model + flags → tok/s, RSS, VRAM, disk, W | — | 2 | **5** | reproducible JSON for ≥2 configs |
| **S8** | **Dense-model streaming scheduler** | How fast can a dense model stream from disk? | arch + hw → residency/bytes per token | — | 3 | 3 | modelled vs measured within 25% (needs S7) |
| **S9** | **GGUF writer / packer (T1, T3, T6)** | Can the file itself be made cheaper to read? | GGUF → repacked GGUF | — | 4 | 3 | byte-exact round-trip + measured cold-load delta |
| **S10** | **Quality evaluation harness** | What did compression cost? | model + config → ppl / MMLU / GSM8K | S7 | 3 | **5** | repro a published quant-quality gap within 0.5 pts |

---

## 2. Dependency graph

```mermaid
graph TD
  S1[S1 trace loader ✅] --> S2[S2 cache policy]
  S1 --> S3[S3 per-layer allocation]
  S1 --> S6[S6 gated pruning]
  S7[S7 bench harness] --> S5[S5 energy model]
  S7 --> S8[S8 dense scheduler]
  S7 --> S10[S10 quality harness]
  S10 --> S4[S4 tier-aware bits]
  S10 --> S6
  S9[S9 GGUF packer]
```

---

## 3. Recommended order

Topological, weighted by (priority − risk-of-waiting):

1. **S7 bench harness** and **S10 quality harness** — nothing else can be *trusted*
   without them; every claim in `PROPOSALS.md`/`RESEARCH.md` currently rests on
   published numbers.
2. **S1 trace loader** — **done**.
3. **S2 cache policy** — **done**. Next: **S3 per-layer allocation** — the round-2
   sweep shows the remaining gap to the ceiling is *allocation across layers*, not
   the eviction rule.
4. **S6 gated pruning** — needs S1 + S10.
5. **S5 energy model**, then **S4 tier-aware bits** — need S7/S10 inputs.
6. **S8**, **S9** — valuable but not on the critical path to a verified result.

**Why risk matters:** S4/S5/S6 all depend on measurements vipera cannot produce
today. Building them first would encode unknown constants — exactly the failure
mode `docs/PROPOSALS.md` §7 warns about.

---

## 4. Autonomous loop — round 1 log

**Picked sub-problem: S1 (highest priority that is unblocked).**

### BƯỚC 1 — Research
The gap was already established in `RESEARCH.md`/`PROPOSALS.md`: routing is
modelled, not measured, so every cache claim is suspect. No new search was needed
beyond confirming the standard cache rule (≈2× active experts) [RESEARCH.md S3].

### BƯỚC 3 — Spec
* **Goal:** consume real routing traces; report the statistics that decide whether a cache can help.
* **Input/Output:** JSONL `{"t": i, "layers": [[e,…], …]}` → `routing[token][layer][slot]`.
* **Algorithm:** line-parse + validate (layer count, top_k, id range); histogram per layer; normalised entropy `H = −Σp·log₂p / log₂E`; static-cache ceiling = Σ(top-F experts)/Σ(all) per layer.
* **Complexity:** O(T·L·K) time, O(E·L) memory.
* **Edge cases:** empty file, malformed JSON, missing `layers`, expert id out of range, wrong `top_k`.

### BƯỚC 4–5 — Code + test
* `vipera/trace.py` — `load_trace`, `validate`, `stats`, `synth_trace`, `save_trace`.
* CLI: `vipera trace <file> --arch <arch.toml> [--json]`; `vipera schedule … --trace <file>`.
* `tests/test_trace.py` — 13 tests: round-trip, limit, 5 validation failures, entropy, ceiling, skew ordering, JSON-serialisability.
* **Result: 95 tests, OK.**

### BƯỚC 6 — Review (hostile)
* `stats()` is O(T·L·E) *memory-free* but rebuilds a histogram per layer per call — fine for planning, not for a hot loop.
* The "static cache ceiling" is an **oracle** (it uses the full trace's frequency); it is an upper bound, **not** an achievable online result. Labelled as such in the docstring/api.
* `load_trace(limit=…)` silently truncates — acceptable for smoke runs, but callers must know; the CLI passes `tokens+2`.
* No streaming/mmap for very large traces — a 100k-token trace is fine, a 10M-token one is not (known limitation).
* `synth_trace` is Zipf by construction, so tests that use it validate the *code*, not the *world*.

### BƯỚC 7 — Integrate
`schedule --trace` validates the trace against the architecture before simulating,
so a mismatched trace fails loudly (exit 2) instead of silently producing nonsense.

### BƯỚC 8 — Reflect (measured, trace-driven)

Reproduce exactly:
```
python3 -c "from vipera.trace import synth_trace, save_trace; \
  save_trace('/tmp/r1.jsonl', synth_trace(64, n_layers=45, n_experts=288, top_k=8, skew=1.2, seed=0))"
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml \
    --quant q4-experts --trace /tmp/r1.jsonl --cache-policy lru   # 33.5%, 0.626 tok/s, bound=io
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml \
    --quant q4-experts --trace /tmp/r1.jsonl --cache-policy lfu   # 43.0%, 0.712 tok/s, bound=compute
```
`--quant q4-experts` is **required**: the default (`fp8`) yields only a 372-slot
cache and 18.0%/33.2%, a different experiment entirely.

Trace: `synth_trace(64 tokens, 45 layers, 288 experts, top_k=8, skew=1.2)`.

| policy | hit rate | tok/s |
|---|---|---|
| LRU | 33.5% | 0.626 |
| LFU | **43.0%** | **0.712** |
| *static-cache oracle ceiling at 6.09% of experts (789 slots — the size actually available)* | **60.7–61.8%** | — |

**Findings.**
1. LFU reaches **0.70× the matched static oracle ceiling** (43.0 / 61.8), not the
   ~0.9× stated in the first draft of this section. That draft compared the achieved
   hit rate at a **6.09%** cache against a ceiling computed at **3%** (`48.1%`) — a
   benchmark at one size against a baseline at another. At the matched size the
   ceiling is ~61%, so the policy is still **~30% off** and remains a real lever:
   S2 is needed, and its `≥0.9×` criterion is currently **unmet**.
2. The modelled-Zipf numbers (21.5%/32.6%, `docs/PROPOSALS.md`) were **pessimistic**
   vs this trace (33.5%/43.0%). The comparison *is* apples-to-apples: same arch,
   same target, same quant, same 789-slot cache, only the routing model differs. A
   real trace is still required before any of it is trusted.
3. The LFU run at 43.0% reports `bound: compute` while LRU at 33.5% reports
   `bound: io` — the workload flips to the compute floor between the two, so past
   ~43% the next lever is S10/S4 (bits) or speculative decoding, not more cache.

**Next sub-problem:** S2 (reach the ceiling with pinned hot-set + admission
control), then S7/S10 to replace modelled inputs with measured ones. *Round 2
did S2 — see §7.*

---

## 5. Open problems (from Prompt 7)

| # | Open problem | Why hard | Why it matters | Anyone solving it? |
|---|---|---|---|---|
| O1 | **Cache policy that generalises across domains** | routing shifts; LFU has no aging | cache claims currently valid only for stationary traces | research on local routing consistency [RESEARCH.md S3] |
| O2 | **Per-layer cache allocation under a global byte budget** | knapsack over layers with shifting demand | the remaining gap to the ceiling is *allocation*, not policy | MoEpic-style per-layer solvers [PROPOSALS.md P2] |
| O3 | **Schedulable energy constraint** | needs a per-tier J/byte model that vipera lacks | 65 W-class problems fail on energy, not time | SSD-energy result exists [RESEARCH.md §6] |
| O4 | **Cheap quality proxy for pruning** | quality is non-separable across components | pruning is the decisive MoE lever but unverified | pruning tooling warns it needs validation [RESEARCH.md S13] |

**Most worth solving:** O1 and O2 — both are reachable with S1 already done, and
both are the difference between "cache helps" and "cache is at its ceiling".

## 6. Research problems (Prompt 9, condensed)

1. **Routing-consistency-aware cache sizing** — predict the useful cache size from a short trace instead of the fixed ≈2× rule. Metric: hit rate at fixed bytes.
2. **Non-stationary expert caching** — a policy with provable regret against an adaptive optimum under distribution shift.
3. **Energy-constrained offload scheduling** — minimise latency s.t. J/token ≤ budget; currently unexplored (latency-only literature).
4. **Sensitivity-estimating quantizer** — learn `σᵢ` (quality per bit) with few probes, then solve P2 TAW.
5. **Prune-then-heal gate** — a proxy guaranteeing ΔQ ≤ ε before committing a layer.
6. **File-layer reorder under a real trace** — does T1 (hot/cold reorder) actually cut disk bytes on a DRAM-less disk?
7. **Cross-layer expert co-activation** — exploit inter-layer routing correlation to prefetch.
8. **KV placement under no-swap** — allocation that never triggers OOM on a swap-less box.
9. **Dense streaming vs offload boundary** — the exact model size where streaming beats partial residency.
10. **Honest evaluation protocol** — reusable, noise-controlled local-inference benchmark (the missing S7+S10).

Each is impact **4–5** if solved; #1/#2 are the cheapest to attack from here.

---

## 7. Autonomous loop — round 2 log (S2)

**Picked sub-problem: S2** (policy), and while in there, the two measurement
defects that make any S2 number untrustworthy.

### The two defects fixed first (both from round 1)
1. **Mismatched ceiling.** `schedule` reported an achieved hit rate at the cache
   size it derived (789 slots = **6.09%**/layer) next to a ceiling computed at
   **3%** (48.1%). Fixed by computing the ceiling *inside* `simulate`, on the same
   routing and at the same slot count — `schedule` now prints
   `static ceiling … policy / ceiling …x` every run. There is no longer a way to
   compare the two at different sizes.
2. **Silent trace truncation.** `schedule --trace` loaded only `--tokens + 2`
   tokens (default **10**). Round 1's headline numbers were a 10-token sample and
   did not say so. With a trace it now consumes the **whole** trace unless
   `--tokens` is set explicitly.

### BƯỚC 3 — Spec
* **Goal:** a policy that reaches ≥0.9× the *matched* static ceiling — including
  after the routing distribution moves, where plain LFU cannot recover.
* **Algorithm (`hot`):** LFU keyed on `(layer, expert)` —
  * **admission filter:** on a full-cache miss, admit the key only if it has been
    seen more often than the key it would evict; otherwise it *bypasses* the cache
    (one-off experts cannot flush the hot set);
  * **pinned hot set:** the most-frequent keys are pinned against eviction;
  * **aging:** counters halve every `aging_interval` accesses so the hot set can
    follow a drifting router.
* **Refactors so the numbers cannot drift again:** `plan_residency()` is now the
  single source of the residency/cache arithmetic, shared by `simulate` and by
  `vipera trace --target`; `global_cache_ceiling()` gives the size-matched bound.

### BƯỚC 4–5 — Code + test
* `vipera/schedule.py` — `ExpertCache` (`lru`/`lfu`/`hot`), `Plan`,
  `plan_residency`, `eval_tail` window, manifest `expert_cache.{slots,
  per_layer_fraction,static_cache_ceiling,ceiling_ratio,eval_window}`.
* `vipera/trace.py` — `global_cache_ceiling()`, `synth_shift_trace()`.
* CLI — `--cache-policy hot`, `--eval-tail`, `trace --target/--quant` (derives the
  matched cache size from the same plan the schedule uses).
* `tests/test_cache_policy.py` (17 new) + 2 CLI tests → **114 tests, OK**.

### BƯỚC 8 — Reflect (measured, reproducible)
Generate the traces:
```
python3 -c "from vipera.trace import synth_trace, synth_shift_trace, save_trace; \
  save_trace('/tmp/st.jsonl', synth_trace(200, n_layers=45, n_experts=288, top_k=8, skew=1.2, seed=0)); \
  save_trace('/tmp/sh.jsonl', synth_shift_trace(200, n_layers=45, n_experts=288, top_k=8, skew=1.2, seed=0))"
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml \
    --quant q4-experts --trace /tmp/sh.jsonl --cache-policy hot --eval-tail 0.3
```
`--quant q4-experts` (789 slots) throughout; `--eval-tail 0.3` = last 30% of the
shift trace, ceiling measured on that same window.

| trace / window | LRU | LFU | **hot** |
|---|---|---|---|
| stationary, whole | 0.66× | 0.96× | **0.94×** |
| shifted, whole | 0.82× | 0.95× | **1.10×** |
| shifted, post-shift tail | 0.65× | 0.71× | **0.92×** |

**Findings.**
1. **A static ceiling is not a universal upper bound.** On the shifted trace the
   `hot` ratio is **1.10×** — an adaptive cache beats a static one under drift.
   The `< 1` bound only holds while the distribution is stationary, which is worth
   stating wherever the word "ceiling" appears.
2. **The S2 criterion is met where it matters.** On the post-shift window `hot`
   reaches **0.92×** while plain LFU stalls at **0.71×** — the round-1 conclusion
   ("LFU ≈ ceiling, policy is not the bottleneck") was an artefact of the 10-token
   sample. Policy *is* a lever once the routing moves.
3. **Aging is the decisive lever, not admission or pinning.** Ablation at the
   chosen aging interval (160-token traces, 2 seeds, ratio stationary/shift):

   | variant | stationary | shifted |
   |---|---|---|
   | `hot` (admission + pin + aging) | 0.932 | 0.901 |
   | no admission filter | 0.916 | 0.888 |
   | no pinning | 0.932 | 0.902 |
   | no aging | **0.960** | **0.695** |

   Admission buys ~1.5 points; pinning is **neutral** on these traces (it is kept
   for the stated design and pinned behaviour is unit-tested, but the synthetic
   sweep does not justify it on its own); aging is what trades stationary
   accuracy for shift recovery.
4. **8× capacity is the aging sweet spot** (aging interval sweep, tail ratio
   stationary / shift-50% / shift-25%):

   | interval | stationary | shift 50% | shift 25% |
   |---|---|---|---|
   | none | 0.960 | 0.695 | 0.702 |
   | 2× | 0.800 | 0.798 | 0.798 |
   | 4× | 0.888 | 0.884 | 0.885 |
   | **8×** | **0.932** | **0.901** | **0.902** |
   | 16× | 0.952 | 0.844 | 0.845 |

   No setting wins both; 8× is the only one that clears 0.9× on all three.

### Known limitations
* `ExpertCache._victim()` is **O(capacity)** per miss (linear scan for the min
  frequency). Fine for planning-sized traces; a long captured trace would want a
  lazy heap. Measured wall-clock on a 200-token trace is seconds, not minutes.
* Pinning is retained and pinned behaviour is tested, but the sweep above shows it
  is neutral on synthetic traces; real traces may differ.
* All numbers remain **synthetic** — `synth_trace`/`synth_shift_trace` are Zipf by
  construction, so they test the *code and the policy*, not the world. A real
  captured trace is still S7.

**Next sub-problem:** S3 (per-layer allocation) — *this prediction was tested in
round 3 and did **not** hold; see §8. No headline claim in this document should
be read as "allocation is the remaining lever".*

---

## 8. Autonomous loop — round 3 log (S7 real trace + S3, and the 100B question)

Round 3 replaces the modelled routing with a **real captured trace**, tests the
round-2 prediction about allocation, and answers the "can ~100B run on this
laptop?" question with the same machinery.

### S7 — capture a real routing trace
llama.cpp has no routing-export hook, but it already **names** the router's
selection tensor per layer (`cb(selected_experts, "ffn_moe_topk", il)` in
`src/llama-graph.cpp`). A ~50-line local patch to `src/llama-context.cpp`
(`LLAMA_MOE_TRACE=<path>`) marks those tensors as graph outputs and reads them
back after each `n_tokens == 1` decode step. It changes **no computation**. The
capture is driven by `bench/capture_trace.py` (tested without a model).

Model: **OLMoE-1B-7B-0924** Q4_K_M (a real small MoE: 16 layers, 64 experts,
top-8) on 8 CPU threads. New profile `arch/olmoe-1b-7b.toml` reproduces the
published totals to −1.2% (6.8B vs 7B).

```
python3 -m bench.capture_trace --bin ~/dev/llama.cpp/build/bin/llama-cli \
    --model olmoe-q4km.gguf --tokens 512 --threads 8 --ngl 0 --out olmoe.jsonl
```

Captured: **511 decode tokens, 16 layers, top-8, 44–58 distinct experts/layer.**

### Result 1 — the real trace is *stationary*, and that flips the policy choice
OLMoE's expert pool (1024 experts ≈ 4 GiB) **fits this laptop's cache entirely**,
so the machine never stresses it. To compare policies on real routing the budget
is constrained directly; the ceiling is recomputed on the same real trace at each
size.

| cache budget | ceiling | LRU | LFU | `hot` |
|---|---|---|---|---|
| 10% (102 slots) | 42.7% | **0.0%** | **41.5% (0.97×)** | 39.7% (0.93×) |
| 25% (256 slots) | 71.1% | 53.6% (0.75×) | 68.6% (0.97×) | 66.5% (0.93×) |
| 50% (512 slots) | 94.1% | 93.1% (0.99×) | 92.6% (0.99×) | 92.5% (0.98×) |

**Findings.**
1. **LRU collapses under a tight budget on real routing** (0.0% at 10%) — one
   pass of 8-of-64 experts evicts everything before reuse. Frequency-based
   policies get **0.97×** there. Against LRU that is not an improvement of a few
   points but a different order of magnitude.
2. **`hot` does *not* win on this trace — LFU does.** The captured trace is a
   single prompt, i.e. *stationary*, and round 3 measured that aging trades
   stationary accuracy for shift recovery. The round-2 `hot` win (0.92× vs 0.71×)
   is real but **regime-specific**: it needs the distribution to move. Policy
   choice is therefore regime-dependent, not a fixed ranking.
3. The real per-layer concentration varies (top-8 mass **0.41–0.61** across
   layers) while the synthetic Zipf trace is uniform (≈0.53) — so the real trace
   *does* have the layer heterogeneity S3 was meant to exploit.

### Result 2 — S3 (per-layer allocation) is a measured negative
The round-2 log predicted the remaining gap was *allocation*. It is not.
Marginal water-filling across layers scored **equal-or-worse** than the single
global pool at every budget (real trace: 0.90× vs 0.93× at 10%, 0.95× vs 0.93% at
25%, 0.98× vs 0.98× at 50%). The reason is not a bug: **a global LFU cache
ordered by frequency already implements exactly the greedy marginal allocation**
(keep the highest-probability keys, wherever they live). Encoding per-layer
quotas re-derives that optimum approximately and pays for it in churn.

The code ships (`ExpertCache(alloc="per-layer")`, `--cache-alloc per-layer`,
unit-tested) so the claim is reproducible, but it is **off by default** and
documented as not-helping.

### Result 3 — can ~100B run on this laptop? Yes, slowly, and cache is not its lever
New profile `arch/glm-4.5-air.toml` (**GLM-4.5-Air**, a real 106B-total / 12B-active
MoE; derived totals +4.3%, active +6.9% vs published). Modelled against
`hw/gtx1650.toml`, `vipera sweep`:

| strategy | hit | disk GB/tok | tok/s | bound |
|---|---|---|---|---|
| baseline fp8 | 34.0% | 4.20 | 0.45 | io |
| q4 experts | 48.6% | 1.63 | **1.07** | compute |
| + prune to 25% experts | 78.5% | 0.68 | 1.75 | compute |
| + prune to 5% experts | 100% | 0.00 | 1.95 | compute |
| + prune 5% + draft ×3 | 100% | 0.00 | **5.85** | compute |
| + drop 20 layers too | 100% | 0.00 | **10.76** | compute |

**Findings.**
1. A 100B MoE **does** fit the thesis: ~**1.07 tok/s** at Q4 from disk, and the
   compression/speculation stack takes it to **5.85 tok/s** (×5.5) at 100% cache
   hit — an improvement *far beyond* 100%, and the place where a large win
   actually exists for this target.
2. At Q4 the model is **compute-bound, not I/O-bound**, so at that point *better
   caching would not speed it up at all*. Cache work stops paying once the disk
   bytes/token fall below the compute floor.
3. **Caveat that matters:** the compute floor is derived from `cpu_gflops = 50`, an
   *estimated* constant in `hw/gtx1650.toml`, never measured here. The
   "compute-bound" verdict is therefore the least trustworthy number in this
   table until S7's bench harness measures real GEMV throughput.
4. The policy/ceiling ratio for this model is **0.55×** on the modelled Zipf — far
   below the 0.94× seen on trace-driven runs. That gap is the *model*, not the
   policy: real routing is more concentrated than `Zipf(skew=1.0)`. It is a
   concrete reason to prefer measured traces over the default skew.

### What is now known vs still unknown
* **Known (measured):** real routing capture works end-to-end; on real stationary
  routing LFU reaches 0.97× of the size-matched ceiling and LRU collapses; ageing
  (`hot`) helps only under drift; per-layer allocation does not help.
* **Unknown:** everything above uses one model, one prompt and CPU-only. A
  100B-class real trace was **not** captured (the model cannot run here), so the
  0.55× and compute-bound numbers for GLM-4.5-Air remain **modelled**.
* **The 100% ceiling:** on a *hit-rate-ratio* metric the improvement is bounded by
  1.0×, so from 0.97× the maximum possible gain is **+3%**. A "100% better"
  target cannot be met on that metric by any cache algorithm — it is achievable
  on bytes/token or tok/s, which is where the sweep above shows ×5.5.

**Next sub-problem:** S7's second half (measure real GEMV/bandwidth so the
compute floor stops being a guess), then S10 (quality harness) — because the
biggest measured win for the 100B target is pruning + speculation, and neither has
a measured quality cost yet. *Round 4 did the measurement half — see §9.*

---

## 9. Autonomous loop — round 4 log (measure the machine, not the guess)

Round 3 ended with a caveat: the "100B is compute-bound" verdict rested on
`cpu_gflops = 50`, an estimate never measured here. This round measures it — and
the verdict changes.

### Method
* `bench/hwprobe.c` — a small C probe. `read`/`copy` stream a 256 MiB buffer
  (≫ 8 MiB L3) so they measure DRAM; `gemv` multiplies a L3-resident 512x1024
  matrix so it is ALU-bound. Both kernels use 8 independent accumulators +
  OpenMP: with a single accumulator the loop is *latency*-bound and the first
  draft of this probe reported **4 GB/s** where the machine does ~44.
* `bench/bench.py` — the existing harness, wrapping `llama-bench` on a real model.

### Measured (this laptop, 2026-10-12)
| quantity | measured | was in `hw/gtx1650.toml` |
|---|---|---|
| streaming read | **43–46 GB/s** | 40.0 (estimate) |
| streaming copy (r+w) | 36 GB/s | — |
| GEMV, 8 threads | **110 GFLOP/s** | — |
| GEMV, 1 thread | 20.8 GFLOP/s | — |
| OLMoE-1B-7B Q4_K_M decode | **41.2 tok/s** (pp512 136.5) | — |

`hw/gtx1650.toml` now carries `ram_bw_gbs = 44.0` and `cpu_gflops = 110.0`, each
annotated with how it was obtained.

### Validation — the model now matches a real run
OLMoE Q4 needs ~1.18B active parameters; at 41.2 tok/s that is an effective
`2·1.18e9·41.2 ≈ 97 GFLOP/s`, so the ALU probe overstates the real quantized
kernel by ~12%. Feeding the measured constant into the scheduler:

| prediction source | modelled | measured | error |
|---|---|---|---|
| real 511-token trace, `cpu_gflops=110` | **44.6 tok/s** | 41.2 | **+8.3%** |
| modelled Zipf, 128 steps | 36.9 tok/s | 41.2 | −10.4% |
| same runs with the old `cpu_gflops=50` | 21 tok/s | 41.2 | −49% |

Both are inside S8's "within 25%" criterion; the old constant was off by **2×**.

### Correction — the 100B verdict flips
Re-running `vipera sweep arch/glm-4.5-air.toml` with the measured constant:

| strategy | round 3 (gflops=50) | round 4 (gflops=110) | bound |
|---|---|---|---|
| q4 experts | 1.07 | **1.08** | ~~compute~~ → **io** |
| + prune 25% experts | 1.75 | **2.12** | io |
| + prune 5% experts | 1.95 | **4.29** | compute |
| + prune 5% + draft ×3 | 5.85 | **12.87** | compute |
| + drop 20 layers | 10.76 | **23.67** | compute |

**The round-3 conclusion "the 100B target is compute-bound, so better caching
would not speed it up" was an artefact of the guess.** At Q4 GLM-4.5-Air is
**I/O-bound** (1.08 tok/s, 1.63 GB/token off disk) — so streaming and caching work
*is* on the critical path for the 100B goal. It only becomes compute-bound after
the expert pool is pruned enough to hit 100% in cache, at which point the levers
are compression and speculation (×12 vs baseline).

### Still unmeasured
* `vram_bw_gbs` (192) and `disk_bw_gbs` (2.0) remain datasheet/earlier values;
  no GPU-bandwidth or `fio`/`iozone` probe has been run on this box.
* The probe's GEMV is **f32**; real kernels dequantize. The 12% gap above is the
  only calibration between the two, from one model at one quant.
* Everything here is CPU-only, single model, and — for GLM-4.5-Air — still
  **modelled**, since a 100B model cannot run on this laptop to be measured.

---

## 10. Autonomous loop — round 5 log (does ageing help on *real* drift?)

Round 2 claimed the `hot` policy wins when routing moves (0.92× vs LFU's 0.71×),
but that was measured on a **synthetic** shift — expert ids rotated by half. Round
3 then found the opposite on a **stationary** real trace. Round 5 tests the claim
the way it would actually be used: real routing, real topic changes.

### Method
Five prompts on unrelated topics (Python code, an irrationality proof, a
Vietnamese prose paragraph, photosynthesis, a SQL query), 109 tokens each,
captured separately and concatenated into **one 545-token session with four topic
changes**. Each segment's routing is real; only the junction is artificial.

```
python3 -m bench.capture_trace --bin ~/dev/llama.cpp/build/bin/llama-cli \
    --model olmoe-q4km.gguf --tokens 110 --threads 8 --ngl 0 \
    --prompts-file prompts.txt --out olmoe-multitopic.jsonl
```

Trace kept at `bench/results/olmoe-multitopic-545tok.jsonl`; the two claims below
are now regression tests (`TestRealMultiTopicShift`).

### Result — the round-2 claim holds on real routing
Ratio to the size-matched static ceiling:

| window | 102 slots (10%) | | | 256 slots (25%) | | |
|---|---|---|---|---|---|---|
| | ceiling | LFU | `hot` | ceiling | LFU | `hot` |
| whole session | 35.5% | 0.97× | **1.06×** | 60.8% | 0.98× | **1.01×** |
| last 30% (post-shift) | 37.1% | 0.93× | **0.96×** | 62.0% | 0.93× | **0.97×** |
| final segment | 40.8% | 0.91× | **0.92×** | 66.2% | 0.89× | **0.90×** |
| each topic alone (mean) | 41.3% | 0.91× | 0.91× | 67.6% | 0.93× | 0.91× |

**Findings.**
1. **Ageing wins under real drift, in every post-shift window and at both
   budgets.** This is the first confirmation of round 2 that is not an artefact
   of the synthetic shift generator.
2. **On a single real topic it wins nothing** (0.91× vs 0.91×), so the effect is
   genuinely about non-stationarity, not a free gain.
3. **The whole-session `hot` ratio exceeds 1.0** (1.06×) — an adaptive cache beats
   a static one under drift. This is the third independent sighting of the same
   point (§8, §9), and it means "static ceiling" must always be read as
   *stationary* upper bound.
4. **Real shifts are far milder than the synthetic one, so round 2 overstated the
   size of the win.** Synthetic: 0.71× → 0.92× (**+21 points**). Real: 0.93× →
   0.97× (**+4 points**). The direction survives; the magnitude does not. Any
   future headline should quote the real number.
5. Real per-topic ceilings at a 10% budget (36–48%) sit **below** the synthetic
   Zipf trace's (~61%) for comparable settings: this model's routing is *less*
   concentrated than `Zipf(skew=1.0)`. The synthetic generator is therefore not
   reliably conservative in either direction — which is the standing argument for
   captured traces over modelled ones.

### Still unknown
* Five prompts is not a workload. Nothing here says how often real traffic shifts,
  which is what decides whether ageing pays in aggregate.
* Only OLMoE (16 layers, 64 experts) was captured; the effect on a 100B model is
  modelled, not measured.

**Next sub-problem:** S10 (quality harness) — the largest measured win for the
100B target is expert pruning + speculative decoding (×12 over baseline, §9), and
neither has a measured quality cost yet. *Round 6 attacked the pruning half and
found the ×12 rests on an assumption real routing refutes — see §11.*

---

## 11. Autonomous loop — round 6 log (what does pruning actually cost?)

§9 offered expert pruning as the lever that takes the 100B target from 1.08 to
12.87 tok/s. Every one of those rows came from `Optimizations.prune_experts`,
which shrinks the pool and then lets the routing model run over the *reduced*
pool — i.e. it assumes the surviving experts carry the same fraction of the
routing as before, **as if pruning were free**. A captured trace can test that
directly.

### Method
`vipera prune <trace> --arch <arch.toml>` ranks each layer's experts by selection
frequency and reports (a) the smallest keep-set that retains a target mass, and
(b) the mass retained at a given sparsity, next to what a **uniform** keep-set of
the same size would retain.

```
python3 -m vipera prune bench/results/olmoe-1b-7b-511tok.jsonl --arch arch/olmoe-1b-7b.toml
python3 -m vipera prune bench/results/olmoe-multitopic-545tok.jsonl --arch arch/olmoe-1b-7b.toml
```

### Result 1 — mass-based selection is worth a great deal
| sparsity | mass kept, 1 topic | if uniform | mass kept, 5 topics | if uniform |
|---|---|---|---|---|
| 5% | 24.7% | 5.0% | 21.3% | 5.0% |
| 10% | 40.7% | 10.0% | 34.0% | 10.0% |
| 25% | 70.9% | 25.0% | 60.4% | 25.0% |
| 50% | 93.5% | 50.0% | 83.9% | 50.0% |
| 75% | 99.7% | 75.0% | 96.2% | 75.0% |

*Which* experts you keep is worth **+16 to +46 points of retained mass** at the
same sparsity. "Prune to 5%" is meaningless until the keep-set is specified.

### Result 2 — and pruning to fit the cache is therefore *not* free
| trace | keep-set for 95% of routing mass |
|---|---|
| one topic (511 tok) | **52.8%** of the pool (541 of 1024 experts) |
| five topics (545 tok) | **71.1%** of the pool (728 of 1024 experts) |

At the 5% sparsity §9 assumed, only **21–25%** of routing mass survives.

**Findings.**
1. **The §9 ×12 figure is not attainable for free.** Its `prune 5% → 100% cache
   hit → 1.95 tok/s` row now stands as: 5% of experts, ~75–79% of the router's
   mass discarded, quality unmeasured. The row should be read as an upper bound
   on speed *for a given quality budget*, not as an available configuration.
2. **The speed/quality trade is now quantitative**, which it was not before: for
   the 100B model, holding 95% of routing mass means keeping ~53–71% of a 47 GiB
   expert pool = **25–33 GiB** — still 2–3× the 11.4 GiB cache, so *even the
   quality-preserving prune does not make it fit*. The pool simply does not shrink
   fast enough per point of mass kept.
3. **Topic diversity makes pruning harder**: the same 95% mass needs 53% of the
   pool on one topic but 71% across five. A pruning policy tuned on one workload
   will over-prune the next.
4. This is why the lever for the 100B target is **streaming, not pruning**: §9
   already showed it is I/O-bound at Q4 (1.08 tok/s, 1.63 GB/token), and pruning
   cannot close the pool/cache gap without spending the quality budget.

### What is still not measured
The honest limit of this round: **retained routing mass is a necessary condition,
not a quality result.** It bounds how much of the router's behaviour survives, but
the real number (perplexity, task accuracy) needs the pruned weights to be built
as a GGUF and run (S9 + S10). `vipera prune` says what a keep-set preserves; it
does not say the model still works. Treat 95%-mass as a design constraint, not as
"95% of quality".

**Next sub-problem:** S9 (GGUF writer/packer) — it is now the blocking dependency
for S10, and therefore for turning every pruning claim above into a quality number.
*Round 7 measured the last two guessed constants and flipped the verdict a second
time — see §12.*

---

## 12. Autonomous loop — round 7 log (the last two constants)

Rounds 4 and 5 replaced the guessed `cpu_gflops` and `ram_bw_gbs`. Two guesses
remained: `disk_bw_gbs = 2.0` and `vram_bw_gbs = 192`.

### Disk — measured, and the guess was 4× wrong
`dd ... iflag=direct` and `os.pread` with `O_DIRECT` on the 4.2 GiB model file:

| pattern | measured |
|---|---|
| sequential, 1 GiB | 8.7 GB/s |
| sequential, 3 GiB | 9.1 GB/s |
| **random 1 MiB blocks** (≈ one Q4 expert) | **8.76 GB/s** |
| sequential 1 MiB blocks (like-for-like) | 7.75 GB/s |
| write, 1 GiB direct | 5.8 GB/s |

Random and sequential are indistinguishable at 1 MiB, and `O_DIRECT` showed the
same rate as buffered — so this is device-level, not page cache. The profile
claimed a DRAM-less SN530 at **2.0 GB/s**; the box does **8–9**. `disk_bw_gbs` is
now **8.0**.

### VRAM — cannot be measured here, and it does not matter for any current result
No CUDA toolkit (`nvcc`), no OpenCL ICD (`/etc/OpenCL/vendors` absent), no Torch,
and the local llama.cpp build is **CPU-only** (no `libggml-cuda`). The GPU cannot
be exercised at all, so `vram_bw_gbs` stays a datasheet 192 — now labelled as
not-measurable rather than merely unmeasured. Every number in this document is
CPU-only, so no conclusion depends on it.

### The verdict flips a second time
`vipera sweep arch/glm-4.5-air.toml --target hw/gtx1650.toml`:

| | round 3 (2 guesses) | round 4 (+cpu_gflops) | round 7 (+disk) |
|---|---|---|---|
| q4 experts, tok/s | 1.07 | 1.08 | **3.21** |
| bound | compute | io | **compute** |
| + prune 5% | 1.95 | 4.29 | **4.29** |
| + prune 5% + draft ×3 | 5.85 | 12.87 | **12.87** |

**Lesson, and it is the point of these two rounds:** the verdicts were never
stable because the *constants* were not measured. Three separate "the 100B target
is compute-bound / I/O-bound" statements appeared across rounds 3, 4 and 7; two of
them were artefacts. A modelled verdict is only as trustworthy as its inputs.

At the measured constants GLM-4.5-Air Q4 sits at **3.21 tok/s** with the compute
floor at 233 ms and I/O at 311 ms per token — **close to balanced**, so neither
"just add cache" nor "just add FLOPs" is right for it; both terms matter, which is
exactly what a value that flipped twice should tell you.

### Still not measured
* Nothing here changes §11: pruning still cannot close the pool/cache gap without
  spending the quality budget, and retained mass is still not a quality result.
* 8 GB/s is one file on one filesystem; a fragmented model or a busier machine
  would differ. The number is a best case, not a floor.
