# plantodo.md — Plan: run GLM 5.3 on the personal machine

> Written **2026-10-13** · companion file to `viperasuport.md` (the PII audit).
> Four parts: **(0) actual repo state when written**, **(1) the task brief**,
> **(2) why the brief is shaped this way**, **(3) newer research directions.**
>
> **Read before doing anything**
> `algorithms-digest.md` *(formerly `thuattoan.md`, renamed and translated)* ·
> `docs/OPTIMIZER.md` · `docs/ROADMAP.md` · `docs/PROPOSALS.md` · `docs/FEASIBILITY.md` ·
> `bench/results/README.md` · `vipera/{analyze,schedule,trace,prune,prefetch,run}.py` ·
> `bench/llama-moe-prefetch.patch` · `report.md`

---

## 0. Actual repo state when this was written (read first — do not redo finished work)

| What the brief assumes | Reality on 2026-10-13 |
|---|---|
| "Write `arch/deepseek-v2-lite.toml`" | ✅ **Already exists** (commit `4db35b7`). `python3 -m vipera arch arch/deepseek-v2-lite.toml` runs: 27 blocks, **1 leading dense**, 26 MoE, 64 experts top-6, hidden 2048 / moe_inter 1408, 15.497 B total · 2.451 B active · −1.3% vs published. |
| "Need a trace ≥ 2000 tokens" | ⚠️ The repo only has OLMoE **511** and **545** tokens. The DeepSeek-V2-Lite **199-token** trace sits at `/tmp/ds_trace.jsonl` (102 KB) — **`/tmp` does not survive**; copy it into `bench/results/` and whitelist it in `.gitignore` first. No trace ≥ 2000 tokens yet. |
| Command `python3 -m vipera prefetch …` | ❌ **Does not exist.** Subcommands are only `analyze plan arch schedule trace score prune prune-gated sweep run`. Build the table with `python3 -m bench.prefetch_table <trace> -o <out> --target-mass 0.95`. |
| Capturing a real trace | `python3 -m bench.capture_trace --bin ~/dev/llama.cpp/build/bin/llama-cli --model <gguf> --tokens 2048 --threads 8 --ngl 0 --out <path>` (needs the `LLAMA_MOE_TRACE` patch in the local fork). |
| Measurements already in hand | CPU-only **12.80** · CUDA **6.90** · Vulkan **3.40** gen t/s (ctx 512, 32 tokens, 8 threads, 9.65 GiB model, experts on CPU) · `--compare` on CPU: default 12.20 → manifest 14.90 = **1.22×**. |
| Theoretical ceilings | ALU **~23 tok/s** · RAM **~37 tok/s** · disk miss **~6.7 tok/s** (measured 2026-10-13). |
| The 12.80 t/s figure | 12.80 > 6.7 (disk ceiling) ⇒ **if that holds, disk is NOT the bottleneck** — exactly what STEP 1 must prove or disprove with numbers. |
| Repository | 100% English, CI green, 225 tests, remote `vipera-iso/vipera` in sync. |

### STEP 1 result: disk is NOT the bottleneck — `[measured]` 2026-10-13

Model `DeepSeek-V2-Lite.Q4_K_M.gguf` (9880 MiB), decode 32 tokens × 5 reps, `-ctk/-ctv q8_0`,
hardware as above. Reproduce any row with
`python3 bench/step1_attrib.py run --threads 4 --load-mode <mmap|dio|mlock>`; full sweep via
`python3 bench/step1_attrib.py sweep`. Raw JSON: `bench/results/step1-attrib.json`.

| load mode | tok/s | ms/token | share of mmap total | decode NVMe | note |
|---|---|---|---|---|---|
| `mmap` (llama.cpp default) | **16.95** ±4.10 | 59.0 | 100% | 3877 MiB extra | page cache thrashes; ±24% CV |
| `dio` (bypasses page cache) | **22.44** ±0.19 | 44.6 | 76% | 67 MiB | file read once at load, 9880 MiB exactly |
| `mlock` (pinned in RAM) | **22.90** ±0.07 | 43.7 | 74% | ~0 | no paging at all |
| ALU ceiling (4.8 GFLOP/token ÷ 110 GFLOP/s) | ~23 | 43.5 | — | — | `[modelled]` from measured `bench/hwprobe.c` |

**Attribution of the 59.0 ms/token the default configuration spends:**

| component | ms/token | share |
|---|---|---|
| compute (ALU) — **at the ceiling**: 43.7 vs the modelled 43.5 ms floor | 43.7 | **74%** |
| paging the default `mmap` causes on its own | 15.3 | **26%** |
| real NVMe bandwidth during decode | ≤0.1 | **≤0.2%** |

**Verdict: the bottleneck is compute, and the ALU is already saturated.** `mlock` reaches
43.7 ms/token against a 43.5 ms ALU floor — i.e. ~0.5% headroom. Disk during decode moved
0.4 MiB/token (0.05 ms at the measured 8 GB/s), so **T1 (GGUF expert reorder) and prefetch
cannot buy more than ~0.2% and should not be built for a RAM-resident model.** The remaining
26% is a self-inflicted paging tax, not a hardware limit.

**Actionable, measured today: `-lm mlock` (or `dio`) over the default `mmap`.** The honest
range, because `mmap` warms up across repetitions and quoting one end of it would be exactly
the trap this document's own convention forbids (`mmap` t4 reps: `[10.30, 17.08, 17.71, 21.26,
19.57]`; `mlock` t4 reps: `[22.78, 22.83, 22.80, 22.84, 22.29]`):

| condition | `mmap` t4 | `mlock` t4 | gain |
|---|---|---|---|
| first repetition (cold) | 10.30 tok/s | 22.78 tok/s | **+121%** |
| steady state (last two reps) | 20.41 tok/s | 22.71 tok/s | **+11%** |
| mean of 5 reps | 17.18 tok/s | 22.71 tok/s | +32% |
| spread | ±4.18 (±24% CV) | ±0.23 (±1% CV) | 18× more stable |

**So the main effect is eliminating cold-start thrash and variance, not steady-state
throughput — steady state is +11%, not +32%.** For an interactive or agent workload that pays
cold-start cost per session, the +121% first-batch number is the relevant one; for a
long-running server, +11% is.

The default `mmap` loses because the 9880 MiB model plus KV plus process overhead exceeds the
~11 GiB available, so the page cache evicts model pages that are then re-faulted: `mmap` moved
13757 MiB from NVMe where `dio` moved exactly the 9880 MiB file and nothing more. Both flags
are accepted by `llama-cli` (`-lm/--load-mode`), so this is a manifest change, not a llama.cpp
patch. **The `--compare` baseline must be re-measured with it, and every previously published
tok/s number (12.80, 11.70, 14.90) is a mmap-under-thrash figure that this supersedes.**

### Thread count: `-t 4` is optimal, not `-t 8` — `[measured]` 2026-10-13

Same conditions, `-lm mlock` (page-cache noise removed), 32 tokens × 3 reps, measured
standalone per count:

| threads | tok/s | scaling vs previous |
|---|---|---|
| 1 | 8.41 ±0.02 | — |
| 2 | 15.11 ±0.07 | 1.80× |
| **4** | **22.90** ±0.07 | 1.52× |
| 8 | 20.75 ±0.37 | **0.91× — a 9% regression** |

The curve flattens then goes **negative**: gains fall 1.80× → 1.52× and then reverse, which is
the shape of hitting a hard ceiling rather than running out of cores. At `-t 4` the run is
already at 22.90 tok/s against the ~23 ALU ceiling, so the 4 extra SMT threads have no
arithmetic left to fill and only contend for the same execution units and cache.
**`-t 8` costs 9%; every previously published figure (12.80, 11.70, 14.90) was measured at
`-t 8` and understates the machine by that margin.** Correct settings for this machine:
`-t 4 -lm mlock`, which together move the measured decode rate from the documented
12.80 tok/s to 22.90 tok/s — a 1.79× improvement that required no change to llama.cpp and no
file-layer transform.

Two methodology corrections recorded so this is not re-derived wrong:

- The `t1/t2/t4` scaling run inside a single `llama-bench -t "1,2,4"` invocation reported
  t4 = 21.75 tok/s; standalone t4 is **16.95**. Sequential tests in one process share an
  increasingly warm page cache, so in-process thread scaling overstates the later threads.
  Thread counts must be measured standalone.
- `-ctk/-ctv q8_0` vs `f16` is worth **nothing at ctx 512** (16.41 vs 16.51 tok/s, i.e. within
  noise). The documented 12.20 → 14.90 CPU win cannot be KV quantization alone at this context
  size; whatever produced it needs re-isolating before it is claimed again.

### The survival question: does the manifest still win on a clean baseline?

**No — it measures ≈1.00×.** `[measured]` 2026-10-13. This was the measurement that decided the
project's direction. The old figure (default 12.20 → manifest 14.90 = **1.22×**,
`algorithms-digest.md`) was taken while the page cache was thrashing, so it answered *"how much
does the manifest help when **both** sides are thrashing"* — not the question that matters.

Re-measured on DeepSeek-V2-Lite Q4_K_M, `-t 4`, ctx 512, 32 tokens, both sides:

| baseline | manifest | speedup | footing |
|---|---|---|---|
| 22.80 tok/s | 22.50 | **0.99×** | `-lm mlock`, page cache clean |
| 22.60 tok/s | 22.50 | **1.00×** | `-lm mlock`, second repetition |
| 13.50 tok/s | 12.90 | **0.96×** | `mmap`, default, thrashing |

```bash
python3 -m vipera run models/DeepSeek-V2-Lite.Q4_K_M.gguf \
    --manifest out/DeepSeek-V2-Lite.Q4_K_M.vipera.json \
    --compare --context 512 --n-predict 32 --threads 4 -- -lm mlock
```

This lands in the second of the three predicted scenarios (**manifest ≈1.00×: the old advantage
was thrash mitigation**), and it is *stronger* than that scenario stated: the manifest is worth
**nothing to −4%** whether the baseline is clean or thrashing, so the old 1.22× is not merely
indirect — **it does not reproduce at all** on either footing.

**The decisive evidence, no inference required.** On the `mmap` run the manifest cut disk read by
**3918 MiB (−26%) and still scored 0.96×.** Fewer disk reads bought no throughput. That is the
direct form of STEP 1's "decode touches 0.4 MiB/token", and it closes the argument the project's
own positioning had left open: for a RAM-resident model, file-layer placement has nothing to act
on — and *demonstrably* has no effect when it acts anyway.

**Why it is structural, not a bug.** `vipera run --dry-run` shows that on a CPU-only build the
manifest maps to exactly `-ngl 0 -ctk q8_0 -ctv q8_0` and nothing else. Every other part of it —
resident VRAM, expert cache VRAM, expert RAM, spill to disk — degrades to warnings, because
`detect_gpu_backend()` is false and the model fits in RAM. There is nothing left to place. And
the one flag that survives, `-ctk/-ctv q8_0`, is worth ~0 at ctx 512 (16.41 vs 16.51 tok/s,
measured above), so its contribution cannot sum to 22%.

**Corrected 2026-10-13 (my own error, worth recording):** an earlier pair of `--compare` runs
at `--context 2048` came back **SIGKILL'd**, and I wrote that up as a property of ctx 2048 — that
the f16 baseline needs ~22.5 GB and dies. **That attribution was wrong.** Re-run sequentially,
the same ctx 2048 completes three times out of three, exit 0: 1.04×, 0.81×, 1.12×. The cause was
**running two benchmarks concurrently** on a 15 GiB box with no swap: each side of `--compare`
loads the 9880 MiB model, so two comparisons in parallel ask for ~20 GiB and the OOM killer
takes one. Two things follow, and both are about method rather than this model:

- **Never run these benchmarks in parallel on this machine.** That is a standing constraint, and
  it invalidates any measurement taken while something else was benchmarking — including,
  possibly, some of the earlier ones. The STEP 1 numbers were taken sequentially, but the
  concurrency mistake here is exactly the kind that is invisible after the fact.
- **An OOM kill surfaces as `n/a` with exit 0**, so a killed baseline looks identical to a
  missing measurement unless the `error:` line is read. `--compare` should print a banner and
  refuse to emit a speedup when either side did not exit 0; it currently prints a table whose
  empty cells carry the whole verdict. (Still to do — see below.)
- The 0.81–1.12× spread at ctx 2048 is `mmap` thrash variance, **not** a ctx effect: the same
  three settings at ctx 512 under `-lm mlock` held 0.96–1.00×.

**What this decides:**

1. The manifest's case must be made where it has something to place — a model that **spills**
   past RAM (Qwen3-30B-A3B, 18.6 GB against ~11 GiB usable), or a long context where KV placement
   matters. Not on a RAM-resident MoE.
2. `-t 4 -lm mlock` is worth **1.79×** (12.80 documented → 22.90 measured) while the manifest is
   worth **0.99×**. The 1.79× needs no llama.cpp patch and no file-layer transform, and it is not
   yet represented in any vipera output. **Vipera should recommend it and currently does not.**
3. `README.md` and `algorithms-digest.md` are corrected: the old 12.20/14.90 pair is marked
   `[superseded]`, and README's `--compare` section now carries the real 0.99× output instead of
   the `~20 / ~20 / 1.00x` placeholder that had been mistaken for a measurement.
4. **Prefetch dies on this footing too.** Its entire purpose is to raise the expert hit rate and
   thereby avoid disk reads; disk is ≤0.2% of decode, and cutting reads by 26% moved throughput
   by −4%.

**Deliverables 2–5 of the brief, answered:**

| # | Deliverable | Answer |
|---|---|---|
| 2 | Bottleneck report | **compute** (43.7 ms/token, 74%), not disk (≤0.2%). Paging is 26% and self-inflicted — removed by `-lm`. |
| 3 | GPU verdict | **No, under every condition measured**: CUDA 6.90 / Vulkan 3.40 vs CPU-only 22.90 tok/s. Closed. |
| 4 | Trace ≥2000 tokens | **Not captured, and no longer needed.** STEP 1 killed the disk hypothesis that the trace feeds. |
| 5 | GLM 5.3 verdict | **Does not run — and the reason is now specific: it is short of ALU, not RAM.** 18B active vs 2.4B is 7.5× the arithmetic against a ceiling already at 99.5%, and ALU cannot be reordered at the file layer. |

**STEP 1 measurement environment (verified 2026-10-13, all `[measured]`):**

- `--moe-cache-mib` is **rejected on CPU-only**: the fork fails context init with
  `llama_init_from_model: MoE cache requires a GPU backend`
  (`python3 ~/dev/llama.cpp/build/bin/llama-cli -m models/DeepSeek-V2-Lite.Q4_K_M.gguf -p hi -n 4 -t 8 -ngl 0 --moe-cache-mib 512`).
  Consequence: the "cache 512 MiB" column in the README table applies to the CUDA path only;
  on CPU-only the experts go through the OS page cache, and any CPU `--compare` win is
  KV-cache quantization (`-ctk/-ctv q8_0`), not expert caching.
- `perf stat` is unavailable (`perf` not installed, `perf_event_paranoid=2`, no sudo), so
  attribution uses `llama-bench -o json` timing plus `/proc/<pid>/{stat,io,status}` and
  `/proc/diskstats` counters, sampled by `bench/step1_attrib.py`.
- `llama-bench -lm dio` (direct I/O, bypasses the page cache) is the clean NVMe lever:
  `mmap` warm = experts served from RAM, `dio` = every expert page-in is a real NVMe read.
- `llama-cli` in one-shot mode prints no `eval time` line in this build — timing must come
  from `llama-bench`, not from parsing `llama-cli` output.

**Confidence convention (mandatory, same as `algorithms-digest.md`):**
`[measured]` = reproducible command, quote it · `[modelled]` = computed by vipera, not run ·
`[proposed]` = not implemented · `[published]` = someone else's number, not re-verified.
**A number without a label is a defect.** Never quote the most favourable end of a range.

---

## 1. The task brief

### CONTEXT

The `vipera` repository — a planner/optimizer for local LLMs. Work in the order below,
no skipping ahead.

### TARGET MACHINE (must not be assumed otherwise)

- CPU i5-11300H, 4C/8T
- RAM 15 GiB, ~10 GiB usable, **no swap**
- GPU GTX 1650 Max-Q 4 GB VRAM (Turing)
- NVMe WD SN530 DRAM-less
- Arch Linux, no sudo, Python 3.14

### TARGET MODELS

- Phase 1: **DeepSeek-V2-Lite** (15.7B / 2.4B active, Q4_K_M ~9.5–10.4 GB)
- Phase 2 (stretch): **GLM 5.3** (`arch/glm-5.3-flash.toml` already present)
- The central question: **can this machine run GLM 5.3, and if so, how?**

### STEP 0 — Remove the blocker

- `arch/deepseek-v2-lite.toml` **already exists** → only **confirm** it, do not rewrite it:
  ```
  python3 -m vipera arch arch/deepseek-v2-lite.toml
  python3 -m vipera trace bench/results/<trace>.jsonl --arch arch/deepseek-v2-lite.toml
  ```
- The real work of this step: **preserve the DeepSeek trace** (`/tmp/ds_trace.jsonl` →
  `bench/results/` + a `.gitignore` whitelist entry) so the trace loop does not close on a
  temp file.

### STEP 1 — Attribute the real bottleneck

Question: of the measured **12.80 t/s** (CPU-only, ctx 512, 32 tokens, experts on CPU),
how much time is (a) disk, (b) RAM, (c) compute?

- Measure separately: expert load time, expert compute time, attention time.
- Compare against the ALU ceiling ~23 t/s and the RAM ceiling ~37 t/s.
- If compute is the bottleneck → raising the hit rate **does nothing**.
- **The conclusion must carry numbers. "Possibly" is not an answer.**
- Existing numbers to check against: 12.80 t/s > 6.7 tok/s (disk-miss ceiling) ⇒ the
  assumption "disk is the bottleneck" **contradicts the measurement** — prove it or kill it.

### STEP 2 — Answer the GPU question

Does the discrete GPU actually help on this machine?

- Numbers in hand: CPU **12.80** / CUDA **6.90** / Vulkan **3.40** t/s `[measured]`.
- Hypothesis: PCIe cost per expert > the attention benefit.
- Test it: measure attention on GPU vs CPU separately, measure PCIe transfer.
- If the GPU is useless for large MoE → **state it plainly and stop all co-execution work**.
- If it helps under some condition → **state the condition**.

### STEP 3 — A real trace for the target model

- Traces in hand: OLMoE 511 and 545 tokens, DeepSeek 199 tokens. **Too short.**
- Need: a trace of **≥ 2000 tokens** on DeepSeek-V2-Lite (or GLM 5.3 if it runs):
  ```
  python3 -m bench.capture_trace --bin ~/dev/llama.cpp/build/bin/llama-cli \
      --model models/<deepseek>.gguf --tokens 2048 --threads 8 --ngl 0 \
      --out bench/results/deepseek-v2-lite-2048tok.jsonl
  ```
- If it cannot be captured → **say so explicitly and propose how to capture it.**
- Build the prefetch table from the new trace (correct command; there is no `vipera prefetch`):
  ```
  python3 -m bench.prefetch_table bench/results/<trace>.jsonl \
      -o bench/results/<trace>-prefetch-table.txt --target-mass 0.95
  ```

### STEP 4 — T1 reorder (only if disk is the bottleneck)

- Only if STEP 1 concludes **disk is the bottleneck**. Otherwise → **T1 is pointless, skip it.**
- T1 = rewrite the GGUF so experts are laid out hot/cold from the trace.
- **Do not modify llama.cpp's core.** Only reorder bytes inside the file.
- Measure before/after: disk read per token, tok/s.
- If it does not win on the target machine → **off by default, record the failure.**

### STEP 5 — Answer the GLM 5.3 question

With numbers from steps 1–4 in hand:

- **Can GLM 5.3 run on this machine?**
- If yes: how much RAM/VRAM/disk, expected tok/s `[modelled]`, and a real trace is needed
  to confirm.
- If no: **what is missing**, and what could compensate (prune, quant, offload) — with numbers.
- **The answer must carry numbers. "It might run" is not an answer.**

### CONSTRAINTS

- Do not run the model before confirming `arch/deepseek-v2-lite.toml`.
- **Do not commit without a measurement.** Every number needs a reproducible command.
- Do not modify llama.cpp's core. Env-driven patches only, as today.
- Do not pretend the loop is closed without a real trace.
- Record every failure — **hide nothing.**
- If step N depends on step N−1 being unfinished → **stop and say so.**

### DELIVERABLE

1. `arch/deepseek-v2-lite.toml` committed *(already done — confirm only)*.
2. Bottleneck report: disk / RAM / compute, with measured numbers and reproducible commands.
3. GPU verdict: does it help, and under what condition.
4. New trace ≥ 2000 tokens *(or the reason it could not be captured)*.
5. GLM 5.3 verdict: runs or not, with numbers.
6. Update the "Status & limits" section (§16) of `algorithms-digest.md` with what was done
   and what was not.

### START

Read `algorithms-digest.md` and `docs/OPTIMIZER.md` first. Then answer:

- Do you understand the task?
- Which step can you do right now, and which needs more data?
- Any questions before starting?

**Do not write code before answering those three questions.**

---

## 2. Why the brief is shaped this way

- **Step 0 removes the blocker first.** `arch/deepseek-v2-lite.toml` is small but blocks
  everything. Without it, every later step is modelled on Zipf — meaningless.
- **Step 1 forces bottleneck attribution.** The documentation used to assume disk was the
  bottleneck. But 90 MiB/token × 8 GB/s ≈ 11 ms/token ≈ 90 tok/s from disk, while only
  12.80 t/s was measured. So the bottleneck **is not disk**. The brief forces that truth out
  before anything gets optimised in the wrong place.
- **Step 2 answers the GPU question.** CPU-only beating CUDA and Vulkan is counter-intuitive.
  If it holds, every co-execution effort is wasted — confirm before writing code.
- **Steps 3–4, trace and T1.** Only if step 1 concludes disk is the bottleneck.
- **Step 5 answers the GLM 5.3 question.** The end goal, dependent on the four steps before
  it. No jumping straight there.
- **The "no commit without a measurement" rule matters most:** it stops an agent from writing
  code and then believing it works.
- **Deliverable 6 forces the status update.** Whatever was achieved gets recorded — and so
  does whatever failed.

### What the brief deliberately does NOT do

- Allow jumping straight to GLM 5.3 (no foundation yet).
- Allow assuming disk is the bottleneck (the measurement disagrees).
- Allow skipping the GPU question (it decides the whole direction).
- Allow writing code before understanding.

---

## 3. Newer research directions `[proposed]`

The centre of gravity is shifting from **tuning existing parameters** (cache policy, prefetch)
to **changing data structures, scheduling and exploiting specialised hardware.**

### 3.1 Expert scheduling and model compression

| Direction | Idea | Relation to vipera |
|---|---|---|
| **Serving 35B MoEs from SSD with trained routing prediction** | predict the next layer's experts with a **learned model** instead of a static trace table | direct upgrade to Algorithm F |
| **SSD-LLaMA** (SSD-native inference, trillion-param MoE) | an inference architecture **designed for SSD** rather than passive offload; target >1 tok/s | same "file layer" philosophy as T1–T6 |
| **ReMoE** (router fine-tuning for expert reuse) | intervene in the **router** to reuse already-loaded experts | outside vipera's scope (no training), but a useful contrast |

### 3.2 CPU–GPU and hardware

| Direction | Idea | Relation to vipera |
|---|---|---|
| **AsymFlow** — prefill/decode disaggregation | heavy prefill ↔ memory-heavy decode, split across devices | relevant to STEP 2 |
| **Layer-wise CPU–GPU scheduling** | compute layer n on CPU **while** transferring layer n+1 to GPU → hides transfer latency | **could overturn the STEP 2 verdict** (GPU loses to CPU today) |
| **NPU (SnowLLM, FusedDQP)** | fuse dequant + projection on an APU | no NPU on the target machine → `[published]`, skip |

### 3.3 New inference algorithms

| Direction | Idea | Relation to vipera |
|---|---|---|
| **SpecQuant** — speculative + multi-parent quantization | draft tokens + adaptive quantization | relates to the `--draft-tokens` knob |
| **NeuroPrefetcher** — delta prefetching | predict the **delta** in MLP activity rather than the expert; `[published]` 7.9–12× vs llama.cpp on memory-limited devices | replaces or complements Algorithm F |
| **SolidAttention** — KV cache on SSD | up to 98% less KV, 3.1× at 128k context | **potentially a bigger lever than the expert cache** at long context |

### 3.4 Alternative frameworks/runtimes

| Direction | Idea |
|---|---|
| **Sovereign Engine** | from-scratch C++20 runtime with a Vulkan backend, aiming to beat llama.cpp on throughput and VRAM use |
| **VectorWeaver** | automatic per-stage kernel optimisation (prefill/decode), **already integrated into llama.cpp** |
| **SpBatch** | recovers sparsity lost to batching; `[published]` 8.89× vs PowerInfer, 2.6× vs llama.cpp |

### 3.5 The four things worth doing first

1. **Add a learned routing predictor** — replace or supplement the static trace prefetch
   (upgrade to Algorithm F).
2. **Investigate delta prefetching** (NeuroPrefetcher) — predict *changes* in MLP activity.
3. **Re-evaluate the GPU** — with CPU-only beating CUDA/Vulkan, measure properly; layer-wise
   scheduling could flip the STEP 2 verdict.
4. **Re-examine the KV cache** — SolidAttention suggests SSD KV offload may be a bigger lever
   than expert caching at long context.

> **Verification note:** every paper/project name in §3 stays `[published]` until the source is
> traced and the numbers are read. Nothing goes into the README or into a vipera claim before
> that.
