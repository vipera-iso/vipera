---
name: local-llm-optimization
description: A decision-tree + playbook + toolset for running large LLMs on personal hardware (PC, laptop, Apple Silicon) without cloud. Covers choosing a quantization format, fitting weights and KV cache into VRAM/RAM, offloading, speculative decoding, and measuring the result honestly. Use when the user asks how to run a local model (7B..405B) on a specific machine, why a model is slow, or how to pick quantization/runtime flags.
---

# SKILL — Optimizing local LLM inference

> Built from public sources (2025–2026). Every non-obvious claim carries a source
> anchor `[S#]` resolved in §Sources. Numbers are labelled **published** (reported
> by the source, not reproduced here) or **vipera** (reproducible with a command
> in this repo). Where no source exists, the text says **"no data"** rather than
> inventing one.

## TL;DR

0. **Fix the invocation first: `-t <physical cores>` and `-lm mlock`.** Together **1.79×** on a
   4C/8T laptop (12.80 → 22.90 tok/s, 18× tighter variance), no rebuild, no code (vipera,
   measured 2026-10-13). Larger than anything below, and independent of the model.
1. **Decode is usually memory-bandwidth-bound** — `tok/s ≈ tier_bandwidth / bytes_read_per_token`. Compression and multi-token prediction win; FLOP tricks do not. **But check: it is not
   universal.** On a 15.7B MoE with the page cache out of the way, decode sat at **99.5% of the
   ALU ceiling** and disk was ≤0.2% of token time — bandwidth had stopped binding and the
   arithmetic was the wall (vipera, measured 2026-10-13). The test is thread scaling: gains that
   fall away (here 1.80× → 1.52× → 0.91×) mean you have hit compute, not bandwidth.
2. **Quantize first.** `Q4_K_M` is the default sweet spot; it is "best of the three 4-bit formats" in an independent 2026 comparison and preserves most task quality [S9][S10].
3. **Fit the KV cache too, not just the weights.** At long context KV can be most of the footprint — quantize it (fp8/q8) or the model "fits" in weights but OOMs in practice [S12].
4. **On MoE, the expert pool is the battle.** Only ~2× the active-expert count fits a cache; if the pool is 30× the cache, expert pruning is the only thing that changes the verdict [S3][S6] (vipera).
5. **Speculative/MTP is the cheapest multiplier** once the model already runs [S8].
6. **Streaming from SSD can violate a power budget** — SSD offload costs ~5–12× more energy/token than DRAM (published) [S3].
7. **Measure or it did not happen.** One number without hardware, model, context and build is not a result.

---

## Table of contents

1. [Overview](#1-overview)
2. [Decision tree](#2-decision-tree)
3. [Technique catalogue](#3-technique-catalogue)
4. [Step-by-step playbooks](#4-step-by-step-playbooks)
5. [Scripts & tools](#5-scripts--tools)
6. [Comparison tables](#6-comparison-tables)
7. [Pitfalls](#7-pitfalls)
8. [Staying current](#8-staying-current)
- [Self-evaluation](#self-evaluation-what-is-weak-here)
- [Cheatsheet](#cheatsheet)
- [Sources](#sources)

---

## 1. Overview

**Problem.** Run an LLM locally at usable speed and quality on hardware with a
small VRAM budget, no swap, and a slow disk, without a cloud round-trip.

**Who this is for.** Anyone with a laptop/workstation (NVIDIA, AMD, Intel, or
Apple Silicon) who wants to choose a model, quantization and runtime flags
defensibly, and to know *why* a config is slow.

**Minimum requirements**
* Software: a runtime (`llama.cpp` / `Ollama` are the CPU-friendly default [S19]), Python ≥3.11 for the scripts here, and a model in `GGUF` (CPU/mixed) or `EXL2`/`GPTQ`/`AWQ` (GPU).
* Hardware: any machine; the limiting factor is memory *capacity* and *bandwidth* — **unless it
  is not.** On-device memory bandwidth is **50–90 GB/s** vs **2–3 TB/s** on datacenter GPUs, a
  **30–50× gap** (published) [S7], so bandwidth is the usual first wall. But once the page cache
  is out of the way a 15.7B MoE on a 4C/8T laptop reached **99.5% of its ALU ceiling** with disk
  at ≤0.2% of token time — there the limiting factor is FLOPs, and no amount of bandwidth,
  caching or file reordering moves it (vipera, measured 2026-10-13). Determine which you are in
  with thread scaling, not assumption.
* Knowledge: how to run a CLI and read a `nvidia-smi` / `free` output.

**Non-negotiable rules**
* Never compare two configs on different machines, context lengths or builds.
* Always report quality *and* speed together, or the comparison is meaningless.

---

## 2. Decision tree

> **STEP 0 — fix the invocation before touching the model.** Two flags, no rebuild, no code,
> measured on a 4C/8T laptop against a 15.7B MoE: **12.80 → 22.90 tok/s = 1.79×**, and variance
> tightened 18× (vipera, measured 2026-10-13). Every step below assumes you have already done
> this, because otherwise you are tuning a model while leaving ~45% of the machine on the table.
>
> ```
> 1. -t <physical cores>, NOT the logical count.
>    Hyperthreads cost 9% here (22.90 → 20.75 tok/s at -t 8): gains fell 1.80x → 1.52x → 0.91x,
>    i.e. the arithmetic ran out before the cores did. More threads is not more speed.
> 2. -lm mlock (or -lm dio) when the model fits in RAM.
>    The default mmap re-faults model pages it just evicted when weights + KV approach usable
>    RAM: 13757 MiB moved from NVMe where dio moved exactly the 9880 MiB file once.
>    Cold start +121%, steady state +11%, spread ±24% → ±1%.
> 3. Then measure the baseline. `--compare` at ctx >= 1024 can report `n/a` because the f16
>    baseline was OOM-killed, which looks like a missing number, not a failure.
> ```
>
> **Why this belongs first:** it is the largest measured lever on this hardware, it costs
> nothing but a flag, and it is *independent of the model*. Nothing in steps 1–4 below can
> return 1.79×. Skip it and you will misattribute the win to whatever you did afterwards.

```
START: what is the model's weight size at your chosen quant?  W  (GB)
       does W fit in VRAM?  ── yes ─►  FULL GPU OFFLOAD  (-ngl 99 / all layers)
        │ no
        ▼
       does W fit in VRAM + RAM (with ~2 GiB headroom)?  ── yes ─► PARTIAL OFFLOAD
        │ no                                                        (-ngl / --n-cpu-moe)
        ▼
       is it a MoE?  ── yes ─► expert cache + prefetch, or KTransformers-style
        │                        CPU/GPU split [S16]; consider expert pruning [S12v]
        ▼ no
       DENSE model that exceeds RAM ─► layer-wise streaming (AirLLM-style);
                                        accept the I/O ceiling; add speculative
                                        decoding; consider layer pruning last
```

> **Where STEP 0 does not apply.** Once the model *spills* past RAM, `-lm mlock` fails outright
> (there is nothing to lock) and mmap paging becomes the real cost — which is exactly the regime
> where placement, expert cache and prefetch start to earn their keep. STEP 0 is for the
> fits-in-RAM case, and it is worth knowing that in that case placement has nothing to act on:
> measured on a RAM-resident MoE, cutting disk reads by 26% changed throughput by −4%
> (vipera, measured 2026-10-13).

### Quick lookup by configuration

| Machine | Weights tried first | Format | Expect |
|---|---|---|---|
| ≤4 GB VRAM, 16 GB RAM | ≤8B | GGUF Q4_K_M, `-ngl` high | 4–8B comfortable; 13B partial |
| 8 GB VRAM, 16–32 GB RAM | ≤13B | GGUF Q4_K_M | 7–8B smooth, 13B with offload [S20] |
| 8 GB VRAM, 32 GB RAM | 70B | GGUF Q4_K_M, MoE offload | partial; ~0.7 tok/s on our modelled box (vipera) |
| 24 GB VRAM, 64 GB RAM | 70B | EXL2/GGUF | 70B runs; add speculative |
| No GPU, 32 GB RAM | 70B | GGUF Q4_K_M | heavy offload; CPU-bound [S19] |
| 16 GB unified (iGPU) | ≤13B | GGUF Q4_K_M | 70B does not fit; KV blows the budget (vipera) |
| Apple ≥32 GB unified | up to 70B | MLX (or GGUF) | unified memory runs models a 24 GB GPU can't [S17][S18] |

> "Expect" column for the 16 GB and 8 GB/32 GB rows comes from vipera's model on
> specific instances; treat them as *what the arithmetic says*, not benchmarks.

---

## 3. Technique catalogue

Each entry: **principle → when to use → pros/cons → tool → quality impact → source**.

### 3.1 Quantization

| Method | Principle | Use when | Tool | Quality (published) |
|---|---|---|---|---|
| **GGUF K-quants** | block-wise affine quant with superblocks | CPU / mixed CPU-GPU | `llama.cpp` `llama-quantize` | Q4_K_M preserves most task quality; best of the 4-bit formats in one 2026 test [S9][S10] |
| **GPTQ** | layer-wise PTQ with Hessian error compensation | GPU-only serving | AutoGPTQ, vLLM | 4-bit; quality varies with calibration data [S9] |
| **AWQ** | activation-aware scaling to protect salient channels | GPU, when calibration data is representative | AutoAWQ | often ≥ GPTQ at 4-bit when well-tuned [S9] |
| **EXL2** | coupled format+runtime, arbitrary avg bits | max control of bits/VRAM on NVIDIA | ExLlamaV2 | "finest bit-width control"; EXL3 claims better quality/GB than GGUF [S10][S11] |
| **HQQ / AQLM / QuIP#** | calibration-free / extreme low-bit | research, extreme compression | respective repos | **no data** reproduced here |

**Rule:** start at `Q4_K_M`; go *up* (`Q5/Q6/Q8`) only if you have headroom, and
*down* (`Q3_K_M`) only if Q4 does not fit — quality degrades faster than size
saves.

### 3.2 Pruning

* **Structured / expert pruning (MoE).** Removes whole experts by routing
  frequency. Effect: shrinks the *pooled* size, so far more fits the cache.
  Tooling exists (`REAP` in vLLM's llm-compressor, prunes to 25%/50% sparsity) [S12v].
  vLLM guidance warns pruning must be validated, not assumed [S13].
  **vipera measured:** pruning to 5% of experts moves cache hit rate 21.5%→95%
  and throughput 0.54→1.39 tok/s on the GLM-5.3-Flash model. Quality cost **not
  measured here**.
* **Layer pruning / depth dropping.** Do not assume; the first layers matter most
  and need "healing". Use only with a quality gate (see §7).
* **Unstructured (SparseGPT/Wanda).** Weight-level; needs sparse kernels to pay
  off — usually **not** a throughput win on commodity hardware. **no data** here.

### 3.3 Distillation

Train a smaller student on a large model's outputs. Highest quality-per-byte, but
it is *training*, not a runtime lever; you cannot apply it to a downloaded model.
Evidence that it works: distilled small models surpassing much larger bases on
math [S7].

### 3.4 KV cache

| Technique | Effect | Tool |
|---|---|---|
| KV quantization (q8/fp8) | halves-ish KV bytes, small quality cost | `llama.cpp --cache-type-k/v`; vLLM `--kv-cache-dtype fp8` [S12] |
| Paged attention | kills fragmentation, enables sharing | vLLM [S8] |
| Compressed/linear attention (**MLA, KDA**) | order-of-magnitude KV reduction | architectural — DeepSeek MLA >90% (published) [S23], Kimi Linear 75% (published) [S15] |
| Eviction (H2O, sliding window) | bounded KV for long context | research/vLLM |

**Key fact:** at 1M tokens KV can be **70–90% of VRAM / 60–85% of RAM** (published) [S14]. Quantize KV *before* blaming the weights.

### 3.5 Offloading & scheduling

* **Layer/partial offload** — necessary to *fit*, not a speedup. `-ngl`, `--n-cpu-moe` [S19].
* **Expert cache + prefetch** — the real MoE lever. MoE-Infinity: **2.7–13.7× per-token**, 46% vs 32% hit; HOBBIT **9.93×** with int4/int2 (up to 4× less transfer, <1% loss); prediction accuracy 84–91% single-layer, >99% with a shadow predictor (all published) [S3][S5]. Cache sizing rule: **≈2× active-expert count** [S3].
* **CPU/GPU hybrid for MoE** — **KTransformers** targets exactly this; reported running a 671B MoE on a single 24 GB GPU by keeping experts in CPU [S16].
* **⚠ Energy:** offloading experts to **SSD costs ~5–12× more energy per token than DRAM** (published) [S3]. Prefer DRAM; check your watt cap.

### 3.6 Dynamic computation

Layer skipping / early exit. Reportedly ~1.5× compute savings with maintained
quality (published) in the general LLM literature; **workload-dependent and
unverified here**. Never ship without a quality gate.

### 3.7 Speculative decoding

A draft proposes K tokens; the target verifies in one pass — **lossless**.

| Method | Reported speedup (published) |
|---|---|
| EAGLE (autoregressive draft) | 2–3× [S8] |
| **P-EAGLE** (parallel draft) | +1.05–1.69× over EAGLE-3 [S8] |
| MTP heads (built into some models) | standard in 2026 releases [S8] |

**vipera:** modelled as tokens-per-pass; turns 1.39 → 4.18 tok/s (×3) on the GLM
instance. It *multiplies* what you have; it does not fix a bandwidth-starved config.

### 3.8 Kernels & runtimes

| Runtime | Best at | Note |
|---|---|---|
| **llama.cpp** | CPU, mixed CPU/GPU, weird/edge hardware, offline | "the portability king" [S19] |
| **vLLM** | GPU serving, batching, PagedAttention, spec-decode | production serving [S8][S12] |
| **ExLlamaV2/EXL2** | NVIDIA, fine bit control | coupled format+runtime [S10][S11] |
| **MLX** | Apple Silicon unified memory | needs 32 GB+ unified to be worthwhile [S18] |
| **KTransformers** | MoE CPU/GPU hybrid | 671B on one GPU (published) [S16] |
| **TensorRT-LLM / MLC** | max NVIDIA perf / cross-platform | **no data** reproduced here |

### 3.9 Hardware notes

* **Apple Silicon:** unified memory lets a 64 GB Mac run models a 24 GB GPU can't [S17]; bandwidth "400+ GB/s" on recent Macs (published) [S18a]. MLX is native; GGUF also works.
* **NVIDIA:** FlashAttention / PagedAttention / CUDA graphs are the levers; consumer cards are bandwidth-limited, so quantization dominates.
* **CPU-only:** expect single-digit tok/s on 7B Q4 and ~1 tok/s or less on big models; the CPU memory bandwidth is the wall [S19].

---

## 4. Step-by-step playbooks

### Scenario A — GPU 8 GB, RAM 16 GB, model 13B

1. **Environment.** Build `llama.cpp` with CUDA (`cmake -B build -DGGML_CUDA=ON`); confirm the binary is not a CPU-only distro build — a silent CPU fallback is a known trap [S21].
2. **Format.** GGUF `Q4_K_M` [S9].
3. **Quant.** Q4_K_M; if it does not fit, Q3_K_M (quality loss) — do not go below Q3.
4. **Run.**
   ```bash
   llama-cli -m model-Q4_K_M.gguf -ngl 99 -c 8192 \
     --cache-type-k q8_0 --cache-type-v q8_0
   ```
   If VRAM overflows, reduce `-ngl` until `nvidia-smi` shows a stable fit.
5. **Extras.** KV q8 (helps at long context) [S12]; consider a small draft model for speculative decoding [S8].
6. **Measure.** `tok/s`, peak VRAM, peak RAM; repeat ≥3 runs, take the median; record context length.
7. **Troubleshooting.** OOM ⇒ lower `-ngl`/context or KV bits. Very slow ⇒ you are CPU-spilling; check `-ngl` actually took effect.

### Scenario B — GPU 24 GB, RAM 64 GB, model 70B

1. **Environment.** `llama.cpp` (CUDA) or vLLM for serving.
2. **Format.** EXL2 for bit control, or GGUF Q4_K_M for portability [S9][S10].
3. **Quant.** 4–5 bit (Q4_K_M / EXL2 ~4.5 bpw).
4. **Run (llama.cpp):**
   ```bash
   llama-cli -m model-Q4_K_M.gguf -ngl 99 -c 32768 \
     --cache-type-k q8_0 --cache-type-v q8_0
   ```
   If 70B does not fully fit in 24 GB, offload some layers (`-ngl <n>`); for MoE use `--n-cpu-moe` [S19].
5. **Extras.** Speculative decoding [S8]; KV quantization for 32k+ [S12].
6. **Measure.** As A, plus `/proc/<pid>/io` read bytes per token.
7. **Troubleshooting.** Long context OOM ⇒ KV is the culprit, quantize it [S14].

### Scenario C — No GPU, RAM 32 GB, model 70B

1. **Environment.** `llama.cpp` CPU build (AVX2/AVX-512). Optionally KTransformers if it is a MoE [S16].
2. **Format.** GGUF Q4_K_M (CPU-friendly) [S19].
3. **Quant.** Q4_K_M; Q3_K_M if it does not fit in 32 GB with KV.
4. **Run.**
   ```bash
   llama-cli -m model-Q4_K_M.gguf -ngl 0 -t $(nproc) -c 4096 \
     --cache-type-k q8_0 --cache-type-v q8_0
   ```
5. **Extras.** Speculative decoding with a small draft model; MMAP the weights.
6. **Measure.** tok/s, peak RSS, disk read bytes/token. Expect **single-digit tok/s at best**; if weights stream from SSD every token, it collapses.
7. **Troubleshooting.** Thrashing ⇒ the working set exceeds RAM; cut context, quantize KV, or pick a smaller model. Do **not** rely on SSD streaming: it is slow *and* energy-expensive [S3].

---

## 5. Scripts & tools

### 5.1 Hardware probe — `scripts/probe.py` (in this skill)

Runs anywhere, stdlib only, degrades to "unknown" instead of crashing:

```bash
python3 scripts/probe.py          # human table
python3 scripts/probe.py --json   # machine-readable
```

Verified output on this repo's target laptop:

```
CPU        11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz  (8 threads)
RAM        15778 MiB
GPU        NVIDIA GeForce GTX 1650  4096 MiB (3730 MiB free)
starting_quant  Q4_K_M
```

### 5.2 Config chooser — `vipera` (this repo)

vipera plans placement and models decode *before* you download 40 GB:

```bash
# model + machine → residency plan + modelled tok/s + bytes/token
python3 -m vipera schedule arch/llama3-70b.toml --target hw/llama70b-pc.toml \
    --quant q4-experts --context 4096
# compare cache policies / pruning / speculative
python3 -m vipera schedule ... --cache-policy lfu --prune-experts 0.05 --draft-tokens 3
```

### 5.3 Benchmark script (wrap llama-bench + resources)

This repo already ships one; it records tok/s **and** peak RSS, peak VRAM and
disk bytes/token, which llama-bench does not:

```bash
python3 -m bench.bench --bin ~/dev/llama.cpp/build/bin/llama-bench \
    --model model-Q4_K_M.gguf --ngl 0 --threads 4 --json bench/results/baseline.json
```

### 5.4 Power / thermal monitor (best-effort, NVIDIA)

```bash
# log power and temperature while a run is active
nvidia-smi --query-gpu=power.draw,temperature.gpu,memory.used \
           --format=csv,noheader -l 1 | tee power.log
```

For a **true** watt figure the problem statements here require a wall wattmeter;
`nvidia-smi` sees only the GPU, not the CPU/platform.

### 5.5 Format conversion

```bash
# GGUF from Hugging Face safetensors, then quantize with an imatrix
python3 convert_hf_to_gguf.py ./model-hf --outfile model-f16.gguf --outtype f16
llama-quantize --imatrix imatrix.gguf model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

---

## 6. Comparison tables

### 6.1 Frameworks

| Framework | Speed | Memory | Quality | Difficulty | Stability | Best for |
|---|---|---|---|---|---|---|
| llama.cpp | high (CPU/mixed) | low | full (any GGUF) | low | very stable [S19] | CPU, edge, offline |
| vLLM | highest (GPU, batched) | higher | full | medium | stable [S8][S12] | serving, batching |
| ExLlamaV2/EXL2 | high (NVIDIA) | fine-grained | EXL3 ≥ GGUF/GB (claim) [S11] | medium | stable | single-user NVIDIA |
| MLX | high (Apple) | unified | full | low | stable | 32 GB+ Macs [S18] |
| KTransformers | high for MoE hybrid | low VRAM | full | high | research [S16] | giant MoE on one GPU |

### 6.2 Quantization levels (llama.cpp)

| Quant | Approx GB / 1B params | Keep when | Notes |
|---|---|---|---|
| Q2_K | ~0.35 | desperate only | large quality loss |
| Q3_K_M | ~0.43 | Q4 does not fit | noticeable loss |
| **Q4_K_M** | **~0.55** | **default** | best size/quality [S9][S10] |
| Q5_K_M | ~0.68 | headroom exists | small gain over Q4 |
| Q6_K | ~0.80 | near-lossless wanted | near-FP16 |
| Q8_0 | ~1.05 | reference quality | ~2× Q4 size |

### 6.3 Techniques — cost vs benefit

| Technique | Deploy cost | Benefit | Quality | Risk |
|---|---|---|---|---|
| Quantization | low | **high** | small at Q4 | below Q3 |
| KV quant | low | medium (long ctx) | small | accuracy at very low bits |
| Expert pruning | medium | **very high (MoE)** | **needs validation** [S13] | silent quality loss |
| Layer pruning | medium | medium | **high risk** | depth collapse |
| Speculative/MTP | medium | **high** | none | drafter overhead |
| Expert cache + prefetch | high | **high (MoE)** | none | routing not always cacheable [S3] |
| SSD offload | low | fits, but slow | none | **energy 5–12×** [S3] |
| Layer skipping | medium | medium | workload-dependent | unverified |

---

## 7. Pitfalls

1. **Quantizing is not free below Q4.** Q3/Q2 lose task accuracy faster than they save bytes; Q4_K_M is the honest default [S9][S10].
2. **Ignoring the KV cache.** Weights "fit" but the run OOMs at long context; quantize KV or shorten it [S12][S14].
3. **Assuming expert pruning is safe.** It needs a diverse evaluation; pruning from a single benchmark is explicitly warned against [S13].
4. **Believing offload is a speedup.** Offloading *fits* a model; it does not make it fast, and SSD offload is energy-harmful [S3].
5. **Silent CPU fallback.** Some ARM64/distro llama.cpp binaries fall back to CPU-only, so the GPU sits idle [S21].
6. **Confusing "fits" with "runs".** A run that streams from disk every token has a hard bandwidth ceiling; check disk bytes/token.
7. **Comparing incomparable runs.** Different context, build, or model hash makes two tok/s numbers meaningless.
8. **Thermal throttling.** Laptops throttle under sustained load; discard the first run and report a median.
9. **No quality check after compression.** Always re-measure perplexity/task score after any lossy transform.

---

## 8. Staying current

**Where to watch (and how often).**

| Source | What | Cadence |
|---|---|---|
| r/LocalLLaMA | practitioner reports, flags | daily |
| llama.cpp / vLLM releases | new kernels, quant types, spec-decode | weekly |
| arXiv (cs.LG/cs.AR) MoE + inference | new offloading/caching algorithms | weekly |
| Open LLM Leaderboard / LMSYS | model quality shifts | monthly |

**How to update this skill.** Re-run the searches in §Sources, diff new findings
against §3, and add a row only with a source and a date. `docs/RESEARCH.md` in
this repo holds the survey this catalogue is built from; `docs/PROPOSALS.md`
holds open algorithm ideas.

**Open problems.** Real routing traces for cache policies; energy-aware
scheduling; per-layer cache allocation; verifiable pruning gates.

---

## Self-evaluation (what is weak here)

* **Numbers are mostly published, not reproduced.** Only the vipera rows are
  reproducible in this repo; everything else is someone else's measurement on
  other hardware.
* **No per-model or per-hardware benchmark table.** Deliberately absent: I do not
  have first-party measurements for most (framework × model × machine) cells.
  Filling them by guessing would be exactly the failure this skill warns about.
* **Speculative-decoding and MLX specifics are thin.** The catalogue states
  mechanisms and points to sources; it does not give tuned flag sets per model.
* **Quant quality gaps.** HQQ/AQLM/QuIP# have **no data** here.
* **Energy is one published ratio.** §3.5's 5–12× figure is not verified on any
  machine in this repo.
* **The decision tree is capacity-based.** It ignores power and thermal envelopes,
  which matter for the "≤65 W" class of problem.

---

## Cheatsheet

```
FIRST MOVES
  quantize           → Q4_K_M (GGUF). Down ONLY if it does not fit.
  quantize KV        → --cache-type-k/v q8_0 at long context
  offload            → -ngl (GPU) / --n-cpu-moe (MoE); offload fits, it does not speed up
  measure            → tok/s + peak RAM/VRAM + disk bytes/token + quality

WHEN IT STILL DOES NOT FIT
  MoE   → expert cache + prefetch, then EXPERT PRUNING (validate quality!)
  dense → layer-wise streaming, accept the I/O ceiling, add speculative decoding
  never → count on SSD streaming for speed or for a tight power budget

MULTIPLIERS (lossless)
  speculative / MTP: ×2–3 typical (published)
  KV quant: frees budget for longer context

RED FLAGS IN ANY CLAIM
  no hardware/model/context stated · no quality number beside the speed ·
  "best" without a baseline · a benchmark that cannot be re-run
```

---

## Sources

Retrieved 2026-10-12 via web search; titles as returned.

- **[S3]** Expert offloading overview (MoE-Infinity, HOBBIT, MoEpic, MoE-SpeQ, DAOP; cache ≈2× rule; SSD energy 5–12×) — https://www.emergentmind.com/topics/expert-offloading
- **[S5]** fMoE — fine-grained expert offloading — https://arxiv.org/html/2502.05370v1
- **[S6]** MoE-Infinity (per-token 2.7–13.7×) — https://openreview.net/forum?id=BL7WMLJKZM
- **[S7]** On-Device LLMs: State of the Union, 2026 (bandwidth gap 30–50×) — https://v-chandra.github.io/on-device-llms/
- **[S8]** P-EAGLE (parallel speculative decoding in vLLM) — https://vllm.ai/blog/2026-03-13-p-eagle
- **[S9]** GPTQ vs AWQ vs GGUF 4-bit comparison, 2026 — https://www.spheron.network/blog/gptq-vs-awq-vs-gguf-4-bit-quantization-comparison-2026/
- **[S10]** Unified llama.cpp quantization evaluation — https://arxiv.org/html/2601.14277v1
- **[S11]** EXL3 vs GGUF benchmark — https://atomic.chat/blog/guides/exl3-vs-gguf
- **[S12]** vLLM — Quantized KV Cache (fp8) — https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- **[S12v]** REAP expert pruning (vLLM llm-compressor) — https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/reap_expert_pruning/
- **[S13]** Live MoE expert pruning in vLLM (validation warnings) — https://forums.developer.nvidia.com/t/why-you-should-reap-it-yourself-live-moe-expert-pruning-in-vllm/364098
- **[S14]** KV cache optimization engineering guide 2026 — https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide
- **[S15]** Kimi Linear / KDA (75% KV reduction) — https://www.alphaxiv.org/abs/2510.26692
- **[S16]** KTransformers (CPU/GPU hybrid MoE) — https://github.com/kvcache-ai/ktransformers , https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/
- **[S17]** Local LLMs on Apple Silicon 2026 — https://www.sitepoint.com/local-llms-apple-silicon-mac-2026/
- **[S18]** Apple Silicon LLM inference optimization (MLX needs 32 GB+) — https://blog.starmorph.com/blog/apple-silicon-llm-inference-optimization-guide
- **[S18a]** Native LLM inference on Apple Silicon (400+ GB/s unified) — https://arxiv.org/html/2601.19139v1
- **[S19]** 10 GB VRAM local LLM setup guide 2026 (llama.cpp portability) — https://www.sitepoint.com/10gb-vram-local-llm-the-complete-setup-guide-2026/
- **[S20]** Local LLM inference optimization: the complete guide (community) — https://www.reddit.com/r/LocalLLaMA/comments/1uc3wg9/local_llm_inference_optimization_the_complete/
- **[S21]** DGX Spark / GB10 stack (ARM64 CPU-only fallback trap) — https://forums.developer.nvidia.com/t/running-a-full-llm-stack-on-dgx-spark-gb10-your-application-litellm-llama-swap-vllm-llama-cpp-ollama/367580
- **[S22]** MoE inference optimization survey (taxonomy) — https://arxiv.org/html/2412.14219v1
- **[S23]** Recent developments in LLM architectures (KV sharing, MLA, mHC) — https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures

> Company blog URLs (Hugging Face, NVIDIA, Apple) and benchmark leaderboards are
> referenced by name only where I did not fetch a specific page; no URL is invented.
