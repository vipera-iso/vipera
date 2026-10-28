# RESEARCH — algorithms for running local LLMs on weak hardware

> Survey of the current (2026) technique landscape for local LLM inference, with
> an **evaluation** attached to each method. Two kinds of numbers appear and are
> labelled on purpose:
>
> * **published** — reported by the cited work; not reproduced here;
> * **vipera** — produced by this repo's model on a named instance
>   (`docs/FEASIBILITY.md`), so it is reproducible with one command.
>
> The evaluation uses the one metric that decides local speed: **bytes read per
> token**, and the tier they come from. Everything else is a multiplier on top.

---

## 0. The frame: why decode is a bandwidth problem

A generated token must touch every active weight once. So for a fixed model:

```
tokens/s  ≈  tier_bandwidth / (active_bytes_per_token)
```

On weak hardware the tiers are far apart — mobile/laptop unified memory is
**50–90 GB/s** where datacenter GPUs are **2–3 TB/s, a 30–50× gap**
(published, On-Device LLMs: State of the Union 2026). That single ratio explains
why *compression* and *multi-token prediction* dominate locally, while *kernel
tuning* barely moves the needle.

**Therefore every technique below is scored primarily on one question: does it
reduce `active_bytes_per_token`, or the bandwidth that serves them?** A method
that only reduces FLOPs, latency, or peak memory without changing bytes/token
does not raise decode throughput on a bandwidth-bound machine.

---

## 1. Taxonomy

| Family | Question it answers | Examples |
|---|---|---|
| **A. Compression** | fewer bytes per parameter / fewer parameters | quantization, pruning, distillation |
| **B. Execution & offloading** | serve bytes from a faster tier, or hide the slow one | layer offload, expert cache, prefetch, split experts |
| **C. Attention & KV** | stop the cache from eating the budget | MLA, linear/KDA attention, KV quantization, eviction |
| **D. Adaptive decoding** | read the weights for *more than one* token | speculative decoding, MTP, layer skipping |

---

## 2. A — Compression

### A1. Quantization (weight-only, block-wise)

**Mechanism.** Store weights in low-bit blocks with per-block scales; decode reads
fewer bytes. This is the highest-leverage technique because it cuts bytes/token
*directly and proportionally* — 16-bit → 4-bit is 4× less traffic, not just 4×
less storage (published).

**Evidence.** A unified llama.cpp study on Llama-3.1-8B-Instruct (3–8 bit
K-quants, GPTQ/AWQ compared) reports **Q4_K_M is the 4-bit sweet spot** and is
"best of the three 4-bit formats" in an independent 2026 comparison
(**6.74 WikiText-2 perplexity, 51.8% HumanEval Pass@1** — published). Sub-4-bit
Q3/Q2 remain usable but degrade noticeably on downstream tasks, while **1.58-bit
ternary** models trade a large quality hit for size (published).

**Evaluation.**

| | |
|---|---|
| Bytes/token | **↓ 2–4×** (the only technique that cuts this cleanly) |
| Quality cost | ~1–2% at Q4_K_M; grows fast below Q4 |
| Maturity | **production** (llama.cpp, vLLM, every runtime) |
| vipera evidence | GLM-5.3-Flash: FP8 0.26 → **Q4 0.65 tok/s**; Llama-70B Q4 is the only config under the 2% budget |
| Verdict | **Always the first move.** It is the baseline the others are measured against, and it does not remove the bandwidth wall — it moves it. |

### A2. Expert / layer pruning

**Mechanism.** Drop rarely-used experts (MoE) or unimportant layers (dense), then
"heal" the survivors with light fine-tuning.

**Evidence.** REAP / LAEP / MoNE / MoE-Pruner all report *structural* expert
removal; a vLLM/llm-compressor recipe prunes to 25% and 50% expert sparsity
(published). Reported retention is high on some setups ("over 96.4% of average
zero-shot accuracy", published) but pruning-from-benchmarks is explicitly warned
against in production reports.

**Evaluation.**

| | |
|---|---|
| Bytes/token | **↓↓ for MoE**: shrinks the *pooled* size, so far more experts fit cache |
| Quality cost | **the real risk** — needs healing and a diverse eval set |
| Maturity | research → tooling (vLLM REAP); not a one-click default |
| vipera evidence | GLM: the *only* lever that changes the verdict — pruning to 5% moves hit rate 21.5%→**95%** and 0.54→1.39 tok/s; dense Llama: **no-op**, only layer pruning helps |
| Verdict | **Decisive for MoE, useless for dense.** It fixes the *cause* (pool ≫ cache) rather than the symptom. |

### A3. Distillation

**Mechanism.** Train a smaller model on a large one's outputs. Compresses by
changing the model, not the file.

**Evidence.** Distilled 8B models surpassing much larger bases on math; Qwen3-30B-A3B
(3B active) outperforming QwQ-32B (published).

**Evaluation.** Highest quality-per-byte, but it is **training, not inference
optimization** — out of scope for a runtime/file-layer tool, and it changes the
model's identity. Verdict: the right answer when you control training; not a
lever a scheduler can pull.

---

## 3. B — Execution & offloading

### B1. Layer-wise / partial offloading (the baseline)

**Mechanism.** Keep hot weights on the fast tier; stream the rest per token.
llama.cpp `-ngl`, AirLLM-style layer streaming.

**Evaluation.** Necessary but *not* a throughput win: it trades peak memory for
I/O. On our dense Llama-70B instance, offloading 24.5 GiB from a 3 GB/s SSD is
**8.9 s/token** (vipera). Verdict: an enabler of feasibility, not of speed.

### B2. Expert cache + prediction + prefetch

**Mechanism.** Cache hot experts; predict next-layer routing; prefetch ahead to
hide I/O behind compute.

**Evidence (published).** MoE-Infinity: **2.7–13.7× per-token** improvement,
cache hit **46% vs 32%**; HOBBIT **9.93×** with mixed-precision (int4/int2)
offload, **up to 4× lower transfer latency at <1% loss**; MoEpic 37–66% latency;
MoE-SpeQ up to 2.34× with 99% hit. Prediction accuracy: **84–91%** for
single-layer lookahead, **>99%** with shadow/speculative predictors. A practical
rule: size the cache at **≈2× the active-expert count** for robust hit rates.

**Evaluation.**

| | |
|---|---|
| Bytes/token | **↓↓** (a hit is a byte not read from the slow tier) |
| Quality cost | none (lossless) |
| Maturity | production (MoE-Infinity, fMoE, llama.cpp MoE offload) |
| vipera evidence | modelled directly: hit rate is the knob; `prefetch` switches `io+compute` → `max(io,compute)` |
| Verdict | **The core MoE lever after quantization**, and it composes with A2. Two caveats: it only helps if routing is *locally consistent* ("not all models suit expert offloading", published), and §5's energy result bites. |

### B3. Split / collapsed experts and mixed-precision loading

**Mechanism.** Split each expert into a VRAM-resident top and a CPU bottom, or
load unimportant experts at lower precision.

**Evidence (published).** HOBBIT: int4/int2 mixed precision, up to 4× transfer
reduction, <1% loss. MoEpic adapts split ratio per layer.

**Evaluation.** A refinement of B2 that buys more hit rate per VRAM byte; vipera
does not model split experts yet (a known gap). Verdict: worth adding to the
model after A2/B2.

---

## 4. C — Attention & KV

**Mechanism.** Long-context KV grows linearly and can dominate the budget — at
1M tokens, **70–90% of VRAM / 60–85% of RAM** (published). Fixes: compress (MLA,
low-rank latent — **>90% KV reduction**, published), go linear/recurrent
(Kimi Linear: **75% KV reduction, up to 6× decode throughput**, published), or
evict/quantize.

**Evaluation.**

| | |
|---|---|
| Bytes/token | neutral for weights; **↓↓↓ for the memory *budget*** |
| Quality cost | architecture change (MLA/KDA need training); quantize/evict ~low |
| Maturity | architecture: research→frontier (DeepSeek MLA, Kimi Linear); KV-quant/evict: production |
| vipera evidence | Llama-70B at 32k: **KV alone = 10.0 GiB** (10.74 fp16 / 5.37 at 8-bit) on a 16 GB machine — KV, not weights, is what makes 32k impossible there |
| Verdict | **A capacity technique, not a bandwidth one.** It rarely raises tok/s; it decides whether long context fits at all. Essential for the LOCAL-LLM-16GB class of problem. |

---

## 5. D — Adaptive decoding

### D1. Speculative decoding / MTP

**Mechanism.** A cheap drafter proposes K tokens; the target verifies them in one
pass. Bytes/token is roughly unchanged, but **tokens per pass** rises.

**Evidence (published).** EAGLE achieves **2–3×** over autoregressive; P-EAGLE
adds **1.05–1.69× over EAGLE-3** by drafting in parallel; MTP heads are standard
in 2026 releases. On-device framing: multi-token prediction is "free" — no latency
penalty (published).

**Evaluation.**

| | |
|---|---|
| Bytes/token | unchanged; **tokens/pass ↑** ⇒ effective tok/s ↑ |
| Quality cost | lossless (verification) |
| Maturity | **production** (vLLM, SGLang, TensorRT-LLM) |
| vipera evidence | modelled as `draft_tokens`: GLM q4+prune 1.39 → **4.18 tok/s** at ×3; Llama-70B 0.72 → 2.16 |
| Verdict | **The cheapest multiplier once the model already runs.** It multiplies whatever throughput you have, so it does not rescue a bandwidth-starved config by itself. |

### D2. Dynamic layer skipping

**Mechanism.** Skip layers for "easy" tokens via a light router.

**Evidence (published).** Reported ~1.5× compute savings with maintained quality,
but gains are workload-dependent and hard to guarantee.

**Evaluation.** Reduces work, not bytes read (unless it skips whole layers, which
then *does* cut bytes). Risk: silent quality drift on out-of-distribution input.
Verdict: promising, **unverified**, and dangerous without a quality gate.

---

## 6. Cross-cutting: energy, and a warning

The problem statements in this repo cap wall power. That makes one published
result decisive and counter-intuitive:

> **Offloading MoE weights to SSDs is currently energetically harmful** — roughly
> **5–12× higher energy per token than DRAM** — unless flash energy drops ~10×
> (published). Best practice: prefer host DRAM over SSD, and use hierarchical
> caches for hot experts.

So the standard "stream from SSD" answer is a *time* fix that can be an *energy*
regression. On a 65 W laptop this can fail a hard constraint. vipera does **not**
model this yet — a real gap.

---

## 7. Evaluation summary

Scores: `bytes/token` impact on a bandwidth-bound decode; quality risk; maturity.
★★★ = decisive, ★ = marginal.

| Technique | bytes/token | Quality risk | Maturity | vipera models it? | Verdict for a weak machine |
|---|---|---|---|---|---|
| Quantization (Q4_K_M) | ★★★ (2–4×) | low | production | yes (`--quant`) | **Do first, always** |
| **MoE expert pruning** | ★★★ (pool→cache) | **high** | tooling | yes (`--prune-experts`) | **Decisive for MoE**; no-op for dense |
| Expert cache + prefetch | ★★★ | none | production | yes (hit rate, `prefetch`) | Core MoE lever; needs local routing consistency |
| Mixed-precision / split experts | ★★ | low | research→tooling | **no** | Next thing to add to the model |
| Speculative / MTP | ★★★ (tokens/pass) | none | production | yes (`--draft-tokens`) | Best multiplier once it runs |
| Layer pruning | ★★ | **high** | research | yes (`--keep-layers`) | Only dense lever; collapses quality if overdone |
| KV compress / linear attention | ★ (capacity) | low→arch | mixed | partly (KV bits, MLA/KDA in arch) | Fits long context; rarely raises tok/s |
| Distillation | ★★★ | medium | production (training) | no | Right answer, different tool |
| Dynamic layer skipping | ★★ | medium | research | no | Unverified; needs a quality gate |
| Layer/partial offload | ★ (enabler) | none | production | yes (residency) | Feasibility, not speed |
| SSD offloading | ★ | none | production | yes (tier cost) | **Energy regression — check §6** |

---

## 8. What this says about the instances in this repo

* **GLM-5.3-Flash (321B-A18B):** quantization + cache move bytes; only **expert
  pruning** changes the verdict (it fixes pool ≫ cache). Verdict stays
  ~0.6 tok/s until pruning is assumed, then compute-bound.
* **Llama-70B dense:** no expert pool to prune, so only layer pruning +
  speculative help; both threaten the quality budget. Bandwidth wall stands.
* **LOCAL-LLM-16GB:** KV (10 GiB at 32k) plus weights cannot co-exist in 16 GB;
  the fix is architectural (linear/MLA attention), not scheduling.

**The one-line conclusion:** for local LLMs, *compression* (quantization, expert
pruning) and *multi-token prediction* decide feasibility; *offloading* decides
whether it runs at all; *KV architecture* decides whether long context fits; and
no algorithm beats the tier bandwidth it is forced to read from.

---

## 9. References (as published; not reproduced here)

* MoE inference optimization survey — https://arxiv.org/html/2412.14219v1
* I/O for LLM inference survey (quantization/compression/offloading) —
  https://link.springer.com/article/10.1007/s10462-026-11651-1
* Expert offloading overview (MoE-Infinity, HOBBIT, MoEpic, MoE-SpeQ, DAOP,
  energy result) — https://www.emergentmind.com/topics/expert-offloading
* MoE-Infinity —
  https://openreview.net/forum?id=BL7WMLJKZM
* fMoE — https://arxiv.org/html/2502.05370v1
* On-Device LLMs: State of the Union, 2026 —
  https://v-chandra.github.io/on-device-llms/
* Unified llama.cpp quantization evaluation —
  https://arxiv.org/html/2601.14277v1
* GPTQ vs AWQ vs GGUF (4-bit, 2026) —
  https://www.spheron.network/blog/gptq-vs-awq-vs-gguf-4-bit-quantization-comparison-2026/
* P-EAGLE (parallel speculative decoding) — https://vllm.ai/blog/2026-03-13-p-eagle
* Kimi Linear / KDA attention — https://www.alphaxiv.org/abs/2510.26692
* Architecture developments (KV sharing, mHC, MLA) —
  https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures
* REAP expert pruning (vLLM llm-compressor) —
  https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/reap_expert_pruning/

> **Caveat.** Every "published" figure above is reported by the cited work on its
> own hardware and workload. This repo has not run any of them. The column that
> *is* reproducible locally is the vipera one.
