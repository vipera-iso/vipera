# FEASIBILITY — can GLM-5.3-Flash run on the target laptop?

> The stretch goal for stage 2: make **GLM-5.3-Flash** (Z.ai, 2026-08) run on a
> 15 GiB / 4 GB-VRAM / DRAM-less-NVMe laptop. This document records what is
> **modelled**, what is **measured**, and the honest verdict.
>
> **Every number in the tables below is modelled by `vipera schedule`** — none of
> it is a real run yet. The model's inputs (bandwidths, expert sizes, routing
> skew) are stated so the estimate can be attacked. See `docs/OPTIMIZER.md` for
> the file-layer transforms this plan feeds into.

---

## 1. The model

| Property | Value | Source |
|---|---|---|
| Total parameters | 321B | published (Z.ai) |
| Active parameters / token | 18B | published |
| Language layers | 45 | published |
| Experts / layer, top-k | 288, top-8 | published |
| Attention | 34 KDA + 11 MLA/DSA (3:1) | published |
| Residual | 4-stream mHC | published |
| Checkpoint (FP8) | ~306 GiB | published |

The exact hidden size / expert intermediate size are **not public**. `vipera/arch.py`
derives a consistent shape from the headline totals:

```
routed experts   312.1 B   (45 layers x 288 experts x 3 x 5120 x 1536)
shared experts     1.08 B
attention (KDA)    6.60 B   (34 layers x 7.4 x 5120^2)
attention (MLA)    0.97 B
router             0.07 B
embeddings         0.78 B
MTP draft          6.96 B
--------------------------------
TOTAL            322.1 B   (+0.3% vs the stated 321 B)
ACTIVE/token      17.96 B  (-0.2% vs the stated 18 B)
```

Run `python3 -m vipera arch arch/glm-5.3-flash.toml` to reproduce the calibration.

## 2. The machine

| Tier | Capacity | Bandwidth |
|---|---|---|
| VRAM (GTX 1650 Max-Q) | 4 GB | ~192 GB/s |
| RAM (DDR4-3200) | 15 GiB, no swap | ~40 GB/s |
| Disk (WD SN530, DRAM-less) | 404 GB free | ~2 GB/s |

The disk is the constraint: the FP8 checkpoint (306 GiB) barely fits, and every
expert that is not cached must be read from it.

## 3. What the scheduler does

`vipera/schedule.py` places every layer's weights and simulates decode steps:

1. **Always-resident set** (attention + router + shared experts + embeddings,
   8.5 GiB) is pinned to VRAM first, then RAM. If it does not fit, it streams
   from disk *every token* and the model is not runnable.
2. **Leftover VRAM+RAM becomes the expert cache.** On the target laptop that is
   8.5 GiB ≈ **378 of 12,960 experts** (~2.9%).
3. Each token routes top-8 experts per layer. A **hit** reads the expert from
   RAM/VRAM; a **miss** reads it from disk and admits it (LRU).
4. `prefetch=True` models a dedicated I/O thread hiding disk reads behind the
   compute of the previous layer, so the token cost is `max(io, compute)`.

## 4. Results (modelled, 2048-token context)

| Quant | Cache hit | Bytes/token | Disk bytes/token | ms/token | **tok/s** |
|---|---|---|---|---|---|
| FP8 (1 B/param) | 13.2% | 16.0 GiB | 6.87 GiB | 3877 | **0.26** |
| Q4 experts, FP8 rest | 21.5% | 11.6 GiB | 3.11 GiB | 1839 | **0.54** |
| Q4 everything | 31.3% | 8.0 GiB | 2.72 GiB | 1536 | **0.65** |

## 5. The verdict

**GLM-5.3-Flash cannot run usefully on this machine.** The best modelled figure
is **~0.65 tokens/s** (roughly one word every two seconds), and it is I/O-bound:
~90% of the per-token time is spent waiting on the DRAM-less disk.

Two independent ceilings cap it, and neither is a software problem:

* **Disk ceiling.** Only ~3% of the experts fit the cache, so ~8B of the 18B
  active parameters are re-read from a 2 GB/s disk every token. Even a perfect
  predictor that prefetched everything would still be limited by 2 GB/s.
* **Active-parameter ceiling.** The model activates 18B parameters per token.
  Even if the *entire* model were resident in RAM, 18B parameters at 40 GB/s is
  `~2.2 tok/s` — and the RAM does not hold 306 GiB anyway.

## 6. What would actually be needed

| Upgrade | Effect |
|---|---|
| ≥ 128 GiB RAM (e.g. a 256 GB workstation) | experts hit RAM; ceiling rises to the 40 GB/s / 18 B ≈ ~2 tok/s, then MTP + batching matter |
| ~4× GB200 (as the vLLM recipe assumes) | the intended deployment; tensor+expert parallel across 300+ GiB of VRAM |
| A quantized laptop variant **of a smaller model** | the realistic path for this hardware |

The honest recommendation stands: keep DeepSeek-V2-Lite (fits RAM) as the
*measurable* target, and treat GLM-5.3-Flash as a **streaming stress test** for
vipera's file layer — not as something this laptop can serve.

## 7. Reproduce

```bash
python3 -m vipera arch     arch/glm-5.3-flash.toml
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --out out/
```

`out/glm-5.3-flash.vipera.json` carries the full per-layer breakdown (bytes and
disk bytes per token for each of the 45 layers).

## 8. Assumptions to attack (in order of leverage)

1. **`cpu_gflops`** was assumed to be 50; it is now **measured at 110 GFLOP/s**
   (ROADMAP §9, `bench/hwprobe.c`), corroborated by a real model running at
   41.2 tok/s. Every `tok/s` figure below therefore predates the measurement and
   is roughly **10% pessimistic**; the *verdicts* (I/O-bound, infeasible) do not
   change, because these models were already I/O-bound.
2. **Routing skew is modelled** (`Zipf(1.0)`, fixed seed), not a real trace.
   Real MoE routing is often more skewed, which raises the cache hit rate; a
   real trace would move the hit-rate column. Round 3 captured one real trace
   (ROADMAP §8) — it made the two policies diverge sharply.
3. **Bandwidths**: RAM is now **measured at 44 GB/s** (was 40). VRAM (192) and
   disk (2.0 GB/s) are still class figures, not measured here. The disk number is
   the one that matters most.
4. **KV cache** is modelled at 2048 tokens (76 MiB). At the model's 1M-token
   context it grows to ~37 GiB and would not fit at all.

---

## 9. Layer-level transforms: what actually moves the needle

The scheduler can model the standard layer-level optimization families
(`vipera/schedule.py`, `Optimizations`) so their effect is compared on one model
and one machine instead of argued in the abstract:

| Technique | Modelled as | Effect on the numbers |
|---|---|---|
| Partial offload / residency | always-resident placement + expert cache | (already in the baseline) |
| Quantization | `--quant fp8\|q4-experts\|q4` | 0.24 → 0.50 tok/s |
| KV compression | `--kv-bits 8` | shrinks KV, so more expert cache |
| **Expert pruning** | `--prune-experts F` | the big one: shrinks the pool so it fits cache |
| Layer pruning | `--keep-layers N` | fewer layers ⇒ fewer reads per token |
| Speculative / MTP | `--draft-tokens N` | multiplies accepted tokens per pass |

`python3 -m vipera sweep arch/glm-5.3-flash.toml --target hw/gtx1650.toml`:

```
strategy                  hit  disk GB/tok  bytes GB/tok     tok/s  bound
baseline fp8            13.2%         7.37         17.19      0.26  io
q4 experts              21.5%         3.33         12.41      0.54  io
q4 experts + KV 8-bit   21.5%         3.33         12.41      0.54  io
+ prune 25% experts     38.5%         2.61         12.41      0.67  io
+ prune 5% experts      95.0%         0.21         12.41      1.39  compute
+ prune 5% + draft x3   95.0%         0.21         12.41      4.18  compute
+ drop 20 layers too    95.4%         0.11          6.91      7.52  compute
```

**The only lever that changes the verdict is expert pruning**, because it is the
only one that fixes the *cause* (a pool 30× the cache). At 5% of experts, 95% of
routes hit the cache, the workload stops being disk-bound, and the ceiling moves
to CPU compute — where **speculative/MTP decoding** and **layer pruning** then
compound. The other families (partial offload, plain KV compression) move bytes,
not the bound.

**But none of this is free or proven here.** Quality is *not* modelled: pruning
5% of experts (or 20 layers) will cost accuracy, and the numbers above assume the
pruned model still works. That has to be measured with perplexity / a task set
before any of these rows can be trusted — see §6 of `OPTIMIZER.md`. A row that
looks fast and a model that got dumber is a failure, not a win.

---

## 10. Instance solved: LLM-PC-OPT on a dense model

The competition instance is **Llama-3-70B, 80 layers, 32k context** on **8 GB VRAM,
32 GB RAM, 3 GB/s NVMe, PCIe 4.0 x16, 8 cores**, targeting **≥ 5 tok/s**, **< 2%
MMLU loss**, **< 250 W**.

vipera can now solve it: a dense model is modelled as *one expert per layer with
top_k = 1*, so the FFN joins the always-resident set and whatever exceeds
VRAM+RAM streams from disk every token — i.e. dense layer-wise streaming.

```bash
python3 -m vipera arch     arch/llama3-70b.toml
python3 -m vipera schedule arch/llama3-70b.toml --target hw/llama70b-pc.toml --quant q4-experts --context 4096
python3 -m vipera sweep    arch/llama3-70b.toml --target hw/llama70b-pc.toml --context 4096
```

The model reproduces the real config: **69.5B params (−0.7%)**, ratio active/total = 1.00
(dense), Q4_K_M-like weights ≈ **41.3 GB**.

| strategy | tok/s | bound |
|---|---|---|
| baseline fp8 | 0.10 | io |
| **Q4_K_M-like (`q4-experts`)** | **0.72** | compute |
| Q4 + KV 8-bit | 0.72 | compute |
| Q4 + expert pruning | 0.72 (no-op: dense) | compute |
| Q4 + speculative ×3 | 2.16 | compute |
| Q4 + drop 20 layers | 0.96 | compute |
| Q4 + keep **25 of 80** layers | 6.91 | compute |

### The answer, and the lower bound

**The instance is infeasible as stated.** The best *quality-preserving* config is
**~0.72 tok/s**; speculative decoding (×3) reaches 2.16 tok/s, still < 5.

Reaching 5 tok/s requires **≤ 0.2 s per token**. A dense 70B must read **~40 GB of
weights per token** (no sparsity to exploit), so 5 tok/s demands

```
40 GB / 0.2 s  ≥  200 GB/s  sustained weight bandwidth.
```

The only tier on this box anywhere near that is VRAM (8 GB at ~200 GB/s). Even in
the impossible best case — every weight resident, split 8 GB VRAM + 32 GB RAM:

```
time/token  ≥  8/200 + 32/50  =  0.68 s   ⇒  ≤ 1.47 tok/s   (assumed tier BW)
```

and using only the constants the problem itself gives (PCIe 4.0 x16 ≈ 32 GB/s):

```
time/token  ≥  40/32 = 1.25 s              ⇒  ≤ 0.8 tok/s
```

Either way the 5 tok/s target is **unreachable by ~4–7×**, with no software
freedom left — it is memory bandwidth, not the algorithm. The only row that clears
5 tok/s keeps **25 of 80 layers**, i.e. discards 69% of the depth; that is a
different, much dumber model and violates the < 2% quality budget.

### Limits of this answer

* **Compute is modelled with one CPU figure** (`cpu_gflops`). On this box a GPU
  does the matmuls, which vipera's flat compute floor understates; the verdict
  does not depend on it (the I/O rows already fail), but a GPU-aware compute term
  is the next modelling gap.
* **Power/thermal (< 250 W) is not modelled at all** — a genuinely missing
  constraint from the problem.
* **Quality is not modelled**; the < 2% MMLU check is an assumption standing in
  for a perplexity/task evaluation.
* RAM/VRAM bandwidths are class assumptions (the instance does not state them);
  the disk and PCIe numbers are the problem's.

---

## 11. Instance solved: LOCAL-LLM-16GB

The open problem: **Llama-3-70B (dense) on a 16 GB unified-memory laptop, no
discrete GPU**, with **32k context**, **≥ 10 tok/s**, **≤ 16 GB peak RAM**,
**≤ 65 W**, and quality within 1% MMLU / 2% GSM8K / 5% perplexity of FP16.
Score `S = (Q/100)·(T/10)·(16/M)·(65/P)·(32k/C)`; solved only if every hard
constraint holds and **S ≥ 1**.

```bash
python3 -m vipera score arch/llama3-70b.toml --target hw/laptop16-igpu.toml \
    --context 32768 --quality 99 --power 65
```

Modelled result of the best quality-preserving config (Q4_K_M-like):

```
Q (quality)   99.0%
T (tokens/s)  0.112
M (peak RAM)  25.77 GB     <- already over the 16 GB hard limit
P (power)     65.0 W
C (context)   32768 tokens
S             0.0069   (FAIL)
```

Residency detail: **38.5 GiB** of Q4 weights, of which only **14 GiB** fit the
16 GB machine; **24.5 GiB stream from the NVMe every token** → 8.9 s/token.
The 32k KV cache alone is **10.0 GiB** (10.74 GB fp16, 5.37 GB at 8-bit), so the
weights and the KV cannot both live in 16 GB regardless of quantization.

### Why it is impossible here

1. **It does not fit.** 41 GB of Q4 weights + 10.7 GB of 32k KV ≫ 16 GB, so at
   least ~36 GB must be read from the **3 GB/s** SSD *per token* → ≥ 12 s →
   **≤ 0.083 tok/s**. That is the floor before any compute.
2. **Even the fit was a fantasy.** If all 41 GB were somehow resident, reading it
   at the machine's ~100 GB/s unified bandwidth is 0.41 s → **≤ 2.4 tok/s** — and
   that ignores the KV cache entirely.
3. **The gate is T ≥ 10.1 tok/s.** With Q=99, M=16, P=65 and C=32k all at target,
   `S ≥ 1` demands **T ≥ 10.1 tok/s**, i.e. **≤ 0.1 s/token**, i.e. **≥ 410 GB/s**
   of weight bandwidth. The machine has ~100 GB/s and a 3 GB/s disk.
4. **The compression needed is extreme.** Fitting weights *and* KV in 16 GB needs
   ~1.6 bits/parameter — far below Q4 — where the < 1% MMLU budget is already
   broken. There is no quality-preserving path.

**Verdict: fails by ~90–100× on throughput and violates the 16 GB RAM limit.**
By the problem's own medal table, even **Bạc** (70B on 32 GB) does not run here;
**Vàng** (70B on 16 GB) requires hardware this problem does not provide. As with
§10, the wall is memory bandwidth and capacity, not the scheduling algorithm.

### Limits

* **Power (< 65 W) is an input, not a measurement** — vipera does not model
  energy, and the problem requires a wattmeter.
* **Compute is a flat `cpu_gflops` term**; the iGPU is not modelled, and no
  custom kernel is assumed.
* **Quality is an input** (`--quality`), not modelled; the 1%/2%/5% budgets are
  assumed, not verified.
* Unified-memory bandwidth (~100 GB/s) is a class assumption.
