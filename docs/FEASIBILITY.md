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
| FP8 (1 B/param) | 13.2% | 16.0 GiB | 6.87 GiB | 4136 | **0.24** |
| Q4 experts, FP8 rest | 21.5% | 11.6 GiB | 3.11 GiB | 1997 | **0.50** |
| Q4 everything | 31.3% | 8.0 GiB | 2.72 GiB | 1683 | **0.59** |

## 5. The verdict

**GLM-5.3-Flash cannot run usefully on this machine.** The best modelled figure
is **~0.6 tokens/s** (roughly one word every two seconds), and it is I/O-bound:
77% of the per-token time is spent waiting on the DRAM-less disk.

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

1. **`cpu_gflops = 50`** in `hw/gtx1650.toml` — the compute floor is *assumed*,
   not measured. If it is too low, the compute-bound ceiling is overstated.
2. **Routing skew is modelled** (`Zipf(1.0)`, fixed seed), not a real trace.
   Real MoE routing is often more skewed, which raises the cache hit rate; a
   real trace would move the hit-rate column.
3. **Bandwidths** (192 / 40 / 2 GB/s) are class figures, not measured here. The
   disk number is the one that matters most.
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
baseline fp8            13.2%         7.37         17.19      0.24  io
q4 experts              21.5%         3.33         12.41      0.50  io
q4 experts + KV 8-bit   21.5%         3.33         12.41      0.50  io
+ prune 25% experts     38.5%         2.61         12.41      0.61  io
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
