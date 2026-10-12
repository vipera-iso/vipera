# ALGORITHMS — Running LLM/MoE on weak hardware

> **See also [`RESEARCH.md`](RESEARCH.md)** — a 2026 survey and *evaluation* of the current
> optimization techniques (quantization, pruning, offloading, KV, speculative), scored on bytes
> per token and cross-referenced to the reproducible vipera model.


> This document is **the most important part** of the project (section 1 of the original prompt).
> Every **bold** number is **measured on the target machine** or **derived from a measurement**;
> other numbers are explicitly marked as estimates. Results attributed to FreeToken/AirLLM/edge0 are
> **as published by them** and have not been re-verified here.

---

## 0. Target machine (measured 2026-10-12)

| Component | Spec | Why it matters for the algorithms |
|---|---|---|
| CPU | Intel i5-11300H, **4 cores / 8 threads**, up to 4.4 GHz | MoE compute that falls back to CPU is slow; DDR4 bandwidth is the real ceiling |
| RAM | **15 GiB**, ~10 GiB available, **no swap** | a model must be ≤ ~11 GiB to stay resident; above that it must stream |
| GPU | **GTX 1650 Max-Q, 4 GB VRAM**, compute 7.5 (Turing) | holds attention + KV + a few experts; **cannot hold the whole model** |
| iGPU | Intel Iris Xe (Tiger Lake) | could co-compute, but there is no unified API with NVIDIA |
| Disk | **WD SN530 NVMe (DRAM-less)**, ~404 GB free | good sequential reads, but **DRAM-less** ⇒ worse burst/random |
| Available | `python3`, `cmake`, `gcc` | missing `llama.cpp`, `ollama`, `cargo`, `nvcc` |

**Theoretical ceilings (derived, not measured):** LLM decode is **memory-bandwidth-bound**.
- DDR4-3200 dual-channel RAM ≈ **40–45 GB/s** practical.
- GTX 1650 VRAM ≈ **192 GB/s**.
- SN530 NVMe sequential read ≈ **1.5–2.0 GB/s** practical (DRAM-less).

⇒ Each token must read the *active* weights. Therefore:
`tokens/s ≈ bandwidth / (active weight bytes per token)`.

---

## 1.a Layer-wise Streaming (AirLLM)

**Mechanism.** Do not load the whole model; load **one layer** into the compute unit, compute, free
it, move to the next layer.

```
peak VRAM = max(size of one layer) + KV + activations + overhead   (INDEPENDENT of total model size)
```

**The central trade-off.** Capacity in exchange for I/O. For a model **larger than RAM**, streaming is
mandatory. For a model that **fits RAM**, streaming is self-defeating for speed.

**Numbers on this machine.** A 30B Q4 MoE (≈18.6 GB) does not fit in 11 GiB of RAM ⇒ it must stream
from the NVMe:
```
18.6 GB / 2.0 GB/s ≈ 9.3 s for ONE full pass
```
If every token needs a full pass ⇒ **~0.1 token/s** — useless. ⇒ Streaming only wins when **most
weights stay resident** and only the rarely used part streams.

**I/O optimization (priority order):**
1. **mmap** the weights (`mmap` + `MADV_WILLNEED`) so the OS page cache handles reads.
2. **Prefetch layer N+1 while computing layer N** (a dedicated I/O thread + ring buffer).
3. **Double buffering**: two buffers, async reads into one while computing on the other.
4. **Pinned (page-locked) host memory** so the H2D copy over PCIe does not block.
5. **The NVMe is a hard ceiling** — no algorithm reads disk faster than 2 GB/s.

**Dense vs MoE:** dense forces streaming *the whole layer*; MoE can stream **per expert**
(finer granularity ⇒ far less I/O). This is why 1.c/1.b matter more than 1.a on weak hardware.

---

## 1.b MoE inference on weak hardware

**Mechanism.** MoE has `N` experts per layer, but each token visits only `k` experts (top-k). Load
only those `k`.

```
needed VRAM/RAM = Σ (shared/attention, resident) + max(k experts) + router   (much less than Σ N experts)
```

| Model | Total | Active/token | Layers | Experts/layer | top-k | Q4_K_M (estimated) |
|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 15.7B | 2.4B | 27 | 64 + 2 shared | 6 | **~9.5–10.4 GB → FITS 11 GiB RAM** |
| Qwen3-30B-A3B | 30.5B | 3.3B | 48 | 128 | 8 | **~18.6 GB → does NOT fit ⇒ stream/Q3** |

**Four engineering pillars:**
1. **Expert offloading** — unused experts live in RAM/disk; only the router-selected expert goes to VRAM.
2. **Expert cache (LRU/LFU)** — keep "hot" experts in VRAM. A cache hit ⇒ zero I/O.
3. **Routing prediction / prefetch** — read the experts the *next layer* will pick, hiding latency.
   MoE is **layer-local**: layer n+1's experts depend on layer n's hidden state ⇒ predictable.
4. **Expert reuse** — tune the router to raise expert overlap across tokens (ReMoE).

**Algorithms in the literature (described, not independently verified):** FloE (frequency-based
offload), EdgeMoE (disk offload for edge), ReMoE (router that increases reuse), MoE-APEX (prefetch),
MergeMoE (merge rarely used experts), BigLittle-MoE (quantize experts by usage tier).

**Speed ceilings on this machine (derived):** decode is ~ memory-bandwidth-bound.
- DeepSeek-V2-Lite with experts **resident in RAM** (40 GB/s): 2.4B × 0.5 B ≈ **1.2 GB/token**
  → ceiling ≈ **~33 tok/s**; with experts in VRAM (192 GB/s) the ceiling is far higher.
- If experts are in RAM but a **cache miss reads from the NVMe** (2 GB/s) → **~1.7 tok/s**. A **~20×**
  difference ⇒ **the project's number-one goal is maximizing cache hit rate**, not "fast streaming".

> **Design conclusion:** pick **DeepSeek-V2-Lite** for Phase 0/1 (fits RAM, actually runnable);
> **Qwen3-30B-A3B** is a Phase-2 goal, framed as "prove streaming + expert cache can exceed RAM".

---

## 1.c CPU–GPU–NPU co-execution

**Mechanism.** The GPU does not wait for the CPU: **the GPU computes layer N while the CPU
reads/computes layer N+1**. Split the work by real-time bandwidth (FreeToken's q* policy): slow PCIe
⇒ push more layers to CPU; fast PCIe ⇒ shift back to GPU.

**Layer-wise scheduling:**
```
[CPU: prefetch layer N+1]  ║  [GPU: compute layer N]   →  swap buffers
```

**On this machine (measured):** there is a dGPU (NVIDIA) + iGPU (Intel Iris Xe), but **no unified API**
to split one model across both GPUs (unlike HeteroMosaic on mobile SoCs with shared memory).
⇒ The realistic design is **CPU + dGPU co-execution**; iGPU/NPU is reserved for the mobile roadmap
(NNAPI/Core ML).

---

## 1.d Quantization for mobile

| Technique | Bits | Use | Note |
|---|---|---|---|
| Block-wise quant (AirLLM) | 4-bit | cut disk reads | blocks are independent ⇒ mmap per block |
| AWQ | 4-bit | mobile | activation-aware, preserves quality |
| GPTQ | 4-bit | mobile | post-training, fast |
| QAT (Squat) | sub-8-bit | SLM | trained with quantization |
| Token-adaptive | 3–4-bit | mobile | per-token bit width |

**On this machine:** use **GGUF K-quants + imatrix** (available in the llama.cpp ecosystem). Consider:
- DeepSeek-V2-Lite: `Q4_K_M` (~10 GB, fits RAM) as default; `Q3_K_M` (~8 GB) if KV needs headroom.
- Qwen3-30B-A3B: `Q3_K_M` (~14.8 GB) is the lowest still acceptable, and still **too large** ⇒
  offloading is unavoidable.

---

## 1.e KV cache management

**Mechanism.** KV cache grows linearly with context → it eats all RAM/VRAM on weak hardware.
- **PagedAttention (vLLM):** split KV into pages, allocate on demand, reduce fragmentation.
- **Compression / quantization:** KV down to 8-bit or 4-bit (nearly free in quality terms).
- **Eviction:** drop unimportant KV (H2O, sliding window).
- **Semantic anchor (FreeToken):** store recurrent-state checkpoints to avoid recompute when context changes.

**On this machine:** default to `--cache-type-k q8_0 --cache-type-v q8_0` in llama.cpp; enable
eviction when context > 4k on a 10 GB model.

---

## 1.f Speculative decoding on weak hardware

**Mechanism.** A small draft model emits `γ` tokens; the target model verifies in one pass.

```
speedup ≈ (1 − α) / (1 − α·β)      α = draft acceptance rate, β = draft/target cost ratio
```

**SS-MoE:** use a smaller model (or a small expert) as the draft. **Draft-then-verify:** a small local
model drafts, the large model verifies.

**On this machine:** draft = a dense **Qwen2.5-0.5B Q4 (~0.4 GB)**; target = DeepSeek-V2-Lite. Both
must share the 11 GiB RAM budget ⇒ feasible, but only worth doing in **Phase 4**.

---

## 1.g Other model compression

- **Pruning** (structured/unstructured): drop rarely used heads/neurons/experts.
- **Low-rank (SVD)**: factorize W ≈ U·Σ·V to reduce parameters.
- **Distillation**: large model → an SLM that runs on mobile.
- **Weight sharing / n-gram embedding compression**: compress the embedding table (large vocab).

On this machine, prioritize **pruning rarely used experts** (alongside MergeMoE) over SVD: MoE is
already sparse, so cutting experts yields a direct memory win.

---

## 2. Summary: which technique buys what on the target machine

| Technique | Reduces | Increases | Cost | 1-month priority |
|---|---|---|---|---|
| MoE expert offloading + cache | RAM/VRAM footprint | cache hit | complex | ⭐⭐⭐ (core) |
| Routing prediction/prefetch | I/O latency | throughput | medium | ⭐⭐⭐ |
| Quantization Q4/Q3 + KV q8 | memory ~4× | — | low | ⭐⭐⭐ |
| Layer-wise streaming | RAM (beyond RAM) | — | disk I/O ceiling | ⭐⭐ (only when > RAM) |
| CPU–GPU co-execution | — | throughput 10–80%* | high | ⭐⭐ |
| Speculative decoding | — | 1.5–2.5×* | medium | ⭐ |
| Pruning/SVD/distill | memory | quality↓ | very high | ✗ (beyond 1 month) |

\* range **as published by FreeToken / the literature**, not measured on this machine.
