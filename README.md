# vipera

**An optimization layer for local LLM model files, so they run on weak hardware** — starting with
**personal computers**, then **phones**.

`vipera` is not another runtime. It is **a layer between the model file and the runtime**: it takes an
artifact (GGUF/safetensors), **rewrites it into an optimized artifact**, and emits a *manifest* that
tells the runtime what lives where (VRAM/RAM/disk), in what load order, and which experts are "hot".

Inspired by [FreeToken](https://github.com/FlashML-org/FreeToken) (MoE serving on consumer hardware)
and [AirLLM](https://github.com/lyogavin/airllm) (layer-wise streaming) — but both assume plenty of
GPU/RAM. `vipera` targets the hard spot: **the model fits in RAM yet overflows VRAM, on a slow disk**.

> Status: **design phase (Phase 0)**. No runnable code yet. Bold numbers are measured on the target
> machine; FreeToken/AirLLM figures are **as published by them**, not re-verified here.

---

## The idea: optimize at the **file** layer, not just at runtime

Runtime tricks (expert cache, prefetch) only get you to the I/O ceiling. `vipera` goes one step
further: **rearrange the model file itself** so a weak runtime reads less and reads sequentially.

| File-layer transform | Input | Output | Benefit on weak hardware |
|---|---|---|---|
| **Expert hot/cold reorder** | routing traces | re-laid-out file | hot experts become contiguous ⇒ one sequential read |
| **Residency manifest** | model + hardware | `*.vipera.json` | runtime knows the VRAM/RAM/disk split instead of guessing |
| **Block-wise quant** | safetensors | 4-bit block GGUF | stream block by block via `mmap` |
| **Expert pruning/merge** | usage frequency | fewer experts | smaller ⇒ fits RAM |
| **KV placement hints** | config | manifest | where KV lives, how many bits |
| **Chunk alignment** | model | page/disk aligned | avoids over-reading on a DRAM-less NVMe |

## Target machine (measured 2026-10-12)

| | |
|---|---|
| CPU | Intel i5-11300H — 4C/8T, up to 4.4 GHz |
| RAM | 15 GiB, ~10 GiB available, **no swap** |
| GPU | NVIDIA **GTX 1650 Max-Q, 4 GB VRAM** (compute 7.5) |
| iGPU | Intel Iris Xe (Tiger Lake) |
| Disk | WD SN530 NVMe **DRAM-less**, 404 GB free |
| OS | Linux |

## Target models

| Phase | Model | Total / active | Q4_K_M (estimated) | Fits 11 GiB RAM? |
|---|---|---|---|---|
| P0–P2 | **DeepSeek-V2-Lite** | 15.7B / 2.4B | ~9.5–10.4 GB | ✅ |
| stretch | Qwen3-30B-A3B | 30.5B / 3.3B | ~18.6 GB | ❌ (needs streaming) |

**The decisive number:** for DeepSeek-V2-Lite, experts resident in RAM imply a ceiling of
**~33 tok/s**; but a cache miss reading from the NVMe (DRAM-less, ~2 GB/s) drops it to **~1.7 tok/s**.
A **~20×** gap ⇒ vipera's primary goal is **fewer disk reads for experts**.

## Documents

- [`docs/OPTIMIZER.md`](docs/OPTIMIZER.md) — **the model-file optimization layer** (vipera's core
  positioning).
- [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — algorithm analysis (streaming, MoE offloading,
  co-execution, quantization, KV cache, speculative decoding, model compression).
- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, FreeToken/AirLLM comparison, 1-month roadmap,
  tech stack, KPIs, risks, **7-day plan**.

## Scope (one month, zero budget)

- ✅ PC-first (no Android device available to test ⇒ "phone" stays at the design/portability level).
- ✅ GGUF + llama.cpp, Python for the optimizer layer.
- ❌ Not this month: ONNX/ExecuTorch/LiteRT runtimes, gRPC, distillation.
