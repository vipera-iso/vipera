# vipera

**An optimization layer for local LLM model files, so they run on weak hardware** — starting with
**personal computers**, then **phones**.

`vipera` is not another runtime. It is **a layer between the model file and the runtime**: it takes an
artifact (GGUF/safetensors), **rewrites it into an optimized artifact**, and emits a *manifest* that
tells the runtime what lives where (VRAM/RAM/disk), in what load order, and which experts are "hot".

Inspired by [FreeToken](https://github.com/FlashML-org/FreeToken) (MoE serving on consumer hardware)
and [AirLLM](https://github.com/lyogavin/airllm) (layer-wise streaming) — but both assume plenty of
GPU/RAM. `vipera` targets the hard spot: **the model fits in RAM yet overflows VRAM, on a slow disk**.

> Status: **stage 2 implemented** — stage 1 (analyzer + residency planner) plus a per-layer
> architecture model and a bandwidth/cache-aware streaming scheduler that models decode speed and
> spits out a residency manifest. Nothing touches tensor data or executes a model yet. Bold numbers
> are measured on the target machine; FreeToken/AirLLM figures are **as published by them**;
> scheduler outputs are **modelled**, and say so.

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

## Quick start

Pure stdlib, Python >= 3.11 (uses `tomllib`). No dependencies to install.

```bash
# inventory a model without reading its tensor data
python3 -m vipera analyze models/deepseek-v2-lite-Q4_K_M.gguf
python3 -m vipera analyze models/deepseek-v2-lite-Q4_K_M.gguf --json

# plan weight placement for the measured target, and write the manifest
python3 -m vipera plan models/deepseek-v2-lite-Q4_K_M.gguf \
    --target hw/gtx1650.toml --out out/
# -> out/deepseek-v2-lite-Q4_K_M.vipera.json

# model a huge MoE against this machine and see whether it can run at all
python3 -m vipera arch     arch/glm-5.3-flash.toml
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --out out/
python3 -m vipera sweep    arch/glm-5.3-flash.toml --target hw/gtx1650.toml

# the LLM-PC-OPT instance: dense Llama-3-70B on 8 GB VRAM / 32 GB RAM / 3 GB/s SSD
python3 -m vipera schedule arch/llama3-70b.toml --target hw/llama70b-pc.toml --quant q4-experts --context 4096

# the LOCAL-LLM-16GB instance: 70B dense on a 16 GB unified-memory laptop
python3 -m vipera score arch/llama3-70b.toml --target hw/laptop16-igpu.toml --context 32768 --quality 99 --power 65

# tests
python3 -m unittest discover -s tests -t .
```

## Benchmark

`llama-bench` reports t/s but not what a weak machine actually pays. `bench/` wraps it and
records peak RSS, peak VRAM and disk read bytes as well.

```bash
python3 -m bench.bench \
    --bin ~/dev/llama.cpp/build/bin/llama-bench \
    --model models/deepseek-v2-lite-Q4_K_M.gguf \
    --ngl 0 --threads 4 --json bench/results/baseline.json
```

Harness validation only — a 260K-parameter toy model, CPU-only, 4 threads:

```
ok True · exit 0 · peak RSS 1.2 MiB · disk read 0.0 MiB
  pp128   92234.97 t/s
  tg64    14775.72 t/s
```

Those confirm the harness runs; they say nothing about the target MoE, which still needs a
real model run.

What it computes today:

| Command | Output |
|---|---|
| `analyze` | layers, expert count, and bytes split into resident / router / shared-expert / "other" / routable-expert buckets |
| `plan` | a `*.vipera.json` residency manifest: how many bytes go to VRAM, RAM, and disk spill, plus warnings |

The planner is **size-driven** for now: it cannot rank *which* experts are hot until routing traces
from Phase 2 exist, and it says so in `placement.expert_order_pending_trace`.

## Documents

- [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) — **can GLM-5.3-Flash (321B/18B) run here?** The
  modelled verdict, per-quant, and the ceilings that make it infeasible.
- [`docs/OPTIMIZER.md`](docs/OPTIMIZER.md) — **the model-file optimization layer** (vipera's core
  positioning).
- [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — algorithm analysis (streaming, MoE offloading,
  co-execution, quantization, KV cache, speculative decoding, model compression).
- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, FreeToken/AirLLM comparison, 1-month roadmap,
  tech stack, KPIs, risks, **7-day plan**.

## Stage 2 — the layer scheduler

`vipera/arch.py` models a hybrid MoE or a dense model per layer (KDA/MLA/full-GQA attention, sparse
experts or a dense FFN, mHC residual, MTP and vision), and `vipera/schedule.py` places every layer's
weights across VRAM/RAM/disk and simulates decode to answer one question: **how many bytes per
token, from which tier, and how fast?**

The headline exercise is a 321B-A18B model on the target laptop. Modelled result: **~0.65 tokens/s**
(Q4), I/O-bound. `vipera sweep` then models the layer-level optimization families (quantization,
expert pruning, layer pruning, KV compression, speculative/MTP) to show which one actually changes
the verdict — expert pruning, because the expert pool is 30× the cache. See
[`docs/FEASIBILITY.md`](docs/FEASIBILITY.md).

## Scope (one month, zero budget)

- ✅ PC-first (no Android device available to test ⇒ "phone" stays at the design/portability level).
- ✅ GGUF + llama.cpp, Python for the optimizer layer.
- ❌ Not this month: ONNX/ExecuTorch/LiteRT runtimes, gRPC, distillation.
