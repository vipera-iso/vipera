---
title: Benchmarks
description: Throughput and memory measurements on the reference machine, and how to reproduce them.
---

# Benchmarks

Vipera measures what the gateway actually produces, on the machine it runs on.
The numbers below come from one reference box; `vipera bench run` produces the
numbers for yours, which is the only set that matters for your deployment.

## When Vipera wins, and when it does not

Vipera optimizes for **unknown hardware**, not for peak tokens/second on a
known-good flag set. That has a direct consequence for where it helps:

| Situation | What Vipera does |
|---|---|
| The model fits in RAM and you already know the right `llama.cpp` flags | Nothing to plan. Throughput matches `llama.cpp` defaults plus the load flags (`mlock`, `n_batch`), roughly 1.4× on the reference box. |
| The model spills VRAM or RAM | Plans per-layer placement and KV budget instead of failing at load. This is the case the planner exists for. |
| Hardware is heterogeneous (small GPU, large CPU) | Picks a hybrid split from measured VRAM free, then falls back in stages if the split does not load. |
| You do not want to memorize forty flags | Emits `-ngl`, KV type, `n_batch`, `mlock`, and KQV offload from a hardware profile. |

## Reference machine

| Component | Value |
|---|---|
| CPU | Intel i5-11300H — 4C / 8T |
| RAM | 15 GiB, ~10 GiB free, no swap |
| GPU | NVIDIA GTX 1650 Max-Q, 4 GB VRAM |
| Model | DeepSeek-V2-Lite Q4_K_M (9.65 GB, MoE 15.7B total / 2.4B active) |
| Runtime | `llama.cpp` CPU build, `-ngl 0` |

## Results

| Config | Gen t/s | Prompt t/s | Peak RSS |
|---|---:|---:|---:|
| CPU only, mmap | 16.0 | 44 | 12.6 GiB |
| CPU only, `-lm mlock` | 22.9 | — | ~10 GiB |
| CUDA offload, experts on CPU | 6.8 | — | — |
| Vulkan offload | 3.4 | — | — |

Reading the table: placement changed throughput on this box only because the
model already fit in RAM. The load-flag layer (`mlock`) did the work. A second
reference run on a 14B dense model that does *not* fit shows the opposite
result: the same 14B model measured 0.1-0.2 t/s CPU-only at 64K context, and a
10-layer hybrid split reached 4.0 t/s at 8K. Treat placement as the lever for
the `spills` case and load flags as the lever for the `fits` case.

### 14B dense on a 4 GB card (asymmetric hardware)

| Config | GPU layers | TTFT | Decode |
|---|---|---:|---:|
| 64K context, q4_0 KV, flash attention on | 9/40 | 5.0 s | 0.13 t/s |
| 8K context, q8_0 KV, flash attention on | 11/40 | 4.8 s | 2.86 t/s |
| 8K context, f16 KV, flash attention off | 10/40 | 3.2 s | 4.00 t/s |

The last row is the useful one for interactive work: with f16 KV at 8K the KV
cache fits in RAM, flash attention stays off, and the CPU-side layers run the
normal attention path. Long context is the limit — at 16K this model does not
create a GPU context on a 4 GB card at all and the engine falls back to CPU.

## Method

```bash
vipera bench run <model> --ctx 8192,16384 --max-tokens 64
```

- **TTFT** is wall-clock time from request to first streamed token; it includes
  prompt prefill, which is why it scales with context size.
- **Prefill t/s** is `prompt_tokens / TTFT`. It is approximate: the reference
  prompt is short, so the measurement is dominated by fixed costs.
- **Gen t/s** is measured after the first token, so it excludes prefill.
- **Peak RSS** is the resident set after load, sampled with `psutil`.
- Runs load one model at a time on an otherwise idle machine. Close browsers
  and other model servers first; VRAM free at plan time is what the planner
  budgets from.

## Caveats

- One machine, one model family. Do not treat these as leaderboard numbers.
- `llama.cpp` build and flags move throughput more than the gateway layer does.
- Prompt-prefix reuse depends on `llama-cpp-python` 0.3.36+ and on the client
  resending an identical prefix; a cold process always pays full prefill.
- MoE models are placed by the same per-layer arithmetic as dense models;
  Vipera does not emit `--n-cpu-moe`-style expert placement flags yet, so an
  expert-only GPU split is not planned for you.
