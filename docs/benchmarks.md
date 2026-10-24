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
model already fit in RAM. The load-flag layer (`mlock`) did the work — **if the
run was made with the memory-lock limit raised**: on this box `ulimit -l` is
8 MiB (soft = hard) and a non-root process cannot raise it, so `-lm mlock` and
`use_mlock=True` both log `failed to mlock … Cannot allocate memory` and fall
back to plain mmap. Re-measure with `RLIMIT_MEMLOCK` unlimited (root:
`/etc/security/limits.conf`, or `LimitMEMLOCK=infinity` for the service) before
quoting an mlock gain; see Caveats. A second
reference run on a 14B dense model that does *not* fit shows the opposite
result: the same 14B model measured 0.1-0.2 t/s CPU-only at 64K context, and a
10-layer hybrid split reached 4.0 t/s at 8K. Treat placement as the lever for
the `spills` case and load flags as the lever for the `fits` case.

### 14B dense on a 4 GB card (asymmetric hardware)

Hermes-4-14B, CUDA `llama-cpp-python`, 45-token prompt, 64 tokens generated,
median of 3, `vipera bench run --ctx 8192`:

| Model | Config | GPU layers | TTFT | Decode |
|---|---|---:|---:|---:|
| Q3_K_M + imatrix (6.8 GiB) | 8K, `q8_0` KV, FA on | **17/40** | 2.7 s | **6.57 t/s** |
| Q3_K_M + imatrix | 16K, `q8_0` KV, FA on | 14/40 | 2.8 s | 6.20 t/s |
| Q4_K_M (8.4 GiB) | 8K, `q8_0` KV, FA on | 14/40 | 2.76 s | 6.18–6.25 t/s |
| Q4_K_M | 16K, `q8_0` KV, FA on | 13/40 | 2.83 s | 5.48 t/s |
| Q4_K_M | 64K, q4_0 KV, FA on | 9/40 | 5.0 s | 0.13 t/s |

The Q3 build (`q8src-imbf-q3km.gguf`) is the better default: its importance
matrix keeps perplexity inside the eval-set error of Q4 (see the quantization
section), it is 1.6 GiB smaller, and its smaller per-layer weights put three
more layers on the card.

8K is the interactive case and `q8_0` is the KV type to use there: it needs
flash attention, which is also what makes 14 layers fit instead of 10 (the
non-FA path materializes a KQ matrix per batch). Long context is the wall — at
64K this model cannot be made fast on a 4 GB card regardless of flags.

Three measured changes moved the 8K row from **4.55 t/s at 10/40** to **6.30
t/s at 14/40**. That pair is measured as an interleaved A/B in one process
(alternating old/new, three runs each, median reported) because this box drifts
±10% between batches: absolute `bench run` medians for the new build ranged
5.35–6.25 t/s over the session. The change set:

1. `cpu_threads` is one per physical core instead of `cores - 1`.
2. The VRAM reserve is calibrated (0.3 GB on this card instead of a flat
   1.3 GB), so four more layers run on the GPU.
3. A plan that does not fit now walks the offload down (`14 → 13 → 12 → 7`)
   instead of dropping straight to CPU-only.

Numbers above are from the CUDA virtualenv
(`~/.venvs/vipera`, `vipera system doctor` → `gpu offload: compiled in`). The
older rows in this section were produced in this environment's CPU-only
sibling and reproduce differently; see the caveats. Raw runs, including the
A/B that isolates each lever, are in
[tools/gpu_results/REPORT.md §G](../tools/gpu_results/REPORT.md).

### Quantization: what an imatrix buys at Q4 and Q3

Hermes-4-14B, requantized from a **Q8_0** source (not from Q4_K_M: `llama-quantize`
copies a tensor verbatim when source and target types match, so an imatrix can
never reach the weights that way). Importance matrix built from BF16. Same
machine as above, CPU-only, `ctx 4096`, ~470-token prompt, median of 3 runs:

| Build | Size | Load | TTFT | Decode | PPL (WikiText, 12k tok) |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 9.00 GB | 8.0 s | 10.1 s | 3.73 t/s | 7.0056 |
| Q3_K_M + imatrix | 7.32 GB | 6.0 s | 7.4 s | 3.85 t/s | 6.9805 |
| Δ | **−18.7%** | **−25%** | **−27%** | +3% (noise) | −0.025 = **0.11σ** |

Read the last column carefully: the two builds are **equivalent within the
eval-set error, not one better than the other**. The claim this supports is
"Q3 with an importance matrix does not lose quality while being 19% smaller and
faster to load", which is what makes it worth shipping on a RAM-tight box.
Decode does not improve because Q3_K_M dequant is more expensive per weight.

The imatrix effect itself was only measurable at Q3: −5.2% PPL (−0.381, ~1.6σ)
against no-imatrix Q3, versus −0.96% at Q4 (0.3σ, indistinguishable from noise).
Raw runs and the per-tensor audit (280/443 tensors changed while types stayed
identical) live in [tools/l1a2_results/REPORT.md](../tools/l1a2_results/REPORT.md).

### KV cache dtype: what it costs in speed

qwen2.5-0.5B, `ctx 4096`, 64 generated tokens. Every row is a fresh engine, so
`load` is included:

| KV dtype | Flash attention | Decode t/s | vs f16 |
|---|---|---:|---:|
| f16 | off | 67.2 | — |
| q8_0 | on (required) | 65.0 | −3% |
| q5_1 | on | 57.5 | −14% |
| iq4_nl | on | 55.1 | −18% |

`q8_0` is the sweet spot: half the KV memory for ~3% decode. The 5- and 4-bit
cache types cost real throughput, and at those widths the model also started
ending generation earlier on the same prompt — treat them as a last resort when
KV size, not speed, is what blocks the context.

KV sizes themselves are **planned, not measured**: the planner's formula
(`2 × kv_layers × kv_heads × head_dim × ctx × bytes_per_elem`) gives 5.00 GB for
Hermes-4 at 32K with f16 KV and 2.50 GB with q8_0. The formula is covered by
`tests/test_planner.py`; the tables above are what was actually run.

### Optimization layer: shaping at the context edge

qwen2.5-0.5B Q4_K_M, `ctx 2048`, one request carrying 38 messages / 25.2 KB of
history (4,727 tokens by llama.cpp's own count) and 16 generated tokens:

| Arm | Result | TTFT |
|---|---|---:|
| `shaping: false` | `Requested tokens (4727) exceed context window of 2048` | — |
| `shaping: true`, first request | 200; 29 messages dropped, 9 left | 3.53 s |
| `shaping: true`, same request again | 200; `/health` `prefix_hits: 1` | 0.024 s |
| Fits without trimming (`ctx 16384`) | 200; full prompt reaches llama.cpp | 43.6 s |

The trimmed request cost 3.5 s to first token; the same history at a context
large enough to hold it cost 43.6 s. The layer's own prompt count is the
conservative 3-chars/token estimate (1,882 tokens for the trimmed request), so
treat it as a budget, not a tokenizer. The repeat request shows the other half:
the prefix was stable, so the backend reused its KV cache and only paid for the
new turn.

Tuning is the load-time half. `vipera bench run hermes-4-14b-q3km-imatrix
--ctx 8192 --max-tokens 32` measured 18/40 layers at 4.90 t/s and wrote the
profile to `tuning.json`; the planner's formula said 17 at the same 3.64 GB of
free VRAM. With 1.5 GB less free, the formula said 9 and `apply_tuning` kept 9:
a split measured with more room is not replayed onto a card that no longer has
it, because a failed load costs more than the extra layer.

### `auto` backend: the gateway vs its own child vs Ollama

Same GGUF, same `ctx 4096`, same 226-token prompt, 128 max tokens, temperature
0, streams read over HTTP. The three arms run **interleaved** (one round hits
all of them) because sequential runs on this box drifted up to 16% with the
fan. Medians of 5 rounds for the 0.5B and 3 for the 14B, both fully warmed
(identical prompt, so prefill is cached).

| Arm | qwen2.5-0.5B Q4_K_M (24/24 layers) | Hermes-4-14B Q3_K_M (16/40 layers) |
|---|---:|---:|
| Vipera, `backend: auto` → `server` (HTTP) | 166.2 t/s | 5.99 t/s |
| the same `llama-server` child, called directly | 169.1 t/s | 6.02 t/s |
| Ollama 0.40.8, defaults | 165.6 t/s | 4.56 t/s |

Absolute numbers drift with the card's thermal state (the same Ollama arm read
205 t/s when the box was cool and 165 t/s an hour of benchmarks later), which is
why every round here hits all three arms. What the comparison supports:

- **The gateway costs ~1%, not 10%.** Before the SSE relay it *was* ~10% at
  this model's 5 ms/token; one reader thread per stream instead of a thread
  round trip per frame removed it. On the 14B it was already ~0.5%.
- **Two Vipera defaults were the rest of the gap, and both were wrong for an
  OpenAI-shaped API.** `--spec-type ngram-map-k` cost 2.1% on fresh prose
  (204.4 → 200.2 t/s, interleaved) and is now opt-in; injecting
  `repeat_penalty: 1.1` into every request cost **10%** (167.4 → 150.6 t/s) and
  is now left to llama.cpp's own 1.0. Ollama charges neither.
- **Vipera's planner still wins where the model does not fit**: 16/40 layers at
  3.42 GB against Ollama's 14/40 at 2.67 GB, and a 14B decoded from system RAM
  is bandwidth-bound, so each extra GPU layer pays: **+31%**. Ollama closes it
  with `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q4_0` and
  `num_gpu=18` — three settings a user has to know, none on by default.

Both numbers are for one box. Run the same three arms on yours before
concluding anything about your hardware.

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
- **`mlock` does not buy throughput here, and the limit is not the reason.**
  `ulimit -l` is 8 MiB (soft = hard), so every locked-memory path warns and
  runs as mmap. Raising it (`ulimit -l unlimited` as root, 14B/Q4, 8K, median-3)
  measured 6.26 t/s against 6.25 t/s without — no change beyond noise. The
  earlier 16.0 → 22.9 t/s row attributed to `mlock` did not come from a working
  lock on this box; do not quote `mlock` as a lever until it is re-measured
  with the limit raised and interleaved against the control.
- `llama.cpp` build and flags move throughput more than the gateway layer does.
- Prompt-prefix reuse depends on `llama-cpp-python` 0.3.36+ and on the client
  resending an identical prefix; a cold process always pays full prefill.
- MoE models are placed by the same per-layer arithmetic as dense models;
  Vipera does not emit `--n-cpu-moe`-style expert placement flags yet, so an
  expert-only GPU split is not planned for you.
