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

### Mixed-quant experts

On an MoE model the routed experts are most of the file and most of the
pressure on a small card. LFM2.5-8B-A1B Q4_K_M is 4.91 GB, of which 4.45 GB is
`*_exps` weights. Quantizing **only** those tensors to Q3_K (imatrix built from
6 chunks of `tools/eval/wikitext_sample.txt`) leaves attention, the KV cache
and the routing layer at Q4_K_M:

| Arm | File | Expert bytes | Expert layers on GPU | VRAM | Decode @64K |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 4.91 GB | 4.45 GB | 12/24 (`-ncmoe 12`) | 3494 MiB | 65.76 t/s |
| `ffn_.*_exps=Q3_K` | 3.82 GB | 3.33 GB | 17/24 (`-ncmoe 7`) | 3504 MiB | **78.05 t/s** |

**+18.7% decode at identical VRAM and context**, median of three requests on
each arm, ranges that do not overlap (65.75–66.27 against 76.97–78.50). The
planner needed no change: it read the smaller `expert_bytes` and spent the
freed 1.2 GiB on five more expert layers.

Quality is a wash, not a win: PPL 29.56 ± 1.88 → 28.69 ± 1.81 over 12 chunks of
a slice of `wikitext_sample.txt` that the imatrix did not see. The imatrix is
what makes that hold — without one, a Q3 requant of an already-Q4 model has
measured worse on this project's L1a arms.

```bash
vipera model optimize LFM2.5-8B-A1B-Q4_K_M.gguf --target-quant Q4_K_M \
  -f tools/eval/wikitext_sample.txt --tensor-type 'ffn_.*_exps=Q3_K'
vipera model add LFM2.5-8B-A1B-Q4K-expQ3K.gguf --name lfm25-expq3
vipera serve start -m lfm25-expq3 --profile auto --ctx 65536
```

### `lowmem` on a MoE model: "no expert in VRAM" is not "no GPU"

`profile: lowmem` means "do not put the weights in VRAM". On a MoE model that was
being read as "do not use the card at all", which threw away more than half the
decode rate: the routed experts are the bulk of the file, and the attention side
fits beside a 4 GB card's reserve. Same model, same context, one request per arm,
64 generated tokens:

LFM2.5-8B-A1B (Q4_K weights, Q3_K experts, 3.82 GB, 3.33 GB of it experts), GTX
1650 4 GB, `ctx 8192`, `-fa on`, q4_0 KV:

| placement | flags | decode | prefill |
|---|---|---:|---:|
| CPU only (what `lowmem` used to plan) | `-ngl 0` | 37.06 t/s | 43.8 t/s |
| experts in RAM, attention on GPU (what `lowmem` plans now) | `-ngl 24 -cmoe` | **49.88 t/s** (+35%) | 45.4 t/s |
| the MoE split `auto` plans | `-ngl 21 -ncmoe 6` | **69.33 t/s** (+87%) | 65.6 t/s |

Read the middle row as the profile's floor: it keeps every expert weight in system
RAM — the promise — and is still a third faster than giving up the card. The last
row is what `profile: auto` gets for the same model, and the reason `lowmem` should
be a memory choice, not a speed one. It also caps context at 2048 where `auto`
plans 8192, so the gap on a real workload is larger than the three columns say.

Through the gateway the same change reads: `vipera bench run lfm25-expq3 --ctx 8192`
went from **37.81 t/s at 0/24 layers** to **51.90 t/s at 24/24 layers** with the
`lowmem` config unchanged (the small difference from the table is the gateway's
measured ubatch and its own prefill).

### Per-tensor placement for dense models: the arithmetic said no, the clock said yes

`llama-server` b11827 accepts `-ot/--override-tensor` and `-ncffn/--n-cpu-ffn`,
which let a plan keep every attention block and the whole KV cache on the GPU
while leaving FFN blocks in system RAM — a finer axis than `-ngl`. This was
rejected here **on arithmetic** before any code was written, and the arithmetic
was wrong. The bytes, off the GGUF for Hermes-4-14B Q3_K_M (attention+
norms 1 193 MiB over 40 layers, FFN 4 856 MiB, token_embd+output 927 MiB):

| Context | Attention + all KV | Left for FFN | FFN layers on GPU | Today (`-ngl`) |
|---|---:|---:|---:|---:|
| 8K, KV q8_0 | 1 856 MiB | 1 874 MiB | 15 | **18** |
| 32K, KV q4_0 | 2 599 MiB | 1 131 MiB | 9 | 16 |
| 64K, KV q4_0 | 3 984 MiB | — | 0 | 18 at 32K, none at 64K |

Read as bytes, it loses at every context, and that is why it was dropped. The
missing term is what the CPU *does* with those bytes: a layer left on the CPU
runs attention over the whole window on four threads, and that cost grows with
the window, while an FFN block streams the same 131.5 MiB per token no matter
how long the history is. Trading 131.5 MiB of extra host traffic for the removal
of one CPU attention block is therefore a win, and the measurement is not close:

Hermes-4-14B Q3_K_M, `ctx 8192`, q8_0 KV, 3 750-token filled prompt:

| Placement | CPU streams | decode | prefill | VRAM |
|---|---:|---:|---:|---:|
| Classic layers `-ngl 18` | 3 562 MiB/token | 3.44 t/s | 37.15 | 3 708 MiB |
| Tensor split `-ngl 40 -ncffn 38` | 4 997 MiB/token | **4.89 t/s** (+42%) | 36.67 | 2 926 MiB |
| Tensor split planned by Vipera `-ncffn 35` | 4 599 MiB/token | **5.38 t/s** (+56%) | 37.15 | 2 926 MiB |

It also unlocks context the classic split cannot configure at all: at
`ctx 16384` the 18/40 plan dies allocating a 578 MiB KV buffer, while the tensor
split serves the same prompt at 4.99 t/s (`-ncffn 38`, 3 620 MiB resident).

| Prompt | Classic 18/40 | Tensor split |
|---|---:|---:|
| 177 tokens | **6.34 t/s** | 5.51 t/s |
| 1 840 tokens | 3.96 t/s | **5.22 t/s** |
| 3 750 tokens | 3.44 t/s | **4.89 t/s** |

The crossover sits between 177 and 1 840 tokens, which is why `planner.py` puts
a floor of 4096 on the *configured* window before emitting `-ncffn`: below it
the classic split is the better bet, and from 4K up the tensor split won every
run. `--load-mode none` (a hint the loader prints for tensor overrides) is
slower here — 4.37 against 4.89 t/s — so mmap stays. Graphs: 4.97 t/s with
capture on against 4.89 with it off, i.e. noise, so a tensor split gives up
capture (a plan that fills the budget cannot promise the 780 MiB of headroom
that measurement had).

### Placement strategy on a 4 GB card: what actually buys tokens per second

Every number here is the reference GTX 1650 with the same harness (one request,
48-96 generated tokens, `GGML_CUDA_DISABLE_GRAPHS=1` for hybrid splits), so the
rows are comparable to each other.

**1. Split by tensor type (dense).** +42 to +56% at a filled window, and it buys
context as well: see the section above. This is what the planner now emits for a
dense model that only partly fits, from `ctx 4096` up.

**2. MoE expert placement is the big lever — 10 to 20x, not 40%.**
LFM2.5-8B-A1B (24 blocks, 32 experts, 4 active, ~1B parameters per token):

| Plan | ctx | decode | prefill | VRAM |
|---|---:|---:|---:|---:|
| dense 14B, classic 18/40 (for scale) | 8 192 | 3.96 t/s | 39.5 | 3 708 MiB |
| `--cpu-moe` (every expert in RAM) | 32 768 | 41.22 t/s | 170.9 | **862 MiB** |
| `--n-cpu-moe 12` | 32 768 | 58.64 t/s | 199.1 | 2 594 MiB |
| `--n-cpu-moe 12` | 65 536 | 65.53 t/s | 201.9 | 2 888 MiB |
| `--n-cpu-moe 12`, q4_0 KV | 131 072 | 65.45 t/s | 202.3 | 3 104 MiB |
| `--n-cpu-moe 12`, 7 081-token prompt | 16 384 | 62.15 t/s | 212.9 | 2 476 MiB |
| `--n-cpu-moe 8`, q8_0 KV | 131 072 | load fails (compute buffer, 408 MiB) | | |

The mechanism is the same one the frontier section finds for dense models: what
matters is bytes streamed per token, and a 4-of-32 expert routing moves ~1B
parameters instead of 14B. Filling the window costs 5% (62.15 against 65.53), so
context is nearly free here — and 128K fits with q4_0 KV.

**3. The newest VRAM expert cache is a loss on this card.**
`--moe-cache-mib` (llama.cpp's `moe-expert-cache`) keeps a persistent pool of
hot experts in VRAM. On a card with a peer-to-peer/pinned copy path it reports
double-digit gains; here it makes decode **3x slower**, 41.22 → 13.96 t/s, with
512 MiB of cache, because every selected expert then crosses PCIe per token
while "experts on the CPU" crosses nothing. Keep expert weights entirely on one
side of the bus (`-ncmoe`), which is what the planner does.

**4. Speculative decoding: one arm works, one is blocked.**

| Arm | Result |
|---|---|
| `--spec-type ngram-simple` on a "rewrite this text" prompt | 4.03 → **6.94 t/s** (+72%), 50/50 drafts accepted, mean run 26 tokens |
| `--spec-type ngram-mod` on fresh prose | 4.12 vs 4.03 t/s — nothing to match, no drafts |
| Draft model Qwen2.5-0.5B at `-ngl 18` | fails to load: `cudaMalloc` 373 MiB |
| Draft model at `-ngl 15` (fits) | refused: "target and draft vocabs are not compatible"; the child then runs **without** speculation at 3.78 t/s, slower than the 18/40 plan it replaced |

Draft-model speculation needs a same-tokenizer draft *and* ~370 MiB of VRAM
(about 2.3 target layers) on a card that has neither to spare. The n-gram
drafter costs ~16 MB and no VRAM, which is why `spec_type` exposes only the
n-gram values — and why it stays opt-in: +72% where the answer copies the
prompt, nothing where it does not.

**5. Reuse the prefix; do not re-prefill it.** Same 1 840-token prompt sent
twice on one child with `cache_prompt`:

| Request | Prefill | Wall |
|---|---:|---:|
| first | 46 597 ms / 1 840 tokens | 70.6 s |
| second | **247 ms / 1 token** | 24.1 s |

For a long-context session this is the largest lever of all, because it removes
the cost rather than reducing it. `--cache-reuse` covers the case where the
prefix shifts instead of repeating.

**6. Things that do not help, with the numbers.**

- **KV quantization buys capacity, not speed.** q4_0 KV is what makes 128K
  context fit on the MoE model above, but it is not a decode lever: [measured
  elsewhere](https://github.com/ggml-org/llama.cpp/discussions/20969) it costs
  12% of decode at 24K and 37% at 110K to per-token dequantization.
- **A lower-bit quant of a dense model is slower here.** Hermes-4-14B IQ3_XXS
  (5.90 GB) at the largest split that loads (`-ngl 21`) decodes 2.92 t/s against
  3.96 for the larger Q3_K_M at 18/40: CPU-side dequantization costs more than
  the bytes saved. `-ngl 24` fails to load.
- **CUDA graphs are worth nothing here** and can kill a hybrid plan mid-request;
  the server backend turns them off for exactly that case.

**What the planner does with this.** The per-tensor bytes it needs are written
when a model is registered (`vipera model import` / `model add`), so a model
registered before this change keeps the classic split until it is registered
again — the planner cannot place by tensor type from a meta that lacks
`attn_bytes` and `ffn_layer_bytes`. A dense model that only partly fits gets
`n_cpu_ffn` (attention on the card, FFN filled from the last layer down) once the
window reaches 4096, at most one FFN block fewer than the bytes allow and never
zero; a MoE model gets `moe_cpu_layers`. The expert count is the one number the
planner cannot derive from bytes alone, because it depends on a card's real
overhead rather than on a formula: with `moe_probe` on (the default), the start
path tries one and then two expert layers more on the GPU than the fit and keeps
the most aggressive split that starts. On the reference card that is worth +6%:
the fit put 17/24 expert layers on the GPU and 69.43 t/s, the probe found 19/24
and 73.82 t/s on a 7 089-token prompt, at 32K with q8_0 KV.

### `auto` backend: the gateway vs its own child vs Ollama

Same GGUF, same `ctx 4096`, same 226-token prompt, 128 max tokens, temperature
0, streams read over HTTP, both engines warmed (identical prompt, so prefill is
cached). Final run on a cool card (42 °C at start, 2 MiB resident).

| Arm | qwen2.5-0.5B Q4_K_M (24/24 layers) | Hermes-4-14B Q3_K_M (18/40 layers) |
|---|---:|---:|
| Vipera, `backend: auto` → `server` (HTTP) | **197.6 t/s** | **5.90 t/s** |
| the same `llama-server` child, called directly | 198.4 t/s | — |
| Ollama 0.40.8, defaults | 203.6 t/s | 5.56 t/s |

The 0.5B row is six interleaved rounds; the 14B alternates four blocks
(Vipera, Ollama, Vipera, Ollama, three generations each) because the two
combined need ~6 GB of a 4 GB card and cannot share it.

What the numbers say:

- **The gateway costs nothing measurable.** 197.6 t/s through Vipera's HTTP
  stack against 198.4 calling the same child directly, on back-to-back runs.
  It was ~10% before the SSE relay replaced a thread round trip per frame with
  one reader thread per stream, and the `read1` batching that came with it.
- **The last 3% against Ollama is the KV cache type, not the build.** q4_0 KV
  costs 2.0% of decode here (198.4 → 202.5 with f16), and it is a deliberate
  memory trade. An earlier revision of this file blamed Ollama's `llama.cpp`
  build for a 5.7% lead; that was wrong — the lead was a bug in Vipera's
  `-ngl`, fixed after 1.3.0 and dissected in the next section.
- **On the hybrid model Vipera wins by 6%, not 31%.** 5.90 against 5.56 t/s:
  the planner put 18/40 layers on the card (3.42 GB) where Ollama's `--fit`
  stops at 14/40 (2.54 GB) to leave its 1 GiB margin. The earlier "+31%" in
  this file was measured with **both engines resident** — Ollama had planned
  2.54 GB for a card that no longer had it and fell back to CPU for layers it
  thought were on the GPU. That is a real failure mode, but it is not a
  throughput comparison; do not quote it.
- **Two Vipera defaults were costing speed and are now opt-in.**
  `--spec-type ngram-map-k` measured 2.1% *slower* on fresh prose (204.4 →
  200.2 interleaved) and `repeat_penalty: 1.1` on every request cost 10%
  (167.4 → 150.6). Neither is on by default now; both are one config key away
  for the workloads that do benefit (chapter 3 says +37% for the drafter when
  the answer rewrites the prompt).

Absolute numbers move with the card: the same Ollama arm read 204 t/s cool and
165 t/s after an hour of benchmarking, so every comparison above shares a round
or an alternating block. Run the arms on your own box before quoting them.

### Two binaries, four flag sets: where the Ollama gap came from

Two `llama-server` binaries were on the box at once — Vipera's (build b11827,
upstream `970b51ce7`) and the one Ollama 0.40.8 spawns
(`~/.local/lib/ollama/llama-server`, fork commit `f174553e1`; run by hand it
needs `GGML_BACKEND_PATH` pointed at `cuda_v13/libggml-cuda.so` or it silently
decodes on the CPU). That makes build and flags separable instead of guessable.
Same GGUF, prompt and measurement, three interleaved rounds:

| Binary | Flags | Offloaded | Decode |
|---|---|---|---:|
| Vipera b11827 | Vipera's planner | 24/25 | 186.7 t/s |
| Ollama f174553e1 | Vipera's planner | 24/25 | 186.6 t/s |
| Vipera b11827 | Ollama's | 25/25 | 203.7 t/s |
| Ollama f174553e1 | Ollama's | 25/25 | 203.3 t/s |

Read down the columns: swapping the **binary** moves throughput by −0.1%,
swapping the **flags** by +9%. The build was never the difference.

The flags differed in one thing that matters. `llama.cpp` counts the output
layer as one group past the GGUF's `block_count`, so `-ngl 24` on a 24-block
model offloads **24 of 25** groups and leaves the output layer on the CPU.
Vipera passed `n_layers` and its banner said `24/24` either way; Ollama passes
no `-ngl` at all, so it gets everything. With `-ngl 25`, one flag removed at a
time:

| Arm | Decode | Δ |
|---|---:|---:|
| all of Vipera's flags | 198.4 t/s | — |
| − `-ctk`/`-ctv q4_0` (f16 KV) | 202.5 t/s | **+2.0%** |
| − `-lm mlock` | 198.6 t/s | +0.1% |
| − `-fa on` (auto) | 198.0 t/s | −0.2% |
| − `-t 4` | 199.0 t/s | +0.3% |
| − `-fit off` | 197.5 t/s | −0.5% |

Everything except the KV dtype is inside noise. The 14B row above is a hybrid
plan (18 of 40 layers), which never triggered the bug, so it is unaffected and
was not re-measured.

### Concurrent requests: slots vs a queue

qwen2.5-0.5B Q4_K_M, `ctx 4096`, four streaming requests of 128 tokens issued at
the same time, aggregate = tokens / wall clock:

| Arm | 1 request | 4 concurrent (aggregate) |
|---|---:|---:|
| Vipera, `parallel: 1` (serialised) | 173 t/s | 178 t/s |
| **Vipera, `parallel: 4`** | 175 t/s | **229 t/s** (+29%) |
| Ollama 0.40.8 (one slot) | 199 t/s | 202 t/s |

One request is unaffected by the slot count; the gain is what an agent loop
with four tools in flight actually experiences — **+29% wall-clock work done**,
not the 2× an earlier, throttled run suggested: continuous batching shares the
same memory bandwidth, so four streams together decode at about a quarter of
one stream's rate each. Slots also cost KV memory (`-c ctx × parallel`), which
is why the default stays at 1 on a 4 GB card: four slots of a hybrid 14B do not
fit, and the load would fall back a rung.

### Long prompts on a hybrid split

Hermes-4-14B Q3_K_M, 18/40 layers, `ctx 4096`, a 24.6 KB log dump:

| Request | Result |
|---|---|
| a trimmed 3 978-token prompt | 200; 30.9 s of prefill for the uncached 1 020 tokens (33 t/s), 32 tokens out in 38.4 s total |
| same, before the planner charged `-ub` | child aborted: `CUDA error: out of memory` |
| same, before shaping measured tokens | 400 `exceed_context_size_error` |

Decode after the load sits at 5.90 t/s (18/40 layers) against Ollama's 5.56 t/s
(14/40), each measured alone on the same card — see the section above for why
those two must not be run at the same time.

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
- Placement is arithmetic plus one probe. A dense model that only partly fits
  keeps every attention block on the card and streams FFN blocks from
  `ctx 4096` up (`n_cpu_ffn`); MoE models place routed experts on every profile.
  The only number that is verified by loading instead of computed is the MoE
  expert count (`moe_probe`). Both axes, with the numbers, are in
  [placement strategy](#placement-strategy-on-a-4-gb-card-what-actually-buys-tokens-per-second).
