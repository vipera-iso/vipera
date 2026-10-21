# GPU sweep report — Hermes-4-14B on GTX 1650 (4 GB), 2026-10-21

Answers one question: **how much does GPU offload actually buy on this box, and
where is the ceiling?** Everything here is measured, nothing is extrapolated.

- **Env (serving):** `~/.venvs/vipera/bin/python` (3.12, `llama-cpp-python` with
  CUDA — `llama_supports_gpu_offload() == True`, `vipera system doctor` PASS).
- **HW:** i5-11300H 4C/8T, 15 GB RAM (no swap), GTX 1650 4 GB
  (`nvidia-smi` reports 3730 MiB total, ~1114 MiB already used by the CUDA
  context at `-ngl 0`).
- **Tool:** `tools/sweep_gpu.py` (fresh engine per row, samples peak VRAM/RSS,
  refuses to run without a GPU backend).
- **Method:** ~470-token prompt, `temperature=0`, 64 generated tokens, decode t/s
  measured after the first token. `n` is given per number: **median-3 where the
  claim needs it, `n=1` flagged** — single runs are labelled, not averaged into
  confidence they do not have.
- Raw JSON + logs: this directory. Reproduce with the commands in each section.

---

## TL;DR

| # | Finding | Evidence |
|---|---|---|
| 1 | **GPU gain is real but modest, not 2–3×.** Best sweep row 6.00 t/s vs 3.95 CPU-only = **+52%**; Q4 tops out at 4.97 (+25%) | Table A/B/C, median-3 |
| 2 | **The shipped pipeline already does 5.16 t/s** (Q4, 8K, `q8_0`, FA on, 10/40 layers, 2.90 s TTFT) — 29% above the 4.00 t/s currently in `docs/benchmarks.md` | Table F, median-3 |
| 3 | **The published 4.00 t/s reproduces only in the binding that has no GPU.** Same command in the CPU-only venv prints `10/40` and yields 3.36 / 4.07 t/s; in the CUDA venv the same command yields 5.14–5.17 | Table F, n=2 vs n=3 |
| 4 | **`threads=4` beats the planner's `threads=3` by ~20%** on both paths (CPU-only 4.78 vs 3.95; GPU 6.00 vs 5.05) — both median-3 | Table C |
| 5 | **The engine's CPU fallback is the worst outcome, not a safety net.** `-ngl 12` on Q4 falls back to **0 layers** → 3.90 t/s, *slower than never trying*; `-ngl 14`/`16` fail outright with RSS ballooning to 40–55 GB | Table A/B |
| 6 | **Planner is calibrated to ±1 layer on Q4 (picks 10, max is 10) but conservative by 3 on Q3** (picks 10, max is 13) and by 4 at 16K (picks 8, 12 loads) | Tables B, E |
| 7 | **16K context is free on Q3 + `q8_0`:** 4.87 t/s @16K vs 4.86 @8K — target T4 (≥4.5) met | Table E, n=1 |
| 8 | **`mlock` does nothing on this box:** `ulimit -l` = 8 MiB (soft = hard), unraisable without root; both `-lm mlock` and `use_mlock=True` warn `failed to mlock … Cannot allocate memory` and run as mmap | see Caveats |

---

## A. `-ngl` sweep — Q4_K_M (9.00 GB), 8K, `kv f16`, `threads 3`

```bash
~/.venvs/vipera/bin/python tools/sweep_gpu.py \
  --model NousResearch_Hermes-4-14B-Q4_K_M-2 --ctx 8192 --kv f16 \
  --ngl 0 4 8 10 12 14 16 --max-tokens 64 \
  --json tools/gpu_results/hermes_q4_8k_f16.json
```

| `-ngl` | actual | TTFT s | decode t/s | peak VRAM MiB | peak RSS MB | note |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 5.29 | 4.00 | 1114 | 10407 | CPU-only baseline (see A1) |
| 4 | 4 | 5.12 | 4.40 | 2126 | 9160 | |
| 8 | 8 | 4.93 | 4.75 | 3034 | 9158 | |
| **10** | **10** | 4.85 | **4.97** | **3476** | 9165 | **max that loads** |
| 12 | **0** | 6.03 | 3.90 | 3476 | 16492 | ⚠ engine **fell back to CPU-only** |
| 14 | — | — | — | 3164 | 39750 | ✗ `Failed to create llama_context` |
| 16 | — | — | — | 3162 | 54978 | ✗ same, RSS at 55 GB |

`n=1` per row (the trend is monotone and the run-to-run spread at fixed config
is <1%, Table A1/C).

**A1 — CPU-only baseline, `-ngl 0`, `threads 3`, median-3:**

| model | runs (t/s) | median |
|---|---|---:|
| Q4_K_M | 4.00, 3.97, 3.97 | **3.97** |
| Q3_K_M+imatrix | 3.95, 3.93, 3.95 | **3.95** |

So Q4's best row (4.97) is **+25%** over its own CPU-only baseline.

## B. `-ngl` sweep — Q3_K_M + imatrix (7.32 GB), 8K, `kv f16`, `threads 3`

```bash
~/.venvs/vipera/bin/python tools/sweep_gpu.py \
  --model ~/.vipera/models/q8src-imbf-q3km.gguf --ctx 8192 --kv f16 \
  --ngl 0 4 8 10 12 14 16 --max-tokens 64 \
  --json tools/gpu_results/hermes_q3_8k_f16.json
```

| `-ngl` | actual | TTFT s | decode t/s | peak VRAM MiB | peak RSS MB | note |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 5.29 | 3.95 | 1092 | 8723 | CPU-only baseline (A1) |
| 4 | 4 | 4.81 | 4.13 | 1978 | 7665 | |
| 8 | 8 | 4.68 | 4.43 | 2708 | 7513 | |
| 10 | 10 | 4.60 | 4.57 | 3074 | 7520 | planner's pick at 8K |
| **12** | **12** | 4.53 | **5.00** | 3440 | 7525 | |
| **13** | **13** | 4.53 | **5.03** | **3624** | 7391 | **max that loads** (97% of VRAM) |
| 14 | **0** | 6.02 | 3.71 | 3324 | 13401 | ⚠ fallback to CPU-only |
| 16 | **0** | 5.77 | 3.71 | 3626 | 13099 | ⚠ same |

`n=1` per row. Q3 tops out at **13 layers** where Q4 stops at 10 (per-layer
weights 0.183 GB vs 0.225 GB), and Q3's peak RSS is 7.4 GB vs 9.2 GB — the same
+25–27% class of gain, 1.7 GB cheaper in RAM.

## C. The `threads` lever (median-3, the strongest free finding)

Same config, only `--threads` changes:

| path | threads 3 | threads 4 | Δ | threads 8 |
|---|---:|---:|---:|---:|
| Q3, 8K, f16, `-ngl 12` | 5.00, 5.05, 5.19 → **5.05** | 5.87, 6.02, 6.00 → **6.00** | **+19%** | 4.86 (n=1) |
| Q3, 8K, f16, `-ngl 0` (CPU) | 3.95, 3.93, 3.95 → **3.95** | 4.75, 4.78, 4.84 → **4.78** | **+21%** | — |

Spread inside each median-3 group ≤ 2.5%, so both deltas clear the Δ>5% gate
with room. TTFT moves the same way (4.92–4.96 s vs 5.01–5.38 s).

The planner sets `threads = cores − 1 = 3` on this 4C/8T box. On 4 physical
cores, **4 threads wins and 8 loses** — the heuristic is leaving a core idle and
costs ~20% on every path measured here.

```bash
~/.venvs/vipera/bin/python tools/sweep_gpu.py --model ~/.vipera/models/q8src-imbf-q3km.gguf \
  --ctx 8192 --kv f16 --ngl 12 --threads 4 --max-tokens 64 --json <out>.json
```

## D. KV dtype and the last layer (8K, Q3, `threads 3`, n=1 each)

| config | decode t/s | peak VRAM MiB |
|---|---:|---:|
| `kv f16`, `-ngl 12` | 5.05 (median-3 from C) | 3440 |
| `kv q8_0`, `-ngl 12` | 4.86 (−3.8%) | **2832** (−608 MiB) |
| `kv f16`, `-ngl 13` | 5.03 | 3624 |

`q8_0` costs <5% decode and buys 608 MiB of VRAM — worth it whenever a layer or
a larger context is what that VRAM buys (it is what makes E possible).

## E. 16K context — target T4 (Q3, `kv q8_0`, `threads 3`, n=1 each)

| ctx | `-ngl` | decode t/s | peak VRAM MiB |
|---:|---:|---:|---:|
| 8192 | 12 | 4.86 | 2832 |
| 16384 | 8 | 4.55 | 2278 |
| 16384 | 10 | 4.79 | 2650 |
| **16384** | **12** | **4.87** | 3018 |

Doubling the context costs **nothing measurable in decode** (4.87 vs 4.86) with
a short prompt, and the planner's 16K pick of 8 layers is 4 short of what loads
(12). T4 (≥4.5 t/s at 16K) is met.

## F. The two-venv trap, demonstrated with one command

`vipera bench run NousResearch_Hermes-4-14B-Q4_K_M-2 --ctx 8192 --max-tokens 64`
— identical model, identical config (`q8_0`, FA on), different virtualenv:

| env | binding | banner | TTFT | GEN t/s | n |
|---|---|---|---:|---:|---:|
| `~/.venvs/vipera` (3.12, CUDA) | GPU capable | `10/40` ✅ true | 2.896 s | 5.14, 5.17, 5.16 → **5.16** | 3 |
| `~/vipera/.venv` (3.14, CPU-only) | **no GPU backend** | `10/40` ❌ **false** | 6.75 / 4.32 s | 3.36, 4.07 | 2 |

Both rows report `10/40 LAYERS`, because the banner prints what the *planner*
decided, not what the binding did. The CPU-only row's 3.4–4.1 t/s is where the
published **4.00 t/s** figure sits, and the CUDA row is 29% higher.

**What this means for `docs/benchmarks.md`:** the row
*"8K context, f16 KV, flash attention off — 10/40 — 4.00 t/s"* must be treated
as a CPU-path number until re-run: today the same shape of run measures 4.97
(`f16`, n=1) on the GPU and 3.97 CPU-only (median-3). The **current shipped
path** (config `q8_0` + FA on) measures **5.16 t/s** (median-3) and should be
quoted instead.

---

## Targets (from `OPTIMIZE-tokps.md` §5)

| # | Target | Measured | Status |
|---|---|---|---|
| T1 | Q4, 8K ≥ 6.0 t/s | 5.16 shipped / 4.97 sweep | ✗ −14% — `-ngl` and threads are exhausted |
| T2 | Q3, 8K ≥ 7.0 t/s | 6.00 (ngl 12, t4, median-3) | ✗ needs a different lever (speculative decode, `n_batch`) |
| T3 | TTFT ≤ 5.5 s @ ~470 tok | 4.38–4.85 s | ✓ (n=1–3); shipped path 2.90 s @45 tok |
| T4 | 16K ≥ 4.5 t/s | 4.87 | ✓ (n=1) |
| T5 | 64K dense agent loop | 0.13 t/s | unchanged — hardware wall, move lane |
| T7 | spec ngram ≥ +10% | not run | pending (P2) |

## Product changes this data justifies

1. **Staged GPU fallback in `Engine`:** on a GPU load failure retry
   `gpu_layers // 2` *before* dropping to CPU-only. Measured reason: `-ngl 12`
   on Q4 lands on 0 layers at 3.90 t/s — worse than the 3.97 it would have got
   by not trying. *(not yet implemented)*
2. **`threads = cores`, not `cores − 1`:** +19–21%, median-3 on both paths.
   *(not yet implemented)*
3. **Serve the Q3+imatrix build:** +25% class gain, 1.7 GB less RSS, 13 layers
   instead of 10, PPL gate unaffected (see `tools/l1a2_results/REPORT.md`).
4. **Never overshoot `-ngl`:** 14+ on Q4 produced a hard failure and 40–55 GB
   RSS — the planner's VRAM formula must stay ≤ the measured wall
   (3624/3730 MiB at the last good row).
5. **Report the binding, not the hardware** — `vipera system doctor` now does
   (`gpu offload` row, P0.2), and `tools/sweep_gpu.py` refuses a CPU-only
   binding.

## Caveats

- **`mlock` is unavailable and untestable here:** `ulimit -l` = 8 MiB
  (soft = hard), unraisable without root. `-lm mlock` and `use_mlock=True` both
  log `failed to mlock … Cannot allocate memory` and fall back to mmap. Any
  mlock-attributed gain in older numbers needs re-measuring with
  `RLIMIT_MEMLOCK` raised (root: `/etc/security/limits.conf` or
  `LimitMEMLOCK=infinity`).
- `load_s` is **warm-page-cache after the first row** (5–7 s cold, 1.2–1.4 s
  warm) — not a load benchmark.
- The sweep prompt (~470 tok) differs from `bench run`'s (45 tok), so TTFT and
  decode columns are comparable **within** a table, not across F.
- Each row is a fresh process/engine; VRAM is sampled every 0.3 s, so peak VRAM
  can under-report by one sample.
- One machine, one card, `n` stated per number. Rows marked n=1 are
  directional: re-run for median-3 before publishing them as figures.

---

# G. 2026-10-22 — planner calibration + threads + placement ladder (shipped)

What changed in `src/` after the sweeps above, and what each change measured.
Same env as §A–F unless a row says otherwise: CUDA venv
(`~/.venvs/vipera`), GTX 1650 4 GB, Hermes-4-14B Q4_K_M, 8K context.

Tool: `tools/measure_knobs.py` (loads `llama_cpp.Llama` with the engine's exact
flags so per-load knobs such as `n_threads_batch` and `n_ubatch` actually apply;
`REPS=1` for single runs). `vipera bench run` numbers come from the CLI itself.

## G1. `-ngl` ceiling with a quantized KV cache (8K, `q8_0`, threads 4)

The §A ceiling of 10 layers was measured with **f16 KV and FA off**. `q8_0` KV
forces flash attention, which is also the low-scratch path, so the card holds
more layers:

```bash
~/.venvs/vipera/bin/python tools/measure_knobs.py NousResearch_Hermes-4-14B-Q4_K_M-2 \
  '[["n12",{"threads":4,"ngl":12}],["n13",{"threads":4,"ngl":13}],["n14",{"threads":4,"ngl":14}],["n15",{"threads":4,"ngl":15}]]'
```

| `-ngl` | loads | VRAM MiB | decode t/s (n=1) |
|---:|:---:|---:|---:|
| 12 | yes | 3274 | 5.41 |
| 13 | yes | 3500 | 5.55 |
| **14** | **yes** | **3696** | **5.96** |
| 15 | no (`Failed to create llama_context`) | — | — |

Q3_K_M + imatrix (7.32 GB) on the same card, same KV type: **17 layers** load
(3668 MiB, 6.17 t/s, n=1), 18 fail. Smaller per-layer weights buy three more
layers than Q4.

So the usable budget for weights+KV is ~3.37 GB of the 3.64 GB free — a
**0.28 GB** reserve, not the 1.3 GB (1.0 driver + 0.3 compute) the planner used.

## G2. `threads` (median-3, 8K, `q8_0`, ngl 10)

| threads | decode t/s | median | TTFT |
|---:|---|---:|---:|
| 3 (old planner) | 4.816, 4.783, 4.805 | **4.805** | 2.93 s |
| 4 | 5.244, 5.191, 5.325 | **5.244** (+9%) | 2.93 s |
| 4 + `n_threads_batch=8` | 5.095, 5.351, 5.260 | 5.260 (+0.3%, noise) | 2.93 s |

`n_threads_batch` is not a lever at this prompt length (prefill is not
thread-bound), so it stays unset. Threads follow physical cores.

## G3. Shipped pipeline, before/after (`vipera bench run --ctx 8192 --max-tokens 64`)

| Build | layers | TTFT | decode t/s (median-3) |
|---|---:|---:|---:|
| before (threads 3, 1.3 GB reserve → 10 layers) | 10/40 | 2.95 s | **4.65** (4.62/4.66/4.65) |
| after (threads 4, 0.28 GB reserve → 14 layers) | 14/40 | 2.76 s | **6.25** (6.25/6.22/6.25) |
| interleaved A/B, batch 1 (n=3 each) | 10 → 14 | 2.92 → 2.78 s | **4.655 → 5.762 (+24%)** |
| interleaved A/B, batch 2 (n=3 each) | 10 → 14 | — | **4.552 → 6.297 (+38%)** |

The interleaved rows are the honest delta: the two CLI batches above were taken
minutes apart and this box drifts ~10% between batches — the same control config
measured 5.15 / 4.50 / 4.66 t/s in three consecutive runs, and the new build's
absolute CLI median ranged 5.35–6.25 t/s over the session. Batch 2 ran in the
slower (throttled) window: both arms dropped, the 10-layer arm dropped more.
Floor the claim at **+20%**, quote the interleaved pair you re-measure.

16K, same build: **13/40, 2.83 s TTFT, 5.48 t/s** (old planner picked 8 layers
and lost to CPU-only).

## G4. Placement ladder (the failure path)

Forcing an offload the card cannot hold used to end at `gpu_layers=0`. With
`engine.load_attempts` it walks `n, n-1, n-2, n/2` first:

| requested | old behaviour | now |
|---:|---|---|
| 15 | CPU-only, 3.9 t/s | **13 layers, 4.4 s to load** |
| 17 | CPU-only, 3.9 t/s | 17/16/15 fail, falls to **8 layers**, 7.6 s |

Note the second row: a failed context creation does **not** return its VRAM
before the next attempt, so after three failures 13 layers no longer fit and
the halving rung is what loads. Keeping the ladder short is deliberate.

## G5. `mlock` with the limit actually raised

Root shell, `ulimit -l unlimited`, identical build, interleaved against the
unprivileged control:

| run | decode t/s | median |
|---|---|---:|
| control (`ulimit -l` = 8 MiB, lock refused) | 6.25, 6.22, 6.25 | **6.25** |
| `ulimit -l unlimited` (lock taken) | 6.26, 6.26, 6.20 | **6.26** |

**No gain (+0.2%, below the 5% gate).** The lock is no longer the blocker, and
the 16.0 → 22.9 t/s figure older docs attribute to `mlock` did not come from a
working lock.

## Caveats for this section

- n is stated per row; `n=1` rows are directional.
- The card was otherwise idle (`nvidia-smi` 2 MiB before each batch).
- `q3_ngl18` / `q4_ngl15` / `q4_ngl16` rows are the ceiling probes; they raise
  `Failed to create llama_context`, which is the signal, not a crash.

## G6. Q3_K_M+imatrix as the served model (2026-10-22)

`vipera model import` copies unless the source already sits at the destination
name, so the file was hard-linked to `~/.vipera/models/hermes-4-14b-q3km-imatrix.gguf`
(no second 7.3 GB on disk) and registered as `hermes-4-14b-q3km-imatrix`, then
set as `default_model`.

| Build | ctx | layers | decode t/s (runs) | median |
|---|---:|---:|---|---:|
| Q3_K_M+imatrix, 8K, q8_0 | 8192 | **17/40** | 6.75 / 6.46 / 6.57 | **6.57** |
| Q3_K_M+imatrix, 16K, q8_0 | 16384 | 14/40 | 6.20 | 6.20 (n=1) |
| Q4_K_M, 8K, q8_0 (control) | 8192 | 14/40 | 6.18 | 6.18 (n=1) |

Same session, so the Q3/Q4 difference (~+3%) is inside this box's ±10% drift;
the reason to serve Q3 is that it does not cost quality (PPL gate 0.3812 < 0.4),
is 1.6 GiB smaller, and holds three more layers.

## G7. The `layer_bytes` correction (found while switching models)

With `-ngl` chosen from `meta["layer_bytes"]` (mean repeating-block size) the
planner asked for **20** layers on the Q3 file where **17** is the ceiling, and
for 20+ on Q4 where 14 is. Fresh process per ask, so nothing is inherited:

| asked | landed on | note |
|---:|---:|---|
| 20 | **0 (CPU)** | 20/19/18 fail, then the quarter rung also fails |
| 19 | 17 | 19 and 18 fail, 17 loads |
| 18 | 16 | 18 and 17 fail, 16 loads |
| 17 | **17** | loads first try |

Two findings:

1. **A failed context creation keeps its VRAM.** After three failures on a 4 GB
   card, a 10-layer rung that normally loads fine could no longer allocate. This
   is why over-estimating is much worse than under-estimating, and why the
   ladder's last rung is now `n/4` instead of `n/2`.
2. **`size_bytes / n_layers` predicts the ceiling better than `layer_bytes`**
   (14 and 16.9 vs a measured 14 and 17) — llama.cpp's per-layer compute buffer
   grows with `-ngl`, and charging the embedding/output bytes to every layer
   approximates it. The planner now always uses `size_bytes / n_layers`.

## G8. `n_ubatch` for prefill (P1.5) — measured, rejected

~470-token prompt, Q4, threads 4, decode measured after the first token:

| config | TTFT | decode t/s |
|---|---:|---:|
| `n_ubatch 512`, ngl 14 | **4.52 s** | **6.40** |
| `n_ubatch 1024`, ngl 14 | load fails | — |
| `n_ubatch 2048`, ngl 14 | load fails | — |
| `n_ubatch 256`, ngl 14 | 4.58 s | 5.01 |
| `n_ubatch 1024`, ngl 12 | 4.66 s | 5.44 |
| `n_ubatch 512`, ngl 12 | 4.66 s | 5.70 |

A larger batch needs VRAM the offload wants, and at equal offload it does not
move TTFT (4.659 vs 4.658 s at ngl 12). Keep 512. Prefill here is bandwidth-bound
on the CPU-resident layers, not batch-size bound, so P1.5 has no lever left;
`n_threads_batch` was equally flat (§G2).
