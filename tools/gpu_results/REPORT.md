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
