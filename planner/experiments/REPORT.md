# REPORT.md — three optimizations, measured on this machine

**Question.** `docs/reference/literature/llm-optimization/` proposes three families of
optimizations for local inference — IO-aware attention (FlashAttention), a quantized KV
cache, and speculative decoding. `plantodo.md` STEP 1 measured this machine's decode at
**22.90 tok/s against a modelled ALU ceiling of ~23 tok/s**, with disk traffic during
decode at ≤0.2%. Which of the three moves decode tok/s here?

**Answer.** FlashAttention does, by about 4% on both models tested. KV quantization does
not. Speculative decoding could not be measured at all — this build loads the draft model
and never engages it. The 30B target cannot even load. Five runs per cell, plus one
discarded warm-up run, median of the five:

| model | cell | runs | median tok/s | std | CV | vs control | perm p | verdict |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite Q4_K_M | baseline (control) | 5 | **22.46** | 0.14 | 0.6% | — | — | — |
| DeepSeek-V2-Lite Q4_K_M | flashattention | 5 | **23.49** | 0.06 | 0.3% | **1.046×** | 0.0079 | **+4.6%, holds in every run** |
| DeepSeek-V2-Lite Q4_K_M | kv-cache-quant | 5 | 23.39 | 0.29 | 1.2% | 0.996× | 0.0556 | no effect |
| Qwen3-4B Q4_K_M | baseline (control) | 5 | **14.96** | 0.06 | 0.4% | — | — | — |
| Qwen3-4B Q4_K_M | flashattention | 5 | **15.56** | 0.10 | 0.6% | **1.040×** | 0.0079 | **+4.0%, holds in every run** |
| Qwen3-4B Q4_K_M | kv-cache-quant | 5 | 15.57 | 0.05 | 0.3% | 1.001× | 0.6825 | no effect |
| Qwen3-4B Q4_K_M | speculative: control arm | 5 | 14.73 | 0.47 | 3.2% | 1.000× | — | (control) |
| Qwen3-4B Q4_K_M | speculative: draft loaded | 5 | 14.60 | 0.09 | 0.6% | 0.991× | 0.9921 | **lever never engaged** |
| Qwen3-30B-A3B Q4_K_M | every cell | 0/1 | — | — | — | — | — | **cannot load (OOM)** |
| DeepSeek-V2-Lite | speculative | 0 | — | — | — | — | — | not applicable (no draft with this tokenizer) |

`p = 0.0079` is the floor of this design: with 5 runs per arm the exact permutation test has
252 splits, and 2 of them are at least as extreme as complete separation. FlashAttention
therefore produces the strongest evidence a 5+5 design can produce — on both models,
independently, and in the same direction.

**This report was wrong once.** An earlier run of the same matrix, without the warm-up run,
put FlashAttention at 0.970× / 1.040× with p = 0.40 / 0.83, and I wrote "no effect" as the
headline. The cause was a measurement-protocol defect, not an effect that appeared later —
§2.4 documents both defects and `logs/evidence_cold_start.log` holds the raw contrast.

---

## 1. Method

| | |
|---|---|
| Machine | i5-11300H (8 threads), 15 GiB RAM, **no swap**, NVMe `nvme0n1`, GTX 1650 (unused) |
| Build under test | llama.cpp `0.6.0-dev (build 1, commit f2cc728)`, **CPU-only** (`llama-bench --list-devices` → none) |
| Threads / context / tokens | `-t 4`, `-c 512`, `-p 0 -n 32` (llama-bench) or 32 generated tokens (llama-cli) |
| Runs | 1 discarded warm-up invocation per cell, then 5 recorded invocations. 1 run = 1 process = 1 log file. |
| Load mode | `-lm mlock` for both measured models (both fit in RAM). STEP 1 measured mlock as the mode that removes the paging tax: 22.90 vs 16.95 tok/s under the default `mmap`. |
| Sequential only | plantodo.md's standing constraint. The harness holds `experiments/.lock` and refuses to start while another harness runs; two concurrent runs OOM-kill each other on this box. |
| Statistic | median of the 5 runs, plus a **two-sided permutation test** on the means. With 5 runs per arm a normal approximation is not defensible, so no t-test is quoted. |

**Controls, one lever at a time.** "Baseline" is *not* the stock configuration: stock is
`-fa auto` with an f16 KV cache, so it would confound the two levers. The control is
`-fa off -ctk f16 -ctv f16`, and each variant differs from it in one flag —
except `kv-cache-quant`, see §4.3.

**Two runners, on purpose, and the seam is stated.** `flashattention` and
`kv-cache-quant` are measured with `llama-bench`, whose JSON reports
`avg_ts`/`stddev_ts`/`samples_ts` for the decode row. This build's `llama-bench` has **no
draft-model flags** (checked against `llama-bench --help`), so `speculative` is measured
with `llama-cli`. Its arm is compared **only** against its own control arm measured in the
same session — never against a `llama-bench` row. Absolute numbers across the two runners
are not comparable.

**What was measured inside each run** (from `/proc`, same technique as
`bench/step1_attrib.py`): peak RSS, major faults, NVMe sectors read, wall time. Decode
traffic stays at ~0–1 MiB per run after load — consistent with STEP 1's "disk is not the
bottleneck" — so no result here is a disk result.

---

## 2. Four things that had to be fixed before the numbers meant anything

Each was found by checking the harness against the tool, and each is recorded with the raw
contrast in `logs/`. In order of how much they could have misled:

**2.1 The lever was never on (§3.3).** The first speculative measurement looked like a
clean null result. It was a broken test: the draft loads and is then never used.

**2.2 KV quantization cannot be isolated from the attention kernel (§4.3).** The first
kv-cache-quant configuration was rejected by the build outright — 5/5 runs, exit 1. The
variant had to be redefined against a different control.

**2.3 `llama-cli`'s reported number is not the lever's number.** Its compact
`Generation: X t/s` line does not respond to the speculative knob, and this build prints no
detailed `eval time` line at all. The harness now reads llama-cli's own response JSON
(`-v`), which carries `predicted_n`, `predicted_ms`, `predicted_per_second` and — when the
speculative path is live — the draft counters.

**2.4 No warm-up run: the first recorded run of every cell was the slowest.**
`logs/evidence_cold_start.log` holds the before/after. In the pre-warm-up session, the first
recorded run of a cell was systematically the worst one:

```
baseline qwen3-4b      run 1..5  10.58  12.76  13.57  13.96  13.66
kv-cache-quant deepseek run 1..5  15.03  20.96  23.08  23.36  23.37
```

That is a cold model file, not noise: each run is a fresh process and the previous cell had
read a *different* model. With one discarded warm-up per cell the same cells measure
14.86–15.02 and 22.79–23.51, CV 0.4–1.2%. The warm-up's own numbers are kept under
`warmup` in each result file as evidence and excluded from every statistic.

That defect is also what hid the FlashAttention result. With few good runs per cell, the
spread swallowed a 4% effect; the same matrix, re-run with the warm-up, separates it
cleanly. **A session also drifts as a whole**: DeepSeek baseline measured 19.54 tok/s in
the pre-warm-up session and 22.46 in the final one, same command. Every comparison in this
report is therefore made against a control measured minutes earlier in the same session,
and no number is compared across sessions.

---

## 3. Result per lever

### 3.1 FlashAttention — +4.0% / +4.6%, and it survives an order swap

DeepSeek 22.46 → 23.49 (1.046×, p = 0.0079); Qwen3-4B 14.96 → 15.56 (1.040×, p = 0.0079).
Both hit the design's p-floor, and the run-to-run spread is under 1%.

The matrix measures each cell once, in sequence, so a cell measured later could benefit
from state the warm-up does not remove. `logs/evidence_fa_order_check.log` interleaves the
two configurations A/B/A/B on DeepSeek, 4 pairs:

| pair | `-fa off` | `-fa on` |
|---|---|---|
| 1 | 22.42 | 23.32 |
| 2 | 22.43 | 23.46 |
| 3 | 22.29 | 23.43 |
| 4 | 22.18 | 23.24 |
| median | 22.36 | 23.38 → **1.046×** |

FlashAttention wins every pair. The effect is small but it is real, and it is the one lever
here that moves the measured number in the direction the literature predicts.

What it is *not*: a reproduction of the papers' speedups. Those are 2–4× at sequence
lengths of thousands of tokens on GPUs; this is 512 tokens on a CPU with 4 threads, where
attention is a small share of the token. If the mechanism is "never materialise the
attention matrix", the effect should grow with context length — that is a hypothesis this
data does not test (§7).

### 3.2 Quantized KV cache — no effect, now with tight enough error bars to say so

Isolated against the FlashAttention arm (same attention kernel, f16 → q8_0): DeepSeek
0.996× (p = 0.0556), Qwen3-4B 1.001× (p = 0.6825). CVs are 0.3–1.2%, so this is not a
noisy null — the two configurations are the same measurement to within the instrument.

That is the predicted direction for an ALU-saturated machine at 512 tokens: a quantized
cache must be dequantized on read (added work), and there is almost no KV traffic to save
yet. What quantization buys is bytes of KV per token — valuable at long context or when KV
must be offloaded, neither of which this experiment exercises. The two-lever difference
against the control is DeepSeek 1.041×, Qwen3-4B 1.000× (i.e. whatever gain is visible there
is FlashAttention's, not KV's).

### 3.3 Speculative decoding — **the lever never engaged, so there is no result**

This is the finding I would most want a reader to keep intact. The measurement looks like a
null result: 14.60 vs 14.73 tok/s (0.991×). It is not a result. The draft model is loaded —

```
I common_speculative_init_result: loading draft model 'models/Qwen3-0.6B-Q4_K_M.gguf'
```

— and then the runtime states, in its own words, that there is nothing to run:

```
I spec common_specu: no implementations specified for speculative decoding
D slot launch_slot_: launching slot : {"id":0,"n_ctx":512,"speculative":false,...}
   ... "speculative.types":"none"
```

Both arms ran the identical non-speculative decoder, which is why the timings agree to
within noise (32 tokens, ~65–71 ms/token, in both). Trying to force acceptance with the
benchmarking-only switch fails for the same reason: `synthetic acceptance requires an
initialized speculative decoding context`.

So: **on this build, speculative decoding cannot be turned on.** The 0.991× is the price of
loading an unused 0.4 GiB draft model — that part is real and reproducible (peak RSS 3012
vs 2533 MiB, wall 5.7 vs 4.9 s). The literature's case for speculative decoding is
untouched; what is refuted is the idea that this machine can test it today.

An earlier version of this report used a second argument — that `--spec-draft-n-max 3` vs
`8` produced the same number — and it does not survive checking: a later reproduction gave
11.9 vs 13.3 tok/s, a 12% swing. That knob swings with the box's noise, so it is not
evidence of anything, and a conclusion resting on it would have been wrong. The runtime
markers above are the evidence.

The harness now enforces the distinction rather than trusting the operator: it parses the
slot and sampler markers out of every run, records `lever_engaged: false` in
`results/variant_speculative.json`, marks the cell `lever-not-engaged` in
`comparison.csv`, and `benchmark.py verify` returns `NOT CONCLUSIVE` instead of signing off
on losslessness — identical output from two identical decoders proves nothing.

### 3.4 Incidental: the two "independent" levers are not independent

`-ctk q8_0 -ctv q8_0` with `-fa off` is rejected outright, 5/5 runs, exit 1:

```
llama_init_from_model: quantized V cache requires flash_attn to be enabled
```

Reproduced on demand into `logs/evidence_kv_quant_requires_fa.log`. The variant was
therefore redefined as *FA on + q8_0 KV*, isolated against the FA-on arm. Anyone planning
to "turn on KV quantization" on a llama.cpp-class stack should know it drags the attention
kernel with it — which, given §3.1, is not the free choice it looks like.

---

## 4. What could not be measured, and why that is a result

**4.1 The 30B MoE cannot run here at all.** Qwen3-30B-A3B Q4_K_M is 18.6 GiB; this box has
15 GiB and no swap. `llama-cli` is killed by the kernel OOM killer during model load, in
both load modes:

```
kernel: oom-kill: ... task=llama-cli ... Out of memory: Killed process (llama-cli)
        total-vm:32478932kB, anon-rss:11798284kB
```

exit −9 / 137 after ~23 s, no timing printed at all; `-lm dio -ub 64 -b 64` dies at the
same point. Logs: `logs/speculative_*_qwen3-30b-a3b_run1.log`.

The consequence matters more than the failure: **the spill/offload regime is unreachable on
this machine.** A model large enough to spill is a model too large to load. Every number in
this report is from a RAM-resident model, so the literature pack's offload family (HOBBIT,
ExpertFlow, Endor, PowerInfer's GPU-CPU split) is untestable here for the same reason. That
is why the planned second model was replaced with Qwen3-4B — a smaller RAM-resident model,
not a second regime.

**4.2 `llama-bench` cannot test drafts, and there is no `llama-server` binary.** This build
ships `libllama-server-impl.so` but no standalone server, so the per-request `timings`
endpoint that could measure speculative decoding properly is not reachable from here either.

**4.3 Two levers share a switch.** §3.4.

**4.4 No paper's numbers were reproduced.** Everything above is this machine. The speedups
in `docs/reference/literature/llm-optimization/SYNTHESIS.md` are their authors'; this report
neither confirms nor contradicts them except where it measures the same lever locally.

---

## 5. What this changes about the earlier conclusion

`SYNTHESIS.md` argued that only work-reducing changes (FLOPs/token) can move the measured
22.90 tok/s, and that byte-saving changes pay only in a regime vipera has not reached. The
measurements split the three levers along exactly that line, but with one correction worth
keeping:

1. **FlashAttention is a work-reducing change, and it works here: +4% on both models.**
   Small, but it is the first optimization in this project to move the number that STEP 1
   said was stuck at the ALU ceiling, and it does so by ~4%. It also pushes decode slightly
   *past* the modelled ~23 tok/s ceiling (23.49), consistent with removing work rather than
   traffic.
2. **KV quantization is a traffic lever and buys nothing here** (1.000×, p = 0.68 on the
   small model) — measured now with error bars tight enough to call it.
3. **Speculative decoding remains untested, not disproven** (§3.3). If it is pursued, the
   prerequisite is a build where `speculative.types` is not `none` — a build question, not
   a model question.
4. **The offload regime remains unreachable**, now demonstrated by an OOM kill rather than
   by argument.

---

## 6. Reproduce

```bash
# dev-only deps, .venv, not project dependencies (vipera stays stdlib-only)
.venv/bin/pip install matplotlib numpy pandas pyyaml

# full matrix: baseline + 3 variants, both models, 1 discarded warm-up + 5 runs per cell
# (sequential; ~9 min). --force re-measures over the stored files.
.venv/bin/python experiments/benchmark.py run --all --force

# one cell only (merges into the existing result file, does not clobber other models)
.venv/bin/python experiments/benchmark.py run --variant flashattention --model qwen3-4b --force

# correctness check for the speculative variant
.venv/bin/python experiments/benchmark.py verify --model qwen3-4b

# tables and figures from the stored JSON (no measurement)
.venv/bin/python experiments/benchmark.py csv && .venv/bin/python experiments/benchmark.py summary
.venv/bin/python experiments/benchmark.py charts
```

Checks run for this report:

| check | result |
|---|---|
| `python -m pytest -q` (repo suite, after all edits) | **227 passed, 4 subtests passed** |
| CSV ↔ REPORT consistency | every median, CV and p-value in the table above is present verbatim in `results/comparison.csv` |
| Every recorded run logged separately, warm-up included | 56 logs on disk: 8 warm-up + 40 run + 2 verify + 2 OOM + 4 evidence. 62 files are committed; the 12 llama-cli `-v` logs (10 measured runs + 2 warm-ups, 8.7 MB of runtime spam) are gitignored for size by `experiments/.gitignore` — their numbers are all in the JSON |
| Failed runs recorded, never imputed | 30B cells: `ok=0, failed=1, status=exit_-9`; the rejected KV config: 5× `exit_1` |
| The FA result survives an order swap | `logs/evidence_fa_order_check.log`, 4/4 pairs |
| The lever audit catches the false null | `lever_engaged:false`, `verify` verdict `NOT CONCLUSIVE` |
| PNGs | 3/3 valid, 61–184 KB |

---

## 7. Limitations

- **5 runs per cell.** Enough to separate a 4% effect when CV < 1% (the p-floor is reached),
  not enough to resolve anything smaller, and the p-value cannot go below 0.0079 at this n.
- **CPU-only build.** Nothing here says anything about these levers on the GTX 1650.
- **512-token context, 32 generated tokens.** Short. This is the friendliest case for a null
  result on the two traffic levers and the *least* favourable case for FlashAttention, so
  the FA gain measured here is a floor, not a ceiling — but that inference is expectation,
  not measurement.
- **Session-level drift.** Same command, different session: DeepSeek baseline 19.54 vs 22.46
  tok/s. Comparisons are valid within a session only; the 30B result is a load failure, not
  a performance result.
- **The warm-up protocol was added mid-experiment**, and this report says so rather than
  presenting one clean set of numbers: the first, noisier version is recorded in
  `logs/evidence_cold_start.log` because the correction is part of the finding.
- **`experiments/` is not part of the package.** `pyproject.toml` still declares
  `dependencies = []`; the repo test suite does not import it. `pandas` was installed but is
  not used — the tables are written with the stdlib `csv` module.
- Charts are drawn from the stored per-run values, so they can be regenerated without
  re-measuring.

---

## 8. Files

```
experiments/
├── benchmark.py                      # run / verify / csv / charts / summary / all
├── config.yaml                       # hardware, software, measurement (+warmup_runs), models,
│                                     #   and models_not_runnable (the 30B OOM, with evidence)
├── baseline/control.yaml             # the control: -fa off -ck f16 -cv f16
├── variants/
│   ├── flashattention/variant.yaml   # -fa on
│   ├── kv-cache-quant/variant.yaml   # -fa on -ck q8_0 -cv q8_0 (FA required by the build)
│   └── speculative/variant.yaml      # llama-cli -md Qwen3-0.6B; documents the not-engaged finding
├── results/
│   ├── baseline.json                 # both models: warm-up + 5 runs each
│   ├── variant_flashattention.json
│   ├── variant_kv-cache-quant.json
│   ├── variant_speculative.json      # both arms, lever_engaged:false, the 30B OOM cells
│   ├── verify_speculative.json       # verdict: NOT CONCLUSIVE, with reasoning
│   ├── comparison.csv                # one row per cell: medians, CV, speedup, perm p, status
│   ├── evidence_lever_and_oom.txt    # the runtime markers behind §3.3 and §4.1, with the greps
│   └── charts/{primary_metric_bar,distribution_box,cross_validation}.png
├── .gitignore                        # keeps the 8.7 MB of llama-cli -v run logs out of git,
│                                     #   keeps every small log and every evidence file in
├── logs/                             # 56 files, each with its argv and wall time:
│   ├── {variant}_{model}_warmup{1}.log        (the discarded run, kept as evidence)
│   ├── {variant}_{model}_run{1..5}.log        (the measured runs)
│   ├── verify_*_qwen3-4b.log                  (correctness check)
│   ├── evidence_fa_order_check.log            (the A/B/A/B interleave, §3.1)
│   ├── evidence_cold_start.log                (before/after the warm-up fix, §2.4)
│   ├── evidence_kv_quant_requires_fa.log      (the rejected configuration, reproduced, §3.4)
│   ├── evidence_draft_max_3_vs_8.log          (the knob test that did NOT reproduce, §3.3)
│   └── speculative_*_qwen3-30b-a3b_run1.log   (the OOM evidence, §4.1)
│                                     # the 12 speculative_*_qwen3-4b logs are on disk but ignored
└── REPORT.md                         # this file
```
