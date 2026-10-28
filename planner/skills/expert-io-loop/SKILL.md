---
name: expert-io-loop
description: A measure → localize → hypothesize → change → re-measure protocol for proving that a change to vipera's placement, expert cache, pruning or manifest actually reduces expert disk I/O — with an explicit noise band, a perplexity gate and a machine-readable verdict. Use when asked whether an optimization helped, how to benchmark a change in this repo, when a tok/s number looks too good, or when two runs disagree.
---

# SKILL — The measure → hypothesis → measure loop

> Scope: **vipera measures and models; it never runs a model itself.** This skill
> covers the *real-run* side of the loop (`bench/` + `llama.cpp`). The modelled
> side (`vipera schedule`, `vipera sweep`) is documented in `README.md` and
> `docs/DESIGN.md`; the two must never be printed in the same column.
>
> Every number below is either **measured here** (file named) or labelled
> **modelled** / **not measured**. Nothing is inherited from a blog post.

## TL;DR

1. **One run is not a result.** Run ≥3, throw away the first (laptop thermals),
   take the median, and report the spread. The repo's own hardware probe repeated
   twice differs by **6.6% on read bandwidth** (`bench/results/hwprobe-run1.json`
   vs `run2.json`) — so **7% is the default noise band**, not 0.
2. **A change is a win only if it clears noise *and* the quality gate.** Gate:
   perplexity rise **≤ 0.3** absolute (`docs/OPTIMIZER.md` §6), otherwise the
   transform stays off by default even if it is fast.
3. **The bottleneck is bytes read per token, not tok/s.** `tok/s` moves for many
   reasons (threads, thermal, MMAP); `bytes_read_per_token` moving down is what
   proves an *expert-I/O* change did something. Report both, always.
4. **Change one variable per round.** Cache policy *and* prune ratio *and* threads
   in one commit produces an unattributable number.
5. **Write the hypothesis before the edit**, with the direction and the threshold.
   A hypothesis you write afterwards is a story, not a test.
6. **Never mix modelled and measured** in one table, and never call
   `peak_gpu_mib` a VRAM measurement — the local `llama.cpp` build is
   `GGML_CUDA:BOOL=OFF` (see `hw/gtx1650.toml`).

---

## 1. The loop

```
  ┌─ P0 freeze baseline ─► P1 localize ─► P2 hypothesis ─► P3 change ─┐
  │        (median of ≥3)     (where bytes      (written DOWN)  (ONE    │
  │                           go)                            variable) │
  └──────────────────── P4 re-measure ◄────────────────────────────────┘
                              │
                              ▼
                    P5 verdict + quality gate + log
```

### P0 — Freeze the baseline

Record everything that makes two runs comparable, in the same JSON you will later
diff: model file + quant, llama.cpp commit, `-t/-ngl/-c`, context, this repo's
`git rev-parse --short HEAD`, and the hw profile TOML.

```bash
# first run is a warm-up: discard it
python3 -m bench.bench --bin ~/dev/llama.cpp/build/bin/llama-bench \
    --model models/deepseek-v2-lite-Q4_K_M.gguf --ngl 0 --threads 8 \
    --pp 512 --tg 128 --json bench/results/base-r1.json
# ... repeat to -r3 (or more on a noisy day)
python3 skills/expert-io-loop/scripts/aggregate.py bench/results/base-r*.json \
    --json > bench/results/baseline.json
```

`aggregate.py` refuses to merge reports with a different model or a different
`llama-bench` command — merging those would manufacture a result.

### P1 — Localize: where do the bytes actually go?

Do this *before* guessing. The repo already answers it:

```bash
python3 -m vipera analyze model.gguf --json        # bucket bytes: resident / router / shared / routable
python3 -m vipera trace trace.jsonl --arch arch/olmoe-1b-7b.toml --target hw/gtx1650.toml
python3 -m vipera schedule arch/olmoe-1b-7b.toml --target hw/gtx1650.toml \
    --quant q4-experts --trace trace.jsonl --cache-policy lru
```

What decides the answer:

| Question | Command | Field |
|---|---|---|
| Are we disk-bound at all? | `schedule` | `bound` (`disk` / `ram` / `compute`) |
| Is the cache the problem? | `schedule` | `hit_rate` **and** the ceiling at the same size — `policy/ceiling` is the honest number |
| Is the routing worth caching? | `trace` | entropy + static-cache ceiling |
| Is pruning the only lever left? | `prune` | keep-set vs uniform at target routing mass |

### P2 — Hypothesis (write it down first)

Template — fill all four lines *before* touching code:

```
HYPOTHESIS  <id, e.g. H07>
IF          <one change: e.g. cache-policy lru -> lfu on the 5-topic trace>
THEN        <primary metric moves: bytes_read_per_token ↓ ≥ 10%>
BECAUSE     <mechanism: routing repeats within a topic window>
FAILS IF    <read/token ↓ < 7% (noise), or ppl rises > 0.3, or peak RSS ↑ > 7%>
```

If you cannot fill BECAUSE with a mechanism from `docs/`, you are guessing —
go back to P1.

### P3 — Change one variable

Levers, cheapest first (detail in `docs/OPTIMIZER.md` §4):

| Lever | Where | Command / flag |
|---|---|---|
| Cache policy / allocation | `vipera/schedule.py` | `--cache-policy lru\|lfu\|hot`, `--cache-alloc global\|per-layer` |
| Prefetch on/off (does overlap pay?) | `vipera/schedule.py` | `--no-prefetch` |
| Expert pruning | `vipera/prune.py` | `--prune-experts 0.05` (after `prune --target-mass`) |
| KV precision | `vipera/schedule.py` | `--kv-bits 8\|4` |
| Speculative / draft | `vipera/schedule.py` | `--draft-tokens 3` |
| Layer dropping | `vipera/schedule.py` | `--keep-layers N` |
| Expert hot/cold reorder (T1), manifest (T2) | `vipera/optimizer.py` | **design only — not implemented** |

Runtime flags (`-t`, `-ngl`, `--mlock`, quant choice) are *confounders*, not
levers: changing them invalidates the baseline (P0) and must start a new round.

### P4 — Re-measure

```bash
python3 -m bench.bench ... --json bench/results/cand-r1.json   # ×3, discard r1
python3 skills/expert-io-loop/scripts/aggregate.py bench/results/cand-r*.json --json \
    > bench/results/candidate.json
python3 skills/expert-io-loop/scripts/compare.py \
    bench/results/baseline.json bench/results/candidate.json --primary avg_ts
```

### P5 — Verdict + quality gate + log

```bash
# quality axis (only for changes that touch weights or expert set)
python3 -m bench.perplexity --bin ~/dev/llama.cpp/build/bin/llama-perplexity \
    --model models/deepseek-v2-lite-Q4_K_M.gguf \
    --file ~/dev/llama.cpp/wikitext-2-raw/wiki.test.raw --chunks 20 --json ppl.json
python3 skills/expert-io-loop/scripts/compare.py base.json cand.json \
    --ppl-base 8.6426 --ppl-cand 8.7011        # gate: rise ≤ 0.3
```

Then log the round the way `docs/ROADMAP.md` §7–13 already does: hypothesis id,
command, both numbers, the verdict, and what you will try if it failed. A round
whose verdict is `INCONCLUSIVE` is a **result** — record it; do not re-run until
it turns green.

---

## 2. Metrics

| Metric | Source | Better | What a move rules out |
|---|---|---|---|
| `avg_ts` (per test) | `bench.bench` | ↑ | nothing by itself — thread count and thermals move it too |
| `bytes_read_per_token` | `bench.bench` (`io.read_bytes / tokens`) | ↓ | **the actual claim**: fewer expert bytes per token |
| `peak_rss_kb` | `bench.bench` sampler | ↓ | "it got faster by no longer fitting" |
| `spread_pct` | `aggregate.py` | ↓ | machine noise; > 7% means do not compare yet |
| `ppl` ± stderr | `bench.perplexity` | ≈ or ↓ | "we bought speed with quality" |
| `hit_rate` / ceiling | `vipera schedule` | ↑ (and `policy/ceiling` → 1) | modelled only — never print beside a measured number |

**Not a metric on this machine:** VRAM. `nvidia-smi` reports the card
(GTX 1650, 4 GB) but the local `llama.cpp` is CPU-only, so `peak_gpu_mib` is a
sampling artefact. `hw/gtx1650.toml` says the same thing from the other side:
VRAM bandwidth there is a datasheet figure "no result in this repo depends on".

---

## 3. Symptom → cause → lever

| You observe | Likely cause | Next lever (P3) |
|---|---|---|
| `bound: disk`, low hit rate, high read/token | expert pool ≫ cache; wrong eviction | `--cache-policy lfu\|hot`, `--cache-alloc per-layer`, then pruning |
| hit rate near ceiling, still slow | cache is *already* optimal for its size; the resident set is too big | shrink resident: `--quant q4-experts`, `--kv-bits 8` |
| hit rate high, read/token low, tok/s still flat | not I/O-bound — CPU/threads/thermal | `-t`, check throttling; **stop touching the cache** |
| read/token high *and* hit rate high | dense part or KV streaming, not experts | `--keep-layers`, `--kv-bits`, re-check `analyze` buckets |
| spread > 7% between identical runs | machine noise (thermal, background load) | more runs, cool-down, median; do **not** tune against it |
| tok/s up, read/token up too | you moved bytes but hid it in throughput | verdict flips to `regressed` in `compare.py` — accept it |
| modelled win, measured flat | model is wrong about bandwidth or overlap | that is a **finding**: fix the model, log it |

---

## 4. Scripts

Both are stdlib-only, Python ≥3.11, and read reports produced by `bench/`.

### `scripts/aggregate.py` — median of N runs

```bash
python3 skills/expert-io-loop/scripts/aggregate.py r1.json r2.json r3.json
python3 skills/expert-io-loop/scripts/aggregate.py --json r*.json > median.json
python3 skills/expert-io-loop/scripts/aggregate.py --noise 0.10 r*.json   # noisier day
```

Prints median `avg_ts` per test with min/max/spread, median read/token, peak RSS,
and whether the spread is inside the band. **Exit:** `0` all inputs ok · `1` some
input `ok:false` · `2` bad usage or incomparable inputs (different model/command).

### `scripts/compare.py` — the verdict

```bash
python3 skills/expert-io-loop/scripts/compare.py base.json cand.json
python3 skills/expert-io-loop/scripts/compare.py base.json cand.json \
    --primary read_per_token --test tg128
python3 skills/expert-io-loop/scripts/compare.py base.json cand.json \
    --ppl-base 8.6426 --ppl-cand 9.2000 --json     # gate trips -> exit 2
```

Verdict rules, in order:

1. quality gate (`ppl_cand - ppl_base > 0.3`) → **REGRESSED**;
2. primary delta beyond the band in the favorable direction **and** neither
   `read/token` nor `peak RSS` regressed beyond the band → **IMPROVED**;
3. primary delta beyond the band in the unfavorable direction → **REGRESSED**;
4. otherwise → **INCONCLUSIVE** (this includes "inside the noise band").

**Exit codes are the verdict:** `0` improved · `1` inconclusive · `2` regressed ·
`3` usage/parse/comparability error. An `IMPROVED` without `--ppl-*` prints
`note: no perplexity supplied — speed claim is UNVALIDATED`; that is deliberate.

---

## 5. What "comparable" means

Two runs may be compared only if **all** hold:

1. same model file and quant (`model_filename` — enforced, exit 3);
2. same llama.cpp build and commit;
3. same `-t`, `-ngl`, context, `-pp/-tg` (command differences are *warned*, fatal
   only with `--strict-command` — a deliberate runtime-flag change starts a new
   baseline instead);
4. same repo `git` state for anything you did not change;
5. ≥3 runs each, median, first discarded;
6. same quality corpus/chunks/context if perplexity is quoted.

---

## 6. Pitfalls

1. **Comparing inside the noise band.** 2% on this box is nothing; the probe
   itself moved 6.6%. Tune `--noise` only with a measured reason.
2. **Cherry-picking one test.** `compare.py` takes the median across matching
   tests for a reason; use `--test` when you have a stated reason (decode-only).
3. **Changing two things.** Threads + cache policy in one round = no attribution.
4. **Reporting a modelled number as a measurement.** `sweep`/`schedule` output
   says "MODELLED … not a real run" — keep that label when quoting it.
5. **Skipping perplexity on a pruning/quant change.** Fast and wrong is a
   regression; the gate exists for exactly this (`docs/OPTIMIZER.md` §6).
6. **Ignoring a failed round.** `INCONCLUSIVE` logged beats `IMPROVED` fabricated.
7. **Boasting VRAM on a CPU-only build.** `peak_gpu_mib` is not evidence here.
8. **Tuning against background load** (browser, other benchmarks, `gnome-shell`).
   Check with `uptime` before blaming your change.

---

## 7. Self-evaluation (what is weak here)

* **The noise band is one sample pair.** 7% comes from two `hwprobe` runs, not a
  distribution. Three or more repeats would let it be an actual σ.
* **No first-party tok/s table yet.** There is no `models/` in the repo (weights
  are gitignored), so every command here is the *recipe*, not a reproduced number.
  `bench/results/` currently holds probes, one perplexity baseline and two traces.
* **Quality gate is perplexity-only.** `docs/ROADMAP.md` already calls the pruning
  perplexity proxy unvalidated; task-level checks are absent.
* **`aggregate.py` assumes identical commands.** A legitimate mid-round flag change
  is a comparability error by design — you must start a new baseline manually.
* **Nothing here measures VRAM or energy** on this machine. Both are modelled.
* **Thermals are handled by "discard the first run", not by temperature.** A real
  gate would watch `coretemp` and drop runs that throttled.

---

## Cheatsheet

```
BASELINE   ≥3 runs → aggregate.py --json → baseline.json   (discard run 1)
LOCALIZE   schedule (bound, hit vs ceiling) · trace (entropy) · analyze (buckets)
HYPOTHESIS IF / THEN (threshold) / BECAUSE / FAILS IF — written before the edit
CHANGE     one variable · record git SHA · runtime flags start a NEW baseline
REMEASURE  ≥3 runs → aggregate.py → compare.py base cand [--ppl-base --ppl-cand]
VERDICT    exit 0 improved · 1 inconclusive · 2 regressed · 3 cannot judge
BAND       7% default (hwprobe run1 vs run2: 6.6% on read) — inside it: no claim
GATE       perplexity rise ≤ 0.3 absolute, else the transform stays off
REPORT     tok/s AND bytes/token AND ppl — one number alone is not a result
```

## Evidence in this repo

| File | What it fixes |
|---|---|
| `bench/results/hwprobe-run1.json`, `hwprobe-run2.json` | the 7% noise band (6.6% read, 1.9% copy, ≤0.7% GEMV) |
| `bench/results/olmoe-ppl-baseline.json` | the perplexity baseline shape (`ppl` 8.64 ± 0.30, 20 chunks, ctx 512) |
| `bench/results/olmoe-*.jsonl` | real routing traces for P1 |
| `hw/gtx1650.toml` | why VRAM is not a metric here |
| `docs/OPTIMIZER.md` §4, §6 | the lever list and the ≤0.3 quality gate |
| `docs/ROADMAP.md` §7–13 | how a logged round is written |
