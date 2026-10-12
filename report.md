# vipera — code review

Review of the **pure-arithmetic optimizer** (stage 1: `analyze` + `plan`, with the stage-2
scheduler checked where relevant). The core parser is right and the project is unusually honest
about what is measured versus assumed. This pass re-checks each finding against the repo as it
stands on **2026-10-12**, and marks whether it is still open, fixed, or partial — the code has
moved on since the first review, so several findings are now stale.

## Scope and method

What was run: `python3 -m unittest discover -s tests -t .` (**136 tests pass**; the first review
saw 35), plus header-only GGUFs built with the official `gguf` writer — one shaped like
DeepSeek-V2-Lite and one like Qwen3-30B-A3B — and compared against the official reader and current
llama.cpp source. The parser matched the official reader on all 377 tensors of the DeepSeek-shaped
file (name, shape, type, size, offset), and the quant block-size table checks out.

What could not be run: a real ~10 GB model, or `llama-bench` on the target. Anything about speed is
therefore reasoning, not measurement.

Status tags: **[open]** still true · **[fixed]** resolved since the first review ·
**[partial]** partly addressed · **[external]** about upstream, not re-checked here.

---

## 1. Upstream llama.cpp has moved under the roadmap — [external]

- `--fit` is on by default in current llama.cpp (`fit_params = true`), and `llama-fit-params`
  prints the `-ngl` / `-ot …exps=CPU` flags for the memory that is actually free. That is the
  "runtime shouldn't guess `-ngl`/`--n-cpu-moe`" job the manifest is meant to do.
- PR #29887 lets host-resident experts compute on the GPU through an LRU cache that transfers only
  the misses, enabled with `--moe-cache-mib N`. That is the LRU half of the planned P2; the
  routing-prediction/prefetch half does not appear to be covered. It shipped in release b11480,
  whose assets include prebuilt Ubuntu x64 CUDA builds, which also eases the "no nvcc" risk.

**Recommendation:** make the P0 baseline today's llama.cpp (default `--fit`, with and without
`--moe-cache-mib`), otherwise "≥2×" is a win over a strawman. Use `llama-fit-params` as the oracle
for the planner (the "predicted vs actual" check `docs/OPTIMIZER.md` already asks for), and have the
manifest emit flags. Upstream does not appear to do the file-layer work, routing traces or
measurement, which is where vipera's opening is. Not verified on a GTX 1650, and the feature is days
old.

## 2. Wrong answers, silently

- **Unknown quant types are sized as fp16, with no warning — [open].** llama.cpp defines types up to
  42; `vipera/gguf.py` stops at 35 (`TQ2_0 = 35`). `TensorInfo.n_bytes` falls back to fp16 density
  for unknown types (`gguf.py:190`), with a comment "rather than silently lying", but still no
  warning is emitted and `type_name` (`gguf.py:204`) is never called anywhere. An MXFP4-expert file
  (the format gpt-oss uses) is sized ~3.76× too big, 26.8 vs 7.1 GiB. Raise or warn, add the newer
  types (block sizes are in `ggml-common.h`), and test `_BLOCK` against `gguf.GGML_QUANT_SIZES`;
  the fixtures are F32-only, so the table is untested.
- **Nothing checks tensor extents against file size — [open].** `GGUF.file_size` is stored
  (`gguf.py:238`) and printed by `analyze` (`cli.py:57`), but never validated against
  `offset + n_bytes`. Truncating a copy to half its size still printed "4.43 GiB on disk" beside
  "8.10 GiB routable experts" and exited 0. Store the data offset and verify
  `data_offset + max(offset + n_bytes) <= file_size`.
- **An infeasible plan writes a manifest that can't say so — [fixed].** `vipera/manifest.py` now
  carries `fits: bool` and `warnings: list[str]` (computed directly, not sniffed from prose), so an
  unfittable plan is self-describing. Add `unplaced_bytes` for completeness.
- **`hw/gtx1650.toml` disagrees with the docs — [open].** `ram_mb = 15360` minus
  `ram_headroom_mb = 1536` gives 13.5 GiB usable, but the README says ~10 GiB is available and
  judges "fits" against ~11. For the Qwen3-shaped file that is a 15× swing in the verdict
  (0.28 GiB of spill vs 4.28 GiB). Use available RAM.
- **The "20x slowdown" warning is a constant — [not re-checked].** The first review reproduced
  identical text for two very different spills, with `Target.disk` only echoed into the manifest.
  The bandwidth inputs have since been measured (`bench/hwprobe.c`; `disk_bw_gbs` moved from an
  assumed 2.0 to a measured 8.0), so the estimate is now on firmer ground, but the scaling per
  spill was not re-checked here. The planner has spill and expert bytes, so it can scale the
  estimate directly.

## 3. Robustness

- **Only a narrow set of errors is caught — [partial].** `vipera/cli.py:439` now catches
  `(GGUFError, TraceError)` and returns exit **2** (the first review saw a raw traceback at exit 1).
  `OSError`, `tomllib.TOMLDecodeError` and `KeyError` (mistyped path, a directory, bad TOML, a
  missing key) are still not mapped. Also bound string lengths by the remaining file size and cap
  array nesting depth: a corrupt string length raises `MemoryError`, and nested arrays raise
  `RecursionError`.
- **`bench` extra flags only pass through after `--` — [open].** `bench/bench.py` uses
  `parse_args` (lines 16, 195), so the documented `-r 1 --no-warmup` is rejected. `parse_known_args`
  fixes it.
- **A wrong `--bin` in `bench` is a traceback at exit 1 — [open].** Usage errors are documented as
  exit 2.
- **`test_label` lives in `bench.py` and is imported into a test module — [open].**
  `bench/bench.py:56` still defines `test_label`, and `tests/test_bench.py` imports it, so `pytest`
  reports an error (the suite passes under `unittest`, which is what `README`/CI run). Rename it.

## 4. The benchmark's key metric isn't per-token — [open]

`bytes_read_per_token` (`bench/bench.py:141-153`) divides the process's whole `read_bytes` by one
pass of tokens. That numerator includes model load, warmup and 5 repetitions (llama-bench's
defaults), and a warm page cache makes reruns read ~0. Use `-r 1 --no-warmup`, diff a short run
against a long one (`-n 16` vs `-n 128`), and evict the file cache between runs. `peak_gpu_mib` is
system-wide `nvidia-smi` memory polled at intervals, so it counts other processes and can miss
spikes; take a pre-run baseline. `os.wait4`'s rusage (`ru_maxrss`, `ru_majflt`) gives exact peak RSS
and major faults for free.

## 5. Planner and design

- **`kv_cache: q8_0/q8_0` is a constant — [open].** It is declared at `vipera/manifest.py:152`. It
  is a good default (current llama.cpp auto-enables flash attention for it and requires K=V types
  for MLA), but then it isn't a plan. Derive it from context length or drop it.
- **llama.cpp always keeps the input embedding on the CPU — [not re-checked].** If so, the planner
  over-reserves ~110 MiB of 3.5 GiB VRAM for `token_embd`.
- **Test T1's premise before building it — [hypothesis].** Stacked `*_exps` tensors already keep
  each expert contiguous, roughly 1.5–3 MiB per projection on V2-Lite, so frequency ordering may
  buy less than `OPTIMIZER.md` expects. T6 alignment saves well under 1% at that size. Interleaving
  gate/up/down per expert (one read instead of three) looks more promising. This is a hypothesis,
  not a measurement.

## 6. Smaller things

- **`ALGORITHMS.md`'s speculative-decoding formula is inverted — [open].** Line 157 reads
  `speedup ≈ (1 − α) / (1 − α·β)`, which *falls* as acceptance rises (0.22 at α=0.8, β=0.1). The
  standard form is `(1 − α^(γ+1)) / ((1 − α)(γβ + 1))`, which gives 2.4 there.
- **Headline numbers derived from assumed bandwidths — [partial].** The 33 tok/s, 1.7 tok/s and 20×
  are bolded under banners saying bold means measured, but they derived from assumed 40 GB/s and
  2 GB/s. The machine constants have since been measured (`bench/hwprobe.c`, `hw/gtx1650.toml`), so
  the inputs are now real; the derived figures should be restated from them.
- **"not another runtime" vs the design — [partial].** `README.md:6` still says it, yet `DESIGN.md`
  plans an API gate, router and execution engine and labels two components "core". Reconcile the
  positioning.
- **No `LICENSE` file despite `license = MIT` in `pyproject.toml` — [open].**

---

## 7. Position vs the expert-level offloading direction

There is a strategic direction (`quest.md`) arguing that for MoE the better target than layer-wise
streaming (AirLLM) is **expert-level offloading — stream only the experts the router picks — plus an
expert cache, routing-aware prefetch, CPU–GPU co-execution, and KV/quantization** (the direction
FreeToken / SGLang / vLLM take). The old `analyze` + `plan` optimizer is pure arithmetic on tensor
sizes: it is *size-driven*, ranks no expert as "hot", and says so in
`placement.expert_order_pending_trace`. Against that direction it is the measured stage, not the
goal.

The repo has since moved toward the direction, still at the file/model layer rather than the runtime:

- **Routing traces** (`bench/capture_trace.py`, `vipera/trace.py`) replace the guess with real
  router choices.
- **Cache policy** (`vipera/schedule.py`, `tests/test_cache_policy.py`) implements LRU/LFU/`hot`
  with admission + pinned set + aging, and reports every hit rate against the **ceiling at the same
  cache size**, so a policy can't be flattered by a differently-sized baseline. On real OLMoE
  routing, LFU beats `hot` when routing is stationary; aging only pays under topic drift.
- **Pruning** (`vipera/prune.py`) quantifies what dropping experts costs — at the 5% sparsity the
  scheduler once assumed, only 21–25% of routing mass survives.

Where the quest and vipera meet: the quest's wins (per-expert streaming, prefetch, co-execution)
are **runtime** behaviours, and vipera's stated scope is *not another runtime* (§6). So vipera's
contribution is the enabling half — the file layout and the residency manifest that let a runtime
do expert-level offloading without guessing — plus the traces and measurements the runtime's cache
and prefetch would need. The parts of the quest vipera does **not** yet touch: actual per-expert
streaming, CPU–GPU co-execution, and KV offload. Worth deciding explicitly whether vipera stays on
the file-layer side of that line or extends a reference runtime consumer.

## Since the first pass

The reading above supersedes the raw draft: it strips the process narration, reports the current
test count (136, not 35), and tags every finding.

| Status | Count | Items |
|---|---|---|
| fixed | 1 | manifest `fits`/`warnings` |
| partial | 3 | `GGUFError`→exit 2 (others still uncaught); bandwidth constants now measured; "not another runtime" positioning |
| open | 10 | unknown-quant sizing, missing extent check, `gtx1650` RAM, bench `--` flags, bench `--bin`, `test_label`, per-token metric, KV constant, ALGORITHMS formula, missing LICENSE |
| not re-checked | 2 | "20×" scaling, input-embedding VRAM reservation |
| hypothesis | 1 | T1 frequency-ordering payoff |

The parser/quant-table and file-size-extent issues remain the highest-value open items, because
they produce wrong numbers with **exit 0**.
