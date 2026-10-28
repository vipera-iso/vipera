# vipera — code review

Review of the **pure-arithmetic optimizer** (stage 1: `analyze` + `plan`, with the stage-2
scheduler checked where relevant). The core parser is right and the project is unusually honest
about what is measured versus assumed.

This pass re-checks each finding against the repo as it stands on **2026-10-13**, **including the
uncommitted working tree** — twelve source/test files are modified and recorded in no commit, so a
finding below can differ from what a clean checkout of `946ca4f` does. (`report.md` itself is the
thirteenth modified file; the rest of this text is the change.) Several findings are now fixed. A few had
gone stale. And two *statuses* from the second pass were plainly wrong rather than stale: `LICENSE`
was marked open although a commit had already added it, and the upstream section stayed `[external]`
although both flags it describes exist in the llama.cpp checkout this repo builds against.

## Scope and method

What was run:

- `python3 -m unittest discover -s tests -t .` → **180 tests pass** (the first review saw 35, the
  second 136).
- `analyze` and `plan` on a **real 9.65 GiB model** — `models/DeepSeek-V2-Lite.Q4_K_M.gguf`,
  377 tensors, header and descriptors parsed in 0.241 s. The second pass could not do this; the
  numbers quoted below come from that run.
- Corrupt-file probes, built by hand in the spirit of the first pass's fixtures: a GGUF with a huge
  string length (2^40 and 2^62), nested arrays at depth 300 and 3000, tensors of unknown quant
  types (36/39/42/99), and a file truncated inside a real data section.
- The upstream claims, re-checked against the llama.cpp checkout the repo builds against
  (`~/dev/llama.cpp`, tree clean, HEAD `e8a3943` = upstream `f2cc728` plus one fork commit that
  touches only `src/llama-context.cpp` and `src/llama-moe-cache.cpp`, 276 lines — the two patches in
  `bench/llama-moe-trace.patch` and `bench/llama-moe-prefetch.patch`) instead of being left as
  `[external]`.

What could not be run: `pytest` (not installed here — the `test_label` finding is therefore verified
by mechanism, not by a pytest run), and `llama-bench` on the target. Note that `llama-bench` **is**
now built at `~/dev/llama.cpp/build/bin/llama-bench`, so that measurement is no longer out of
reach; it was not taken in this pass. Anything about speed is still reasoning, not measurement.

Status tags: **[open]** still true · **[fixed]** resolved since the first review · **[partial]**
partly addressed · **[confirmed]** a hypothesis from an earlier pass that this pass settled ·
**[hypothesis]** unmeasured · **[external]** about upstream, not re-checked here.

---

## 1. Upstream llama.cpp has moved under the roadmap — [confirmed] in the checkout, [external] for the PR ids

- `--fit` is on by default in current llama.cpp: `fit_params = true` in the params struct
  (`common/common.h:481`), with the flag `-fit, --fit [on|off]` (`common/arg.cpp:2881`). That is the
  "runtime shouldn't guess `-ngl`/`--n-cpu-moe`" job the manifest is meant to do.
- The host-resident-experts GPU cache exists: `--moe-cache-mib N` — "GPU cache size in MiB for the
  MoE experts kept in the CPU. with multiple GPUs, it is split among them like the layers
  (`--tensor-split`) (default: 0, disabled)" (`common/arg.cpp:2780`). That is the LRU half of the
  planned P2, transferring only the misses.
- `tools/fit-params/` is present in the tree (so `llama-fit-params` exists as an upstream tool); no
  binary for it is built here — `build/bin/` holds `llama-bench` and `llama-cli` only.
- **Still unverified:** the PR number (#29887) and the release id (b11480). There is no PR or release
  metadata in a source checkout, and this pass did not go to the network. Treat those two ids as
  reported, not checked.

**Recommendation status — [open].** Nothing in `bench/` or `docs/` passes `--fit`, so the P0
baseline is still not "today's llama.cpp default", and "≥2×" remains a win over a strawman until it
is. One half of the recommendation did land, in a different form: `--moe-cache-mib` is now driven by
vipera's own prefetch work (`bench/prefetch_table.py:19`, `bench/results/README.md:24`) through
`bench/llama-moe-prefetch.patch`, which prefetches *predicted* experts rather than only transferring
misses. Upstream still does not appear to do the file-layer work, the traces or the measurement,
which is where vipera's opening is.

## 2. Wrong answers, silently

- **Unknown quant types are rejected, not mis-sized — [fixed].** The fp16 fallback is gone:
  `TensorInfo.n_bytes` raises `UnknownQuantType` (`gguf.py:247`, class at `gguf.py:35`) and
  `read_gguf` rejects the file up front (`gguf.py:364`) rather than returning plausible totals.
  Probed with hand-built files: tensor types 36, 39 (the MXFP4 gpt-oss uses), 42 and 99 all raise
  `UnknownQuantType` instead of being sized ~3.76× too big. Tests cover it
  (`tests/test_gguf.py:71-87`).
  Two leftovers, neither of them a wrong answer any more: types 36–42 are still unsupported, so an
  MXFP4-expert file now fails loudly instead of lying — better, but still not analyzable — and the
  quant-table hygiene item below.
- **Tensor extents are checked against file size — [fixed].** `GGUF.validate_extents`
  (`gguf.py:273`) verifies `data_offset + offset + n_bytes <= file_size` and is called from
  `read_gguf` (`gguf.py:368`). Probe: a file truncated inside its data section now raises
  `GGUFError: tensor 't' needs bytes up to 160, past the 106-byte end of file: the file is truncated
  or its tensor offsets are wrong`. An intact file and a descriptor-only fixture (`file_size <=
  data_offset`) still load, so the fixtures that motivated the exemption keep working.
- **An infeasible plan writes a self-describing manifest — [fixed].** `Plan.fits` (`manifest.py:81`)
  and `warnings` are computed from bytes, not sniffed from prose. `unplaced_bytes` is still absent —
  completeness, not correctness.
- **`hw/gtx1650.toml` now agrees with the docs — [fixed].** `ram_headroom_mb` went 1536 → 5120, so
  `usable_ram` is the documented 10 GiB (`README.md:45`) instead of the 13.5 GiB the machine does not
  have. Measured effect on a 15.9 GiB Qwen3-30B-shaped header: **0 MiB of spill** before, **2456
  MiB** after — the planner no longer calls a model the docs mark ❌ "needs streaming" resident.
  DeepSeek-V2-Lite still plans with no spill, as the docs say it should. A regression test pins the
  budget (`tests/test_manifest.py::TestTargetProfile`).
- **The slowdown warning is derived from the target's tiers — [fixed].** `manifest.py` computes
  `ram_bw_gbs / disk_bw_gbs` for the target in hand and weights the average by the spilled share of
  the expert pool, so the number cannot be a constant: the shipped profile reads **~5.5x** (44 GB/s
  RAM against 8 GB/s disk) and **~1.7x** averaged over the pool, the same plan against a 22 GB/s disk
  reads **~2.0x**, and a target declaring `disk_bw_gbs = 0` now says the cost cannot be derived
  instead of dividing by zero (`tests/test_manifest.py::TestSpillPenalty`).

## 3. Robustness

- **The common failure modes are mapped to exit 2 — [fixed].** `vipera/cli.py:496` catches
  `(GGUFError, TraceError)`, `:499` catches `KeyError`, `:502` catches
  `(OSError, tomllib.TOMLDecodeError)`, all returning **2** with an `error:` line. The mistyped
  path, the directory, the bad TOML and the missing key from the first review are all handled now.
- **Malformed input is rejected, not crashed on — [fixed].** Every length and count in a header is
  now checked against the file's own size *before* anything is allocated: `_read_exact` refuses a
  read larger than the bytes left (`gguf.py:150`), a metadata array may not claim more elements than
  the remaining bytes, header counts get the same test, and nesting is capped at
  `MAX_ARRAY_DEPTH = 32` (`gguf.py:28`). The two crashes this review reproduced — a corrupt string
  length (2^40 / 2^62) raising **`MemoryError`** and 3000-deep arrays raising **`RecursionError`** —
  both raise `GGUFError` now, and so do the adjacent cases (a 2^62 array count, a 2^62
  `kv_count`/`tensor_count`). Fuzzing every truncation point and every single-byte corruption
  (0x00 / 0xFF / 0x7F) of two fixtures — **36,272 malformed variants** — produced no exception
  other than `GGUFError`. Tests: `tests/test_gguf.py::TestMalformedHeader`.
- **`bench` extra flags reach llama-bench without `--` — [fixed].** `split_args` (`bench/bench.py:197`)
  uses `parse_known_args` and drops the separator. Verified: `-r 1 --no-warmup -ngl 0` arrives as
  `['-r', '1', '--no-warmup', '-ngl', '0']` on the built command line.
- **A wrong `--bin` in `bench` is still a traceback at exit 1 — [open], reproduced.** Exit status
  measured as **1**, with a raw `FileNotFoundError: [Errno 2] No such file or directory:
  '/nonexistent/llama-bench'` traceback — while the module documents 2 for bad usage
  (`bench/bench.py:10`).
- **`test_label` still lives in `bench/bench.py` and is still imported into a test module — [open].**
  Defined at `bench/bench.py:56`, imported at `tests/test_bench.py:16`. The mechanism is confirmed:
  the imported function is named `test_*` and its signature `(r: dict) -> str` has a required
  argument, which is exactly what makes pytest's collection fail. **Not reproduced**, because pytest
  is not installed in this environment. `unittest` — what `README` and CI run — is unaffected and
  passes 180 tests. Rename it.

## 4. The benchmark's key metric isn't per-token — [open]

`bytes_read_per_token` is still `bytes_per_token(read_bytes, tokens)`
(`bench/bench.py:153`, `bench/measure.py:140`): the process's whole `read_bytes` divided by one pass
of tokens. The numerator still includes model load, warmup and llama-bench's repetitions, and a warm
page cache still makes reruns read ~0. `peak_gpu_mib` is still system-wide `nvidia-smi` memory
polled at intervals (`bench/measure.py:41`) with no pre-run baseline, so it counts other processes
and can miss spikes.

One thing did improve: the extra flags now reach llama-bench, so `-r 1 --no-warmup` is usable
without the `--` separator. The rest of the recommendation stands — diff a short run against a long
one (`-n 16` vs `-n 128`), evict the file cache between runs, take a pre-run GPU baseline, and use
`os.wait4`'s rusage (`ru_maxrss`, `ru_majflt`) for exact peak RSS and major faults for free.

## 5. Planner and design

- **`kv_cache: q8_0/q8_0` is still a constant, not a plan — [open].** Declared at
  `vipera/manifest.py:152`. It is a good default (current llama.cpp auto-enables flash attention for
  it and requires K=V types for MLA), but then it isn't a plan. Derive it from context length or
  drop it.
- **llama.cpp does keep the input embedding on the CPU — [confirmed], and it costs real VRAM.**
  `pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list }` (`src/llama-model.cpp:1665`) hard-wires the
  input layer to the CPU backend, and that list is what the loader uses for input-layer tensors —
  selected at `src/llama-model-loader.cpp:1218-1219` from the argument the call site passes
  (`src/llama-model.cpp:2363`). So the planner's reservation is real
  waste: on the real file, `token_embd.weight` is **112.5 MiB** (Q4_K), and `plan` puts the whole
  always-resident set in VRAM (`resident_vram_bytes: 833058816` = 794.5 MiB), making **112.5 MiB ≈
  3.1% of the planner's 3584 MiB usable VRAM** reserved for a tensor the runtime keeps in RAM. The
  earlier "~110 MiB" estimate was accurate. Either discount it or record that the plan is not what
  llama.cpp will do.
- **Test T1's premise before building it — [hypothesis], unchanged.** Stacked `*_exps` tensors
  already keep each expert contiguous, roughly 1.5–3 MiB per projection on V2-Lite, so frequency
  ordering may buy less than `docs/OPTIMIZER.md` expects. T6 alignment saves well under 1% at that size.
  Interleaving gate/up/down per expert (one read instead of three) looks more promising. Still a
  hypothesis, not a measurement.

## 6. Smaller things

- **`docs/ALGORITHMS.md`'s speculative-decoding formula is still inverted — [open].**
  `docs/ALGORITHMS.md:157` reads `speedup ≈ (1 − α) / (1 − α·β)`, which *falls* as acceptance rises
  (0.22 at α=0.8, β=0.1). The standard form is `(1 − α^(γ+1)) / ((1 − α)(γβ + 1))`, which gives 2.4
  there.
- **The headline numbers rested on an assumed bandwidth — [fixed].** The README's target-model
  section (previously `README.md:59`) said a cache miss on the DRAM-less NVMe ran at "**~2 GB/s**"
  and derived "**~1.7 tok/s**" and a "20×" gap from it, while the measured tier is 8 GB/s
  (`hw/gtx1650.toml:7`). `README.md` and `docs/ALGORITHMS.md`
  §1.b now use the measured constant: ~1.2 GB ≈ 0.15 s/token ≈ **6.7 tok/s**, a **~3×** gap, not
  ~20×. The same stale constant still sits in `docs/FEASIBILITY.md:50,86-87`, `docs/DESIGN.md:152`
  and `docs/OPTIMIZER.md:20`; the restatement stopped at the two files whose claims the measurement
  itself changes.
- **"not another runtime" vs the design — [partial].** `README.md:10` still says it, while
  `docs/DESIGN.md:127` reserves ONNX/MLC/ORT "for the router, not built yet" and `DESIGN.md:200-215`
  sketches "Flow of one decode step (P2 target)" with a router and a predictor. The design does mark
  the runtime work as not-yet-built, which softens rather than resolves the tension. Reconcile the
  positioning.
- **Quant-table hygiene — [open].** `TensorInfo.type_name` (`gguf.py:257`) is still defined and still
  never called. `_BLOCK` (`gguf.py:107`) is still not cross-checked against the official reader: no
  test imports `gguf.GGML_QUANT_SIZES`. Both are cheap and both guard the table that every byte
  total depends on.
- **`LICENSE` exists — [fixed].** An MIT `LICENSE` (1067 B) was added in `63014bb`, so
  `license = { text = "MIT" }` in `pyproject.toml` is no longer a claim without a file.

---

## 7. Position vs the expert-level offloading direction

There is a strategic direction — written up outside this repo, at `~/quest.md` — arguing that for MoE
the better target than layer-wise streaming (AirLLM) is **expert-level offloading: stream only the
experts the router picks — plus an expert cache, routing-aware prefetch, CPU–GPU co-execution, and
KV/quantization** (the direction FreeToken / SGLang / vLLM take). The old `analyze` + `plan` optimizer is pure arithmetic on tensor
sizes: it is *size-driven*, ranks no expert as "hot", and says so in
`placement.expert_order_pending_trace`.

The repo has moved further toward the direction since the second pass:

- **Routing traces** (`bench/capture_trace.py`, `vipera/trace.py`) replace the guess with real
  router choices.
- **Cache policy** (`vipera/schedule.py`) implements LRU/LFU/`hot`, and reports every hit rate
  against the **ceiling at the same cache size**, so a policy can't be flattered by a differently
  sized baseline.
- **Pruning** (`vipera/prune.py`) quantifies what dropping experts costs, and P4 (gated pruning with
  per-layer rollback, `d8c01f6`) is now implemented.
- **Prefetch exists as of this pass** — the part the second pass listed as untouched. The quest's
  "routing-aware prefetch" is built as a fork patch: `bench/llama-moe-prefetch.patch` uploads
  predicted experts into llama.cpp's MoE cache (`--moe-cache-mib`) while the previous layer
  computes, driven by a table from `bench/prefetch_table.py` and kept as evidence at
  `bench/results/olmoe-1b-7b-prefetch-table.txt` (0.95 routing mass = **541 experts across 16
  layers**, mean 33.8 of 64). That is the honest scale: routing is concentrated enough to help a
  cache already holding roughly half the experts, and too flat to help a much smaller one.

Where the quest and vipera meet: the wins are **runtime** behaviours, and vipera's stated scope is
*not another runtime* (§6). With the prefetch patch, that line is no longer clean — vipera now
modifies llama.cpp's runtime, not just the file layout. Still untouched: actual per-expert
streaming, CPU–GPU co-execution, and KV offload. Worth deciding explicitly whether vipera stays on
the file-layer side of that line or continues to extend a runtime consumer.

---

## Since the first pass

The second pass superseded the raw draft: it stripped the process narration, reported the then-current
test count (**136**, not 35), and tagged every finding. Its ledger:

| Status | Count | Items |
|---|---|---|
| fixed | 1 | manifest `fits`/`warnings` |
| partial | 3 | `GGUFError`→exit 2; bandwidth constants measured; "not another runtime" positioning |
| open | 10 | unknown-quant sizing, missing extent check, `gtx1650` RAM, bench `--` flags, bench `--bin`, `test_label`, per-token metric, KV constant, ALGORITHMS formula, missing LICENSE |
| not re-checked | 2 | "20×" scaling, input-embedding VRAM reservation |
| hypothesis | 1 | T1 frequency-ordering payoff |

## Since the second pass

This pass re-read every finding against the working tree. Of the ten items the second pass left
**open**, four are now fixed — three in the uncommitted changes, one (`LICENSE`) in a commit the
second pass had already missed — and the single item it tagged **partial** that was a real defect
(`GGUFError` → exit 2) is now fully mapped rather than half-mapped. The uncommitted work is confined
to the parser, the CLI, the bench harness and their tests.

| Status | Count | Items |
|---|---|---|
| fixed | 6 | unknown-quant sizing now raises, extent check, manifest `fits`/`warnings`, cli error mapping (exit 2), bench flags without `--`, `LICENSE` |
| partial | 2 | headline numbers still on an assumed disk bandwidth (README says ~2 GB/s, the profile says 8.0), "not another runtime" positioning |
| open | 10 | `gtx1650` RAM, constant "20×", malformed-input bounds (`MemoryError`/`RecursionError`), bench `--bin` exit code, `test_label`, per-token metric, KV constant, ALGORITHMS formula, quant-table hygiene, `--fit` baseline |
| confirmed | 2 | upstream `--fit` on by default and `--moe-cache-mib` both exist in the checkout the repo builds against; input embedding pinned to CPU — 112.5 MiB ≈ 3.1% of usable VRAM reserved for nothing |
| external | 1 | PR #29887 and release b11480 — no PR/release data offline, so still reported rather than checked |
| hypothesis | 1 | T1 frequency-ordering payoff |

The parser items that used to produce wrong totals with exit 0 are closed, and so are the two ways
its malformed input used to crash. The verdict-changing findings were fixed straight after this pass
(below), which left the README's own prose — deriving `~2 GB/s`, `~1.7 tok/s` and `~20×` from the
pre-measurement disk figure — as the one place where the stale assumption was still published. The
real-model run below restates that prose from measurement, and corrects a second ceiling the same
prose got wrong.

## Since the third pass

Everything this review found that could change a verdict, or crash instead of returning one, was
closed immediately, and each fix was measured on the same input before and after.

- **RAM budget** (`hw/gtx1650.toml`): the headroom now holds back the ~5 GiB the OS and the working
  environment actually keep, so `usable_ram` is 10 GiB as documented. Through `vipera plan` on a
  15.9 GiB Qwen3-30B-shaped header: **0 MiB of spill and no warning** before, **2456 MiB of spill**
  after. The real DeepSeek-V2-Lite file still plans with 0 spill both ways, so the docs' ✅ is
  untouched.
- **Spill penalty** (`vipera/manifest.py`): the per-miss cost is computed from the target's tiers and
  the averaged figure is weighted by the spilled share, verified through the CLI on one model and
  three targets — shipped profile **~5.5x** / **~1.7x averaged**, a 22 GB/s disk **~2.0x** / **~1.2x
  averaged**, and `disk_bw_gbs = 0` reported rather than divided by.
- **Malformed headers** (`vipera/gguf.py`): string lengths, array counts, nesting depth and header
  counts are all bounded by the file's own size, so the two reproduced crashes and their neighbours
  return `GGUFError`. Verified by the 36,272-variant fuzz above, and by the real 9.65 GiB model file
  still parsing to the same 377 tensors.

| Status | Count | Items |
|---|---|---|
| fixed since the third pass | 3 | `gtx1650` RAM budget, derived spill penalty, malformed-header bounds |
| still open | 7 | the remaining open items in the ledger above |
| test suite | 190 | was 180; ten new tests cover the three fixes |

## Since the real-model run

The headline numbers were then checked by running the machine instead of re-deriving them: the real
9.65 GiB `DeepSeek-V2-Lite.Q4_K_M.gguf`, `llama-bench` on CPU at 8 threads, the captured OLMoE
routing trace, and the official `gguf` reader as an independent cross-check of the parser.

- **The `~33 tok/s` ceiling was wrong by 1.65× — [fixed].** It derived the ceiling from bandwidth
  alone (1.2 GB/token at 40 GB/s). Decode is bounded by *both* bandwidth (measured 44 GB/s → ~37
  tok/s) and arithmetic (4.8 GFLOP/token against the measured 110 GFLOP/s → **~23 tok/s**), and the
  arithmetic bound is the lower one, so the published figure was never reachable. Measured: 19.24 tg128,
  19.60 tg32, 20.03 tg64 under `llama-bench`, then 19.7, 20.1 and **17.4** under `llama-cli` — **17.4–
  20.1 tok/s over six runs**, the top being 87% of the ALU ceiling and 61% of the published one. The
  spread is real and not fully explained (the later `llama-cli` runs were slower and read more from
  disk, consistent with page-cache pressure from the runs before them); separating load from steady
  state needs a cold cache and `drop_caches` needs root. `README.md` and `docs/ALGORITHMS.md` §1.b now
  state ~23 and the range rather than a single point.
- **The disk-derived figures restated — [fixed].** At the measured 8 GB/s a miss read costs ~1.2 GB
  ≈ 0.15 s/token ≈ **6.7 tok/s**, so the gap to the resident case is **~3×**, not ~20×: still a real
  target, but the old constant overstated it by ~4×. This supersedes the *headline numbers* row of
  the third-pass ledger above; its other `partial` (`README.md:10` positioning) is untouched.
- **What the run confirmed rather than contradicted:** the parser and the verdict. `analyze` on the
  real file takes 0.374 s for 377 tensors; the official reader agrees on every tensor — 0/377
  mismatched on offset, type, bytes and element count, the same 10,360,420,352-byte total, the same
  `data_offset`, 45/45 metadata keys. `plan` says 0 spill on both targets and the runs honour it:
  **11.8–148 MiB/token** of disk read against ~1.2 GB/token of weight traffic (three runs of
  192/32/32 tokens, page cache differently warmed each time), and ~20 tok/s where streaming the
  8.87 GiB expert pool would give 0.84 tok/s. The first pass through these notes quoted only the
  favourable end of that range; the range is what the evidence supports, and the weakness in it is
  stated rather than smoothed over.
- **The ✅ "fits 11 GiB RAM" is a claim about the file, not the process — [open].** Running the
  9.65 GiB model needs **13.0–13.3 GiB** of resident memory: 12.97 GiB peak RSS through `bench`,
  13.22 GiB sampled from `/proc`, and 13.28 GiB of `anon-rss` in the kernel's OOM record. `llama-cli`
  at its **default** context is OOM-killed on this box (`-c 4096` → SIGKILL after 7 s; `-c 512`
  completes), on a machine with 15.4 GiB total and no swap. The target table and `plan`'s 10 GiB
  `usable_ram` budget describe the weights, not the runtime's footprint, and the table's "Fits 11 GiB
  RAM?" column is read as if they were the same number.
- **Two things the run does not settle, now stated as open.** Decode scales oddly with threads (4 → 8
  threads buys +6%), so how much of the per-token cost is bandwidth rather than ALU is unknown;
  `docs/ALGORITHMS.md` §1.b now says that instead of asserting "decode is ~ memory-bandwidth-bound".
  And separating model load from steady state needs a cold page cache: `drop_caches` needs root, which
  this box does not grant, which is why the disk-read figures are given as a range.

**The real model now has an architecture profile** (`arch/deepseek-v2-lite.toml`), which it did not
before: values read from the file's own metadata (`block_count`, `expert_count`, `expert_used_count`,
`expert_shared_count`, `key_length` split into nope/rope, `kv_lora_rank`), `stated_*` from the
published 15.7B/2.4B. `vipera arch` reports the residual — **+1.9% total, +2.2% active** — which is
the modelled cost of not having a per-layer dense flag (see the blocker below). That profile makes
`vipera sweep` answerable for the model actually on disk, and the answer lands on the measurement:
**q4-experts predicts 20.04 tok/s** for this target against **20.03–20.2 tok/s** measured. That is the
first place in this repo where a modelled number and a real run of the same model on the same machine
agree to 0.1%.

**Blocker: the fork's trace could not drive those commands — [fixed].** The trace patch records
`ffn_moe_topk` per layer, so a trace carries **MoE layers only**: 26 for DeepSeek-V2-Lite, whose
`block_count` is 27 with `leading_dense_block_count = 1`. vipera had no way to express a dense leading
block, so `validate` compared 26 against `n_layers = 27` and `trace`, `prune` and `schedule --trace`
all rejected a trace captured from the real model with exit 2
(`error: token 0: trace has 26 layers, architecture has 27`). `arch/glm-4.5-air.toml` had carried the
same unexpressible fact since the first review (`first_k_dense_replace = 1`, "a small, one-layer
approximation").

`Architecture` now has `leading_dense_layers` + `dense_inter`: the dense blocks keep their attention,
gain one always-resident FFN, and contribute **no** router, shared experts or pool slots, so
`moe_layers = n_layers - leading_dense_layers` (26 here) is the dimension a trace carries. The
existing profiles are untouched by construction — `leading_dense_layers = 0` reduces every count to
the old `n_layers * ...` — and that is checked, not asserted: all four pre-existing profiles, the
OLMoE trace run and the GLM sweep are byte-identical before and after (133 lines of CLI output), and
`dense_ffn` is absent from the parameter summary unless a profile has one.

**The loop is now closed on the real model**, and both halves of it line up:

| check | modelled | measured |
|---|---|---|
| decode speed, `sweep` on the profile | 20.04 tok/s | 20.03–20.2 tok/s |
| decode speed, `schedule --trace` on the real 199-token trace | 22.35 tok/s | 20.2 tok/s |
| routed/shared/dense/MLA/router parameters | within 0.02% of the file's own `n_elements` | — |

The trace-driven figure is the honest one to quote first: it is 10% optimistic where the
bandwidth-and-ALU-only model was 0.1% off, because it charges the measured 110 GFLOP/s in full while
llama.cpp delivers ~90% of it. `schedule --trace` on that trace reports a **95.6% hit rate** at 3156
slots, 804.2 MiB resident and 6.70 GiB of experts, and `prune` on the same routing says 1292 of 1664
experts (77.6% of the pool) retain 95.3% of the routing mass.

**The profile now matches the file's own tensors, which explains the residual.** Summing `n_elements`
per component from the real GGUF gives routed 14.395B, shared 0.450B, dense FFN 0.067B, MLA 0.372B,
router 0.003B — every one of them matching the profile to 0.00%. The remaining −1.3% on the total is
one tensor the model does not count: the file carries **both** `token_embd.weight` and a separate
`output.weight` (209.7M each, 1.34% of the model), and `params_summary` counts `vocab × hidden` once.
That untied output head is still unmodelled — it is read once per token, so it affects modelled I/O
as well as the total — and is left as an open item rather than quietly absorbed.

| Status | Count | Items |
|---|---|---|
| fixed since the real-model run | 3 | the `~33 tok/s` ceiling, the disk-derived miss cost, the dense leading block blocking every trace command |
| open since the real-model run | 2 | the ✅ RAM verdict reads as a process footprint when it is a file size; the untied output head is not counted (−1.3%) |
| left for a doc sweep | 1 | 2 GB/s still quoted in FEASIBILITY §1/§4, DESIGN §152, OPTIMIZER §20 |
| test suite | 190 | unchanged — this pass touched no code |
