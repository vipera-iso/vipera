# FINDINGS — what six rounds of measurement established

> Every headline number vipera published came from a benchmark taken while the page cache was
> thrashing, on a thread count that was leaving a third of the machine idle, and in one case
> compared against a figure that was copied without its conditions. Re-measured on a controlled
> footing, the two figures the project led with both fell: the manifest's 1.22× does not
> reproduce, and the GPU verdict — which was the one that held — turns out to be a wider gap than
> reported.
>
> **Status labels:** `[measured]` = reproducible, command quoted · `[modelled]` = computed, not
> run · `[proposed]` = not implemented · `[published]` = someone else's, not re-verified.
> A number without a label is a defect. No range is quoted at its favourable end.
>
> Written 2026-10-13. All measurements on one machine, described below, and none of them
> generalise past it without saying so.

---

## 1. The environment these numbers belong to

Nothing here transfers to a different machine without re-measuring. `[measured]`

| | |
|---|---|
| CPU | Intel i5-11300H, 4 physical cores / 8 threads |
| RAM | 15 GiB total, **MemAvailable 12.4 GiB** (`12974684 kB`), no swap |
| VRAM | GTX 1650 Max-Q, 4 GB |
| Disk | WD SN530 NVMe, ~8 GB/s |
| Build | `llama.cpp`, CPU-only (`build/`), plus `build-cuda/` and `build-vulkan/` |
| Model | `DeepSeek-V2-Lite.Q4_K_M.gguf`, 9880 MiB, 15.7 B total / **2.4 B active**, MoE, 64 experts top-6 |
| OS | Arch, no sudo, `perf` unavailable (`perf` not installed, `perf_event_paranoid=2`) |

`MemAvailable` matters more than `MemTotal` and is the figure used throughout. It was quoted as
"~11 GiB" in earlier notes; the measured value is 12.4 GiB, and the correction is what makes
§6 visible at all.

Because `perf` is unavailable, attribution uses `llama-bench -o json` timing against
`/proc/<pid>/{stat,io,status}` and `/proc/diskstats` counters, sampled by `bench/step1_attrib.py`.
Because `ulimit -l = 8192`, `-lm mlock` is usable on the CPU-only build.

---

## 2. The rule that produced every correction

**Never run two benchmarks at once on this machine.** Each side of `vipera run --compare` loads
the 9880 MiB model, so two comparisons in parallel ask for ~20 GiB against 12.4 GiB available,
and the OOM killer takes one.

This is not merely a hygiene rule. An OOM kill surfaces as an `n/a` cell with **exit 0**, so a
killed run is indistinguishable from a missing measurement unless the `error:` line is read. Two
earlier `--compare` runs at ctx 2048 were killed this way and were written up as a property of the
model — that the f16 KV baseline "needs ~22.5 GB and dies". Re-run sequentially, the same ctx
completes three times out of three. **The attribution was mine and it was wrong.**

`vipera run --compare` now prints a `!! COMPARISON INVALID` banner naming the side that died and
its exit code, and suppresses the speedup outright rather than leaving an `n/a` cell to be skimmed
past. The exit code was already correct; the table was not.

---

## 3. Six rounds

### Round 1 — Attribute the decode bottleneck

**Question:** is disk, RAM, or compute the limit?

59.0 ms/token at the default invocation decomposes as: `[measured]`

| component | ms/token | share |
|---|---|---|
| ALU / compute | 43.7 | 74% |
| self-inflicted `mmap` paging | 15.3 | 26% |
| real NVMe during decode | **≤0.1** | **≤0.2%** |

Decode touches **0.4 MiB/token**. The ALU figure sits at the modelled ceiling of 43.5 ms, leaving
**~0.5% headroom**.

**There is no disk left for a file-layer transform to reclaim.** The 26% that is not compute is
paging the default invocation causes on its own.

### Round 2 — Load mode

`llama-bench`, five repetitions each. `[measured]`

| `-lm` | tok/s | spread | NVMe moved |
|---|---|---|---|
| `mmap` (default) | 16.95 | **±4.10** | **13757 MiB** |
| `dio` | 22.44 | ±0.19 | 9880 MiB (exactly the file) |
| `mlock` | **22.90** | ±0.07 | — |

`mmap` moves 13757 MiB for a 9880 MiB file — the file paged ~1.4× over. The honest mlock gain is
not one number: **+121% cold** (10.30 → 22.78), **+11% steady-state** (20.41 → 22.71), +32% as a
five-repetition mean. The main effect is killing cold-start thrash and an 18× variance, not
raising steady-state throughput.

### Round 3 — Thread count

Standalone runs, under `mlock`. `[measured]`

| `-t` | tok/s | gain over previous |
|---|---|---|
| 1 | 8.41 | — |
| 2 | 15.11 | 1.80× |
| **4** | **22.90** | 1.52× |
| 8 | 20.75 | **0.91× (−9%)** |

Gains fall 1.80× → 1.52× → 0.91× and then go negative: **the arithmetic ran out before the cores
did**. More threads is not more speed. Thread counts must be measured standalone — an in-process
`llama-bench -t "1,2,4"` reported t4 = 21.75 against a standalone 16.95.

### Round 4 — The manifest, on a clean baseline

**Question:** does vipera's manifest still beat llama.cpp defaults once the thrash is gone?

`vipera run --compare`, ctx 512, `-n 32`, sequential. `[measured]`

| footing | baseline | manifest | ratio |
|---|---|---|---|
| `-t 4 -lm mlock` | 22.80 | 22.50 | **0.99×** |
| `-t 4 -lm mlock`, repeat | 22.60 | 22.50 | **1.00×** |
| `-t 4`, `mmap` | 13.50 | 12.90 | **0.96×** |
| `-t 4`, fresh process, `mmap` | 20.70 | 19.10 | **0.92×** |

**The documented 1.22× does not reproduce on either footing.** And the mmap run carries direct
proof of why: the manifest cut disk read by **3918 MiB (−26%)** and still scored **0.96×**.
Fewer disk reads bought no throughput, which is Round 1's ≤0.2% in its most direct form.

At ctx 2048, three sequential runs: 1.04×, 0.81×, 1.12× — mean ≈ 0.99×. The 0.81–1.12× spread is
`mmap` thrash variance, not a context effect: the same settings at ctx 512 under `mlock` hold
0.96–1.00×.

### Round 5 — The GPU verdict, re-measured

The verdict read "CUDA 6.90 / Vulkan 3.40 vs CPU-only 22.90". Its provenance was the same defect
found elsewhere: the Vulkan run was issued with **no `--threads`** and compared against a CPU
figure of 11.7 t/s — a number now known to be 1.79× off. `-t` reaches the GPU builds too, because
with `--n-cpu-moe` the experts execute on the CPU.

Re-measured at `-t 4`, ctx 512, sequential. `[measured]`

| build | load mode | generation | vs CPU `mmap` |
|---|---|---|---|
| CPU-only | `mmap` | **16.00 t/s** | 1.00× |
| CPU-only | `-lm mlock` | **22.90 t/s** | 1.43× |
| CUDA | `mmap` | **6.80 t/s** | **0.43×** |
| Vulkan | `mmap` | **3.40 t/s** | **0.21×** |

The ranking holds and the gap widens. Comparing on the footing each build can reach: **CUDA
returns 43% and Vulkan 21% of what an unaided CPU already does.**

`-lm mlock` aborts the Vulkan build (SIGABRT after 1.4 s, peak RSS 146 MiB — it dies during
load), so the two sides cannot share a load mode, and the table names the footing per row rather
than implying one.

### Round 6 — The spill test, and why it cannot be run

**Question:** does the manifest win where placement has something to act on — a model that
overflows RAM?

`vipera plan` on `Qwen3-30B-A3B-Q4_K_M.gguf` (17.3 GiB) predicts a spill of **3867 MiB, 23% of
the expert pool** — squarely the regime. `[modelled]` The dry-run then maps it to exactly:

```
mapped to     -ngl 0 -ctk q8_0 -ctv q8_0
warning       … 3867 MiB of experts are planned to spill to disk, and llama.cpp has no
              per-expert disk streaming — it will fall back to mmap paging …
```

**Byte-identical to the flags for the model that fits.** In `vipera/run.py:plan_flags()`, the
non-GPU branch sets `ngl = 0` and the `spill > 0` branch appends a *warning* and **no flags**.

The timing half is not performable. `[measured]`

| attempt | result |
|---|---|
| `--compare -c 512 -n 32 -t 4` | both sides SIGKILL'd; peak RSS 10529 / 12664 MiB; disk 29748 / 30445 MiB |
| `llama-cli -c 256 -n 8 -t 4` | **EXIT=137 during model load** — before inference starts |

17.3 GiB against MemAvailable 12.4 GiB, with `overcommit_memory = 0` and a `CommitLimit` of
7.7 GiB. Not a tuning problem; a wall.

**That wall was mine, not the model's — corrected 2026-10-15.** The kill was not the expert pool
overflowing RAM. It was `--repack`, on by default. llama.cpp's CPU backend copies **3D** weight
tensors used by `GGML_OP_MUL_MAT_ID` — the MoE expert matmul — into a `CPU_REPACK` buffer, and
that buffer is **anonymous** memory, not file-backed (`ggml-cpu.cpp`
`ggml_backend_cpu_repack_buffer_type`; `repack.cpp` `extra_buffer_type::supports_op` accepts
`MUL_MAT_ID` with `ggml_n_dims == 3`). Repacking is a CPU *speed* feature that silently converts
a disk-streamable model into a must-fit-in-RAM one. `-nr` turns it off. `[measured]`

`bench/moe_spill.py` runs the two arms back to back, 3 repetitions each, one flag apart:

| arm | outcome | exit | peak anon RSS | peak file RSS | tok/s |
|---|---|---|---|---|---|
| default (repacking on) | **OOM-killed 3/3** | −9 | **11552 MiB** | 10392 MiB | — |
| `-nr` | **ran 3/3** | 0 | **112 MiB** | 11327 MiB | **5.6** (5.0–6.2) |

The RSS columns are the mechanism, not decoration: the default arm's ~11.5 GiB is *anonymous* and
therefore unreclaimable, so it exhausts a machine with 11.5 GiB available; the `-nr` arm's
~11.3 GiB is *file-backed* and therefore reclaimable, and the kernel pages the expert slabs in and
out as they are needed. The anonymous figure was identical to the MiB across all three `-nr` runs,
which is what makes it a property of the flag rather than of the run. No `dmesg` is available
without root, so the kill is identified from the signal plus peak RSS sitting at the machine's
limit.

A second measurement, on models that **fit**: disabling repacking is not even a tradeoff here.
`llama-bench`, `-t 4 -lm mlock`, 512 prompt / 32 generated. `[measured]`

| model | `--repack` | pp512 | tg32 |
|---|---|---|---|
| DeepSeek-V2-Lite | 1 | 50.51 ± 0.29 | 22.95 ± 0.09 |
| DeepSeek-V2-Lite | 0 | **72.31 ± 0.44** | 23.20 ± 0.03 |
| Qwen3-4B | 1 | 69.05 ± 0.47 | 15.43 ± 0.25 |
| Qwen3-4B | 0 | **94.63 ± 2.65** | 15.47 ± 0.12 |

Prompt processing is **+37–43%** faster with repacking disabled and generation is unchanged, on
both models. The DeepSeek row was re-run with the arms in the opposite order (72.31/50.51), because
the first pair looked like an ordering artefact; the effect reproduces either way, with error bars
tight enough that the gap is well outside the noise. **The flag's effect is measured; the reason a
*repack* makes prompt processing slower is not established**, and it is not claimed. `-nr` also
disables every CPU-specific extra buffer (AMX, KleidiAI), so on a machine that has those the
direction could reverse — which is why vipera emits it for the spill case, where it is the
difference between running and not running, and does not apply it by default.

**A third arm, and the plan it cancelled.** Skipping the repack of *only* the 3D expert tensors
(`bench/llama-cpu-repack-moe.patch`, `GGML_CPU_REPACK_MOE=0`) leaves every 2D weight repacked and
puts the expert slabs back in the file. Anonymous RSS lands at **585 MiB** — the non-expert weights
and nothing else — which confirms the account of the memory above, and the model runs:

| arm | outcome | peak anon RSS | peak file RSS | tok/s |
|---|---|---|---|---|
| default | OOM-killed 2/2 | 11445 MiB | 8289 MiB | — |
| `-nr` | ran 2/2 | 112 MiB | 11350 MiB | 5.85 (5.5–6.2) |
| gate (`GGML_CPU_REPACK_MOE=0`) | ran 2/2 | 585 MiB | 11074 MiB | 5.55 (5.0–6.1) |

It is a cleaner mechanism and **not a faster one**: 5.55 against 5.85 with overlapping ranges, so
repacking the 2D weights buys nothing measurable here. `-nr` therefore stays the recommendation,
because it needs no rebuild.

**And it cancelled the obvious next project.** With `-nr`, the run reads **0.0 MiB per token** from
storage after load completes (`bench/moe_spill.py`, the `disk_mib_per_token` column): the working set
fits in the resident file pages, so the kernel charges the disk for nothing. There is no disk
traffic for an explicit host-side expert cache to win back, which means the cache — vipera's own
LFU/hot policies, trace-derived frequency tables, an explicit slot store — would have been a large
build for no gain. That is now known rather than assumed, and it cost one column in a harness.

The remaining gap is **5.6 tok/s against a modelled ALU ceiling of ~16.7** for 3.3 B active
parameters, and it is **not disk**: it is CPU time, and the candidates are `mmap` minor-fault
overhead on the expert pages and the non-repacked Q4_K matmul the expert path falls back to.
Neither has been separated yet, so no attribution is claimed.

**But the spill regime is already occupied.** The 9.65 GiB model sits *on* the margin of 12.4 GiB
rather than comfortably inside it — `mmap` moved 13757 MiB for a 9880 MiB file (§Round 2), i.e.
mildly spilling, and it is exactly where `-lm mlock` bought its +121%. **The manifest measured
0.99× there.** The usable window between "fits, mildly pages" and "does not start" is roughly
9.65 → 12.4 GiB, and contains no model that separates the manifest from `-ctk q8_0 -ctv q8_0`.

---

## 4. The four findings

**1. The bottleneck is compute, not disk.** 74% ALU, ≤0.2% NVMe during decode, ~0.5% headroom.
Determined by thread scaling, which is the test that distinguishes a compute wall from a
bandwidth wall: gains that fall away mean compute. `[measured]`

**2. Two flags are worth 1.79×, cost nothing, and vipera does not recommend them.**
`-t <physical cores> -lm mlock` takes the documented 12.80 to 22.90 tok/s with an 18× tighter
variance, no rebuild and no code. It is larger than anything in the file layer, independent of
the model, and until now absent from every vipera output. This is the largest gap between what
vipera knows and what vipera says. `[measured]`

**3. The manifest measures 0.92–1.00× on a CPU-only build, and the 1.22× is retracted.** Cutting
disk reads by 26% moved throughput by −4%. `[measured]`

**4. The manifest represents more than the runtime can execute.** `[measured]` + `[modelled]`

| vipera can express | llama.cpp can execute |
|---|---|
| per-expert VRAM placement | `-ngl N` — whole layers only |
| per-layer KV placement | `-ctk/-ctv` — all of the KV, uniformly |
| per-expert disk streaming | none; falls back to `mmap` paging |

On CPU-only this gap is invisible, because the manifest degrades down to flags llama.cpp already
has. On GPU it shows as "cannot be honoured". On spill it shows as a warning with no flags. **The
manifest's only real lever on a CPU-only build is KV quantization**, and against a multi-GiB
spill that lever is worth ~770 MB `[modelled]` (48 layers × 8 GQA KV heads × 128 dim ≈ 192 KiB/token
f16 → q8_0 at ctx 8192).

This is a limit of the runtime, not a defect in vipera. But it bounds what a planning layer can
be worth, and it should be stated in the claim rather than discovered by the reader.

---

## 5. Two further results

**GPU offload does not help, under every condition measured.** 0.43× (CUDA) and 0.21× (Vulkan) of
an unaided CPU — Round 5. With `--n-cpu-moe` the 2.4 B active expert params run on the CPU
either way, so the GPU accelerates attention while adding a device transfer at every layer
boundary, and does not come close to paying for it.

**GLM 5.3 does not run, and the reason is specific: it is short of ALU, not RAM.** 18 B active
against 2.4 B is 7.5× the arithmetic, on a ceiling already at 99.5%. ALU cannot be reordered at
the file layer, so no quantization, cache, or placement choice moves it. `[modelled]`

---

## 6. What is not claimed

- **Nothing here generalises past the machine in §1.** The CPU-bound verdict is a statement about
  a 4C/8T laptop running a 2.4 B-active MoE that nearly fills RAM. A machine where the model fits
  comfortably in VRAM is a different problem with different degrees of freedom.
- **The spill regime was reached, but only for the load flag.** The model now runs (§Round 6
  correction), so "unreachable" was wrong. What is still unmeasured is whether the manifest's
  *placement* pays once the model is running: with no per-expert disk streaming in llama.cpp, the
  spill falls back to `mmap` paging, and the manifest's modelled hit rate does not describe it.
- **`-lm mlock` was not tested on a spilling model**, because it cannot be — there is nothing to
  lock once the model exceeds RAM. Its +121% / +11% / +43% figures apply to the fits-in-RAM case.
- **The CUDA/Vulkan rows are `mmap` only**, because `mlock` aborts the Vulkan build. Whether the
  GPU gap narrows under a load mode the CPU can use and they cannot is untested, and would only
  widen further.
- **"CUDA + prefetch = 2.00 t/s" is `[superseded]` and not re-measured.** It is kept only for the
  patch's own verdict: 11.2% of predicted experts useful, +44 GiB of extra upload.

---

## 7. Reproduction index

All commands assume `cd ~/dev/vipera`, and **none of them may be run concurrently** (§2).

| result | command |
|---|---|
| Round 1 attribution | `python3 bench/step1_attrib.py` (raw output: `bench/results/step1-attrib.json`) |
| Round 2 load modes | `llama-bench -m models/DeepSeek-V2-Lite.Q4_K_M.gguf -p 512 -n 128 -t 4 -lm mmap\|dio\|mlock -o json` |
| Round 3 threads | same, `-lm mlock -t 1\|2\|4\|8`, one process per thread count |
| Round 4 manifest | `python3 -m vipera run models/DeepSeek-V2-Lite.Q4_K_M.gguf --manifest out/DeepSeek-V2-Lite.Q4_K_M.vipera.json --compare --context 512 --n-predict 32 --threads 4 -- -lm mlock` |
| Round 5 GPU | §3 Round 5 table, one row at a time |
| Round 6 spill | `python3 -m vipera plan models/Qwen3-30B-A3B-Q4_K_M.gguf --target hw/gtx1650.toml --out out/` then the same `run … --dry-run` |
| GPU detection | `ls ~/dev/llama.cpp/build/bin/` — no `ggml-cuda`/`ggml-vulkan` ⇒ `detect_gpu_backend()` is false |
