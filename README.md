# vipera

[![CI](https://github.com/vipera-iso/vipera/actions/workflows/ci.yml/badge.svg)](https://github.com/vipera-iso/vipera/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**An optimization layer for local LLM model files, so they run on weak hardware** — starting with
**personal computers**, then **phones**.

`vipera` is not another runtime. It is **a layer between the model file and the runtime**: it takes an
artifact (GGUF/safetensors), **rewrites it into an optimized artifact**, and emits a *manifest* that
tells the runtime what lives where (VRAM/RAM/disk), in what load order, and which experts are "hot".
`vipera run` then **drives an existing runtime** (llama.cpp) from that manifest — it executes the
model instead of only predicting its speed, without reimplementing a runtime.

Inspired by [FreeToken](https://github.com/FlashML-org/FreeToken) (MoE serving on consumer hardware)
and [AirLLM](https://github.com/lyogavin/airllm) (layer-wise streaming) — but both assume plenty of
GPU/RAM. `vipera` targets the hard spot: **the model fits in RAM yet overflows VRAM, on a slow disk**.

> Status: **stage 3 — it runs.** stage 1 (analyzer + residency planner) and stage 2 (per-layer
> architecture model + a bandwidth/cache-aware streaming scheduler) plan a residency manifest, and
> `vipera run` now **executes the model for real**: it reads that manifest back, translates it into a
> llama.cpp command line, launches the binary and measures throughput, peak RSS and disk reads. The
> tensor data is not rewritten yet and vipera does not reimplement a runtime — it drives the
existing one from its own manifest — but the layer no longer only predicts. Everything `schedule`
> reports is still **modelled** and says so; `run`'s numbers are measured.

---

## The idea: optimize at the **file** layer, not just at runtime

Runtime tricks (expert cache, prefetch) only get you to the I/O ceiling. `vipera` goes one step
further: **rearrange the model file itself** so a weak runtime reads less and reads sequentially.

| File-layer transform | Input | Output | Benefit on weak hardware |
|---|---|---|---|
| **Expert hot/cold reorder** | routing traces | re-laid-out file | hot experts become contiguous ⇒ one sequential read |
| **Residency manifest** | model + hardware | `*.vipera.json` | runtime knows the VRAM/RAM/disk split instead of guessing |
| **Block-wise quant** | safetensors | 4-bit block GGUF | stream block by block via `mmap` |
| **Expert pruning/merge** | usage frequency | fewer experts | smaller ⇒ fits RAM |
| **KV placement hints** | config | manifest | where KV lives, how many bits |
| **Chunk alignment** | model | page/disk aligned | avoids over-reading on a DRAM-less NVMe |

## Target machine (measured 2026-10-12)

| | |
|---|---|
| CPU | Intel i5-11300H — 4C/8T, up to 4.4 GHz |
| RAM | 15 GiB, ~10 GiB available, **no swap** |
| GPU | NVIDIA **GTX 1650 Max-Q, 4 GB VRAM** (compute 7.5) |
| iGPU | Intel Iris Xe (Tiger Lake) |
| Disk | WD SN530 NVMe **DRAM-less**, 404 GB free |
| OS | Linux |

## Target models

| Phase | Model | Total / active | Q4_K_M (estimated) | Fits 11 GiB RAM? |
|---|---|---|---|---|
| P0–P2 | **DeepSeek-V2-Lite** | 15.7B / 2.4B | ~9.5–10.4 GB | ✅ |
| stretch | Qwen3-30B-A3B | 30.5B / 3.3B | ~18.6 GB | ❌ (needs streaming) |

**The decisive number:** for DeepSeek-V2-Lite the RAM-resident ceiling is **~23 tok/s**, and it is an
*arithmetic* ceiling, not a bandwidth one: 2.4B active × 2 = 4.8 GFLOP/token against the measured
**110 GFLOP/s**. The real 9.65 GiB Q4_K_M file decodes at **17.4–20.1 tok/s** across six runs — three
with `llama-bench`, three with `llama-cli` — and the top of that range is 87% of the ceiling. During
decode the same runs read **11.8–148 MiB/token** from disk across three runs (192/32/32 tokens, page
cache in a different state each time). Both ranges are far below the ~1.2 GB/token of weight traffic,
which is what `plan`'s "0 spill" means in practice: nothing streams. The spread itself is not fully
explained — separating load from steady state needs a cold cache, and `drop_caches` needs root, which
this box does not grant. One measured setting, reproducible with:

```bash
llama-bench -m models/DeepSeek-V2-Lite.Q4_K_M.gguf -ngl 0 -t 8 -p 32 -n 64 -r 1   # tg64 20.0, pp32 43.7
```

The earlier **~33 tok/s** here came from bandwidth alone; the measured ALU bound rules it out.

A cache miss reading the same weights from the NVMe — measured at **8 GB/s**, not the 2 GB/s this
paragraph used to assume — costs ~1.2 GB ≈ **0.15 s/token**, i.e. **~6.7 tok/s**. That is a **~3×**
gap, not the ~20× the old constant implied: still worth removing, which keeps vipera's primary goal
**fewer disk reads for experts**.

## Quick start

Pure stdlib, Python >= 3.11 (uses `tomllib`). No dependencies to install.

```bash
# inventory a model without reading its tensor data
python3 -m vipera analyze models/deepseek-v2-lite-Q4_K_M.gguf
python3 -m vipera analyze models/deepseek-v2-lite-Q4_K_M.gguf --json

# plan weight placement for the measured target, and write the manifest
python3 -m vipera plan models/deepseek-v2-lite-Q4_K_M.gguf \
    --target hw/gtx1650.toml --out out/
# -> out/deepseek-v2-lite-Q4_K_M.vipera.json

# model a huge MoE against this machine and see whether it can run at all
python3 -m vipera arch     arch/glm-5.3-flash.toml
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --out out/
python3 -m vipera sweep    arch/glm-5.3-flash.toml --target hw/gtx1650.toml

# routing traces: inspect a trace, then simulate decode driven by it
python3 -m vipera trace trace.jsonl --arch arch/glm-5.3-flash.toml \
    --target hw/gtx1650.toml --quant q4-experts     # cache size matched to the schedule
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml \
    --quant q4-experts --trace trace.jsonl --cache-policy hot --eval-tail 0.3

# which experts survive a given sparsity, and how much routing mass that keeps
python3 -m vipera prune trace.jsonl --arch arch/olmoe-1b-7b.toml --target-mass 0.95

# the LLM-PC-OPT instance: dense Llama-3-70B on 8 GB VRAM / 32 GB RAM / 3 GB/s SSD
python3 -m vipera schedule arch/llama3-70b.toml --target hw/llama70b-pc.toml --quant q4-experts --context 4096

# the LOCAL-LLM-16GB instance: 70B dense on a 16 GB unified-memory laptop
python3 -m vipera score arch/llama3-70b.toml --target hw/laptop16-igpu.toml --context 32768 --quality 99 --power 65

# run the model for real, with the manifest choosing the flags, and measure it
python3 -m vipera run models/deepseek-v2-lite-Q4_K_M.gguf \
    --target hw/gtx1650.toml --context 1024 --n-predict 32
# reuse a manifest instead of re-planning, or print the command without running it
python3 -m vipera run models/deepseek-v2-lite-Q4_K_M.gguf \
    --manifest out/deepseek-v2-lite-Q4_K_M.vipera.json --dry-run

# tests
python3 -m unittest discover -s tests -t .
```

## Benchmark

`llama-bench` reports t/s but not what a weak machine actually pays. `bench/` wraps it and
records peak RSS, peak VRAM and disk read bytes as well.

```bash
python3 -m bench.bench \
    --bin ~/dev/llama.cpp/build/bin/llama-bench \
    --model models/deepseek-v2-lite-Q4_K_M.gguf \
    --ngl 0 --threads 4 --json bench/results/baseline.json
```

## Real routing traces (S7)

llama.cpp does not export the router's choices, so a local ~50-line patch to
`src/llama-context.cpp` (`LLAMA_MOE_TRACE=<path>`) marks the already-named
`ffn_moe_topk` tensor as a graph output and reads it back after each decode step.
It changes no computation. `bench/` drives it and writes the JSONL the scheduler
reads:

```bash
python3 -m bench.capture_trace \
    --bin ~/dev/llama.cpp/build/bin/llama-cli \
    --model /path/to/olmoe-q4km.gguf --tokens 512 --threads 8 --ngl 0 \
    --out bench/results/olmoe.jsonl
python3 -m vipera trace bench/results/olmoe.jsonl --arch arch/olmoe-1b-7b.toml --target hw/gtx1650.toml
```

On a real OLMoE-1B-7B trace (511 tokens, 64 experts, top-8), evaluated at a
constrained budget against the size-matched ceiling:

| cache | ceiling | LRU | LFU | `hot` |
|---|---|---|---|---|
| 10% | 42.7% | 0.0% | **0.97×** | 0.93× |
| 25% | 71.1% | 0.75× | **0.97×** | 0.93× |
| 50% | 94.1% | 0.99× | 0.99× | 0.98× |

Three corrections this bought: on *stationary* real routing **LFU beats `hot`**
(aging only pays when the distribution moves — confirmed on real topic drift in
§10), **per-layer allocation does not help** (§8), and expert pruning is **not
free**: at the 5% sparsity the scheduler assumed, only 21–25% of routing mass
survives, so ~53–71% of experts must be kept to hold 95% of the mass (§11).
Details in [`docs/ROADMAP.md`](docs/ROADMAP.md) §8–§11.

Harness validation only — a 260K-parameter toy model, CPU-only, 4 threads:

```
ok True · exit 0 · peak RSS 1.2 MiB · disk read 0.0 MiB
  pp128   92234.97 t/s
  tg64    14775.72 t/s
```

Those confirm the harness runs; they say nothing about the target MoE, which still needs a
real model run.

What it computes today:

| Command | Output |
|---|---|
| `analyze` | layers, expert count, and bytes split into resident / router / shared-expert / "other" / routable-expert buckets |
| `plan` | a `*.vipera.json` residency manifest: how many bytes go to VRAM, RAM, and disk spill, plus warnings |
| `trace` | a routing trace's skew (entropy) and the **static-cache hit ceiling at the size the schedule will actually use** |
| `prune` | the per-layer keep-set that retains a target routing mass, next to what uniform pruning would keep |
| `schedule` | modelled bytes/token and tok/s, plus the cache hit rate **and the ceiling at the same cache size** (`policy / ceiling` ratio) |
| `run` | a real llama.cpp run driven by a manifest: the mapped flags, throughput, peak RSS and disk read bytes |

The planner is **size-driven** for now: it cannot rank *which* experts are hot until routing traces
from Phase 2 exist, and it says so in `placement.expert_order_pending_trace`.

## Running a model (`vipera run`)

Everything above decides *where weights should live*. `vipera run` is the half that actually
**executes** — it takes a manifest, maps it onto llama.cpp flags, launches the binary, and measures
what happened. vipera drives the existing runtime from its own manifest; it does not reimplement one.

The mapping is deliberately small and inspectable:

| Manifest field | llama.cpp flag |
|---|---|
| every MoE layer's experts in RAM | `--n-cpu-moe <n_layers>` |
| `expert_cache_vram_bytes` | `--moe-cache-mib <MiB>` |
| `resident_vram_bytes` (attention/norms) | `-ngl <n_layers>` |
| `kv_cache.type_k` / `type_v` | `-ctk` / `-ctv` |

Two things it will **not** pretend to do: a build with no GPU backend cannot honour VRAM placement
(so it warns and runs `-ngl 0`), and llama.cpp has no per-expert disk streaming, so a plan with
experts spilled to disk is reported as unmappable rather than silently approximated.

A real run on this machine — the 9.65 GiB DeepSeek-V2-Lite Q4_K_M, the shipped manifest, the
CPU-only llama.cpp build, 4 threads, 1024-token context:

```
$ python3 -m vipera run models/DeepSeek-V2-Lite.Q4_K_M.gguf --target hw/gtx1650.toml \
      --context 1024 --n-predict 24 --threads 4
manifest      computed from DeepSeek-V2-Lite.Q4_K_M.gguf + gtx1650-laptop
mapped to     -ngl 0 -ctk q8_0 -ctv q8_0
warning       llama.cpp was built without a GPU backend, so the manifest's VRAM placement
              (794 MiB resident + 2790 MiB expert cache) cannot be honoured; running CPU-only
ok            True   (exit 0)
wall          9.09 s
peak RSS      12681.5 MiB
disk read     5264.7 MiB
prompt        10.0 t/s
generation    11.70 t/s
```

That is a measurement, not a model.

### `--compare`: the baseline, measured, not assumed

`--compare` runs plain llama.cpp defaults and the manifest-driven command back to back and prints the
difference, so a "beats the default" claim is a measurement rather than a target:

```
metric                  baseline      manifest         delta
------------------------------------------------------------
generation t/s              ~20           ~20           1.00x
peak RSS                  ~13000        ~13000         +0 MiB
disk read                  ~5000         ~5000        +0 MiB

manifest / baseline generation speedup: 1.00x
```

### Prefetch from a routing trace

`--trace` learns the prefetch table from a real routing trace *and* arms the MoE prefetch patch in
one step, replacing the manual `bench/prefetch_table.py` → `LLAMA_MOE_PREFETCH=…` workflow (the patch
itself is untouched — vipera drives it, it does not fork llama.cpp's core):

```bash
python3 -m vipera run model.gguf --target hw/gtx1650.toml \
    --trace bench/results/olmoe-1b-7b-511tok.jsonl --prefetch-depth 2 --prefetch-top 32
```

If the binary does not carry the patch, the runner says so (`has no MoE prefetch support`) instead of
letting a run look like a prefetch that simply did not help. `--prefetch-table PATH` uses a
pre-built table; `--prefetch-mass N` sets how much of each layer's routing mass the table must cover.

### Exercising the VRAM half

Without a GPU backend the manifest's VRAM placement is reported and skipped. To exercise it, build
llama.cpp's Vulkan backend (works on an NVIDIA driver through its Vulkan ICD, no CUDA toolkit needed):

```bash
cmake -S ~/dev/llama.cpp -B ~/dev/llama.cpp/build-vulkan -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON -DLLAMA_CURL=OFF \
    -DVulkan_INCLUDE_DIR=$HOME/vksdk/Vulkan-Headers/include \
    -DCMAKE_PREFIX_PATH=$HOME/vksdk/install          # SPIRV-Headers
cmake --build ~/dev/llama.cpp/build-vulkan --target llama-cli -j$(nproc)
python3 -m vipera run model.gguf --target hw/gtx1650.toml \
    --binary ~/dev/llama.cpp/build-vulkan/bin/llama-cli
```

With a GPU backend present the runner maps the manifest to `-ngl <n_layers> --n-cpu-moe <n_layers>
--moe-cache-mib <MiB>`: attention in VRAM, every expert on the CPU, and the VRAM expert cache sized
from the manifest.

On this box the Vulkan build sees both GPUs and the run is pinned to the discrete one:

```
$ python3 -m vipera run models/DeepSeek-V2-Lite.Q4_K_M.gguf --target hw/gtx1650.toml \
      --binary ~/dev/llama.cpp/build-vulkan/bin/llama-cli --context 512 --n-predict 16 \
      -- --device Vulkan1 --moe-cache-mib 512
mapped to     -ngl 27 --n-cpu-moe 27 --moe-cache-mib 2789 -ctk q8_0 -ctv q8_0
generation    2.60 t/s          # GTX 1650, ~10.4 GiB peak RSS
```

Two honest notes. The manifest's own 2790 MiB expert cache does **not** fit next to the compute
buffers — the first attempt died with `failed to allocate Vulkan1 buffer`, so the run needed a
smaller `--moe-cache-mib` through the `--` pass-through. And with experts on the CPU the GPU path is
**slower** than CPU-only here (2.60 vs 11.7 t/s): shuffling experts across a 4 GB link costs more
than the attention it accelerates. Exercising the VRAM half is not the same as winning with it.

Armed with a real trace captured from this model, the prefetch patch reports its own verdict:

```
prefetch      474 of 5440 predicted experts used (8.7%), +28798.0 MiB upload
generation    1.30 t/s
```

A 199-token trace and a 512 MiB cache make the prediction mostly wrong — 8.7% useful for 28 GiB of
extra upload, and the run got slower. That is the same lesson `docs/ROADMAP.md` §11 records: a
table concentrated enough to fit a small cache does not come from a short trace. The number is the
patch's, not vipera's, and the runner only surfaces it (`-v` is added automatically so the verdict
is not hidden).

## Skill

[`skills/local-llm-optimization/SKILL.md`](skills/local-llm-optimization/SKILL.md) — a packaged,
sourced skill for running LLMs locally: decision tree, technique catalogue, three hardware
playbooks, comparison tables, pitfalls and a runnable hardware probe
([`scripts/probe.py`](skills/local-llm-optimization/scripts/probe.py)). Claim-to-source anchors are
in its §Sources; unpublished numbers are marked "no data".

[`skills/expert-io-loop/SKILL.md`](skills/expert-io-loop/SKILL.md) — the **measurement protocol**:
baseline → localize → hypothesis → one change → re-measure → verdict. Ships two stdlib scripts,
[`aggregate.py`](skills/expert-io-loop/scripts/aggregate.py) (median of ≥3 runs + spread, refuses
incomparable inputs) and [`compare.py`](skills/expert-io-loop/scripts/compare.py) (verdict against a
7% noise band and the ≤0.3 perplexity gate; the exit code *is* the verdict). The 7% band is measured
in this repo, from `bench/results/hwprobe-run{1,2}.json`.

## Documents

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — the project decomposed into sub-problems with a
  dependency graph, priorities, success criteria, open problems, and the logs of three
  autonomous solving rounds (S1 trace loader, S2 cache policy, S7 real trace + the 100B question).

- [`docs/PROPOSALS.md`](docs/PROPOSALS.md) — **research proposals**: four new/improved
  algorithms (skew-adaptive cache, tier-aware bit allocation, energy-aware tiering, gated pruning)
  with pseudocode, complexity, experiment design and a hostile self-review. P1 is implemented.
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — **survey + evaluation of local-LLM optimization
  algorithms** (quantization, pruning, offloading, KV, speculative), scored on bytes/token and
  tied to reproduciable vipera numbers.
- [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) — **can GLM-5.3-Flash (321B/18B) run here?** The
  modelled verdict, per-quant, and the ceilings that make it infeasible.
- [`docs/OPTIMIZER.md`](docs/OPTIMIZER.md) — **the model-file optimization layer** (vipera's core
  positioning).
- [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — algorithm analysis (streaming, MoE offloading,
  co-execution, quantization, KV cache, speculative decoding, model compression).
- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, FreeToken/AirLLM comparison, 1-month roadmap,
  tech stack, KPIs, risks, **7-day plan**.

## Stage 2 — the layer scheduler

`vipera/arch.py` models a hybrid MoE or a dense model per layer (KDA/MLA/full-GQA attention, sparse
experts or a dense FFN, mHC residual, MTP and vision), and `vipera/schedule.py` places every layer's
weights across VRAM/RAM/disk and simulates decode to answer one question: **how many bytes per
token, from which tier, and how fast?**

The headline exercise is a 321B-A18B model on the target laptop. Modelled result: **~0.65 tokens/s**
(Q4), I/O-bound. `vipera sweep` then models the layer-level optimization families (quantization,
expert pruning, layer pruning, KV compression, speculative/MTP) to show which one actually changes
the verdict — expert pruning, because the expert pool is 30× the cache. See
[`docs/FEASIBILITY.md`](docs/FEASIBILITY.md).

Expert caching over `(layer, expert)` supports three eviction policies: `lru`, `lfu`, and `hot`
(admission filter + pinned hot set + aging). Every scheduled run reports the achieved hit rate
**against the static-cache ceiling at the same slot count**, so a policy can never be flattered by a
baseline at a different cache size. On a synthetic non-stationary trace, `hot` reaches **0.92×** of
the matched ceiling over the post-shift window where plain `lfu` stalls at **0.71×**. Numbers,
ablation and the aging sweep are in [`docs/ROADMAP.md`](docs/ROADMAP.md) §7.

## Scope (one month, zero budget)

- ✅ PC-first (no Android device available to test ⇒ "phone" stays at the design/portability level).
- ✅ GGUF + llama.cpp, Python for the optimizer layer.
- ❌ Not this month: ONNX/ExecuTorch/LiteRT runtimes, gRPC, distillation.

## License

[MIT](LICENSE) © 2026 vipera-iso

