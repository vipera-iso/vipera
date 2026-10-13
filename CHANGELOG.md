# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project intends to follow [Semantic Versioning](https://semver.org/).

> **No tagged releases yet.** `pyproject.toml` reads `0.0.1` and everything
> below is unreleased. The entries were reconstructed from the commit history
> when this file was added, so they summarise rather than enumerate.

## [Unreleased]

### Added

- **`vipera run` — the runtime half** — reads a residency manifest, maps it onto
  llama.cpp flags (`--n-cpu-moe`, `--moe-cache-mib`, `-ngl`, `-ctk`/`-ctv`),
  launches the binary and measures throughput, peak RSS and disk read bytes. It
  refuses to silently approximate what llama.cpp cannot do: a GPU-less build and
  a disk-spilling plan are both reported instead of faked.
- **`vipera run --compare`** — runs plain llama.cpp defaults and the
  manifest-driven command back to back and reports the difference, so the
  "beats the default" claim is measured against the real baseline.
- **Expert prefetch wired into the runner** — `vipera run --trace` learns the
  prefetch table from a routing trace and arms the MoE prefetch patch
  (`LLAMA_MOE_PREFETCH`) in one step. The patch is driven through its env vars and
  is not forked; a binary without it is detected and reported. The table logic
  moved to `vipera/prefetch.py`, with `bench/prefetch_table.py` kept as its CLI.
- **GPU build paths** — `vipera run` targets a `GGML_VULKAN` or `GGML_CUDA`
  llama.cpp build so the manifest's VRAM placement is actually exercised. Both are
  verified on the target machine; CUDA installs root-free from NVIDIA's PyPI
  wheels (no `nvcc` was present before). Measured on the same model: CUDA 7.00
  tok/s against Vulkan 2.60 and CPU-only 11.70, with experts on the CPU because
  the model does not fit 4 GB of VRAM.
- **GGUF analyzer and residency planner** — reads a model's header and tensor
  descriptors (never the tensor data) and splits bytes into always-resident
  versus routable-expert buckets.
- **Per-layer architecture model** — hybrid attention (MLA / KDA / GQA),
  sparse experts, shared experts, multi-stream residual and MTP, as pure
  parameter arithmetic, so feasibility can be reasoned about without a model.
- **Streaming scheduler** — places every layer's weights across VRAM/RAM/disk
  and simulates decode to report bytes/token and modelled tok/s, plus a
  `*.vipera.json` residency manifest.
- **Expert cache** with `lru`, `lfu` and `hot` policies (admission filter,
  pinned hot set, aging) and both a global pooled and a per-layer
  water-filling allocation. Every reported hit rate is paired with the static
  ceiling **at the same slot count**.
- **Routing traces** — a JSONL format, a loader/validator, entropy and
  ceiling statistics, and a llama.cpp patch (`bench/`) that captures real
  router choices so the cache claims can be checked against measured routing.
- **Expert pruning** — the retained routing mass for a keep-set, next to what
  uniform pruning would retain; plus gated pruning (P4) that rolls a layer back
  whole when its gate fails instead of damaging it quietly.
- **Benchmark and measurement harnesses** — a `llama-bench` wrapper that
  samples peak RSS/VRAM and disk read bytes, a hardware probe, and a perplexity
  harness.
- **Packaged skills** — `skills/local-llm-optimization` (decision tree,
  technique catalogue, hardware playbooks) and `skills/expert-io-loop` (the
  measurement protocol, with a measured 7% noise band).
- **Documentation** — `docs/RESEARCH.md`, `PROPOSALS.md`, `ROADMAP.md`,
  `FEASIBILITY.md`, `OPTIMIZER.md`, `ALGORITHMS.md`, `DESIGN.md`.
- **Install section** — the README now states what running vipera actually takes:
  the optimizer is pure standard library and needs only Python >= 3.11; the
  `vipera` command goes into a venv because a PEP 668 "externally managed" Python
  refuses an editable install; and `vipera run` needs a `llama-cli` binary plus a
  GGUF file, with GPU backends optional (CPU is fastest on the target machine).
- **Repository hygiene** — `LICENSE` (MIT), `CONTRIBUTING.md`, `.editorconfig`,
  this changelog, and a CI workflow running the suite on Python 3.11–3.14.

### Changed

- **Model and machine constants are now measured** rather than assumed
  (`bench/hwprobe.c`), including the disk bandwidth that drives every
  disk-bound verdict.
- **The comparison baseline was corrected** — a policy's hit rate is no longer
  flattered by a ceiling computed at a different cache size.
- **The simulator's hot paths were sped up 3–21×** without changing a single
  reported number: the Zipf weight table is built once per simulation, LRU
  victim selection is no longer a quadratic scan, and `lfu`/`hot` no longer
  maintain a recency list they never read. Ledger in `PERF.md`.
- Project documents were translated to English and the project was renamed to
  `vipera` when the model-file optimizer layer was introduced.

### Fixed

- The S1 hit-rate baseline compared against a ceiling at the wrong cache size,
  which made a correct policy look better than the bound it was measured against.
- `sweep` requested 25 kept layers on a 16-layer model, leaving the layer list
  shorter than the count and crashing the simulator.
- A file whose tensor offsets pointed past its end, or that used a ggml quant
  type with no known block size, previously analysed to a clean exit with wrong
  byte totals; both now fail loudly.
