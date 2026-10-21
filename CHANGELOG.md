# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Opt-in `llama-server` backend** — `backend: "server"` in the config runs
  the `llama-server` binary as a child process and serves the same
  OpenAI-compatible API from it (`src/vipera/server_backend.py`). The flags
  come from the same `Plan` as the in-process engine, the child's output goes
  to `~/.vipera/logs/llama-server-<port>.log`, a GPU start failure retries
  CPU-only, and the child is stopped with the gateway. `inprocess` stays the
  default until this path has been measured on real hardware.
- **`Engine.tokenize()`** — one call that works on both backends, instead of
  reaching into `engine.llm`.
- **Hybrid-attention KV sizing** — models that mix attention with recurrent
  layers (KDA/BCSA, Mamba, linear attention) are detected from their tensor
  names, and the planner now sizes the KV cache from the attention layers only
  instead of all layers. A 72-layer hybrid with 18 KV-bearing layers no longer
  reserves ~4x its real KV; `model inspect` reports the split and `vipera serve`
  shows it in the allocation panel. Models with no detectable attention layers
  keep the conservative dense estimate.
- **2026 GGUF quant types** — `model inspect` now names `TQ2_0` (35), `MXFP4`
  (39), `NVFP4` (40), and `Q1_0` (41), and reserves `Q2_0` (42). Ids missing
  from the static table are resolved through `gguf.GGMLQuantizationType`, an
  unknown id no longer ranks as 0 (which produced a false "below Q4_K"
  warning), and `--target-size` uses block-table bytes-per-weight instead of
  nominal bit width.

### Changed
- **Throughput: 4.55 → 6.3 t/s on the reference 14B** (8K, interleaved A/B,
  median-3). Three measured changes: `cpu_threads` is now one per physical core
  instead of `cores - 1` (+9%, and +19-21% at f16 KV); the VRAM reserve is
  calibrated from the card instead of a flat 1.3 GB (0.3 GB here, which runs
  four more layers on the GPU); and a plan that does not fit walks the offload
  down instead of jumping to CPU-only, which was the slowest outcome (3.9 t/s
  against 5.0 t/s at the last split that loaded).
- **The offload budget no longer trusts `meta["layer_bytes"]`** — only the
  repeating `blk.N` block size was charged per layer, which is optimistic by
  3-6 layers because llama.cpp's per-layer compute buffer grows with `-ngl`
  (measured: a Q3_K_M file has a 17-layer ceiling where the old arithmetic
  asked for 20, and asking too much ends on the CPU). `size_bytes / n_layers`
  lands on the measured ceiling for both files. The placement ladder's last
  rung is a quarter of the plan rather than half, because a failed context
  creation keeps its VRAM.
- **Four more KV cache types** — `bf16`, `q5_0`, `q5_1`, and `iq4_nl` are
  accepted next to the original five, and the planner sizes them from their
  ggml block size instead of falling back to the f16 estimate.
- **Flash attention only where it is required** — `f32`, `f16`, and `bf16` no
  longer enable the CPU flash-attention path, which measured ~30x slower than
  the default path on a 14B; quantized cache types still do, because llama.cpp
  refuses to create a context without it.
- A GGUF whose tensor type the installed `gguf` cannot size now fails with
  `GGUFAnalysisError: failed to read ...` instead of a traceback.

### Fixed
- **`serve start` warns for every non-loopback `--host`** — `0.0.0.0` was
  treated as safe, so the most common "expose to the network" bind got no
  warning at all, and the grace-period message claimed the server was aborting
  when it actually started (Ctrl+C is what aborts).
- **A corrupt `config.json`/`registry.json` is no longer erased in silence** —
  the unreadable file is copied to `<name>.corrupt` and a warning names the
  backup, instead of defaults being saved over the user's settings on the next
  write.
- **`stream_options.include_usage` now reaches the backend** — the request
  model parsed the field but never forwarded it, so the trailing
  `choices: []` usage frame could not fire. It is passed on now; llama-server
  fills it with real counts, and the in-process engine still drops it.
- **`vipera bench` no longer crashes on an engine without `.llm`** — it called
  `engine.llm.tokenize()` directly, so any engine wrapper (or the server
  backend) failed before measuring. It now calls `engine.tokenize()`.
- **`tools/ppl_baseline.py` parses on Python 3.11** — a nested-quote f-string
  only worked on 3.12+, while the package declares `requires-python >=3.11`.
- **`vipera model rm` keeps a file another alias still references** —
  downloading the same file twice via `model search` created two aliases for
  one path; removing one of them deleted the shared file out from under the
  other.

## [1.2.0] - 2026-10-20

### Changed
- **Planner budgets VRAM from real per-layer sizes** — registering a model now
  stores `layer_bytes` (mean size of one repeating `blk.N` block, read from the
  GGUF index). `size_bytes / n_layers` charged the ~1 GB of `token_embd` +
  `output.weight` to every layer, so a 14B/Q4 model on a 4 GB card now offloads
  one or two more layers at the same VRAM headroom (11 vs 9 at 8K with 3.7 GB
  free).
- **VRAM plan keeps room for GPU compute buffers** — the planner reserves
  0.3 GB on top of the 1 GB driver reserve. Sizing to the last byte of VRAM
  produced a plan llama.cpp cannot create a context for, and the engine's only
  fallback is CPU-only: on the GTX 1650, 11 offloaded layers load and 12 fail.
- **Prompt-prefix reuse is now guaranteed by the dependency floor** —
  `llama-cpp-python` is pinned to `>=0.3.36`, the release that keeps the KV
  cache for a shared prompt prefix. On 0.5B, a 1230-token prompt cost 1.5 s
  cold and 0.18 s for the +19-token follow-up turn.
- **`vipera bench` reports the engine's actual placement** — a GPU load that
  fell back to CPU no longer prints the planned layer count (16K/f16 on a
  4 GB card falls back and now shows 0/40 rather than 9/40).

## [1.1.0] - 2026-10-20

### Added
- **Tool calling** — `/v1/chat/completions` accepts `tools` and `tool_choice`,
  forwards `tool_calls` back in both streaming and non-streaming responses, and
  accepts assistant tool-call turns plus `tool` results with `tool_call_id`.
- **Full OpenAI sampling pass-through** — `stop`, `seed`, `presence_penalty`,
  `frequency_penalty`, `top_k`, `min_p`, `logit_bias`, `logprobs`,
  `top_logprobs`, `response_format`, and `stream_options.include_usage` now
  reach `llama.cpp` instead of being dropped by the request model.
- **GGUF inspection** — `vipera model inspect <path> [--tensors]` reads the GGUF
  index only and reports architecture, per-layer/category size distribution,
  quantization mix, and warnings for quality-sensitive tensors quantized below
  Q4_K. Backed by `vipera.gguf_analyzer.analyze_gguf()`.
- **GGUF optimization** — `vipera model imatrix`, `model quantize`, and
  `model optimize` wrap `llama-imatrix` / `llama-quantize` with imatrix support,
  MoE `--prior-weight`, layer pruning, target-size quant selection, binary
  discovery (`--bin-dir`, `VIPERA_LLAMA_BIN`, `PATH`), dry runs, and partial
  output cleanup. Planning helpers live in `vipera.optimizer`.

### Changed
- **Faster readiness** — `serve start` no longer runs the 20-token speed check
  before binding, and the fixed 1.2 s boot animation is gone. The check is now
  opt-in via `vipera serve start --bench`; the `show_bench_on_start` config key
  was removed.

## [1.0.0] - 2026-10-17

First public release.

### Added
- **OpenAI-compatible HTTP gateway** — `/v1/chat/completions`,
  `/v1/models`, `/health`. Real SSE streaming.
- **Hardware-aware planner** — reads CPU cores, RAM, GPU name, VRAM and emits
  a `Plan` (`gpu_layers`, `context`, `kv_cache`, `flash_attn`, `n_batch`,
  `use_mlock`, `offload_kqv`).
- **Engine fallback** — retries with a CPU-safe plan when GPU offload fails at
  load; retries without `mlock` when `mlock` is restricted.
- **Model manager** — `model search` (HuggingFace), `model import`,
  `model list`, `model rm`. Auto-sets the imported model as default.
- **CLI chat** — interactive `chat repl` and scriptable `chat oneshot`.
- **Benchmark** — `bench run` measures prefill (TTFT-based) and generation
  throughput across context sizes.
- **Diagnostics** — `system doctor` for runtime, library, and hardware;
  `system config` for typed, validated config access.
- **Rich CLI** — banner, model picker, typewriter streaming.

### Known limitations
- Tool calls / function calling are not implemented.
- Multi-model serving is not implemented.
- The `model` field in `/v1/chat/completions` is ignored; one gateway serves
  one model.

[Unreleased]: https://github.com/vipera-iso/vipera/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/vipera-iso/vipera/releases/tag/v1.2.0
[1.1.0]: https://github.com/vipera-iso/vipera/releases/tag/v1.1.0
[1.0.0]: https://github.com/vipera-iso/vipera/releases/tag/v1.0.0
