# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/vipera-iso/vipera/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/vipera-iso/vipera/releases/tag/v1.1.0
[1.0.0]: https://github.com/vipera-iso/vipera/releases/tag/v1.0.0
