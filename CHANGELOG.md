# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet.

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

[Unreleased]: https://github.com/vipera-iso/vipera/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/vipera-iso/vipera/releases/tag/v1.0.0
