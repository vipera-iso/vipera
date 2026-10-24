# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Ollama SDK integration test** — `tests/integration/test_ollama_sdk.py`
  starts the gateway on a loopback port and drives it with the official
  `ollama` client: `/api/tags`, `/api/chat` (streaming and not), `/api/generate`,
  `/api/embed`, `/api/ps` and the 404 error body. Compatibility is checked
  against the real client parser instead of hand-written JSON, and the test
  needs no model or Ollama daemon (`ollama` is now a dev dependency).
- **Protocol layers** — the gateway's HTTP surface is now composed of layers
  selected by `protocols` (or `vipera serve start --protocol`). `openai` keeps
  `/v1/*`; the new `ollama` layer serves `/api/tags`, `/api/ps`, `/api/chat`,
  `/api/generate`, `/api/embed`, `/api/embeddings` and `/api/version`, so
  Ollama clients and Open WebUI can point at Vipera unchanged. Both layers use
  the same engine and lock; layers can also be installed as plugins through
  the `vipera.protocols` entry-point group.
- **`/v1/embeddings`** — OpenAI-shaped embeddings from either backend, plus
  Ollama's `/api/embed`. Off by default: enable it with
  `vipera system config embeddings true`, because llama.cpp builds the
  embedding context at load time (the server backend passes `--embeddings
  --pooling mean`, the in-process engine uses mean pooling). A model loaded
  without it answers 501 and explains how to turn it on.
- **Ollama models without a second download** — `vipera model add
  ollama:qwen2.5:7b` (or `model import`) reads the manifest in
  `~/.ollama/models` — both the current `manifests-v2/ollama.com` layout and
  the older `manifests/registry.ollama.ai` one, honouring `OLLAMA_MODELS` —
  takes the GGUF's original filename (so the quant is preserved), and
  hardlinks the blob into `models/`. Nothing is downloaded and no second copy
  is stored; a copy happens only when the two stores are on different
  filesystems. Removing the Vipera entry leaves Ollama's blob untouched, and a
  model that was never pulled reports `ollama pull …`. `vipera model add` is
  now an alias for `model import`, and `ollama list`-style model names are
  listed by the source's local search.
- **Model sources** — `vipera/model_cmd.py` no longer talks to Hugging Face
  directly. `vipera.sources` defines the `ModelSource` contract and ships
  `hf`, `local`, `ollama` and `url`; `vipera model import` accepts `hf:repo`,
  `local:path`, `ollama:name:tag` and `https://…/model.gguf` references, and
  `huggingface_hub` is imported on first use instead of at module import, so
  `vipera --version` and the commands that never download no longer pay for it.
- **Multi-model serving** — `vipera serve --models a,b --max-loaded N` (or
  `models`/`max_loaded` in the config) serves several aliases from one gateway.
  A model is planned and mounted on its first request, and the least recently
  used engine is closed once more than `--max-loaded` are resident. `GET
  /v1/models` lists every served alias; a request naming anything else returns
  404 with the available aliases. `GET /health` reports the loaded set.
- **`python -m vipera`** — same entry point as the console script, used by the
  end-to-end tests so they do not depend on `PATH`.
- **Split test tiers** — `tests/unit`, `tests/integration` (real files and
  processes) and `tests/e2e` (CLI in a subprocess), with `integration`/`e2e`
  markers registered in `pyproject.toml`.
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
- **Registry schema v2** — `registry.json` is written as
  `{"_schema": 2, "models": {...}}` and each entry records `source`,
  `source_ref` and `format` next to the path and GGUF metadata. 1.x files
  (a flat alias → entry map with no version) are read as-is and upgraded on
  the next write, so no migration step is needed. `list_models()` and
  `get_model()` now return a frozen `ModelSpec` instead of a dict; internal
  callers were updated, and `vipera.registry` had no public API before v2.
- **Backends can be installed as plugins** — `vipera.backends.names` lists the
  built-ins plus anything registered under `[project.entry-points."vipera.backends"]`,
  and `create_engine` loads an unknown-but-registered name from that group
  instead of failing. The config validates a backend name against the same
  list. The factory contract is `factory(plan, model_path) -> ChatEngine`;
  planning stays with the built-in planner until a second real backend shows
  what a backend-specific plan needs.
- **Backends split into `vipera/backends/`** — `protocol` (the `ChatEngine`
  contract and the accepted chat parameters), `loading` (offload ladder and
  retry rungs), `registry` (backend selection), `inprocess` (llama-cpp-python)
  and `server` (the llama-server child). The server backend no longer imports
  the engine module from inside a function to dodge a cycle. `vipera.engine`
  and `vipera.server_backend` remain importable as shims for 1.x callers.
- **Error codes** — `vipera.errors` adds `ViperaError` with stable codes.
  `PlannerError` (E3001) and `BackendError` (E4001) subclass `ValueError` and
  `RuntimeError`, so existing `except` clauses and callers keep working while
  new code can match on the type.
- **`flash_attn` is no longer a config key** — the planner decides it from
  `kv_cache` (quantized KV requires it), so the key never had an effect and
  only suggested it did. `vipera system config flash_attn ...` now rejects it.
  Existing `config.json` files keep the stale key harmlessly; drop the line.
- **`profile` in the config is now read** — `serve start` used to pass `auto`
  unconditionally, ignoring a configured `profile`.
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
- **`vipera serve stop` now stops the `llama-server` child** — uvicorn exits
  on SIGTERM without unwinding, so neither `serve start`'s `finally` nor the
  engines' `atexit` hooks ran and a `backend: server` child was left holding
  the model (reparented to init, invisible to `serve status`). `serve start`
  installs a SIGTERM handler that raises `SystemExit`, so the normal unwind
  releases the engine, the child and the pid file. `stop --force` (SIGKILL)
  still cannot clean up from inside the gateway.
- **Streaming no longer emits two usage frames** — with
  `stream_options.include_usage`, the gateway forwarded llama-server's trailing
  `choices: []` frame and then synthesized another one. Usage is now captured
  and emitted exactly once, in the OpenAI shape.
- **A `null` or mistyped config value no longer crashes the planner out of
  sight** — `"ctx": null` survived the merge and raised `TypeError` deep inside
  `plan()`; it now falls back to the default, and an uncastable value logs
  `vipera: config ...` to stderr instead of being kept as a string.
- **`model recommend` suggests a command that can run** — the emitted
  `vipera model quantize` line targets the model's current quant, which
  llama-quantize refuses to requantize without `--allow-requantize`.
- **`Q8_K` sizes as 8-bit** — `_quant_bpw` had no `Q8` branch, so `Q8_K` fell
  through to the Q4 fallback (4.58 bpw) and halved the estimate.
- **The hybrid-attention docstring matches the code** — a layer matching both
  markers counts as KV-bearing, not recurrent.
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
