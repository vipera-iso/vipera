# Architecture

Vipera is a thin, opinionated layer between a GGUF file and an OpenAI-compatible
HTTP API. There is no second inference engine: `llama-cpp-python` does the work,
the planner decides how it is invoked, and the CLI and server share one
`Engine` instance per model.

```
clients ──HTTP/SSE──▶ FastAPI server ──chat()──▶ Engine ──▶ llama.cpp ──▶ GGUF
                                                   ▲
                              Plan ────────────────┘
                                ▲
                    Planner ◀───┴── Hardware probe
```

## Module map

All code lives in `src/vipera`.

| Module | Responsibility | Depends on |
|---|---|---|
| `hardware.py` | Probe CPU topology, RAM, GPU name, VRAM. Returns `HardwareInfo`. No side effects beyond optional NVML init. | `psutil`, `pynvml` module (optional `nvidia` extra) |
| `planner.py` | Turn `(model metadata, HardwareInfo, overrides)` into a `Plan` dataclass. Pure function. | `hardware` |
| `engine.py` | Wrap `llama_cpp.Llama` from a `Plan`; perform the GPU→CPU and `mlock` retries; expose `chat()`. | `llama-cpp-python` (optional extra), `planner` |
| `server.py` | Build the FastAPI app around an engine: `/health`, `/v1/models`, `/v1/chat/completions`, SSE framing. | `fastapi`, `pydantic` |
| `registry.py` | JSON model index at `$VIPERA_HOME/registry.json`: add/list/get/remove. | `config` |
| `config.py` | Paths, default config, load/save with type coercion. | stdlib |
| `gguf_analyzer.py` | Read GGUF metadata and tensor index without loading weights: per-layer/category sizes, quantization mix, low-quant warnings. | `gguf` (optional extra) |
| `optimizer.py` | Pure helpers for quantization: locate `llama-imatrix`/`llama-quantize`, build their command lines, pick a quant for a size target, name outputs. No subprocess execution. | stdlib |
| `cli.py` | Typer root app, banner, default-model resolution, `help`. | all commands |
| `commands/*.py` | One module per command group: `model`, `serve`, `chat`, `bench`, `system`. | engine, planner, registry, ui |
| `ui.py` | Rich console, banner, panels, model picker, typewriter streaming. | `rich` |

Dependency direction is one-way: commands → engine/planner/registry → config.
`server.py` knows nothing about the planner; it receives an already-loaded
engine plus the model alias.

## Request flow

### Starting the gateway

1. `vipera` (no subcommand) loads config and resolves a model: the
   `default_model`, an auto-selection when exactly one model is installed, or an
   interactive picker. `vipera -s` forces the picker. It then calls the same
   function as `vipera serve start --model <alias>`.
2. `serve start` resolves `--model`, `--host`, `--port`, `--profile`, `--ctx`
   against the config and looks the model up in the registry.
3. `hardware.detect()` builds a `HardwareInfo`; `planner.plan()` produces a
   `Plan` from the model's stored metadata and the hardware profile.
4. `Engine(plan, model_path)` loads the model. A boot panel prints the plan
   (offloaded layers, KV cache type, flash attention), then an optional quick
   speed check runs (`show_bench_on_start`).
5. `server.build_app(engine, alias)` is handed to `uvicorn.run(...)`, binding
   the configured host/port. If FastAPI/uvicorn cannot be imported, the CLI
   falls back to the chat REPL with the loaded engine.

### A chat completion

**Non-streaming:** the FastAPI handler runs `engine.chat(...)` in a worker
thread via `run_in_threadpool`, under a process-wide `Lock`, and returns the
dictionary from `llama.cpp` unchanged.

**Streaming:** the handler returns a `StreamingResponse` wrapping a synchronous
generator. Each `llama.cpp` chunk becomes `data: <json>\n\n`; a final
`data: [DONE]\n\n` is always emitted. The lock is held for the whole
generation, so concurrent clients queue rather than interleave tokens.

Because the generator is synchronous, Starlette iterates it in a threadpool and
each chunk is flushed as it is produced. That is what makes per-token streaming
work without an async inference API.

## Planner

`plan(meta, hw, profile="auto", ctx=8192, kv_cache="q8_0", n_batch=512)`
produces:

| Field | Meaning |
|---|---|
| `gpu_layers` | Number of transformer layers offloaded to the GPU. |
| `cpu_threads` | Worker threads passed to `llama.cpp` (`cores - 1`, minimum 1). |
| `context` | `n_ctx`, possibly reduced on low-memory machines. |
| `kv_cache` | KV cache type string, mapped to a `llama.cpp` enum in the engine. |
| `flash_attn` | Whether flash attention is requested. |
| `n_batch` | Prompt batch size, capped by placement. |
| `offload_kqv` | Whether the KV cache may live in VRAM. |
| `use_mlock` | Whether to lock the model in RAM. |
| `note` | Human-readable placement summary (`CPU ONLY`, `Full GPU`, `Hybrid: n/N layers`). |

The algorithm, in order:

1. **Metadata defaults.** Missing `n_layers` → 32; missing `size_bytes` → 4 GB.
2. **Threads.** `cpu_threads = max(1, cpu_cores - 1)`.
3. **mlock.** Enabled when free RAM minus model size is greater than 2 GB.
   This leaves headroom so page-locking the weights does not push the machine
   into swap.
4. **VRAM budget.** `safe = max(0, vram_free - 1.0)` reserves 1 GB for the
   display/driver, then 90% of that is treated as usable.
5. **Layer placement.** `layer_gb = size_gb / n_layers`;
   `gpu_layers = min(n_layers, floor(usable / layer_gb))`. This assumes uniform
   layer sizes, which is a deliberate simplification.
6. **CPU-only path** (`gpu_layers == 0`): flash attention and KQV offload off,
   `n_batch = min(n_batch, 128)`, KV cache forced to `f16`, and if total RAM is
   under 8 GB the context is capped at 2048.
7. **GPU path:** flash attention on when total VRAM exceeds 4 GB and the
   profile is not `speed`; KQV offload on; `n_batch = min(n_batch, 512)`.
   The note distinguishes full and hybrid offload.

`profile` currently affects only the flash-attention decision. The planner does
not emit MoE-specific flags (`--n-cpu-moe` or equivalents); MoE models are
placed by the same per-layer arithmetic.

## GGUF optimization

Optimization is offline: it rewrites files, it does not participate in serving.
The split is deliberate — `optimizer.py` is pure and testable (command
construction, binary discovery, size math), while `commands/model_cmd.py` owns
subprocess execution so it can stream output, handle Ctrl+C, and delete a
partial output when the child fails.

`model optimize` chains the steps:

1. `gguf_analyzer` reads the source index to get size and dominant quant.
2. `estimate_quant_for_target` converts the size budget to a target
   bits-per-weight and picks the highest-quality K-quant at or below it.
3. If calibration data is supplied, `llama-imatrix` writes a matrix next to the
   source (or a numbered variant if that name exists).
4. `llama-quantize` runs with `--imatrix` when present, `--allow-requantize`
   because the source is often already quantized, and optional
   `--prune-layers` / `--prior-weight`.

`--prior-weight` is checked against the installed binary's `--help` output
before being passed, so an older llama.cpp fails with a clear message instead
of an unknown-argument error. `--dry-run` prints the plan and the exact commands
without creating anything.

## Engine

`Engine.__init__` loads `llama_cpp.Llama` with `n_gpu_layers`, `n_ctx`,
`n_threads`, `n_batch`, `n_ubatch=min(n_batch, 512)`, `flash_attn`,
`offload_kqv`, `type_k`/`type_v` (from `_KV_TYPES`), `use_mmap=True`,
`use_mlock`, and `verbose=False`.

Two retries exist, and they are independent:

1. **mlock retry** (inside `_load`): if loading with `use_mlock` raises, retry
   once with `use_mlock=False`. Common on systems with a low `RLIMIT_MEMLOCK`.
2. **GPU→CPU fallback** (in `__init__`): if the plan fails outright, rebuild a
   conservative plan — `gpu_layers=0`, flash attention and KQV offload off,
   `kv_cache="f16"`, `n_batch=128`, `context=min(context, 2048)` — and load that.
   A warning is printed; the successful plan replaces `self.plan`, so callers
   and status output describe what actually ran.

If both loads fail, the engine raises `RuntimeError`. Note that the message for
the CPU-failure case is currently in Vietnamese while the rest of the runtime
output is English.

`chat()` forwards `messages`, `stream`, and sampling parameters
(`temperature`, `max_tokens`, `top_p`, `repeat_penalty`, with defaults
0.7 / 2048 / 0.95 / 1.1) to `create_chat_completion`. Any other keyword
arguments are ignored.

Supported KV cache names: `f32`→0, `f16`→1, `q4_0`→2, `q4_1`→3, `q8_0`→8. An
unknown name maps to `None`, which lets `llama.cpp` use its default.

## Concurrency model

- One process, one loaded model, one `Engine`.
- One `threading.Lock` serializes every completion, streaming or not.
- Non-streaming requests occupy a Starlette threadpool worker while waiting for
  the lock; streaming requests hold the lock until the last token.
- There is no request queue limit, cancellation, timeout, or per-client
  accounting. A slow generation blocks other clients.

This is intentional for a single-user local gateway. If you need parallel
decoding, you need a different serving stack.

## Hardware probing

`hardware.detect()` reports:

- **CPU**: physical cores and logical threads via `psutil`.
- **RAM**: total and available via `psutil.virtual_memory()`.
- **Apple Silicon**: identified as macOS on `arm64`; VRAM is reported as unified
  memory, with 70% of available RAM treated as usable VRAM.
- **NVIDIA**: via the `pynvml` module, provided by the `nvidia` extra
  (`pip install -e ".[nvidia]"`, which installs `nvidia-ml-py`). Without it,
  GPU fields stay at their defaults
  and the machine is planned as CPU-only. Detection alone does not add a GPU
  backend: acceleration still requires a CUDA-enabled `llama-cpp-python` build.

Probing never raises: the NVML import and calls are wrapped in
`contextlib.suppress(Exception)`.

## Known gaps

These are observable in the current code and worth knowing before filing a bug:

- Tool/function calling, multimodal message content (arrays of parts), and the
  remaining OpenAI endpoints (`/v1/completions`, `/v1/embeddings`) are not
  implemented.
- The `model` field of a chat request is accepted and ignored; the response
  `model` field comes from `llama.cpp`, not from the gateway alias.
- No authentication, TLS, CORS, or rate limiting. The default bind address is
  loopback; do not expose the port directly.
- VRAM detection does not guarantee GPU acceleration. With the `nvidia` extra
  and a CPU-only `llama-cpp-python` build, the planner can report GPU layers
  that the runtime cannot actually use.
- `temperature`, `max_tokens`, `top_p`, `flash_attn`, and `profile` exist in the
  config file, but `flash_attn` and `profile` are not read by `serve start`
  (flash attention is decided by the planner; `--profile` defaults to `auto`
  rather than the stored value). Per-request sampling parameters default to the
  values in `server.ChatRequest`, not to config.
- The planner does not emit MoE-specific offload flags despite the README's
  description of the flag set.
- `VIPERA_HOME` is read once at import time; changing it after the package is
  imported has no effect (tests patch the module constants instead).
