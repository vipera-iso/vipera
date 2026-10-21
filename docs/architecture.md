---
title: Architecture
description: Module map, request flow, planner, engine fallback, concurrency model, and known gaps.
---

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
| `engine.py` | Wrap `llama_cpp.Llama` from a `Plan`; perform the GPU→CPU and `mlock` retries; expose `chat()`/`tokenize()`. `create_engine()` picks the backend. | `llama-cpp-python` (optional extra), `planner` |
| `server_backend.py` | Opt-in `llama-server` child process: translate a `Plan` into flags, wait for `/health`, proxy `chat()`/`tokenize()`, stop the child. | `planner`, `optimizer` (binary lookup) |
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
4. `create_engine(plan, model_path, backend)` loads the model (`inprocess` →
   `Engine`, `server` → `ServerEngine`). A boot panel prints the plan
   (offloaded layers, KV cache type, flash attention). The gateway binds as
   soon as `uvicorn` starts; the 20-token speed check only runs when
   `serve start --bench` is passed.
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
| `cpu_threads` | Worker threads passed to `llama.cpp` (one per physical core, minimum 1). |
| `context` | `n_ctx`, possibly reduced on low-memory machines. |
| `kv_cache` | KV cache type string, mapped to a `llama.cpp` enum in the engine. |
| `flash_attn` | Whether flash attention is requested. |
| `n_batch` | Prompt batch size, capped by placement. |
| `offload_kqv` | Whether the KV cache may live in VRAM. |
| `use_mlock` | Whether to lock the model in RAM. |
| `note` | Human-readable placement summary (`CPU ONLY`, `Full GPU`, `Hybrid: n/N layers`). |

The algorithm, in order:

1. **Metadata defaults.** Missing `n_layers` → 32; missing `size_bytes` → 4 GB.
2. **Threads.** `cpu_threads = max(1, cpu_cores)` — one thread per physical
   core. Leaving a core idle measured 19-21% slower than using it (14B, both
   CPU-only and with 12 GPU layers, median-3); using every logical thread is
   slower still, so SMT siblings are not used.
3. **mlock.** Enabled only when free RAM minus the host-resident weights and
   host-resident KV cache exceeds 3 GB. Pinning with less headroom leaves
   nothing for llama.cpp compute buffers and can trigger the OOM killer on a
   no-swap machine. (On the reference box the kernel refuses the lock anyway —
   `ulimit -l` is 8 MiB — and raising that limit measured no gain; see
   [benchmarks](benchmarks.md#caveats).)
4. **VRAM budget.**
   `safe = max(0, vram_free - max(0.25, min(1.0, 0.07 * vram_total)))`, plus
   0.9 GB when the plan cannot use flash attention. The reserve covers the
   CUDA/driver context and llama.cpp's compute buffers. It is calibrated from
   measurement, not guessed: on a GTX 1650 with 3.64 GB free, a 14B/Q4 model at
   8K/`q8_0` offloads 14 layers (3.37 GB of weights + KV) and 15 fail to create
   the context, so the real reserve is ~0.3 GB. The flat 1.3 GB this used to
   reserve left four layers on the CPU and measured 5.0 t/s against 6.0 t/s at
   14 layers. It scales with the card so a 24 GB card is not over-reserved,
   and it stays near the measured figure on small ones.
5. **Layer placement.** llama.cpp allocates the KV cache per layer: only the
   offloaded layers keep their slice of KV in VRAM. So
   `per_layer_kv = kv_gb / kv_layers`,
   `gpu_layers = min(n_layers, floor(safe / (layer_gb + per_layer_kv)))`.
   Subtracting the whole KV cache before offloading anything would report 0
   layers for models that actually fit a useful hybrid split. `kv_layers` is
   every layer for a dense model and only the attention subset for a hybrid one
   (see below); the host-side KV charge follows the same split.
   `layer_gb` is `size_gb / n_layers`. `meta["layer_bytes"]` (the mean size of
   one repeating `blk.N` block, excluding `token_embd`/`output.weight`) is
   deliberately **not** used any more: charging only the repeating block was
   optimistic on both files it was tried with, because llama.cpp's per-layer
   CUDA compute buffer grows with the number of offloaded layers. Measured on
   the reference card at 8K/`q8_0`: a Q4_K_M model has a 14-layer ceiling where
   `layer_bytes` arithmetic asked for 20, and a Q3_K_M has a 17-layer ceiling
   where it asked for 20. `size_gb / n_layers` lands on 14 and 17. Over-
   estimating is the expensive direction: a failed context creation keeps its
   VRAM, so three failures in a row left even a 10-layer attempt unable to
   load.
6. **CPU-only path** (`gpu_layers == 0`): KQV offload off,
   `n_batch = min(n_batch, 128)`, and the requested KV type is kept. Flash
   attention is enabled whenever that type is quantized (llama.cpp requires
   it), so a 64K context does not silently become a 10 GB f16 cache. If total
   RAM is under 8 GB the context is capped at 2048.
7. **GPU path:** flash attention on when KV is quantized or total VRAM exceeds
   4 GB (without it the KQ matrix is materialized per batch, which on a 4 GB
   card cut the offload from 14 layers to 10); KQV offload on;
   `n_batch = min(n_batch, 512)`. The note distinguishes full and hybrid
   offload.

`profile=cpu` forces zero offload and `profile=lowmem` returns a fixed
2048-context plan; the other profiles only influence the placement note. The
planner does not emit MoE-specific flags (`--n-cpu-moe` or equivalents); MoE
models are placed by the same per-layer arithmetic.

### Hybrid attention

Stacks that mix attention with recurrent layers (KDA/BCSA, Mamba, linear
attention) keep a KV cache only on their attention layers; the others carry a
fixed-size state that does not grow with context. Sizing KV with `n_layers`
over-reserves a 72-layer hybrid by roughly 4x and makes the planner offload far
too few layers.

The analyzer derives the split from the tensor index, not from an architecture
whitelist: a layer counts as KV-bearing when it has an `attn_k*` (including
MLA's `attn_k_b`/`attn_kv_a_mqa`), `attn_v*`, or `attn_qkv*` tensor, and as
recurrent when it has `ssm_`, `linear_attn`, `kda`, `mamba`, or `conv1d`.
Recurrent wins when both match, which is the safe side: a state-only layer
counted as KV over-reserves rather than under-reserves.
`metadata.hybrid_attention` is true only when both groups are non-empty, which
keeps dense models on the old path. If no layer is detected as KV-bearing at
all, the split is treated as unknown and every layer keeps paying for KV — the
conservative direction for a VRAM budget, since missing KV is an OOM while an
over-estimate costs a layer or two of offload.

The estimate covers K/V tensors only. The recurrent state itself (chunked
`conv1d` windows, `A_log` decay, SSM widths) is not sized from the GGUF, so the
plan remains an underestimate for those layers. Treat `Plan.kv_cache_gb` as a
KV figure, not as total activation memory.

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

### Quant type ids

The analyzer labels every id the installed `gguf` package knows, including the
2026 additions `TQ2_0` (35), `MXFP4` (39), `NVFP4` (40), and `Q1_0` (41), and
reserves `Q2_0` (42), which `gguf` 0.19 cannot size yet. Unknown ids resolve
through `gguf.GGMLQuantizationType` when available, so a future type is named
instead of reported as `TYPE_nn`. An id the reader cannot size fails with
`GGUFAnalysisError: failed to read ...` rather than a traceback: a half-parsed
index must never reach the planner.

Quality warnings fire only when the bytes-per-weight is known and below Q4_K; an
unknown id is not assumed to be low quality. `--target-size` math uses the
block-table bytes-per-weight (NVFP4 4.5, MXFP4 4.25, TQ2_0 2.0625, TQ1_0
1.6875, Q1_0 1.125), not the nominal bit width.

### Layer classification

`layer_distribution` lists `blk.N` blocks only; token embeddings and
`output.weight` are counted separately because they never follow a layer to the
GPU. Tensor categories are matched by name: `attn_*` → attention, `ffn_*` →
feed-forward, `ssm_*`/`linear_attn*`/`kda*`/`conv1d` → recurrent, `token_embd`
and `output` → embeddings. The same attention markers feed the hybrid split
described under [Hybrid attention](#hybrid-attention).

## Engine

Before loading, `Engine.__init__` checks `Plan.ram_estimate_gb` against
`psutil.virtual_memory().available` and raises `RuntimeError` with a concrete
suggestion when the plan clearly will not fit. This is a preflight, not a
guarantee: an actual OOM is a `SIGKILL` and cannot be caught by the fallback
logic below.

`Engine.__init__` loads `llama_cpp.Llama` with `n_gpu_layers`, `n_ctx`,
`n_threads`, `n_batch`, `n_ubatch=min(n_batch, 512)`, `flash_attn`,
`offload_kqv`, `type_k`/`type_v` (from `_KV_TYPES`), `use_mmap=True`,
`use_mlock`, and `verbose=False`.

Two retries exist, and they are independent:

1. **mlock retry** (inside `_load`): if loading with `use_mlock` raises, retry
   once with `use_mlock=False`. Common on systems with a low `RLIMIT_MEMLOCK`.
2. **Placement ladder** (`engine.load_attempts`), in stages: `n`, `n-1`, `n-2`,
   `n/2` GPU layers, then CPU with the caller's `context` and `kv_cache` intact
   (`gpu_layers=0`, KQV offload off, `n_batch` capped at 128), then
   `context=min(context, 2048)`, and only as a last resort `kv_cache="f16"`
   with flash attention off. Every reduction prints a warning, and dropping
   below the requested context says so explicitly. The successful plan
   replaces `self.plan` (including its `note`), so callers and status output
   describe what ran.

The layer ladder exists because the old behaviour — any GPU failure jumps to
CPU-only — is the worst outcome on asymmetric hardware: 3.9 t/s CPU-only where
the last loadable split gave 5.0 t/s (14B/Q4, 8K, GTX 1650). Asking for 15
layers where 14 fit now loads at 13 in ~4 s instead of serving from the CPU.
A failed attempt does not hand its VRAM back before the next one, so the ladder
is short and ends on a halving rung rather than walking down one layer at a
time.

If every load fails, the engine raises `RuntimeError`.

`chat()` forwards `messages`, `stream`, and sampling parameters
(`temperature`, `max_tokens`, `top_p`, `repeat_penalty`, with defaults
0.7 / 2048 / 0.95 / 1.1) to `create_chat_completion`. Any other keyword
arguments are ignored.

Supported KV cache names: `f32`→0, `f16`→1, `q4_0`→2, `q4_1`→3, `q5_0`→6,
`q5_1`→7, `q8_0`→8, `iq4_nl`→20, `bf16`→30. An unknown name maps to `None`,
which lets `llama.cpp` use its default.

### Server backend

`ServerEngine` (`server_backend.py`) exposes the same surface as `Engine`.
It translates the `Plan` into `llama-server` flags (`-c`, `-ngl`, `-t`, `-b`,
`-ub`, `-fa`, `-ctk`/`-ctv`, `--jinja`), starts the child on a free loopback
port, polls `/health` for up to 900 s, and talks the OpenAI HTTP API
(`/v1/chat/completions`, streaming or not, plus `/tokenize`).

Differences from the in-process engine:

- `Plan.use_mlock` is not forwarded: `llama-server` rejects `--mlock` (only
  `llama-cli` has it) and its replacement, `--load-mode`, changed name across
  releases.
- The child's stdout/stderr go to `~/.vipera/logs/llama-server-<port>.log`; a
  failed start raises with the log tail instead of an opaque exception.
- A GPU start failure walks the same placement ladder as the in-process engine
  (`n`, `n-1`, `n-2`, `n/2` layers) before going CPU-only (`gpu_layers=0`, KQV
  offload off, `n_batch` capped at 128). The context and KV type are kept.
- `close()` terminates the child (SIGTERM, then SIGKILL after 10 s) and is
  registered with `atexit`; `serve stop` stops the gateway, which stops it.
- `stream_options.include_usage` is forwarded when a caller asks for it, so
  `vipera bench` gets an exact token count instead of a chunk count.

### Prompt-prefix reuse

`llama-cpp-python` 0.3.36 (the package floor) keeps the token ids it last
evaluated. On the next `generate` it finds the longest common token prefix,
removes only the KV entries past it, and decodes the remainder — so a follow-up
turn that repeats the conversation prefills just the new tokens, not the whole
system prompt. Measured on qwen2.5-0.5b: a 1230-token prompt took 1.5 s cold
and 0.18 s when only 19 tokens were new.

Two consequences for this codebase:

- `chat()` must forward `messages` unmodified and in order. Pre-rendering,
  reordering, or injecting per-request text into earlier turns changes the
  token prefix and silently disables the reuse.
- The cache belongs to the loaded engine: `serve stop`/`start`, a model, `ctx`,
  or `kv_cache` change all start from an empty cache.

## Concurrency model

- One process, one loaded model, one `Engine`.
- One `asyncio.Lock` serializes every completion, streaming or not.
- The sync engine is driven through `asyncio.to_thread`, so the event loop stays
  free between chunks. Streaming and non-streaming requests hold the lock until
  generation finishes.
- If a client disconnects while a worker thread is inside `next()` (or inside a
  non-streaming call), the lock is not released until that thread returns; the
  stream iterator is closed then too. This prevents a second request from
  entering the same llama.cpp context concurrently.
- There is no request queue limit, timeout, or per-client accounting. A slow
  generation delays other clients.

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
- Speculative decoding is unavailable. A `llama-server` backend now exists
  (`backend: "server"`), but it does not pass `--spec-type` or `-md` yet, and
  the in-process `llama_cpp.Llama` backend cannot enable DFlash at all;
  `draft_model=` would be classic speculative decoding, not DFlash. See
  [design/llama-server-backend.md](design/llama-server-backend.md).
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
- The VRAM model assumes the KV cache stays per layer, and its reserve is
  calibrated on one card. Verified for a 14B/Q4 at 8K/`q8_0` (14 layers load,
  15 fail) and 16K/`q8_0` (13 load); the f16 non-FA path is covered by an extra
  flat reserve rather than by a measured curve. An over-estimate is now
  recoverable through the placement ladder, at the cost of a failed load.
- The planner does not emit MoE-specific offload flags despite the README's
  description of the flag set.
- Expert-slab reordering (I/O-locality optimization) is not implemented. It
  needs an expert-usage trace, and this project defines no trace format yet, so
  `model optimize` only handles quantization, imatrix, and layer pruning.
- `VIPERA_HOME` is read once at import time; changing it after the package is
  imported has no effect (tests patch the module constants instead).
