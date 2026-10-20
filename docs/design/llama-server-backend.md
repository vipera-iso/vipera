---
title: llama-server backend
description: Proposal for a subprocess + loopback HTTP backend that unlocks DFlash speculative decoding.
---

# Design proposal: `llama-server` backend

**Status:** proposal, not implemented. Nothing in this page describes current
behaviour; see [architecture.md](../architecture.md) for that.

## Why

Vipera drives `llama_cpp.Llama` in-process. That API cannot run DFlash
speculative decoding:

- DFlash lives in `llama-server` (`--spec-type draft-dflash`), not in the
  `llama_cpp` Python binding. The binding exposes `draft_model=` for classic
  speculative decoding only — a different algorithm with different flags,
  different memory behaviour, and no `-fa on` requirement.
- DFlash requires flash attention and a draft model trained for the exact
  target. `llama_cpp.Llama(flash_attn=True)` alone does not enable it, so
  bolting `draft_model=` onto `engine.py` would benchmark a different feature
  and produce a misleading "DFlash is not faster" result.
- `llama-server` also brings prompt caching, parallel slots, grammar, and
  `/v1/chat/completions` compatibility for free. None of that is reachable
  through the binding.

The alternative — keep the in-process backend and accept classic speculative
decoding — is cheaper but cannot meet the DFlash acceptance criteria. This
proposal keeps both backends and makes the choice explicit.

## Goal

Add a second backend that:

1. starts `llama-server` as a child process bound to **loopback only** on an
   ephemeral port,
2. translates a `Plan` into `llama-server` flags,
3. proxies chat completions over HTTP, preserving streaming and the gateway's
   existing OpenAI surface,
4. exposes optional DFlash flags when the registry entry declares a compatible
   draft model.

Non-goals: multi-model routing, replacing the in-process backend as the
default, GPU sharing between two backends, or changing `/v1` semantics.

## Current shape

`server.py` calls `engine.chat(messages, stream=..., **params)` and serializes
what it gets back with `json.dumps`. `engine.chat` returns either a dict or a
**sync** generator of dicts, and the lock in `server.py` assumes one generation
at a time. Any backend must keep that contract exactly, including the
"close the sync iterator to stop generation" rule.

```
cmd/serve_cmd ──> Engine(plan, path) ──> llama_cpp.Llama
                       │
server.build_app ──────┘   chat() -> dict | Iterator[dict]
```

## Proposed shape

```
serve_cmd ──> Backend (Protocol) ──┬── LlamaCppBackend  (moved from Engine)
                                   └── LlamaServerBackend
                                            │ spawn
                                            ▼
                                   llama-server :ephemeral
                                   /health, /v1/chat/completions
```

New files:

| File | Contents |
|---|---|
| `src/vipera/backends/base.py` | `Backend` protocol: `chat()`, `close()`, `describe()`, `healthy()`. |
| `src/vipera/backends/llama_cpp.py` | Today's `Engine` body, unchanged behaviour, renamed. |
| `src/vipera/backends/llama_server.py` | Spawn, readiness, request/stream proxy, teardown. |
| `src/vipera/backends/__init__.py` | `create_backend(kind, plan, path, options)` factory. |

`engine.py` becomes a thin compatibility shim (`Engine = LlamaCppBackend`) so
existing imports and tests keep working; the module can be removed in a later
release.

### Process lifecycle

1. **Binary resolution.** `VIPERA_LLAMA_SERVER` env var, then `llama-server` on
   `PATH`, then `~/.vipera/bin/llama-server`. Never `shell=True`. If the binary
   is missing, fail before binding the gateway port with an install hint.
2. **Capability probe.** Run `llama-server --help` once per binary (cache by
   mtime + size in `$VIPERA_HOME/cache/llama-server.json`) and record whether
   `--spec-type`, `-md`, `-fa`, `--api-key`, `--no-webui` are supported. A
   DFlash request against an old binary fails with "this build lacks
   --spec-type" instead of a child that exits 1 mid-load.
3. **Port selection.** Bind `socket(AF_INET, SOCK_STREAM)` to
   `127.0.0.1:0`, read the port, close, pass it to `--port`. The TOCTOU race is
   acceptable on loopback and far cheaper than parsing startup logs; if the
   child exits with "address already in use", retry once with a new port.
4. **Auth.** Generate a 32-byte `secrets.token_urlsafe` per process and pass
   `--api-key`. The proxy sends `Authorization: Bearer <token>`. Loopback plus
   a per-process key means a stale process cannot be driven by another local
   user by accident.
5. **Readiness.** Poll `GET /health` every 100 ms until `{"status":"ok"}` or
   `startup_timeout_s` (default 180, configurable). Capture stderr into a ring
   buffer of the last 200 lines; on timeout include them in the exception,
   because `llama-server` reports OOM and unknown-quant errors only there.
6. **Teardown.** `terminate()` → wait 5 s → `kill()`. Register `atexit` and a
   `SIGINT`/`SIGTERM` handler in `serve_cmd` so Ctrl+C does not orphan the
   child. `close()` is idempotent, matching `Engine.close()`.

### Flag mapping

| `Plan` field | `llama-server` flag |
|---|---|
| `model_path` | `-m <path>` |
| `gpu_layers` | `-ngl <n>` |
| `context` | `-c <n>` |
| `cpu_threads` | `-t <n>` |
| `n_batch` | `-b <n>` |
| `min(n_batch, 512)` | `-ub <n>` |
| `flash_attn` | `-fa on` / `-fa off` |
| `kv_cache` (`f32`→`f32`, `f16`→`f16`, `q4_0`, `q4_1`, `q8_0`) | `-ctk <type> -ctv <type>` |
| `offload_kqv=False` | `--no-kv-offload` |
| `use_mlock=True` | `--mlock` |
| always | `--host 127.0.0.1`, `--port <p>`, `--api-key <tok>`, `--no-webui`, `--jinja` |
| draft config | `-md <draft.gguf>`, `--spec-type draft-dflash` |

`--jinja` is unconditional because tool calling through the proxy must behave
like the in-process path; without it `llama-server` uses its built-in chat
templates and tool calls silently stop working.

### DFlash configuration

DFlash drafts are trained per target, so auto-detection from `models/` is not
safe: a GGUF draft carries no `draft_for` metadata, and pairing the wrong draft
either fails inside `llama_context` with an opaque message or decodes garbage.
The registry entry must declare it:

```json
{
  "name": "qwen3-4b",
  "path": "/models/Qwen3-4B-Q4_K_M.gguf",
  "meta": {
    "backend": "llama-server",
    "spec": {
      "type": "draft-dflash",
      "draft": "/models/Qwen3-4B-DFlash-Q8_0.gguf",
      "draft_for": "Qwen3-4B"
    }
  }
}
```

Validation before spawn compares target and draft metadata: architecture must
match, `draft_for` must be a case-insensitive substring of the target file name
or alias, and the draft's vocabulary size must equal the target's. Anything
else is a hard error with the two values printed. This is a compatibility
*heuristic*, not a proof; the proposal accepts that a wrong-but-similar draft
still fails at load time, which the stderr ring buffer then explains.

`-fa on` is forced when `spec.type` is set, overriding a plan that disabled
flash attention, because DFlash is invalid without it. That override must be
reported in the serve panel.

### Memory accounting

The planner does not know about drafts, and DFlash adds real VRAM:

- draft weights (Q8_0 0.5–2.7 GiB for the drafts seen so far),
- a second KV cache for the draft,
- flash-attention buffers for both.

A backend that silently adds 2 GiB to a plan sized for a 4 GB card turns a
working serve into an OOM kill. Minimum viable version: `Plan` gains
`draft_vram_gb` (0.0 by default) and `serve` subtracts it from `safe_vram` via
the same `plan()` call, before offload is computed. Until that exists, DFlash
is refused when `vram_free_gb` is below a configured floor (default 6 GiB) and
the message says why. CPU-only DFlash is refused outright: flash attention on
the CPU backend measured 0.13 t/s at 64K on the reference box, so the feature
would appear to "work" while being unusable.

### Request proxy

Non-streaming: `httpx.Client.post(.../v1/chat/completions, json=body,
timeout=None)` inside the caller's worker thread; return `response.json()`.
Errors map to gateway errors: child 4xx → 400 with the child's `error.message`,
child 5xx → 502, connection refused → 503 with "backend exited, see logs".

Streaming: open `client.stream("POST", ...)`, iterate lines, parse `data:`
frames, `yield` each decoded object, stop at `[DONE]`. The proxy must **not**
re-emit the child's SSE text: `server.py` already serializes dicts, and passing
raw bytes through would double-encode. Closing the generator (client
disconnect) closes the HTTP response, which tells `llama-server` to stop — the
same cancellation semantics the lock already relies on.

Not forwarded: `model` (already validated by `server.py`), `stream_options`
(the proxy asks the child for usage when the gateway needs it), and anything
outside `engine._CHAT_PARAMS`. `functions`/`function_call` are legacy and are
not supported by `llama-server`; they stay filtered out.

### Health

`/health` stays cheap: return `{"status":"ok"}` only when the child answers its
own `/health`; otherwise `503` with the child's state (`starting`, `exited`,
`unreachable`). Do not restart on the health endpoint; restart once on the next
chat request if the process is gone, and log the restart.

## Testing

The whole backend can be tested without model weights:

- A fake `llama-server` is a 40-line Python script using `http.server` that
  answers `/health` and emits a fixed SSE stream. Tests point
  `VIPERA_LLAMA_SERVER` at it.
- Unit tests assert the exact argv for a given `Plan` (including `-fa on`
  override and `--no-kv-offload`), that `--port` is in range, and that the API
  key is not logged.
- Lifecycle tests: readiness timeout reports stderr; early exit reports exit
  code; `close()` reaps the PID; double `close()` is a no-op.
- Proxy tests: streaming yields the same dict sequence the in-process backend
  yields; a child 400 becomes a gateway 400; `[DONE]` is not forwarded.
- Optional integration test, skipped unless `VIPERA_DFLASH_MODEL` and
  `VIPERA_DFLASH_DRAFT` are set, loads a real pair and asserts tokens/s.

## Rollout

1. **Probe + spawn** (half day): binary resolution, capability cache, health
   polling, teardown, fake-server tests.
2. **Chat proxy** (1 day): non-streaming then streaming, error mapping, flag
   mapping, `vipera serve --backend llama-server`.
3. **Planner awareness** (half day): draft VRAM accounting, refusal floor,
   serve panel line.
4. **DFlash flags** (half day): registry `spec` block, validation, `-md` +
   `--spec-type draft-dflash`, `-fa on` override.
5. **Benchmark** (half day, separate PR): prompt t/s, gen t/s, TTFT, and VRAM
   with and without the draft on a machine with ≥8 GiB VRAM. The acceptance
   target (≥10% generation speedup) cannot be measured on the 4 GB reference
   box and must name the hardware it was measured on.

## Risks

| Risk | Mitigation |
|---|---|
| Orphaned child after a crash | `atexit`, signal handler, PID file under `$VIPERA_HOME/run/`, startup sweep of stale PIDs. |
| Port race | Retry once on bind failure; never reuse a port across restarts. |
| Child logs swallow the real error | Ring buffer of stderr, included in the exception and in `/health` detail. |
| Planner over-commits VRAM with a draft | Draft VRAM subtracted before offload; hard refusal below the floor. |
| `llama-server` flag drift across versions | Capability probe per binary, not a hard-coded version check. |
| Two SSE implementations diverge | Proxy yields dicts, `server.py` remains the only serializer. |
| Prompt cache semantics differ (`cache_prompt`) | Keep the child's default on; document that a model or `ctx` change restarts the child. |
| Security: local port exposed | Loopback bind, per-process API key, `--no-webui`, no shell interpolation. |

## Open questions

- Should `serve start` prefer `llama-server` when the binary exists and the
  model has a `spec` block, or always require `--backend`? Proposal: require
  the flag, and print a hint when a `spec` block is present but unused.
- Should the child inherit the gateway's environment? Proposal: pass a scrubbed
  environment (PATH, HOME, CUDA_VISIBLE_DEVICES) so `HF_TOKEN`-style variables
  do not leak into a subprocess that only needs a file path.
- Is a persistent PID file worth the complexity versus `atexit` plus a
  `pkill -f` hint in the error message? Proposal: PID file, because orphaned
  servers holding VRAM are the failure users cannot diagnose themselves.
