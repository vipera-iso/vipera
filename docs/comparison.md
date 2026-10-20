---
title: Comparison
description: How Vipera differs from Ollama, llama.cpp's server, and LM Studio, row by row.
---

# Compared to the alternatives

| | **Vipera** | Ollama | `llama.cpp/server` | LM Studio |
|---|:---:|:---:|:---:|:---:|
| GGUF-native | ✅ | ✅ | ✅ | ✅ |
| OpenAI-compatible API | ✅ | ✅ | ✅ | ✅ |
| Hardware-aware planner (VRAM/RAM/disk) | ✅ | ⚠️ partial | ❌ | ⚠️ partial |
| Real SSE streaming | ✅ | ✅ | ✅ | ✅ |
| First-class CLI (no GUI) | ✅ | ✅ | ✅ | ❌ |
| HuggingFace search built-in | ✅ | ✅ | ❌ | ✅ |
| Diagnostics (`doctor`) | ✅ | ⚠️ partial | ❌ | ❌ |
| Python-native, embeddable | ✅ | ❌ | ❌ | ❌ |

Every row below explains the mark, with the file that implements it.

## GGUF-native

All four load `.gguf` directly; none of them needs a conversion step to a
framework-specific format. Vipera reads the GGUF index itself for metadata and
tensor sizes, so a model can be inspected without loading weights
([`gguf_analyzer.py`](../src/vipera/gguf_analyzer.py)).

## OpenAI-compatible API

All four accept `POST /v1/chat/completions`. The difference is what reaches the
backend: Vipera forwards the full sampling surface — `stop`, `seed`,
`logit_bias`, `logprobs`, penalties, `response_format`, `tools`/`tool_choice` —
instead of dropping unknown fields
([`engine.py`](../src/vipera/engine.py), [`server.py`](../src/vipera/server.py)).

## Hardware-aware planner

This is the row that separates the tools.

- **Vipera ✅** — `plan()` turns `(model metadata, hardware)` into
  `gpu_layers`, `context`, `kv_cache`, `flash_attn`, `n_batch`, `offload_kqv`,
  and `use_mlock`, budgeting VRAM per layer and reserving room for compute
  buffers ([`planner.py`](../src/vipera/planner.py),
  [`hardware.py`](../src/vipera/hardware.py)).
- **Ollama ⚠️ partial** — it offloads layers and picks a context size, but the
  decision is a heuristic over the model and detected VRAM rather than a
  per-layer budget with an explicit KV-cache estimate.
- **`llama.cpp/server` ❌** — it exposes `-ngl`, `-c`, `-ctk`/`-ctv` and leaves
  the choice to the operator. Correct defaults exist, but nothing plans them.
- **LM Studio ⚠️ partial** — the GUI estimates whether a model fits and picks
  an offload ratio; the estimate is not exposed as a reviewable plan or used by
  a headless server.

## Real SSE streaming

All four stream tokens as Server-Sent Events. Vipera returns a synchronous
generator directly to `StreamingResponse`, so each `llama.cpp` chunk is flushed
as it is produced rather than buffered
([`server.py`](../src/vipera/server.py)).

## First-class CLI

Ollama and `llama.cpp` both ship complete CLIs; LM Studio is GUI-first with a
`lms` companion. Vipera's CLI covers the full lifecycle — model management,
serving, chat, benchmarking, diagnostics — and every command is discoverable
with `--help` ([`commands/`](../src/vipera/commands/)).

## HuggingFace search built-in

Vipera and Ollama both search and pull GGUF repositories from the CLI. LM
Studio searches from the GUI. `llama.cpp/server` takes a local path and nothing
else ([`model_cmd.py`](../src/vipera/commands/model_cmd.py)).

## Diagnostics

`vipera system doctor` reports the runtime, whether `llama-cpp-python` and
`gguf` import, and the detected hardware, and tells you which install command
fixes a missing piece ([`system_cmd.py`](../src/vipera/commands/system_cmd.py)).
Ollama has `ollama ps` and debug logging but no single environment check;
`llama.cpp` and LM Studio have none.

## Python-native, embeddable

Vipera is a Python package: `from vipera.planner import plan` and
`from vipera.engine import Engine` work in a script, and `build_app(engine,
alias)` returns a FastAPI app you can mount
([`server.py`](../src/vipera/server.py)). Ollama is a Go daemon with a Python
client; `llama.cpp` has Python bindings but no gateway layer; LM Studio is
closed-source.

## What Vipera does not do

- One model per gateway process; no multi-model routing.
- No authentication, TLS, CORS, or rate limiting. Bind to loopback.
- No expert-placement flags for MoE models, and no disk-offload planner.
- No web UI.

See [architecture.md](architecture.md#known-gaps) for the maintained list.
