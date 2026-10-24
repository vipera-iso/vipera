---
title: Configuration
description: config.json keys, storage layout, registry schema, and VIPERA_HOME.
---

# Configuration and storage

Vipera keeps everything under one directory: `VIPERA_HOME`, which defaults to
`~/.vipera`. Configuration is a single JSON file; there is no environment
variable per setting and no global/system-wide config.

## Layout

```text
~/.vipera/
├── config.json      # gateway settings
├── registry.json    # model index (alias → path + metadata)
├── models/          # *.gguf artifacts
└── logs/            # llama-server child output (backend = server)
```

`logs/` holds one `llama-server-<port>.log` per server-backend run. The
in-process backend writes nothing there yet.

### `VIPERA_HOME`

```bash
VIPERA_HOME=/srv/vipera vipera serve start
```

The environment variable is read **once, at import time**, in
`src/vipera/config.py`. Changing it after the package has been imported (for
example in a long-lived Python process) has no effect; set it before launching
`vipera`. Tests patch the module constants rather than the environment for this
reason.

Directories and files are created with the process umask; Vipera does not
tighten permissions on model files or config.

## `config.json`

Create or edit it with `vipera system config`, or write JSON directly:

```bash
vipera system config                 # show all
vipera system config port            # show one
vipera system config port 9000       # set one
```

Unknown keys are rejected by `vipera system config` (exit 1 with the valid key
list). A key present in the file but absent from the defaults is preserved when
the file is loaded and rewritten by the next `save_config` call, but the CLI
cannot create it.

If `config.json` is missing or unparseable, `load_config()` silently falls back
to the defaults. Corrupt JSON does not produce an error.

### Keys

| Key | Type | Default | Read by | Meaning |
|---|---|---|---|---|
| `host` | `str` | `127.0.0.1` | `serve start`, `serve status` | Gateway bind address. |
| `port` | `int` | `8686` | `serve start`, `serve status` | Gateway port. |
| `default_model` | `str \| null` | `null` | root command, `serve start`, `chat`, `bench` | Alias used when `--model` is absent. |
| `backend` | `str` | `inprocess` | `serve start`, `chat`, `bench` | Where inference runs: `inprocess` (llama-cpp-python) or `server` (a `llama-server` child). See [backends](#backends). |
| `protocols` | `str` | `openai` | `serve start` | HTTP layers to mount: `openai` (`/v1/*`), `ollama` (`/api/*`), comma-separated. See [protocols](#protocols). |
| `embeddings` | `bool` | `false` | `serve start`, `chat`, `bench` | Load the model with embedding support, so `/v1/embeddings` and `/api/embed` can answer. Off by default because llama.cpp builds the embedding context up front. |
| `profile` | `str` | `auto` | nothing currently | Planner profile. `serve start --profile` defaults to `auto` independently. |
| `ctx` | `int` | `8192` | `serve start`, `chat` | Context size for the planner. `bench run` uses its own `--ctx` list. |
| `kv_cache` | `str` | `q8_0` | `serve start`, `chat`, `bench` | KV cache type: `f32`, `bf16`, `f16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`. Honored on CPU-only plans too; flash attention is enabled only for the quantized types, which require it. |
| `n_batch` | `int` | `512` | `serve start`, `chat`, `bench` | Prompt batch size; capped by the plan. |
| `spec_type` | `str` | `ngram-map-k` | `serve start`, `chat`, `bench` with `backend: server` | llama-server's `--spec-type`, i.e. draft-free n-gram speculation. `none` turns it off; `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` are the draft-free values. Measured +37% decode when the answer rewrites text from the prompt, -1% on fresh prose; ignored by `inprocess` (no equivalent exists: the Python binding's prompt lookup measured -29%). Background in [techniques](../OPTIMIZE-techniques.md). |
| `temperature` | `float` | `0.7` | nothing currently | Per-request default comes from `server.ChatRequest`. |
| `max_tokens` | `int` | `2048` | nothing currently | Per-request default comes from `server.ChatRequest`. |
| `top_p` | `float` | `0.95` | nothing currently | Per-request default comes from `server.ChatRequest`. |

The "read by nothing" rows are accepted and persisted so the schema is stable,
but changing them does not alter runtime behavior today. See
[known gaps](architecture.md#known-gaps).

### Backends

`backend` selects where the tokens come from:

| Value | Runtime | When to use it |
|---|---|---|
| `inprocess` | `llama-cpp-python` in the gateway process | Default. No extra binary, one process to stop. |
| `server` | `llama-server` as a child process, spoken to over its OpenAI/SSE API | You have a newer or differently built llama.cpp (CPU kernels, flags) than the Python wheel bundles. |

The gateway API is identical for both; the planner produces the flags in both
cases. The server backend looks for `llama-server` in `VIPERA_LLAMA_BIN`, then
`~/.vipera/bin`, `~/llama.cpp/build/bin`, `/usr/local/bin`, `/opt/homebrew/bin`,
then `PATH`. It writes the child's output to `logs/llama-server-<port>.log` and
shuts the child down with the gateway. If the GPU plan fails to start it
retries CPU-only, exactly like the in-process engine.

```bash
vipera system config backend server
vipera serve start                 # child llama-server, same :8686 API
vipera system config backend inprocess
```

### Protocols

`protocols` selects which HTTP layers the gateway mounts. `--protocol`
overrides the config for one run.

| Layer | Routes | Client compatibility |
|---|---|---|
| `openai` | `/v1/models`, `/v1/chat/completions`, `/v1/embeddings` | OpenAI SDKs, LangChain, LlamaIndex |
| `ollama` | `/api/tags`, `/api/ps`, `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `/api/version` | Ollama SDKs, Open WebUI |

```bash
vipera system config protocols openai,ollama
vipera serve start --model <alias>          # both /v1/* and /api/*
vipera serve start --model <alias> --protocol ollama   # /api/* only
```

Unknown names are dropped on load (with a warning) and rejected by
`vipera system config`. Additional layers can be installed as plugins through
the `vipera.protocols` entry-point group.

### Type coercion

When loading, known keys are coerced:

- `port`, `ctx`, `n_batch`, `max_tokens`, `n_ubatch` → `int`
- `temperature`, `top_p` → `float`
- `embeddings` → `bool`
- While loading, strings `"1"`, `"true"`, `"yes"`, `"on"` (case-insensitive)
  become `True`; every other string becomes `False`.
- `backend`, `protocols` and `kv_cache` are checked against their allowed
  sets: an unknown value falls back to the default on load (a protocol is
  dropped from the list), and `vipera system config` rejects it with the list
  of valid values.
- `vipera system config` uses strict coercion: an unrecognized boolean raises
  an error instead of silently becoming `False`.

```bash
vipera system config port 9000        # stored as 9000, not "9000"
vipera system config embeddings true  # stored as true
```

Coercion failures during load (for example `port = "abc"`) are ignored, leaving
the raw value in memory. `save_config` writes whatever is in memory, so a bad
value can round-trip and fail later when it is used. Prefer valid values.

`save_config` writes a temporary file and atomically renames it, matching the
registry, so a crash mid-write cannot truncate `config.json`.

## Registry

`registry.json` is a flat object keyed by alias:

```json
{
  "deepseek-v2-lite": {
    "name": "deepseek-v2-lite",
    "path": "/home/user/.vipera/models/deepseek-v2-lite.gguf",
    "added": "2026-10-17T18:04:11.123456Z",
    "meta": {
      "size_bytes": 10363723776,
      "n_layers": 27,
      "n_params": 15700000000,
      "context_length": 163840,
      "quant": "Q4_K_M",
      "architecture": "deepseek2"
    }
  }
}
```

- `name` duplicates the key; consumers index by key.
- `path` is stored as a string and may be relative if a download returned a
  relative path.
- `added` is UTC, ISO 8601, with `+00:00` normalized to `Z`.
- `meta` is written by `model import` / `model search`. Missing fields are
  tolerated by the planner, which falls back to defaults.

`size_bytes` is always the on-disk size. `n_layers`, `n_params`,
`context_length`, and `architecture` require the optional `gguf` package; when
it is absent, `n_layers` stays `0` and the CLI displays `?`. `quant` is parsed
from the filename when GGUF metadata is unavailable.

### Registry operations

| Operation | Trigger | Notes |
|---|---|---|
| Add/overwrite | `model import`, `model search` | Same alias overwrites the entry. Both commands generate a unique alias first. |
| Set default | `model import`, `model search` | `default_model` is updated in `config.json`. |
| Remove | `model rm` | Deletes the registry entry and the file at `path`; clears `default_model` if it matched. |

The registry has no file locking and writes are not atomic. Running two Vipera
commands that mutate the registry at the same time can lose an entry or leave a
partially written file; the next read then falls back to `{}` because parse
errors are suppressed. Do not run `model import`, `model search`, or `model rm`
concurrently.

## Environment variables

| Variable | Effect |
|---|---|
| `VIPERA_HOME` | Root storage directory. Read at import time. |
| `VIPERA_LLAMA_BIN` | Directory searched for `llama-server` (and the quantize tools). |

Nothing else is configurable via the environment. In particular, `HOST` and
`PORT` have no effect; use `vipera serve start --host/--port` or the config
file.

## Typical recipes

**Change the port and keep the model default:**

```bash
vipera system config port 9000
vipera serve start
```

**Serve a different model without changing the default:**

```bash
vipera serve start --model other-model --port 9000
```

**Keep two independent setups on one machine:**

```bash
VIPERA_HOME=~/.vipera-work vipera model list
VIPERA_HOME=~/.vipera-personal vipera model list
```

Each home has its own config, registry, and model files. Models are copied, not
shared, so disk usage doubles.
