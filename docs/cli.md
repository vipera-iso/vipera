# CLI reference

Vipera installs one entry point, `vipera` (mapped to `vipera.cli:app`). Every
group supports `--help`.

```text
vipera [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|---|---|
| `-v`, `--version` | Print `vipera <version>` and exit. |
| `-s`, `--select` | Force the interactive model picker when starting the gateway. |
| `--install-completion` / `--show-completion` | Typer's shell completion helpers. |

## Starting the gateway

With no subcommand, `vipera` boots the gateway:

1. Prints the banner.
2. If the registry is empty, prints a warning and exits with status 1.
3. Selects a model:
   - `--select` → always show the picker.
   - `default_model` set and `--select` absent → use it.
   - no default and exactly one model installed → auto-select it.
   - no default and several models → show the picker.
4. Delegates to `vipera serve start --model <alias>`.

```bash
vipera            # default model, or picker if needed
vipera -s         # always pick
vipera --version
vipera help       # rich usage guide
```

## `vipera model`

Manage GGUF artifacts and the local registry. Importing or downloading a model
sets it as `default_model`.

### `vipera model import SRC [--name ALIAS]`

Copy a local `.gguf` into `$VIPERA_HOME/models/`, read its metadata, register
it, and make it the default.

| Argument/Option | Description |
|---|---|
| `SRC` | Path to a `.gguf` file. `~` is expanded; the path must exist. |
| `-n`, `--name` | Alias. Defaults to the file stem (with a trailing `.gguf` removed). |

If the alias is already taken by a registry entry or an existing
`models/<alias>.gguf`, a numeric suffix is appended: `alias-2`, `alias-3`, ...

The destination is `models/<alias>.gguf`. If the source already is that path,
the copy is skipped. Metadata extraction uses the optional `gguf` package when
installed; otherwise size and quantization (parsed from the filename) are still
recorded. Output reports parameter count, size, layer count, and quantization.

Exit status is 1 if the file does not exist.

### `vipera model search`

Interactive Hugging Face search and download.

1. Prompts for a keyword.
2. Queries Hugging Face for up to 10 repositories tagged `gguf`, sorted by the
   API's default relevance.
3. Prompts for a repository, then lists every `.gguf` file in it.
4. Downloads the chosen file into `$VIPERA_HOME/models/` with
   `hf_hub_download` and registers it under a unique alias derived from the
   filename.
5. Sets it as `default_model` and suggests running `vipera`.

There are no non-interactive flags; this command is for humans. Connection
failures and downloads that raise are reported and the command exits (download
errors print a message without re-raising).

### `vipera model inspect PATH [--tensors]`

Read-only analysis of a GGUF file. The index is parsed with the optional `gguf`
package; tensor payloads are never read into memory.

| Argument/Option | Description |
|---|---|
| `PATH` | Path to a `.gguf` file. `~` is expanded. |
| `--tensors` | Also print every tensor: layer, name, quantization, shape, size. |

Output is three tables — summary (architecture, layers, context, embedding,
quantization, tensor count, file size), size by category (embedding, attention,
FFN, expert, norm, other), and quantization mix — plus a `QUALITY WARNINGS`
panel when `attn_v` or `ffn_down` tensors are quantized below Q4_K. The warning
is a heuristic, not a quality measurement.

Exit status is 1 when the file is missing or `gguf` is not installed; the
message includes the install command.

### `vipera model imatrix PATH --calibration-data FILE`

Run `llama-imatrix` to compute an importance matrix from a plain-text
calibration file.

| Argument/Option | Default | Description |
|---|---|---|
| `PATH` | required | Source `.gguf` file. |
| `-f`, `--calibration-data` | required | UTF-8 text used for calibration. |
| `-o`, `--output` | `<model>.imatrix.gguf` | Output matrix path; refuses to overwrite. |
| `--ctx` | `512` | Calibration context size. |
| `-t`, `--threads` | llama.cpp default | CPU threads. |
| `--bin-dir` | auto | Directory containing `llama-imatrix`. |
| `--dry-run` | off | Print the command without running it. |

`llama-imatrix` requires at least twice as many calibration tokens as `--ctx`,
so a very small calibration file fails fast. Search order for the binary is
`--bin-dir`, `$VIPERA_LLAMA_BIN`, `~/.vipera/bin`, common llama.cpp build
paths, then `PATH`.

### `vipera model quantize PATH [OPTIONS]`

Run `llama-quantize` on a GGUF file.

| Argument/Option | Default | Description |
|---|---|---|
| `PATH` | required | Source `.gguf` file. |
| `-q`, `--target-quant` | `q4_k_m` | Target type (`q4_k_m`, `q5_k_m`, `q8_0`, …). |
| `--imatrix` | none | Importance matrix from `model imatrix`. |
| `-o`, `--output` | `<model>-vipera-<quant>.gguf` | Output path; refuses to overwrite. |
| `--prune-layers` | none | Comma-separated layer indices to drop. |
| `--prior-weight` | `1` | MoE expert prior; rejected if the build lacks the flag. |
| `--allow-requantize` | off | Permit quantizing an already-quantized source. |
| `--bin-dir` | auto | Directory containing `llama-quantize`. |
| `--dry-run` | off | Print the command without running it. |

llama.cpp disables requantization by default; without `--allow-requantize` the
tool exits with an error. Vipera deletes a partial output file when the
underlying command fails. Pruning warns because it can substantially reduce
quality.

### `vipera model optimize PATH [OPTIONS]`

High-level pipeline: inspect → choose a quant → optionally build an imatrix →
quantize. Requires `--target-size` (decimal GB) or `--target-quant`.

| Option | Default | Description |
|---|---|---|
| `--target-size` | none | Target file size in decimal GB. |
| `--target-quant` | none | Skip sizing and use this quant. |
| `-f`, `--calibration-data` | none | Build an imatrix before quantizing. |
| `-o`, `--output` | `<model>-vipera-<quant>.gguf` | Output path. |
| `--prune-layers` | none | Passed through to `llama-quantize`. |
| `--prior-weight` | `1` | MoE expert prior. |
| `--bin-dir` | auto | Directory containing both tools. |
| `--dry-run` | off | Print the full plan and both commands. |

The chosen quant is the highest-quality entry in `Q8_0, Q6_K, Q5_K_M, Q4_K_M,
Q3_K_M, Q2_K` whose estimated size fits the budget. The estimate scales the
source file size by bits-per-weight ratios, so it is approximate for mixed-quant
sources and models with large embeddings. `optimize` passes
`--allow-requantize` because its source is often already quantized.

### `vipera model list`

Print a table of registered models: alias, file size, quantization, layer count,
and training context length. Shows a hint when the registry is empty.

### `vipera model rm NAME [--yes]`

Delete the model's file, remove its registry entry, and clear `default_model`
if it pointed at this model.

| Option | Description |
|---|---|
| `-y`, `--yes` | Skip the confirmation prompt. |

Exit status is 1 if the alias is unknown; cancelling the prompt aborts.

## `vipera serve`

### `vipera serve start [OPTIONS]`

Load a model and run the OpenAI-compatible HTTP server.

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | `default_model` | Registry alias to serve. Required if no default is set. |
| `--host` | `config.host` (`127.0.0.1`) | Bind address. |
| `--port`, `-p` | `config.port` (`8686`) | Bind port. |
| `--profile` | `auto` | Planner profile (`auto`, `cpu`, `gpu`, `lowmem`). Note this is **not** read from config. |
| `--ctx` | `config.ctx` (`8192`) | Context size passed to the planner. |
| `--bench` | off | Run the 20-token speed check **before** serving. This delays readiness, so it is opt-in. |

Startup sequence: resolve model → print plan → construct `Engine` →
`uvicorn.run(log_level="warning")`. The gateway is ready as soon as uvicorn
binds; no benchmark blocks it.

`--bench` generates up to 20 tokens from a fixed prompt and labels the result
`FAST` (≥20 t/s), `OK` (≥8 t/s), or `SLOW`. It is informational and never fails
startup. Use `vipera bench run` for a real measurement across context sizes.

If `uvicorn`/`fastapi` are missing, the CLI warns and drops into the chat REPL
with the already-loaded engine instead of serving HTTP.

Exit status is 1 when the model is unspecified, unknown, or fails to load.

### `vipera serve stop`

Stop a gateway that was started by `vipera serve start`. The running process
writes a pid file (`$VIPERA_HOME/serve-<port>.pid`); `stop` reads it, refuses to
signal a PID whose command line does not contain `vipera`, sends `SIGTERM`, and
waits up to 10 seconds. `--force` escalates to `SIGKILL`.

| Option | Default | Description |
|---|---|---|
| `--port`, `-p` | `config.port` | Port whose pid file should be used. |
| `--force` | off | Kill if the process ignores `SIGTERM` for 10 seconds. |
| `-y`, `--yes` | off | Skip the confirmation prompt. |

Exit status is 1 when there is no pid file, the pid is gone, or the process
cannot be identified as vipera.

### `vipera serve status`

Open a TCP connection to `config.host:config.port` and print `ONLINE` or
`OFFLINE`. This checks the port, not the HTTP `/health` payload, so any process
listening on that port is reported as online.

## `vipera chat`

Both subcommands resolve the model as `--model`, then `default_model`, then the
picker. Loading uses the planner with `config.ctx`, `config.kv_cache`, and
`config.n_batch`.

### `vipera chat repl [-m ALIAS]`

Interactive multi-turn REPL seeded with a single system message:
`You are a helpful assistant.`

- `exit`, `quit`, or `:q` leaves; Ctrl+C / EOF also leaves.
- Empty input is ignored.
- An error during generation is printed and the failed user turn is removed from
  history.

Responses are rendered through a Rich live panel with a per-character typewriter
effect. That effect adds artificial delay, so the chunk rate shown under a
response is a rendering rate, not inference throughput. Use `vipera bench run`
or the HTTP API for real timings.

### `vipera chat oneshot "PROMPT" [-m ALIAS]`

Send one user message, stream the reply, print it, exit. Useful inside scripts;
output is still rendered with the typewriter effect.

## `vipera bench`

### `vipera bench run [MODEL] [--ctx LIST] [--max-tokens N]`

Measure prefill and generation throughput across context sizes.

| Argument/Option | Default | Description |
|---|---|---|
| `MODEL` | `default_model` (or picker) | Registry alias. |
| `--ctx` | `512,2048` | Comma-separated context sizes. A fresh engine is loaded for each. |
| `--max-tokens` | `128` | Generation cap per run. |

For each context size the command plans and loads a new engine, then streams a
fixed English prompt and reports:

| Column | Definition |
|---|---|
| `GPU LAYERS` | `gpu_layers / total_layers` for that plan. |
| `PROMPT TOK` | Tokenized prompt length (0 if tokenization fails). |
| `TTFT` | Time to first output token, in milliseconds. |
| `PREFILL T/s` | `prompt_tokens / TTFT` — an approximation that ignores prefill/queue overlap. |
| `GEN TOK` | Number of non-empty streaming deltas (a chunk count, not a tokenizer count). |
| `GEN T/s` | `(GEN TOK - 1) / (total_time - TTFT)`, i.e. excluding first-token latency. |
| `TOTAL` | Wall-clock seconds for the run. |

A load or generation failure is reported per context size and the command
continues with the next size. Engines are released between runs.

## `vipera system`

### `vipera system doctor`

Print a diagnostics panel with three sections and a PASS/WARN/FAIL summary:

- **Runtime**: OS and release, Python version, whether `llama_cpp` is
  importable (FAIL if missing), whether `gguf` is importable (WARN if missing).
- **Hardware**: CPU cores/threads; on Apple Silicon the Metal GPU and unified
  memory; on NVIDIA the GPU name and VRAM; otherwise a CPU-only warning plus
  available RAM.

If `llama-cpp-python` or `gguf` is missing, `doctor` prints the corresponding
`pip install` command below the panel. `doctor` does not load a model, so it can
be run before downloading anything.

### `vipera system config [KEY] [VALUE]`

Read or write `$VIPERA_HOME/config.json`.

| Invocation | Behavior |
|---|---|
| `vipera system config` | Print all current keys and values. |
| `vipera system config KEY` | Print one value. Unknown keys exit 1 and list valid keys. |
| `vipera system config KEY VALUE` | Set and persist. Values are coerced to the key's type. |

See [configuration.md](configuration.md) for keys, types, and coercion rules.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success, including `--version` and `help`. |
| `1` | No models installed, unknown model, missing file, unknown config key, invalid selection, failed load. |
| `2` | Typer/Click usage error (bad flag, missing argument). |

## Global behavior notes

- All output is Rich-formatted to the terminal; there is no `--json` or plain
  mode. Script against the HTTP API when you need machine-readable output.
- The model picker is interactive and requires a TTY-like stdin. Commands that
  may invoke it (`chat`, `bench`) accept `--model`/positional arguments to avoid
  the prompt.
