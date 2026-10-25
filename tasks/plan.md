# Implementation Plan: Ollama parity for Vipera

## Overview

Vipera today beats Ollama on one axis (a per-layer, measured hardware plan) and
loses on most of the surface a real Ollama user touches: no Modelfile / model
identity, no vision, no reasoning (`think`), a narrower API, one NVIDIA GPU only,
and no daemon. This plan closes those gaps in an order that keeps the system
shippable after every task, starting with the cheapest high-leverage work
(docs truth, reasoning, `capabilities`) and ending with the expensive ones
(vision, multi-GPU, packaging).

Target: an Ollama user can point the official `ollama` Python SDK and the
OpenAI SDK at Vipera, run their existing models with their existing settings,
and get the same answers. Not a target: Ollama Cloud, an official model
registry, or a desktop app — those are a different product.

## Current state (measured, not assumed)

| Area | Vipera today | Evidence |
|---|---|---|
| Routes | `/v1/{chat/completions,models,embeddings}`, `/api/{version,tags,ps,show,chat,generate,embed,embeddings}` | `src/vipera/protocols/ollama.py`, `openai.py` |
| Multi-model | ✅ pool, lazy load, keep_alive, LRU eviction | `src/vipera/pool.py` |
| Model identity | alias + GGUF meta only, no system prompt / template / params | `src/vipera/registry.py` |
| Vision | rejected: `400 image input is not supported` | `ollama.py:387`; `openai.py:36` |
| Reasoning | absent: no `think`, no `thinking` field, no `/api/show.thinking` | grep across `src/` |
| `capabilities` | hardcoded `["completion"]` | `ollama.py:171` |
| `num_ctx` | accepted, silently ignored (not in `_OPTION_MAP`) | `ollama.py:88`, `_chat_kwargs` |
| GPU | device 0 only, NVML/NVIDIA only, no tensor split | `hardware.py:44` |
| Lifecycle | foreground `serve start` + pid file for status/stop | `commands/serve_cmd.py` |

## Architecture Decisions

1. **Registry schema v3 carries the "model as a product" fields.**
   `ModelSpec` grows `system`, `template`, `parameters`, `license`, `digest`,
   `parent`. `SCHEMA_VERSION` 2 → 3, `_load` accepts 1.x / 2 / 3, `_save` writes
   3. Reason: `/api/show`, `/api/tags`, `create`, and per-model load overrides
   all read the same record; a second store would drift.

2. **Modelfile is parsed into the registry, not into a new file format.**
   `vipera model create -f Modelfile` writes a `ModelSpec`. No new on-disk
   manifest/blob tree: GGUF stays the artifact, the registry stays the catalog.

3. **Per-model `parameters` reach the engine as load-time overrides, not
   per-request rewrites.** `num_ctx`, `kv_cache`, `n_batch` change the loaded
   context and therefore must be applied at load; sampling params (`temperature`,
   `stop`, `top_p`) become request defaults. This mirrors `plan()` owning the
   load and the HTTP layer owning sampling.

4. **`TEMPLATE` is parsed and surfaced, not silently ignored.** llama-server
   takes `--chat-template` at startup; a per-request template is not portable
   across backends. Decision: store it, return it in `/api/show`, apply it via
   `--chat-template` when the server backend is active, warn loudly otherwise.
   Silent ignore is the bug class we just removed from `config.py`.

5. **`think` maps onto the backend's own reasoning controls** (`--reasoning-format`
   / `--chat-template-kwargs` on llama-server, capability-probed like the
   existing `--help` probes). When the backend cannot honour it, answer `400`
   with the reason — never pretend.

6. **`capabilities` is computed, not a constant.** Derived from GGUF metadata
   (vision when an `mmproj` is resolvable, embedding when the model is an
   embedding GGUF), the registry (thinking defaults), and the gateway config
   (tools always, since `tools` already works).

7. **Vision is server-backend-first.** `mmproj` becomes a `Plan` field and a
   `--mmproj` flag; the in-process path is attempted only if the installed
   `llama-cpp-python` exposes a chat handler (version probe), otherwise `400`
   pointing at `backend: server`.

8. **Versioning:** `0.x` until Phase 4 lands, then `1.0.0` only if the parity
   suite in P0-1 reports ≥ 90% of the Ollama SDK surface green.

## Task List

### Phase 0 — Baseline and truth (1 day)

#### Task P0-1: Ollama SDK parity suite

**Description:** Turn `tests/integration/test_ollama_sdk.py` into a harness that
drives every method of the official `ollama` Python client against Vipera and
reports which ones work. Methods that are not implemented yet are `xfail` with a
reason, so the suite is a live scoreboard rather than a red wall.

**Acceptance criteria:**
- [ ] `pytest -q tests/integration/test_ollama_sdk.py` passes and prints/records a count of supported vs `xfail` methods.
- [ ] Covering: `chat`, `generate`, `embed`, `embeddings`, `list`, `ps`, `show`, `create`, `copy`, `delete`, `pull`, `push`.
- [ ] Each `xfail` has `reason=` naming the phase/task that will fix it.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/integration/test_ollama_sdk.py -q`
- [ ] Manual check: the scoreboard numbers are quoted in `tasks/todo.md`.

**Dependencies:** None
**Files likely touched:** `tests/integration/test_ollama_sdk.py`, `pyproject.toml` (if the client needs a pin)
**Estimated scope:** M (3 files)

#### Task P0-2: Fix the docs that sell the product short

**Description:** `docs/comparison.md` claims "one model per gateway process; no
multi-model routing" and "no expert-placement flags for MoE" — both false. It
also links `src/vipera/engine.py`, which does not exist. README documents the
old `/health` body. Correct all of it against the code.

**Acceptance criteria:**
- [ ] `comparison.md` rows for multi-model, MoE placement, `/health` match `pool.py`, `backends/server.py:server_command`, `server.py:health`.
- [ ] No link in `README.md`/`comparison.md` points at a non-existent path.
- [ ] README `/health` example includes `engines` and the 503 rule.

**Verification:**
- [ ] Manual check: `grep -rn "engine.py" docs README.md` returns nothing.
- [ ] Manual check: every relative link in the two files resolves (`python -c` link checker or eyeball).

**Dependencies:** None
**Files likely touched:** `docs/comparison.md`, `README.md`, `docs/http-api.md`
**Estimated scope:** S (3 files)

#### Task P0-3: Baseline measurements

**Description:** Record what "caught up" must not regress: decode/prefill on a
small model, gateway overhead share, and cold-start time, on the reference
machine, before touching the request path.

**Acceptance criteria:**
- [ ] `vipera bench run` numbers recorded in `docs/benchmarks.md` with date, model, hardware.
- [ ] Gateway overhead measured with and without a client (existing methodology in `docs/benchmarks.md`).
- [ ] Numbers reproducible by one documented command.

**Verification:**
- [ ] Manual check: rerun the documented command, values within noise.
- [ ] Tests pass: full suite still green.

**Dependencies:** None
**Files likely touched:** `docs/benchmarks.md`
**Estimated scope:** S (1 file)

### Checkpoint: after P0
- [ ] Parity suite runs and reports a scoreboard.
- [ ] Docs no longer state false gaps.
- [ ] `pytest`, `ruff`, `mypy` clean; baseline recorded.
- [ ] Human reviews the scoreboard before Phase 1 starts.

### Phase 1 — Model identity and Modelfile (4–6 days)

#### Task P1-1: Registry schema v3

**Description:** Add `system`, `template`, `parameters`, `license`, `digest`,
`parent` to `ModelSpec`; bump `SCHEMA_VERSION` to 3; keep reading 1.x flat maps
and v2 files; migrate on next write.

**Acceptance criteria:**
- [ ] A v2 registry file loads unchanged (existing models keep alias/path/meta).
- [ ] A v1 flat map still loads.
- [ ] New fields round-trip through `add_model`/`list_models`/`_save`.
- [ ] Unknown fields survive a write (forward compatibility).

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_registry.py -q`
- [ ] Manual check: write a v2 file by hand, load, save, diff shape.

**Dependencies:** None
**Files likely touched:** `src/vipera/registry.py`, `tests/unit/test_registry.py`
**Estimated scope:** S (2 files)

#### Task P1-2: Modelfile parser

**Description:** New `src/vipera/modelfile.py` parsing `FROM`, `SYSTEM`,
`TEMPLATE`, `PARAMETER`, `ADAPTER`, `LICENSE`, comments and line continuations.
Unknown instructions and malformed `PARAMETER` lines raise `ModelfileError`
with line numbers.

**Acceptance criteria:**
- [ ] Parses a Modelfile with all supported instructions into a dataclass.
- [ ] Multi-line `"""` blocks for SYSTEM/TEMPLATE are preserved byte-exactly.
- [ ] Unknown instruction → `ModelfileError` naming the line.
- [ ] `PARAMETER stop "a"` and `PARAMETER temperature 0.2` produce typed values.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_modelfile.py -q`
- [ ] Manual check: parse a real Ollama Modelfile (e.g. from a model page) (`--dry-run` path).

**Dependencies:** None
**Files likely touched:** `src/vipera/modelfile.py` (new), `tests/unit/test_modelfile.py` (new)
**Estimated scope:** M (2 files)

#### Task P1-3: `vipera model create`

**Description:** `vipera model create -f Modelfile -n alias [--dry-run]`:
resolve `FROM` (local path, registry alias, or `hf:` ref), parse (P1-2), write a
registry entry with the P1-1 fields. Put it in a new command module so
`commands/model_cmd.py` (845 lines) does not grow.

**Acceptance criteria:**
- [ ] `FROM` accepts a `.gguf` path and an existing alias; both create a new alias.
- [ ] `--dry-run` prints the resolved record and writes nothing.
- [ ] Creating over an existing alias requires `--force`.
- [ ] Missing `FROM` file → exit 1 with the path in the message.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_model_create_cmd.py tests/e2e/test_cli.py -q`
- [ ] Manual check: `vipera model create` then `vipera model list` shows system/params.

**Dependencies:** P1-1, P1-2
**Files likely touched:** `src/vipera/commands/modelfile_cmd.py` (new), `src/vipera/cli.py`, `tests/unit/test_model_create_cmd.py` (new)
**Estimated scope:** M (3 files)

#### Task P1-4: `vipera model cp`

**Description:** Copy a registry entry under a new alias, keeping path and
inheriting system/template/parameters (Ollama's `cp`).

**Acceptance criteria:**
- [ ] `vipera model cp a b` creates alias `b` pointing at the same GGUF.
- [ ] Unknown source → exit 1, existing destination without `--force` → exit 1.
- [ ] The copy records `parent` = source digest.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/e2e/test_cli.py -q`
- [ ] Manual check: `vipera model list` shows both aliases, one file on disk.

**Dependencies:** P1-1
**Files likely touched:** `src/vipera/commands/modelfile_cmd.py`, `tests/e2e/test_cli.py`
**Estimated scope:** S (2 files)

#### Task P1-5: Per-model load overrides

**Description:** `plan()` (and the pool factory) take the model's stored
`parameters` and apply the load-time subset (`num_ctx`, `kv_cache`, `n_batch`,
`n_ubatch`, `n_gpu_layers`), leaving sampling params for the request path.

**Acceptance criteria:**
- [ ] A model with `num_ctx: 16384` loads with a 16384 context; default models are unchanged.
- [ ] `kv_cache` from the Modelfile wins over the global config, and the plan says so.
- [ ] An invalid override is repaired with a warning (same policy as `config._coerce`).
- [ ] `Plan` records which values came from the model vs the config.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_planner.py tests/unit/test_pool.py -q`
- [ ] Manual check: `vipera serve start` on a model with `num_ctx` prints the new context.

**Dependencies:** P1-1
**Files likely touched:** `src/vipera/planner.py`, `src/vipera/pool.py`, `src/vipera/backends/loading.py`, `tests/unit/test_planner.py`
**Estimated scope:** M (4 files)

#### Task P1-6: Per-model request defaults

**Description:** When a request omits a sampling parameter, use the model's
stored value before the global default (Ollama behaviour): `temperature`,
`top_p`, `top_k`, `repeat_penalty`, `stop`.

**Acceptance criteria:**
- [ ] A model with `PARAMETER temperature 0.2` answers at 0.2 when the client sends none.
- [ ] A client-provided value still wins.
- [ ] Applied on both `/v1/chat/completions` and `/api/chat|generate`.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_protocols.py tests/unit/test_server.py -q`
- [ ] Manual check: run the same prompt on two models with different `temperature`.

**Dependencies:** P1-1, P1-3
**Files likely touched:** `src/vipera/protocols/ollama.py`, `src/vipera/protocols/openai.py`, `src/vipera/registry.py`, `tests/unit/test_protocols.py`
**Estimated scope:** M (4 files)

#### Task P1-7: System prompt from the registry

**Description:** A model created with `SYSTEM` injects that system message when
the client sends none. Explicit client system messages win, and the injected one
is visible in `/api/show`.

**Acceptance criteria:**
- [ ] `SYSTEM` is prepended as a `system` message only when the request has none.
- [ ] `/api/show` returns `system` and the stored `template`.
- [ ] Shaping/prefix accounting sees the injected message (no double count).

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_protocols.py -q`
- [ ] Manual check: `curl /api/chat` with no system → answer follows the model's system prompt.

**Dependencies:** P1-1, P1-3
**Files likely touched:** `src/vipera/protocols/ollama.py`, `src/vipera/protocols/openai.py`, `src/vipera/optimize/layer.py`
**Estimated scope:** S (3 files)

#### Task P1-8: `/api/create` + `/api/show` fields

**Description:** Serve Ollama's `POST /api/create` (`{name, modelfile}`) on top
of P1-3, and complete `/api/show` with `system`, `template`, `parameters`,
`license`, `modified_at` from the registry.

**Acceptance criteria:**
- [ ] `ollama.create(model="x", modelfile="FROM ...")` works through the official SDK.
- [ ] `/api/show` matches Ollama's documented keys for the fields we have.
- [ ] `create` over an existing model returns Ollama's 4xx shape, not a stack trace.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/integration/test_ollama_sdk.py -q`
- [ ] Manual check: P0-1 scoreboard shows `create` as supported.

**Dependencies:** P1-3, P0-1
**Files likely touched:** `src/vipera/protocols/ollama.py`, `tests/integration/test_ollama_sdk.py`, `tests/unit/test_protocols.py`
**Estimated scope:** S (3 files)

#### Task P1-9: `TEMPLATE` handling decision, implemented

**Description:** Store the template, return it in `/api/show`, and apply it via
`--chat-template` on the server backend (flag-probed). On a backend that cannot
take it, emit one startup warning naming the model — never silently ignore.

**Acceptance criteria:**
- [ ] Server backend plan contains `--chat-template` when the model has one and the binary supports it.
- [ ] Without support: warning printed once at load, request still served.
- [ ] `ollama.show` returns the template verbatim.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_server_backend.py tests/unit/test_planner.py -q`
- [ ] Manual check: `server_command` output for a model with a template.

**Dependencies:** P1-1, P1-5
**Files likely touched:** `src/vipera/backends/server.py`, `src/vipera/planner.py`, `tests/unit/test_server_backend.py`
**Estimated scope:** M (3 files)

### Checkpoint: after P1
- [ ] An Ollama Modelfile with SYSTEM + PARAMETERs produces a working Vipera model.
- [ ] `ollama create/show/list/ps` all green in the parity suite.
- [ ] Full suite + `ruff` + `mypy` clean; docs updated for `model create`.
- [ ] Human reviews before Phase 2.

### Phase 2 — Reasoning and capabilities (2–3 days)

#### Task P2-1: Probe the backend's reasoning controls

**Description:** Before coding `think`, establish what this `llama-server` build
accepts (`--reasoning-format`, `--chat-template-kwargs`) with the existing
`--help` probe pattern, and what `llama-cpp-python` exposes.

**Acceptance criteria:**
- [ ] `supports_reasoning_format(binary)` and a kwargs probe exist with tests, same shape as `_supports_fit`.
- [ ] A short note in `docs/design/` records what the reference build supports and what each value does.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_server_backend.py -q`
- [ ] Manual check: probe output on the installed binary quoted in the design note.

**Dependencies:** None
**Files likely touched:** `src/vipera/backends/server.py`, `docs/design/llama-server-backend.md`, `tests/unit/test_server_backend.py`
**Estimated scope:** S (3 files)

#### Task P2-2: Accept `think` and return `thinking`

**Description:** Add `think: bool | str | None` to the Ollama chat/generate
models, forward it as the backend control, and split the reasoning trace into
`message.thinking` / `thinking` while streaming.

**Acceptance criteria:**
- [ ] `think: true` returns a non-empty `message.thinking` for a reasoning model and unchanged `content`.
- [ ] Streaming yields interleaved `thinking` chunks then `content` chunks.
- [ ] `think` on a model that cannot think returns the model default (Ollama semantics), not an error.
- [ ] Backend without support → `400` naming the reason.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_protocols.py -q`
- [ ] Manual check: `curl /api/chat -d '{"think": true}'` on a reasoning GGUF.

**Dependencies:** P2-1
**Files likely touched:** `src/vipera/protocols/ollama.py`, `src/vipera/backends/server.py`, `tests/unit/test_protocols.py`
**Estimated scope:** M (3 files)

#### Task P2-3: `/api/show.thinking` metadata

**Description:** Expose `thinking: {values, default}` when the model/backend
supports it, and omit the key otherwise (Ollama's contract).

**Acceptance criteria:**
- [ ] Reasoning model → `values` non-empty, `default` present.
- [ ] Non-reasoning model → key absent, not `values: []`.
- [ ] `capabilities` includes `"thinking"` for those models.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_protocols.py -q`
- [ ] Manual check: `ollama.show` through the SDK.

**Dependencies:** P2-2
**Files likely touched:** `src/vipera/protocols/ollama.py`, `tests/unit/test_protocols.py`
**Estimated scope:** S (2 files)

#### Task P2-4: Real `capabilities`

**Description:** Replace the hardcoded `["completion"]` with a computed list:
`completion`, `tools` (always), `embedding` (embedding GGUF/config), `thinking`
(P2-3), `vision` (Phase 4 hook).

**Acceptance criteria:**
- [ ] An embedding-capable model reports `embedding`; a reasoning model reports `thinking`.
- [ ] `capabilities` never lies: a capability only appears if the request path works.
- [ ] `/api/tags` and `/api/show` agree.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_protocols.py -q`
- [ ] Manual check: Open WebUI/Continue show tool affordances (or at minimum, the JSON inspected).

**Dependencies:** P2-3
**Files likely touched:** `src/vipera/protocols/ollama.py`, `tests/unit/test_protocols.py`
**Estimated scope:** S (2 files)

### Checkpoint: after P2
- [ ] A qwen3/gpt-oss-class reasoning model answers with a separate `thinking` field, streamed correctly.
- [ ] `ollama.show` reports thinking levels and honest capabilities.
- [ ] Parity scoreboard: `chat` + `show` fully green.
- [ ] Full suite green; human review before Phase 3.

### Phase 3 — API surface (4–6 days)

#### Task P3-1: `num_ctx` and `options` honesty

**Description:** Map `options.num_ctx` to the model's load context, or reject it
with `400` when it would require a reload the gateway cannot do mid-request.
Also audit every `OllamaOptions` field against `_OPTION_MAP` so nothing else is
silently dropped.

**Acceptance criteria:**
- [ ] `num_ctx` either changes the effective context (documented path) or returns a 400 naming the supported path.
- [ ] A test enumerates every `OllamaOptions` field and asserts each is mapped or explicitly rejected.
- [ ] No field is silently ignored.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_protocols.py -q`
- [ ] Manual check: `curl` with `num_ctx` shows the documented behaviour.

**Dependencies:** P1-5
**Files likely touched:** `src/vipera/protocols/ollama.py`, `tests/unit/test_protocols.py`, `docs/http-api.md`
**Estimated scope:** S (3 files)

#### Task P3-2: Real `digest`

**Description:** Replace `sha256(alias)` with a content digest: hash the GGUF
once at import/create, store it in the registry, serve it from `/api/tags` and
`/api/ps`.

**Acceptance criteria:**
- [ ] Two aliases for the same file share one digest; two different files never collide.
- [ ] The digest is computed once (cached in the registry), not per request.
- [ ] `/api/tags` and `/api/ps` return the same digest for a loaded model.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_registry.py tests/unit/test_protocols.py -q`
- [ ] Manual check: `ollama.list()` digests match `sha256sum` of the file (with the `sha256:` prefix).

**Dependencies:** P1-1
**Files likely touched:** `src/vipera/registry.py`, `src/vipera/protocols/ollama.py`, `src/vipera/commands/model_cmd.py`, `tests/unit/test_registry.py`
**Estimated scope:** S (4 files)

#### Task P3-3: `/v1/responses` (stateless)

**Description:** Add the OpenAI Responses endpoint, translated onto the existing
chat path: `input` (string or message list), `output_text`, `stream` with the
documented event sequence. Stateless only, like Ollama's local server.

**Acceptance criteria:**
- [ ] `openai.responses.create(model=..., input="...")` returns `output_text`.
- [ ] Streaming emits `response.created`, output deltas, and a terminal event in the documented order.
- [ ] Invalid `input` shapes return 400 before any model load.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/integration/test_openai_sdk.py -q` (new file)
- [ ] Manual check: the SDK example from OpenAI docs, pointed at Vipera.

**Dependencies:** None (chat path already correct)
**Files likely touched:** `src/vipera/protocols/openai.py`, `src/vipera/protocols/__init__.py`, `tests/integration/test_openai_sdk.py` (new)
**Estimated scope:** M (3 files)

#### Task P3-4: `/api/delete`, `/api/copy`, `/api/blobs`

**Description:** Model lifecycle routes over the registry: delete an alias
(never the file unless `--purge` at the CLI), copy = P1-4, and `HEAD/GET
/api/blobs/:digest` for clients that probe blob existence.

**Acceptance criteria:**
- [ ] `ollama.delete("x")` removes the alias and leaves the GGUF on disk.
- [ ] `ollama.copy("a", "b")` matches the CLI behaviour.
- [ ] Blob routes answer 200/404 consistently with the registry digest.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/integration/test_ollama_sdk.py -q`
- [ ] Manual check: scoreboard shows delete/copy supported.

**Dependencies:** P1-4, P3-2
**Files likely touched:** `src/vipera/protocols/ollama.py`, `tests/unit/test_protocols.py`
**Estimated scope:** M (2 files)

#### Task P3-5: `pull` / `push` / `create` download story

**Description:** `/api/pull` streams progress for a model reference Vipera can
resolve (HuggingFace repo or `hf:` ref). `/api/push` has no upstream registry —
return a documented `501` with the reason instead of a 404.

**Acceptance criteria:**
- [ ] `ollama.pull` on a supported ref streams progress lines and installs the model.
- [ ] Unsupported ref → error naming the supported syntax.
- [ ] `/api/push` → 501 with an explanation, not a stack trace.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_protocols.py -q`
- [ ] Manual check: pull a small GGUF end to end.

**Dependencies:** P3-4
**Files likely touched:** `src/vipera/protocols/ollama.py`, `src/vipera/sources/hf.py`, `tests/unit/test_protocols.py`
**Estimated scope:** M (3 files)

#### Task P3-6: Anthropic `/v1/messages` (non-streaming) — conditional

**Description:** Only if the open question below says yes. Add an Anthropic
compatible `POST /v1/messages` translated onto the existing chat path,
non-streaming first, so a client that appends `/v1/messages` to the base URL
gets an answer. Streaming and `tool_use` blocks are a follow-up task once the
protocol shape is proven; until then they answer a documented 400.

**Acceptance criteria:**
- [ ] `client.messages.create(model=..., max_tokens=..., messages=[{"role":"user","content":"hi"}])` returns Anthropic's response shape (`content[0].text`, `stop_reason`, `usage`).
- [ ] `stream=True` and `tools` return 400 naming the follow-up task.
- [ ] System prompts, `stop_sequences`, and `max_tokens` map onto the chat parameters.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/integration/test_anthropic_sdk.py -q` (new file, `anthropic` as a dev dependency)
- [ ] Manual check: the official `anthropic` Python client pointed at Vipera.

**Dependencies:** P3-1
**Files likely touched:** `src/vipera/protocols/anthropic.py` (new), `src/vipera/protocols/__init__.py`, `tests/integration/test_anthropic_sdk.py` (new)
**Estimated scope:** M (3 files)

### Checkpoint: after P3
- [ ] OpenAI SDK Responses example works unchanged.
- [ ] Parity scoreboard: every Ollama SDK method is green or an explicit, documented 501.
- [ ] No request field is silently ignored anywhere on the Ollama layer (test-enforced).
- [ ] Human review before Phase 4.

### Phase 4 — Vision (5–8 days)

#### Task P4-1: `mmproj` discovery and planning

**Description:** Detect the projector (`mmproj*.gguf` next to the model, or an
explicit registry field), add `mmproj` to `Plan`, and emit `--mmproj` for the
server backend when supported.

**Acceptance criteria:**
- [ ] A model with a sibling `mmproj` plans with it; without one, nothing changes.
- [ ] `--mmproj` appears only after a `--help`/flag probe says the binary takes it.
- [ ] `plan()` failure modes documented in `Plan` note.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_planner.py tests/unit/test_server_backend.py -q`
- [ ] Manual check: `server_command` for a VL model.

**Dependencies:** None
**Files likely touched:** `src/vipera/planner.py`, `src/vipera/backends/server.py`, `src/vipera/gguf_analyzer.py`, `tests/unit/test_planner.py`
**Estimated scope:** M (4 files)

#### Task P4-2: OpenAI vision input

**Description:** `Message.content` becomes `str | list[ContentPart]` with
`text` and `image_url` parts; `data:` URLs only (no remote fetch → no SSRF);
size/count limits enforced before a model load.

**Acceptance criteria:**
- [ ] `content: [{type:"text"},{type:"image_url",image_url:{url:"data:..."}}]` is accepted and forwarded.
- [ ] Remote `http(s)` image URLs are rejected with a 400 explaining `data:` only.
- [ ] Oversized/malformed base64 → 400 before the engine is acquired (P0-style ordering).
- [ ] Non-vision model → 400 naming the model and the `mmproj` requirement.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_server.py -q`
- [ ] Manual check: a real prompt against a VL GGUF, answer refers to the image.

**Dependencies:** P4-1
**Files likely touched:** `src/vipera/protocols/openai.py`, `tests/unit/test_server.py`
**Estimated scope:** M (2 files)

#### Task P4-3: Ollama `images`

**Description:** Accept `images: [base64]` on `/api/chat` and map it onto the
same parts, removing the current hard 400. Keep a clear 400 for text-only models.

**Acceptance criteria:**
- [ ] `ollama.chat(..., messages=[{"role":"user","content":"...","images":[b64]}])` works.
- [ ] Text-only model → 400 with the reason (existing behaviour preserved).
- [ ] `/api/show.capabilities` includes `vision` for those models.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_protocols.py tests/integration/test_ollama_sdk.py -q`
- [ ] Manual check: same image through both `/api/chat` and `/v1/chat/completions`.

**Dependencies:** P4-2, P2-4
**Files likely touched:** `src/vipera/protocols/ollama.py`, `tests/unit/test_protocols.py`
**Estimated scope:** S (2 files)

#### Task P4-4: In-process vision, or an honest refusal

**Description:** Try `llama-cpp-python`'s chat handler for the detected
projector. If the installed version cannot do it, return a 400/501 that names
`backend: server` — with a test pinning that decision.

**Acceptance criteria:**
- [ ] Version probe is explicit; unsupported binding → documented error, not a traceback.
- [ ] Server backend remains the recommended path in docs.
- [ ] No silent image drop anywhere.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_engine.py -q`
- [ ] Manual check: run one VL prompt on each backend and record the result in `docs/benchmarks.md`.

**Dependencies:** P4-1, P4-2
**Files likely touched:** `src/vipera/backends/inprocess.py`, `src/vipera/backends/loading.py`, `tests/unit/test_engine.py`
**Estimated scope:** M (3 files)

### Checkpoint: after P4
- [ ] A VL GGUF answers about an image over both APIs, streaming and not.
- [ ] Text-only models still reject images with a clear message.
- [ ] Perf: vision adds no measurable cost to text requests (baseline from P0-3).
- [ ] Human review before Phase 5.

### Phase 5 — Hardware breadth (4–6 days)

#### Task P5-1: Multi-GPU hardware model

**Description:** `HardwareInfo` gains `gpus: list[GpuInfo]` (name, vram total/free,
backend). NVIDIA via `pynvml` (all devices), Apple Silicon as one unified device,
AMD
attempted via `amdsmi`/`rocm-smi` and skipped cleanly when absent.

**Acceptance criteria:**
- [ ] Two NVIDIA cards → two entries with correct VRAM.
- [ ] No GPU → empty list, planner falls back exactly as today.
- [ ] AMD path never crashes when the library is missing.
- [ ] `system doctor` prints every device.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_hardware.py tests/unit/test_system_cmd.py -q`
- [ ] Manual check: `vipera system doctor` on the reference machine.

**Dependencies:** None
**Files likely touched:** `src/vipera/hardware.py`, `src/vipera/commands/system_cmd.py`, `tests/unit/test_hardware.py`
**Estimated scope:** M (3 files)

#### Task P5-2: Per-GPU budget and tensor split

**Description:** Distribute layers across devices using a per-device VRAM budget
and emit `--tensor-split` / `--main-gpu` / `-sm layer` for the server backend,
keeping the single-GPU plan byte-identical to today's.

**Acceptance criteria:**
- [ ] Plan for a 2-GPU box splits layers and states the split in its note.
- [ ] Single-GPU plans are unchanged (golden test on `server_command` output).
- [ ] Uneven cards are respected (larger card gets more layers).

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_planner.py tests/unit/test_server_backend.py -q`
- [ ] Manual check: `vipera bench run` on the 2-GPU box if available.

**Dependencies:** P5-1
**Files likely touched:** `src/vipera/planner.py`, `src/vipera/backends/server.py`, `tests/unit/test_planner.py`
**Estimated scope:** M (3 files)

#### Task P5-3: `doctor` and docs for the new hardware paths

**Description:** Update `system doctor` output, `docs/architecture.md#planner`,
and `configuration.md` so the multi-GPU story is documented rather than implied.

**Acceptance criteria:**
- [ ] Docs describe the split rule and its limits (same-VRAM assumption dropped).
- [ ] Every new config/env knob is in the `configuration.md` table.
- [ ] No claim in docs without a test or a measured run behind it.

**Verification:**
- [ ] Manual check: every documented key exists in `DEFAULT_CONFIG`.
- [ ] Tests pass: full suite.

**Dependencies:** P5-2
**Files likely touched:** `docs/architecture.md`, `docs/configuration.md`, `src/vipera/commands/system_cmd.py`
**Estimated scope:** S (3 files)

### Checkpoint: after P5
- [ ] Two-GPU plan measured and documented; single-GPU numbers unchanged.
- [ ] `doctor` accurate on NVIDIA / Apple / no-GPU.
- [ ] Human review before Phase 6.

### Phase 6 — Lifecycle and packaging (3–4 days)

#### Task P6-1: `serve start --daemon`

**Description:** Detach the gateway (double fork or `--daemon` flag), write the
existing pid file, redirect logs to `LOGS_DIR`, and make `serve status/stop`
work on it unchanged.

**Acceptance criteria:**
- [ ] `serve start --daemon` returns to the shell, `/health` answers within 3s or the command fails.
- [ ] `serve status` reports the daemon; `serve stop` stops it and removes the pid file.
- [ ] A crashed daemon leaves a stale pid file that `status` detects (existing logic).

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_serve_cmd.py -q`
- [ ] Manual check: start, `curl /health`, stop, repeat twice.

**Dependencies:** None
**Files likely touched:** `src/vipera/commands/serve_cmd.py`, `tests/unit/test_serve_cmd.py`
**Estimated scope:** M (2 files)

#### Task P6-2: `vipera run <model>`

**Description:** One command that pulls if missing, starts (or reuses) a
gateway, and opens a chat REPL — the `ollama run` equivalent.

**Acceptance criteria:**
- [ ] `vipera run qwen2.5:7b` with the model missing offers to pull it, then chats.
- [ ] Ctrl-C exits the REPL without killing a shared daemon.
- [ ] Works against an already-running gateway (no double load).

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/e2e/test_cli.py -q`
- [ ] Manual check: full journey on a clean `VIPERA_HOME`.

**Dependencies:** P6-1, P3-5
**Files likely touched:** `src/vipera/commands/chat_cmd.py`, `src/vipera/cli.py`, `tests/e2e/test_cli.py`
**Estimated scope:** M (3 files)

#### Task P6-3: systemd/launchd unit generator

**Description:** `vipera serve install-service` writes a user unit (systemd) or
LaunchAgent plist, prints the enable command, and refuses to overwrite without
`--force`.

**Acceptance criteria:**
- [ ] Generated unit runs the same config/`VIPERA_HOME` as the shell.
- [ ] `--dry-run` prints the unit and writes nothing.
- [ ] Unsupported platform → clear message, exit 1.

**Verification:**
- [ ] Tests pass: `.venv/bin/python -m pytest tests/unit/test_serve_cmd.py -q`
- [ ] Manual check: install, `systemctl --user start`, `/health`, uninstall.

**Dependencies:** P6-1
**Files likely touched:** `src/vipera/commands/serve_cmd.py`, `docs/install.md`
**Estimated scope:** M (2 files)

### Checkpoint: after P6
- [ ] `vipera run` journey works from a clean machine.
- [ ] Daemon survives a shell exit and is inspectable via `status`.
- [ ] Linux + macOS units generated and manually verified where available.

### Phase 7 — Positioning and release (1–2 days)

#### Task P7-1: `docs/ollama-compat.md`

**Description:** One page: every Ollama route/method Vipera serves, every one it
deliberately does not (with the reason), and the known behavioural differences
(template application, digest timing, push).

**Acceptance criteria:**
- [ ] Table generated from a test (route list) so it cannot drift silently.
- [ ] Every "not supported" row links to the 501/400 that enforces it.
- [ ] Linked from README and `comparison.md`.

**Verification:**
- [ ] Tests pass: a test asserts the route list matches the doc's first column.
- [ ] Manual check: click every link.

**Dependencies:** Phase 3, Phase 4
**Files likely touched:** `docs/ollama-compat.md` (new), `tests/unit/test_protocols.py`
**Estimated scope:** S (2 files)

#### Task P7-2: Rewrite `comparison.md` against the finished state

**Description:** Second pass after parity: honest table with the remaining real
gaps (no cloud models, no registry, no desktop app, no MLX) and the rows where
Vipera wins, each with a source path.

**Acceptance criteria:**
- [ ] Every ✅ in the table has a file or a measured run behind it.
- [ ] Every ❌ names the ticket/phases that would close it, or says "by design".
- [ ] Reviewer can reproduce one claim per row.

**Verification:**
- [ ] Manual check: spot-check five rows against the code.
- [ ] Tests pass: full suite.

**Dependencies:** P7-1
**Files likely touched:** `docs/comparison.md`, `README.md`
**Estimated scope:** S (2 files)

#### Task P7-3: Release gate

**Description:** Bump the version, write the CHANGELOG entry, and run the
release checklist (parity scoreboard %, benchmarks, docs build, `pip install`
from sdist in a clean venv).

**Acceptance criteria:**
- [ ] Parity scoreboard ≥ 90% of the SDK surface, with every miss documented.
- [ ] `pip install` from a built sdist works in a fresh venv and serves `/health`.
- [ ] CHANGELOG states what changed for Ollama users specifically.

**Verification:**
- [ ] Manual check: clean-venv install + `vipera run` smoke test.
- [ ] Tests pass: full suite, `ruff`, `mypy`.

**Dependencies:** All phases
**Files likely touched:** `pyproject.toml`, `CHANGELOG.md`
**Estimated scope:** S (2 files)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `--reasoning-format` / `--chat-template-kwargs` differ per llama.cpp build | High (P2 blocks) | P2-1 probes first and records findings; `think` degrades to a documented 400 |
| Vision needs `mmproj` that users do not have next to the GGUF | Medium | Discovery + explicit registry field + clear error naming the file |
| Modelfile `TEMPLATE` semantics (Go template) cannot be honoured per request | Medium | Store+surface always, apply only where a backend flag exists, warn otherwise (Decision 4) |
| No AMD or 2-GPU hardware to test P5 | High (untested paths) | Keep the code table-driven with fakes; mark the AMD path experimental in docs; do not claim it works |
| Phase 3 scope creep (Anthropic compatibility, push) | Medium | Anthropic `/v1/messages` is an open question (below); push is 501 by decision |
| Registry migration breaks existing installs | High | P1-1 tests 1.x/v2 fixtures; write-then-rename already in place |
| Perf regression from vision/thinking in the hot path | Medium | Baseline in P0-3, re-measure at each checkpoint |
| 35 tasks becomes an unbounded project | High | Hard target is the parity scoreboard + Phases 0–3; Phases 4–6 can ship separately |

## Open Questions

- Does "catching up" mean **80% of real usage** (Phases 0–3 + 1.0 release) or **API-complete** (includes Anthropic `/v1/messages` and `/api/push`)?
- Anthropic-compatible `/v1/messages`: in or out? Ollama has it; it is a second protocol layer to maintain forever.
- Is there AMD or multi-GPU hardware available to validate Phase 5? Without it, P5 ships as "tested against fakes only" and must say so.
- Should `serve start` default to daemon mode (matching Ollama's always-on daemon) or keep foreground and require `--daemon`?
- Do we want a Vipera-hosted registry for `push` at all, or is 501 the permanent answer?
- Version: `1.0.0` at end of Phase 4, or stay `0.x` until vision is measured on real hardware?
