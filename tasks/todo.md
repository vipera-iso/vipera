# Ollama parity — task list

Detailed task specs (acceptance criteria, files, verification): [plan.md](plan.md).

**Rule of the list:** each task leaves `pytest` + `ruff` + `mypy` green. A task is
done only when its acceptance criteria in `plan.md` are checked *and* the parity
scoreboard below moved (or the task says why it does not move it).

## Parity scoreboard (update after every task; P0-1 owns the numbers)

| SDK method | Status | Owning task |
|---|---|---|
| `chat` | ✅ works | — |
| `generate` | ✅ works | — |
| `embed` / `embeddings` | ✅ works (config `embeddings true`) | — |
| `list` | ✅ works | — |
| `ps` | ✅ works | — |
| `show` | ⚠️ partial: no `system` / `thinking` / real capabilities | P1-8, P2-3, P2-4 |
| `create` | ❌ | P1-3, P1-8 |
| `copy` | ❌ | P1-4, P3-4 |
| `delete` | ❌ | P3-4 |
| `pull` | ❌ | P3-5 |
| `push` | ❌ (planned: documented 501) | P3-5 |
| `think` / `message.thinking` | ❌ | P2-2 |
| vision (`images`, `image_url`) | ❌ (400 today) | P4-2, P4-3 |
| `/v1/responses` | ❌ | P3-3 |

Score today: **5 / 14 fully green** (chat, generate, embed, list, ps), 1 partial
(`show`), 8 missing. Target: **≥ 13 / 14** with every miss documented.

Measured by the suite: **9 passed, 8 xfailed** (2026-10-25). Each `xfail` names
its owning task in `reason=`, so adding a feature is expected to flip exactly
one test to green.

---

## Phase 0 — Baseline and truth

- [x] **P0-1** Rewrite `tests/integration/test_ollama_sdk.py` as a parity suite: every SDK method exercised, unsupported ones `xfail` with the owning task in `reason=`. Fill in the scoreboard above.
  - Verify: `pytest tests/integration/test_ollama_sdk.py -q` passes; counts quoted here. → **9 passed, 8 xfailed**.
- [x] **P0-2** Fix docs that name non-existent files and false gaps: `comparison.md` (multi-model routing, MoE flags), `engine.py` links, README `/health` body.
  - Verify: `grep -rn "engine.py" docs README.md` → empty; `/health` example includes `engines` + 503.
  - Done: module tables point at `backends/` and `pool.py`, the multi-model and
    MoE rows are corrected, and the README `/health` row shows `engines` + 503.
    `engine.py` survives only in `docs/design/` as a historical decision record,
    plus the test filename it names.
- [ ] **P0-3** Record baseline: `vipera bench run` decode/prefill + gateway overhead share + cold start, in `docs/benchmarks.md` with hardware and date.
  - Verify: one documented command reproduces the numbers.

### Checkpoint P0
- [ ] Parity suite green and reporting a scoreboard.
- [ ] Docs no longer state false gaps.
- [ ] `pytest`, `ruff`, `mypy src/vipera` clean.
- [ ] Human reviewed the scoreboard.

## Phase 1 — Model identity and Modelfile

- [ ] **P1-1** Registry schema v3: `system`, `template`, `parameters`, `license`, `digest`, `parent`; read 1.x/v2, write v3.
- [ ] **P1-2** `src/vipera/modelfile.py`: parse `FROM/SYSTEM/TEMPLATE/PARAMETER/ADAPTER/LICENSE`, line-numbered errors, multi-line blocks.
- [ ] **P1-3** `vipera model create -f Modelfile -n alias [--dry-run] [--force]` in a new command module (do not grow `model_cmd.py`).
- [ ] **P1-4** `vipera model cp <src> <dst>`.
- [ ] **P1-5** Load-time overrides from `parameters` (`num_ctx`, `kv_cache`, `n_batch`, `n_ubatch`), with a warning on invalid values and provenance recorded on `Plan`.
- [ ] **P1-6** Request-time defaults from `parameters` (`temperature`, `top_p`, `top_k`, `repeat_penalty`, `stop`), client wins.
- [ ] **P1-7** `SYSTEM` injected only when the client sends no system message; shaping/prefix accounting must not double count.
- [ ] **P1-8** `POST /api/create` over P1-3; `/api/show` returns `system`, `template`, `parameters`, `license`.
- [ ] **P1-9** `TEMPLATE`: store + surface always, apply via `--chat-template` where supported, one startup warning where not. Never silent.

### Checkpoint P1
- [ ] A real Ollama Modelfile (SYSTEM + PARAMETERs) produces a working Vipera model.
- [ ] Scoreboard: `create` ✅, `show` ✅.
- [ ] `vipera model create` documented in `docs/cli.md`.
- [ ] Full suite + `ruff` + `mypy` clean; human review.

## Phase 2 — Reasoning and capabilities

- [ ] **P2-1** Probe the installed build for `--reasoning-format` / `--chat-template-kwargs` (same `--help` pattern), record findings in `docs/design/llama-server-backend.md`.
- [ ] **P2-2** `think: bool|str|None` on `/api/chat` + `/api/generate`; `message.thinking` / `thinking`, interleaved when streaming; unsupported backend → 400 with the reason.
- [ ] **P2-3** `/api/show.thinking = {values, default}` when supported, key absent otherwise.
- [ ] **P2-4** Compute `capabilities` (`completion`, `tools`, `embedding`, `thinking`, `vision`) instead of hardcoded `["completion"]`; `/tags` and `/show` agree.

### Checkpoint P2
- [ ] A reasoning GGUF returns a separate, correctly streamed `thinking` field.
- [ ] Scoreboard: `think` ✅, `show` ✅.
- [ ] Human review.

## Phase 3 — API surface

- [ ] **P3-1** `options.num_ctx` mapped or explicitly 400'd; a test enumerates every `OllamaOptions` field and proves none is silently dropped.
- [ ] **P3-2** Content `digest` computed once at import/create, cached in the registry, served by `/tags` and `/ps`.
- [ ] **P3-3** `/v1/responses` (stateless): `input`, `output_text`, streaming event order; validation before load.
- [ ] **P3-4** `/api/delete`, `/api/copy`, `/api/blobs/:digest`.
- [ ] **P3-5** `/api/pull` for resolvable refs with progress; `/api/push` → documented 501.
- [ ] **P3-6** *Conditional on the open question:* Anthropic `/v1/messages`, non-streaming only; `tools` and `stream` return a 400 pointing at the follow-up task.

### Checkpoint P3
- [ ] OpenAI `client.responses.create` example works unchanged.
- [ ] Scoreboard: every method ✅ or an explicit documented 501.
- [ ] `docs/ollama-compat.md` drafted with the enforced list.
- [ ] Human review.

## Phase 4 — Vision

- [ ] **P4-1** `mmproj` discovery (sibling file or registry field) → `Plan.mmproj` → `--mmproj` after a flag probe.
- [ ] **P4-2** OpenAI `content` parts: text + `data:` image URLs only (no remote fetch), limits enforced before engine acquisition.
- [ ] **P4-3** Ollama `images` accepted and mapped; text-only models keep a clear 400; `capabilities` gains `vision`.
- [ ] **P4-4** In-process vision via a probed chat handler, or an explicit 400 pointing at `backend: server`.

### Checkpoint P4
- [ ] A VL GGUF answers about an image over both APIs, streaming and not.
- [ ] Text request latency unchanged vs P0-3 baseline.
- [ ] Human review.

## Phase 5 — Hardware breadth

- [ ] **P5-1** `HardwareInfo.gpus: list[GpuInfo]` (all NVIDIA devices, Apple unified, AMD best-effort without crashing when absent); `doctor` prints every device.
- [ ] **P5-2** Per-GPU budget + `--tensor-split` / `--main-gpu` / `-sm layer`; single-GPU `server_command` output unchanged (golden test).
- [ ] **P5-3** Document the split rule and its limits in `architecture.md` + `configuration.md`.

### Checkpoint P5
- [ ] 2-GPU plan measured and documented (or explicitly marked untested).
- [ ] `doctor` accurate on NVIDIA / Apple / no-GPU.
- [ ] Human review.

## Phase 6 — Lifecycle and packaging

- [ ] **P6-1** `serve start --daemon` (detach, log file, pid file, `status`/`stop` unchanged).
- [ ] **P6-2** `vipera run <model>`: pull if missing, reuse or start a gateway, REPL.
- [ ] **P6-3** `serve install-service` generating systemd user unit / launchd plist, `--dry-run`, no overwrite without `--force`.

### Checkpoint P6
- [ ] `vipera run` works from a clean `VIPERA_HOME`.
- [ ] Daemon survives shell exit; `status`/`stop` correct.
- [ ] Units generated and manually verified where the platform exists.

## Phase 7 — Positioning and release

- [ ] **P7-1** `docs/ollama-compat.md`, route table generated from a test so it cannot drift.
- [ ] **P7-2** Rewrite `comparison.md` against the finished state; every ✅ backed by a file or a measured run.
- [ ] **P7-3** Release gate: scoreboard ≥ 90%, clean-venv install from sdist serves `/health`, CHANGELOG entry for Ollama users.

### Checkpoint P7 — Done
- [ ] Scoreboard ≥ 13/14 with documented misses.
- [ ] `docs/ollama-compat.md` + `comparison.md` reviewed by a human.
- [ ] Clean-venv `pip install` + `vipera run` smoke test recorded.
- [ ] `pytest`, `ruff`, `mypy` clean.

---

## Start here (first three, in order)

1. **P0-1** — the parity suite is the measuring stick; without it "caught up" is an opinion.
2. **P0-2** — two hours to stop the docs lying about the product.
3. **P1-1** — registry v3 unblocks six other tasks; everything else in Phase 1 waits on it.

Parallel-safe after P0: **P1-2** (parser) beside **P0-3** (benchmarks); **P2-1**
(binary probe) beside **P1-1**; **P3-3** (`/v1/responses`) beside Phase 2 —
it touches only the OpenAI layer.

Must stay sequential: registry migrations (P1-1), per-model overrides (P1-5 →
P1-6), and anything touching `protocols/ollama.py` (`_messages`, `_chat_kwargs`,
`capabilities`) — that file is already 755 lines and merged by hand.
