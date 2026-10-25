# Ollama lifecycle parity — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the first, self-contained parity gap with Ollama: make `keep_alive` real, make `/api/ps` report a real expiry, and add `/api/show` so Ollama clients can inspect a model.

**Architecture:** `ModelPool` owns an idle deadline per loaded model and a small reaper thread. The gateway exposes two optional callbacks (`keep_alive`, `expiries`) so both protocol layers stay independent of the pool. The Ollama layer parses the request's `keep_alive`, asks the gateway to apply it, reads expiries for `/api/ps`, and serves `/api/show` from the registry metadata.

**Tech Stack:** Python 3.11+, FastAPI, pytest, existing `ModelPool` / `Gateway` / Ollama protocol.

**Spec:** the parity analysis in the session that produced this plan (five gaps: multimodal, lifecycle, model management, distribution, API maturity). This plan implements only the lifecycle + `/api/show` slice.

## Global Constraints

- No new runtime dependency.
- Single-model mode must keep working unchanged; lifecycle is a pool-mode feature.
- `keep_alive` values follow Ollama: duration string (`5m`, `1h`), seconds number, `0` = unload after the request, negative = never unload.
- Do not change the default OpenAI protocol mount.
- All tests, `ruff check`, `ruff format --check`, and `mypy src/vipera` must stay green.

## Review Focus

- A request with an invalid `keep_alive` must not load a model first.
- A model that is mid-generation must never be reaped.
- `keep_alive: 0` must unload after the response, not before it.
- `keep_alive` omitted must use the configured default, not "forever".
- `/api/show` for a valid alias with no registry metadata must still return a parseable response.

---

### Task 1: Duration parser

**Files:**
- Create: `src/vipera/keepalive.py`
- Test: `tests/unit/test_keepalive.py`

**Interfaces:**
- Produces: `parse_keep_alive(value: str | int | float | None, default: float | None = None) -> float | None`.
  `None` means "use the caller's default", negative means "never expire", `0` means "expire now".

- [ ] **Step 1: Write the failing tests**

```python
def test_keep_alive_defaults_to_the_caller_value():
    assert parse_keep_alive(None, default=300) == 300


def test_keep_alive_parses_ollama_durations():
    assert parse_keep_alive("5m") == 300
    assert parse_keep_alive("1.5h") == 5400
    assert parse_keep_alive("500ms") == 0.5
    assert parse_keep_alive(90) == 90


def test_keep_alive_zero_and_negative_are_special():
    assert parse_keep_alive("0") == 0
    assert parse_keep_alive(-1) == -1


def test_keep_alive_rejects_garbage():
    with pytest.raises(ValueError):
        parse_keep_alive("soon")
```

- [ ] **Step 2: Run tests; expect ImportError / failure**
- [ ] **Step 3: Implement `parse_keep_alive`** with a single regex and unit multipliers `ms=0.001, s=1, m=60, h=3600, d=86400`.
- [ ] **Step 4: Run tests; expect pass**
- [ ] **Step 5: Commit** `feat: parse Ollama keep_alive durations`

---

### Task 2: Pool TTL and reaper

**Files:**
- Modify: `src/vipera/pool.py`
- Test: `tests/unit/test_pool.py`

**Interfaces:**
- `ModelPool.__init__(..., max_loaded=2, keep_alive: float | None = 300.0, reap_interval: float = 1.0)`
- `ModelPool.keep_alive(engine, seconds: float | None) -> None` — `None` applies the pool default, negative means forever.
- `ModelPool.expire(now: float | None = None) -> list[str]` — close idle engines whose deadline passed; returns aliases.
- `ModelPool.expiries() -> dict[str, str | None]` — alias -> ISO timestamp, `None` for forever.
- `ModelPool.close()` stops the reaper.

- [ ] **Step 1: Write failing tests** for: release + `expire(now + 31)` closes; in-use engine survives; `keep_alive=0` closes on release; negative never expires.
- [ ] **Step 2: Run; expect failure**
- [ ] **Step 3: Implement** `_expires_at`, `_pending_ttl`, `expire()`, `expiries()`, `_reap_loop`; clear deadlines on acquire and eviction; close outside the condition lock.
- [ ] **Step 4: Run pool tests; expect pass**
- [ ] **Step 5: Commit** `feat(pool): unload idle models after keep_alive`

---

### Task 3: Gateway + `/api/ps`

**Files:**
- Modify: `src/vipera/protocols/gateway.py`, `src/vipera/server.py`, `src/vipera/protocols/ollama.py`
- Test: `tests/unit/test_protocols.py`

**Interfaces:**
- `Gateway.keep_alive: Callable[[Any, float | None], None] = _noop_keep_alive`
- `Gateway.expiries: Callable[[], dict[str, str | None]] = _no_expiries`
- `/api/ps` uses `gateway.expiries()`; unknown expiry stays `9999-12-31T23:59:59Z`.

- [ ] **Step 1: Write failing test** that builds a pool-backed gateway, calls `/api/chat` with `keep_alive: "5m"`, then asserts `/api/ps` reports a finite `expires_at`.
- [ ] **Step 2: Run; expect failure**
- [ ] **Step 3: Wire the callbacks** and update the route.
- [ ] **Step 4: Run; expect pass**
- [ ] **Step 5: Commit** `feat(ollama): expose real model expiry in /api/ps`

---

### Task 4: `/api/show`

**Files:**
- Modify: `src/vipera/protocols/ollama.py`
- Test: `tests/unit/test_protocols.py`, `tests/integration/test_ollama_sdk.py`

**Interfaces:**
- `POST /api/show` body `{"model": "<alias>"}`.
- Response includes `details`, `model_info`, `capabilities`, `modified_at`, `template`, `parameters`, `license`.
- Unknown alias returns the existing `{"error": ...}` 404 shape.

- [ ] **Step 1: Write failing unit tests** for known and unknown models.
- [ ] **Step 2: Run; expect 404 for the route**
- [ ] **Step 3: Implement** the route and a shared `_details` helper.
- [ ] **Step 4: Run; expect pass**, then add the SDK `client.show(...)` test.
- [ ] **Step 5: Commit** `feat(ollama): add /api/show`

---

### Task 5: Config, CLI, docs, full verification

**Files:**
- Modify: `src/vipera/config.py`, `src/vipera/commands/serve_cmd.py`, `docs/http-api.md`, `docs/configuration.md`, `CHANGELOG.md`
- Test: `tests/unit/test_config.py`, `tests/unit/test_serve_cmd.py`

- [ ] **Step 1: Add `keep_alive: "5m"` to `DEFAULT_CONFIG`** and `--keep-alive` to `serve start`; pass the parsed value to `ModelPool`.
- [ ] **Step 2: Add a config/CLI test** that an invalid value exits before loading a model.
- [ ] **Step 3: Document** the three behaviours and `/api/show`.
- [ ] **Step 4: Run** `pytest -q`, `ruff check src tests`, `ruff format --check src tests`, `mypy src/vipera`.
- [ ] **Step 5: Commit** `docs: document Ollama lifecycle parity`

## Self-review

- Spec coverage: lifecycle (Tasks 1–3), API maturity (`/api/show`, Task 4), config/docs (Task 5).
- Deliberately out of scope: multimodal, model management CRUD, runner distribution, thinking.
- Type consistency: `keep_alive` returns seconds or `None`; pool accepts the same; ISO expiry is a string or `None`.
