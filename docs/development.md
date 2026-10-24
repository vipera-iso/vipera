---
title: Development
description: Dev setup, tests, lint and type checks, CI, and the release process.
---

# Development

Vipera targets Python 3.11 and 3.12, uses a `src/` layout, `hatchling` for
builds, `ruff` for lint/format, and `pytest` for tests.

## Setup

```bash
git clone https://github.com/vipera-iso/vipera
cd vipera

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev,llama,gguf]"
pre-commit install
```

On an NVIDIA host, add `,nvidia` to the extras to enable VRAM detection
(`pip install -e ".[dev,llama,gguf,nvidia]"`).

The extras:

| Extra | Provides |
|---|---|
| `dev` | `pytest`, `pytest-cov`, `httpx`, `ruff`, `mypy`, `pre-commit` |
| `llama` | `llama-cpp-python` — required to actually load a model |
| `gguf` | `gguf` — required for layer/parameter/context metadata |
| `nvidia` | `nvidia-ml-py` (the `pynvml` module) — enables NVIDIA VRAM detection (pair with a CUDA build for offload) |

Tests run without either optional extra; runtime commands that load a model do
not.

**One venv per job.** A CUDA build of `llama-cpp-python` and a plain CPU build
can sit side by side on the same machine, and nothing warns you when you bench
the wrong one — a CPU-only binding silently ignores `gpu_layers`, so a GPU
sweep reported through it is a CPU measurement with a GPU-shaped label. Pick
one virtualenv for serving and one for tests, and confirm which is which
before quoting a number:

```bash
vipera system doctor | grep 'gpu offload'
# serving env:  gpu offload  compiled in (CUDA/Metal)  [ PASS ]
# test env:     gpu offload  CPU-only build            [ WARN ]
```

`tools/sweep_gpu.py` refuses to run in a binding without a GPU backend for the
same reason (pass `--allow-cpu` only when the CPU fallback is what you meant to
measure). Every throughput table should name the environment it came from.

Verify the install:

```bash
vipera --version          # vipera 1.1.0
vipera help
vipera system doctor
```

## Tests

```bash
pytest -q
pytest -q --cov=vipera
```

Test layout:

| File | Covers |
|---|---|
| `tests/test_planner.py` | CPU-only, full-GPU, hybrid, per-layer VRAM budgeting, and low-RAM planning. Uses a synthetic `HardwareInfo`. |
| `tests/test_engine.py` | Staged GPU→CPU fallback using a fake `llama_cpp.Llama`. |
| `tests/test_server.py` | `/health`, `/v1/models`, non-streaming and SSE completion against a fake engine. |
| `tests/test_config.py` | Defaults, save/reload, type coercion, corrupt-file fallback. |
| `tests/test_registry.py` | Add/get/list/remove. |
| `tests/integration/test_ollama_sdk.py` | The `/api/*` layer over a real socket, driven by the official `ollama` client (installed by `.[dev]`). Starts uvicorn on a loopback port with the stub engine; no model and no Ollama daemon. |

Rules:

- Tests must not need a `.gguf` file. Anything that requires a real model is
  marked `@pytest.mark.slow` (registered in `pyproject.toml`).
- Isolate storage with `tmp_path` and monkeypatch the path constants, the way
  the existing config/registry fixtures do. Just setting `VIPERA_HOME` after
  import is not enough.
- Server tests use `fastapi.testclient.TestClient` with a stub engine exposing
  `chat(messages, stream=False, **kwargs)`; keep the stub contract in sync with
  `Engine.chat`. The Ollama SDK test reuses that stub but over a real HTTP
  connection, so a framing or error-shape change breaks it too.
- `--strict-markers` is enabled, so a marker that is not registered is an
  error.

## Lint, format, types

```bash
ruff check .
ruff check --fix .
ruff format .
ruff format --check .
mypy
```

Configuration lives in `pyproject.toml`:

- line length 100; rules `E, W, F, I, B, UP, SIM, RUF`; `E501` and `B008`
  ignored (Typer's `Option(...)` defaults legitimately appear in signatures).
- `tests/*` additionally ignores `B011`.
- mypy checks only `src/vipera`, with missing imports ignored, unused ignores
  warned, and redundant casts warned.

Pre-commit runs whitespace/EOF/YAML/TOML/JSON/merge-conflict/large-file checks,
`ruff --fix`, and `ruff-format`. Markdown is exempt from trailing-whitespace
cleanup and should keep hard line breaks where intended.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`:

- **lint**: Ubuntu, Python 3.12, `pip install ruff`, `ruff check .`.
- **test**: matrix of Ubuntu/macOS/Windows × Python 3.11/3.12, `pip install -e
  ".[dev]"`, `pytest -q`, then a CLI smoke test (`vipera --version`,
  `vipera help`).

`fail-fast: false` keeps the full matrix informative. CI does not install
`llama-cpp-python` or `gguf`, so neither may be required by the test suite.
Neither mypy nor `ruff format --check` runs in CI; run them locally if you touch
typed code.

## Release

Maintainers:

1. Land everything for the release on `main`.
2. Update `CHANGELOG.md` under a new version heading.
3. Bump `version` in **both** `pyproject.toml` and `src/vipera/__init__.py`.
4. Commit as `chore: release vX.Y.Z`, tag `vX.Y.Z`.
5. Push the branch and the tag.

`.github/workflows/release.yml` builds the sdist/wheel with `python -m build` on
the tag and publishes with the `PYPI_API_TOKEN` repository secret when it is
set, falling back to PyPI trusted publishing (`pypi` environment,
`id-token: write`) when the secret is empty. The secret is the durable path:
it keeps working even when the OIDC publisher is missing or misconfigured, and
it is never written to the repository. Rotate it on PyPI (Account settings →
API tokens → revoke, then add a new project-scoped token) and update the
secret with `gh secret set PYPI_API_TOKEN`.

The distribution name is **`vipera-ai`**; the import package and CLI remain
`vipera`. The name `vipera` on PyPI belongs to an unrelated project, so it must
not be used in install commands or publishing configuration.

### PyPI trusted publisher (one-time, per repository)

The workflow cannot publish until PyPI knows this repository. `vipera-ai` 1.2.0
was uploaded with a project-scoped API token, so the project now exists and the
publisher to add is a **normal** one (a *pending* publisher, added under the
account's Publishing page, is only for a project that does not exist yet — and
PyPI invalidates it if someone else registers the name first).

1. Sign in to PyPI, open the `vipera-ai` project → **Publishing** →
   **Add a new publisher**.
2. Fill in exactly these values — every one is part of the OIDC claim set:

   | Field | Value |
   |---|---|
   | Owner | `vipera-iso` |
   | Repository name | `vipera` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The project name is fixed by the page you are on; it is not a field here.

3. Push a new tag. Do not re-run the 1.2.0 job: that version is already on PyPI,
   so its upload would be rejected as an existing file. No secret is involved:
   the job's `id-token: write` permission is exchanged for a short-lived upload
   token.

The failure this fixes is exact: `##[error]Trusted publishing exchange failure`
followed by `invalid-publisher: valid token, but no corresponding publisher`.
The action prints the claims it presented (`sub`, `workflow_ref`,
`environment`) right below that error — compare them against the table above
before changing anything. A `403 File already exists` is a different problem:
the version was already uploaded, so bump it instead of re-running.

After publishing, create the GitHub release:

```bash
gh release create vX.Y.Z --notes-from-tag
```

## Demo GIF

The README embeds `docs/assets/demo.gif`, recorded from
`docs/assets/demo.tape` with [VHS](https://github.com/charmbracelet/vhs). The
tape opens with a `figlet -f block` wordmark, so `figlet` has to be installed
for the first frame to render.

```bash
# Arch
sudo pacman -S vhs ffmpeg figlet gifsicle ttf-jetbrains-mono-nerd
fc-cache -f

# Debian/Ubuntu (vhs itself needs the Charm apt repository)
sudo apt install figlet

# macOS
brew install vhs ffmpeg figlet gifsicle
brew install --cask font-jetbrains-mono-nerd-font

make demo
```

`make demo` runs the tape twice over: `vhs` records `docs/assets/demo-raw.mp4` at
1800x1200 with a 32px font, then `ffmpeg` downscales it to 900x600 with lanczos
and a 128-colour palette. Recording directly at 900x600 works but leaves grey
fringes on every glyph; rendering at 2x and resampling is what makes the text
crisp when the README shows it at 720px. Running `vhs docs/assets/demo.tape` by
hand only produces the intermediate MP4. If a recording runs long enough that
the GIF would pass 950 KiB, the target silently re-encodes at 96 colours and
fails only if it still exceeds 1000 KB.

Recording needs a model registered as `qwen2.5-0.5b-instruct-q4_k_m`;
`vipera serve start` in the tape fails without one, and the GIF then stops
before the request. Target: under 17 seconds, under 1 MB, terminal width
around 110 columns.

There is no CI job that re-records the GIF. `.github/workflows/vhs.yml` still
lists the pre-move paths and has no model to load, so treat recording as a
local, manual step.

## Changing the docs

Documentation pages live in `docs/` and are linked from the index at
[docs/README.md](README.md).

- Change behavior → update the page that specifies it in the same PR.
- Runtime claims must come from the code. If you document a flag, parameter,
  default, or error path, point at where it is implemented.
- When you discover that the README and the code disagree, fix the README or
  add the gap to [architecture.md#known-gaps](architecture.md#known-gaps); do
  not document the intended behavior as if it already shipped.
- Keep commands copy-pasteable and paths repository-relative.
