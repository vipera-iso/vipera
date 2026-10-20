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

Rules:

- Tests must not need a `.gguf` file. Anything that requires a real model is
  marked `@pytest.mark.slow` (registered in `pyproject.toml`).
- Isolate storage with `tmp_path` and monkeypatch the path constants, the way
  the existing config/registry fixtures do. Just setting `VIPERA_HOME` after
  import is not enough.
- Server tests use `fastapi.testclient.TestClient` with a stub engine exposing
  `chat(messages, stream=False, **kwargs)`; keep the stub contract in sync with
  `Engine.chat`.
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
the tag and publishes through PyPI trusted publishing (`pypi` environment,
`id-token: write`). No API token is stored in the repository.

The distribution name is **`vipera-ai`**; the import package and CLI remain
`vipera`. The name `vipera` on PyPI belongs to an unrelated project, so it must
not be used in install commands or publishing configuration.

After publishing, create the GitHub release:

```bash
gh release create vX.Y.Z --notes-from-tag
```

## Demo GIF

The README embeds `docs/images/demo.gif`, recorded from `docs/demo.tape` with
[VHS](https://github.com/charmbracelet/vhs).

```bash
# Arch
sudo pacman -S vhs ffmpeg gifsicle ttf-jetbrains-mono-nerd
fc-cache -f

# macOS
brew install vhs ffmpeg gifsicle
brew install --cask font-jetbrains-mono-nerd-font

vhs docs/demo.tape
```

Target: under 15 seconds, under 3 MB, terminal width around 110 columns.
`.github/workflows/vhs.yml` re-records and commits the GIF when `docs/demo.tape`
or `src/**` changes on `main`, compressing with `gifsicle` when it exceeds
3.5 MB.

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
