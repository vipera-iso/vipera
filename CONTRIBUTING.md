# Contributing to Vipera

Thanks for taking the time to contribute.

## Development setup

Vipera requires **Python 3.11+**. `src/` layout, `hatchling` build backend,
`ruff` for lint + format, `pytest` for tests, `pre-commit` for local checks.

```bash
git clone https://github.com/vipera-iso/vipera
cd vipera
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e ".[dev,llama,gguf]"
pre-commit install
```

Verify:

```bash
vipera --version           # vipera 1.0.0
vipera help
vipera system doctor
```

## Running tests

```bash
pytest -q
pytest -q --cov=vipera
ruff check .
ruff format --check .
```

Tests must not require a `.gguf` file. Anything that needs a real model goes
behind `pytest.mark.slow`.

## Code style

- Line length 100 (`ruff` enforces).
- Sorted imports (`ruff` isort-compatible).
- Types added; `py.typed` ships.
- Docstrings on every non-trivial module.
- Comments explain *why*, not *what*.

## Commit messages

Conventional Commits:

```
feat: add /v1/completions endpoint
fix(server): flush SSE chunks before generator exit
docs: describe the planner fallback path
refactor(engine): move KV lookup into _load
test: cover mlock retry
chore: bump ruff to 0.6
```

One logical change per commit.

## Pull requests

1. Fork, create a topic branch off `main`.
2. Make the change. Add or update tests.
3. Run `pre-commit run --all-files` and `pytest -q`.
4. Open the PR against `main`.
5. CI must pass. One maintainer approval required.

**Merged quickly:** reproduction case, failing-then-passing test,
before/after `vipera bench run` table.

**Sent back:** drive-by reformatting mixed with a functional change, new
dependencies without justification, changes to the planner without a
hardware profile.

## Reporting bugs

Open an issue with the
[bug report template](.github/ISSUE_TEMPLATE/bug_report.yml). Include:

- `vipera --version`
- `vipera system doctor` output
- The exact command and its full output
- `uname -a` / `systeminfo` if it's a load failure

## Recording the demo GIF

The README embeds `docs/assets/demo.gif`. To regenerate:

```bash
# Arch
sudo pacman -S vhs ffmpeg figlet gifsicle ttf-jetbrains-mono-nerd
fc-cache -f

# Debian/Ubuntu (vhs itself needs the Charm apt repository)
sudo apt install figlet

# macOS
brew install vhs ffmpeg figlet gifsicle
brew install --cask font-jetbrains-mono-nerd-font

# record and downscale
make demo
```

`figlet` draws the opening wordmark with the `block` font; without it the first
frame shows only the typed command. `make demo` records at 1800x1200 and
downscales to 900x600 with lanczos, which is what keeps the glyphs sharp at the
README's display size; running `vhs` alone only writes the intermediate MP4.

Recording also needs a model registered as `qwen2.5-0.5b-instruct-q4_k_m`,
so this is a local step — `.github/workflows/vhs.yml` is best-effort and does
not download a model.

Target: **under 17 s**, **under 1 MB**, terminal width ~110 cols.

## Release process (maintainers)

1. Land everything for the release on `main`.
2. Update `CHANGELOG.md`.
3. Bump `version` in both `pyproject.toml` and `src/vipera/__init__.py`.
4. `git commit -m "chore: release vX.Y.Z"` and `git tag vX.Y.Z`.
5. `git push && git push --tags` — `release.yml` publishes to PyPI.
6. `gh release create vX.Y.Z --notes-from-tag`.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
