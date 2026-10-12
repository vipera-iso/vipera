# Contributing to vipera

Thanks for looking. vipera is small, stdlib-only, and deliberately
measurement-first — most of what follows is that culture, not bureaucracy.

## Requirements

- **Python >= 3.11** (the code uses `tomllib`).
- **No dependencies.** `pyproject.toml` declares `dependencies = []` on purpose:
  vipera has to run where installing packages is hard. A change that adds a
  runtime dependency needs a strong reason and should be raised as an issue first.

## Running the tests

```bash
python -m unittest discover -s tests -t .
```

The suite is pure stdlib and needs no install. Run it from the repo root — `-t .`
is what makes `vipera` and `tests` importable.

Two classes of test matter most, and both run by default:

- `tests/test_cache_policy.py` — cache policies, including the two white-box
  guards that fail if the recency-list optimisation is reverted.
- `tests/_fixture.py` — GGUF fixtures written by hand, checked against the real
  format rather than a round-trip through our own writer.

## The two rules this project actually enforces

**1. Numbers are measured, not assumed.** Every figure in the docs is tagged as
*measured* or *modelled*, and modelled numbers say so. If you add a number,
record how it was obtained. `PERF.md` is the ledger of what was tried and what
was rejected — read it before proposing a performance experiment.

**2. Behaviour stays put unless the change is the point.** The manifest is the
project's product, so a refactor or optimisation must leave it **byte-identical**.
When changing the simulation or planner, capture the output before and after and
diff it:

```bash
python -m vipera schedule arch/glm-5.3-flash.toml \
    --target hw/gtx1650.toml --quant q4-experts --out /tmp/before
# ...make the change...
python -m vipera schedule arch/glm-5.3-flash.toml \
    --target hw/gtx1650.toml --quant q4-experts --out /tmp/after
diff -r /tmp/before /tmp/after
```

Behaviour that *is* meant to change needs a test that pins the new behaviour and
a note explaining why the old one was wrong.

## Adding a model or machine profile

- Architecture: `arch/<name>.toml` (layer counts, experts, attention params).
- Hardware: `hw/<name>.toml` (memory, bandwidths, compute). Prefer values you
  measured (`bench/hwprobe.c` reports the machine constants) over defaults.

Both are plain TOML read by `Architecture.from_toml` / `Target.from_toml`, and
unknown keys are ignored, so adding one cannot break anything else.

## Commits

- Subject in the **imperative, capitalised**, no trailing period —
  `Add a routing-trace loader`, not `Added` or `adding`.
- Body explains **why**, not what. The diff already shows what.
- One logical change per commit. A refactor and a feature are two commits.

## Branches and pull requests

- Branches: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`.
- Keep the diff reviewable; a change that touches the planner and the CLI reads
  better split in two.
- A PR is ready when the suite is green and the before/after diff above is
  clean (or the change is deliberately and explicitly a behaviour change).
- Expect review to ask "was this measured?" — that is the normal bar here, not
  a suspicion.

## Where things live

| Path | Owns |
|---|---|
| `vipera/gguf.py` | GGUF header/metadata/tensor reader — never reads tensor data |
| `vipera/analyze.py` | descriptors → resident/expert byte inventory |
| `vipera/manifest.py` | `Target`, the size-driven placement planner |
| `vipera/arch.py` | per-layer architecture model (parameter arithmetic) |
| `vipera/schedule.py` | residency planner, expert cache, decoder simulator |
| `vipera/trace.py` | routing-trace format, loader, validation, statistics |
| `vipera/prune.py` | mass-based and gated expert pruning |
| `vipera/cli.py` | the `vipera` command, one `cmd_*` per subcommand |
| `bench/` | measurement harnesses (llama-bench wrapper, probe, perplexity) |

Logic belongs in the module that owns the concept, not in `cli.py`. If a
subcommand needs a new computation, put it in the relevant module and give it a
unit test.
