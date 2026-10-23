---
title: Repository structure
description: What is wrong with the current layout, the proposed tree, and what it costs to get there.
---

# Repository structure

**Status:** proposal. The visual pass changed `README.md`, `src/vipera/ui.py`,
`docs/index.html`, and added this directory; it did **not** move files, because
every move below breaks links in a commit whose purpose was presentation.

## What the repository currently is

Two things in one directory: a Python package (134 tracked files, ~4,000 lines
of `src/`) and a research archive. 35 MB of `docs/` is mostly
`docs/references/` — vendored HTML snapshots of llama.cpp and FastAPI pages —
plus `fullcode.md`, a generated 4,796-line dump of `src/` that is 13% of the
repository's text.

That is a legitimate way to run a measurement-driven project: the runs and the
sources stay next to the code. The problem is that **the two are sorted
alphabetically together**, so the first thing a visitor sees in the file list
is `OPTIMIZE-hermes4.md`, `OPTIMIZE-ollama.md`, `OPTIMIZE-techniques.md`,
`OPTIMIZE-tokps.md`, `fullcode.md`, and then `README.md`.

## Four concrete problems

1. **The root is a junk drawer.** `OPTIMIZE-*.md` are research notebooks about
   the runtime, not about Vipera's interface. `fullcode.md` is generated
   output. Both belong under `docs/`, and the generated file belongs in
   `.gitignore` with a `make fullcode` target, not in a review diff.
2. **`docs/` mixes product documentation with primary sources.** The nine
   guides at the top level are the product; `docs/references/` is an archive.
   Nothing in the docs says which is which, and link checking walks both.
3. **Six result directories, no index.** `tools/*_results/` hold the raw
   `.jsonl` behind the published numbers (~1 MB total, which is fine), but
   nothing links a number in `docs/benchmarks.md` to the file that produced it
   except a single `REPORT.md`.
4. **Two palettes, one brand.** Fixed in this pass: the terminal now uses the
   `docs/assets` ink/paper/cyan tokens instead of its own slate.

## Proposed tree

```text
docs/
  index.md            product documentation (unchanged)
  index.html          landing page (added)
  design/             proposals: backend, visual system, this file
  research/           OPTIMIZE-techniques.md, -tokps.md, -hermes4.md, -ollama.md
  references/         vendored sources (keep, but mark as archive)
tools/
  results/
    README.md         one table: number → file that produced it
    gpu/ iq3/ l1a/ l1a2/ layer/ server/
  server_ab.py        harnesses stay top-level, outputs move under results/
```

Root keeps: README, LICENSE, CHANGELOG, CONTRIBUTING, SECURITY,
CODE_OF_CONDUCT, pyproject.toml, Makefile, and the four package directories.

## What moving costs

| Move | References to update |
|---|---|
| `OPTIMIZE-*.md` → `docs/research/` | 7 (inside `docs/`, `README.md`, `OPTIMIZE-tokps.md`) |
| `fullcode.md` → generated, ignored | 1 (`OPTIMIZE-tokps.md`), plus a `make fullcode` target and a `.gitignore` line |
| `tools/*_results/` → `tools/results/` | 6 path strings in `tools/*.py` and 3 in `docs/benchmarks.md` |
| `docs/references/` → `docs/references/` | none (already isolated), but exclude it from the link check or keep its links relative |

Total: about 17 touch points. That is a small, mechanical change — worth doing
as its own commit with `git mv`, so the diff is a rename and reviewers see
nothing else.

## Why it was not done here

A redesign commit should be reviewable as a design commit. Mixing it with file
moves makes both harder to review and both harder to revert. The proposal is
written down so the move can be a single follow-up with a green CI run and no
open questions about intent.
