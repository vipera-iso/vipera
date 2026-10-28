# Source: addyosmani/agent-skills

Provenance for every file under `docs/reference/agent-skills/upstream/` and for the
six skills copied into `skills/`.

| | |
| --- | --- |
| Repository | <https://github.com/addyosmani/agent-skills> |
| Commit | `1be8e34187e34647bb83adc3a1323b26ae6f6abe` (2026-10-10) |
| License | MIT — see `upstream/LICENSE` (Copyright (c) 2025 Addy Osmani) |
| Vendored on | 2026-10-12 |
| Published | No. Local vendoring only; nothing was pushed anywhere. |

## What was installed where

**Loaded by OpenCode** — copied verbatim into `skills/`, discovered through the
project `opencode.jsonc` entry `"skills": ["./skills", "~/dev/vipera/skills"]`:

> Both forms are listed on purpose. `./skills` resolves from the *active working
> directory*, which is empty when the skill list is queried without a session in
> this repo; the `~/` entry always resolves. Verified: `GET /api/skill
> ?location[directory]=~/dev/vipera` returns these 8 IDs plus the 2 built-ins,
> and none of the 19 upstream skills parked in `upstream/`.

| Skill ID | Why it applies to vipera |
| --- | --- |
| `spec-driven-development` | Freeze what the optimizer must and must not do before more code lands. |
| `planning-and-task-breakdown` | Cut `docs/PROPOSALS.md` items into ordered, individually verifiable tasks. |
| `doubt-driven-development` | Adversarial review of every unvalidated claim — same rule vipera already states in `skills/expert-io-loop`. |
| `source-driven-development` | Matches the repo's existing rule: every claim carries an anchor back to a real source, "no data" where there is none. |
| `performance-optimization` | Profiling-first method; vipera's numbers are modelled unless a bench run says otherwise. |
| `code-review-and-quality` | Multi-axis review gate before a change is treated as settled. |

**Reference only** — the other 19 upstream skills live in `upstream/` and are
deliberately **not** on any skill source path, so they never enter a prompt:

```
api-and-interface-design, browser-testing-with-devtools, ci-cd-and-automation,
code-simplification, constraint-driven-development, context-engineering,
debugging-and-error-recovery, deprecation-and-migration, documentation-and-adrs,
frontend-ui-engineering, git-workflow-and-versioning, idea-refine,
incremental-implementation, interview-me, observability-and-instrumentation,
security-and-hardening, shipping-and-launch, test-driven-development,
using-agent-skills
```

Read one by hand when it fits, or copy it into `skills/` to make it loadable.

## Why 19 are parked

Each advertised skill costs its `description` on every model step. Upstream
descriptions run 300–500 characters; loading all 25 would spend roughly 6–8k
characters of prompt before any work starts, for skills that mostly target web
frontends, CI pipelines and browser testing — none of which vipera has.

## Updating

```sh
git clone --depth 1 https://github.com/addyosmani/agent-skills /tmp/agent-skills
diff -ru /tmp/agent-skills/skills/<id> skills/<id>
```

Re-copy only what changed, and record the new commit hash above. Upstream files
are unmodified: `upstream/` is a byte-for-byte copy apart from the renamed
`UPSTREAM-README.md`.
