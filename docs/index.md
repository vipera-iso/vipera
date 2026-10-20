---
title: Vipera documentation
description: Entry point for installation, CLI, HTTP API, configuration, architecture, and benchmarks.
---

# Vipera documentation

Vipera turns a `.gguf` file into an OpenAI-compatible HTTP endpoint, with a
hardware-aware planner in front of `llama.cpp`. If you are new to the project,
read the top-level [README](../README.md) first, then come back here for the
details.

## Guides

| Document | What it covers |
|---|---|
| [install.md](install.md) | Step-by-step install for non-technical users, model download, troubleshooting, uninstall. |
| [cli.md](cli.md) | Every command, flag, default, and side effect. |
| [http-api.md](http-api.md) | Endpoints, request/response schemas, SSE framing, client examples. |
| [configuration.md](configuration.md) | `config.json` keys, storage layout, registry schema, `VIPERA_HOME`. |
| [architecture.md](architecture.md) | Module map, request flow, planner algorithm, engine load/fallback, concurrency, known gaps. |
| [benchmarks.md](benchmarks.md) | Reference measurements, method, and when placement actually helps. |
| [comparison.md](comparison.md) | Row-by-row comparison with Ollama, `llama.cpp/server`, and LM Studio. |
| [development.md](development.md) | Dev setup, tests, lint/type checks, CI, release process, demo GIF. |
| [design/llama-server-backend.md](design/llama-server-backend.md) | Proposal (not implemented): subprocess `llama-server` backend, DFlash speculative decoding, draft validation, migration. |

Supporting files in the repository root:

- [CONTRIBUTING.md](../CONTRIBUTING.md) — contributor workflow and review expectations.
- [SECURITY.md](../SECURITY.md) — vulnerability reporting and supported versions.
- [CHANGELOG.md](../CHANGELOG.md) — release history.
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) — Contributor Covenant 2.1.

## Reading paths

**I want to run a model.**
[install.md](install.md) → [cli.md](cli.md) → [configuration.md](configuration.md).

**I want to call the gateway from my app.**
[http-api.md](http-api.md).

**I want to understand why a model was placed the way it was.**
[architecture.md](architecture.md#planner) → [benchmarks.md](benchmarks.md).

**I want to choose between Vipera and another tool.**
[comparison.md](comparison.md).

**I want to change the code.**
[development.md](development.md) → [architecture.md](architecture.md#module-map).

**I want to know where speculative decoding is going.**
[design/llama-server-backend.md](design/llama-server-backend.md).

## Documentation conventions

- Commands are written as they are actually invoked: `vipera serve start`.
- Source references use repository-relative paths, e.g. `src/vipera/planner.py`.
- Statements about runtime behavior are derived from the code in `src/vipera`,
  not from aspirational design notes. Where the README and the code disagree,
  the code is described here and the discrepancy is listed under
  [known gaps](architecture.md#known-gaps).
- When you change behavior, update the page that describes it in the same pull
  request. See [development.md](development.md#changing-the-docs).
