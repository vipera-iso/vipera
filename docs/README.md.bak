# Vipera documentation

Vipera is a local AI gateway: it turns a `.gguf` file into an OpenAI-compatible
HTTP endpoint, with a hardware-aware planner in front of `llama.cpp`.

This directory holds the reference material. If you are new to the project,
read the top-level [README](../README.md) first, then come back here for the
details.

## Guides

| Document | What it covers |
|---|---|
| [install.md](install.md) | Step-by-step install for non-technical users, model download, troubleshooting, uninstall. |
| [architecture.md](architecture.md) | Module map, request flow, planner algorithm, engine load/fallback, concurrency model, known gaps. |
| [cli.md](cli.md) | Every command, flag, default, and side effect. |
| [http-api.md](http-api.md) | Endpoints, request/response schemas, SSE framing, compatibility notes, client examples. |
| [configuration.md](configuration.md) | `config.json` keys, storage layout, registry schema, `VIPERA_HOME`. |
| [development.md](development.md) | Dev setup, tests, lint/type checks, CI, release process, demo GIF. |

Supporting files in the repository root:

- [CONTRIBUTING.md](../CONTRIBUTING.md) — contributor workflow and review expectations.
- [SECURITY.md](../SECURITY.md) — vulnerability reporting and supported versions.
- [CHANGELOG.md](../CHANGELOG.md) — release history.
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) — Contributor Covenant 2.1.

## Reading paths

**I want to run a model.**
[install.md](install.md) → [cli.md](cli.md#vipera-model) →
[configuration.md](configuration.md).

**I want to call the gateway from my app.**
[http-api.md](http-api.md).

**I want to understand why a model was placed the way it was.**
[architecture.md](architecture.md#planner).

**I want to change the code.**
[development.md](development.md) → [architecture.md](architecture.md#module-map).

## Documentation conventions

- Commands are written as they are actually invoked: `vipera serve start`.
- Source references use repository-relative paths, e.g. `src/vipera/planner.py`.
- Statements about runtime behavior are derived from the code in `src/vipera`,
  not from aspirational design notes. Where the README and the code disagree,
  the code is described here and the discrepancy is listed under
  [known gaps](architecture.md#known-gaps).
- When you change behavior, update the page that describes it in the same pull
  request. See [development.md](development.md#changing-the-docs).
