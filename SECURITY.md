# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| `1.0.x` | ✅ |
| `< 1.0` | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

- GitHub Security Advisories: https://github.com/vipera-iso/vipera/security/advisories/new
- Email the maintainer listed on the GitHub profile.

Include: description, impact, steps to reproduce, version and platform,
whether you want credit.

We acknowledge within **72 hours**, provide a fix or mitigation within
**14 days** for high-severity issues, and publish an advisory when a fix is
released.

## Scope

- **In scope:** auth bypass, path traversal in `vipera model import`, command
  injection in the llama.cpp launch line, unsafe deserialization of the
  registry or config, DoS via unbounded `/v1/chat/completions` requests.
- **Out of scope:** local privilege escalation via llama.cpp itself, DoS by
  very large prompts, absence of auth on the default bind address
  (`127.0.0.1`, intentional).

## Threat model

- Default bind is `127.0.0.1`. The gateway has **no authentication** by
  design — it is meant to be reached via `localhost`, or behind your own
  reverse proxy that adds auth.
- Model files are trusted inputs. Do not `vipera model import` a `.gguf` from
  an untrusted source.
- Registry and config are plain JSON in `~/.vipera/`. File permissions are the
  user's responsibility.

If you expose Vipera beyond localhost:

1. Set `host` to the interface you actually want (never `0.0.0.0` in default).
2. Front it with nginx / caddy / Traefik and add auth.
3. Add rate limits at the proxy.
4. Consider containerizing.

## Disclosure

Coordinated disclosure: we work with the reporter, ship a fix, and publish an
advisory after the fixed version is available. Reporters who want credit will
be named in the advisory.
