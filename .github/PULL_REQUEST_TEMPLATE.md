## What does this PR do?

<!-- One or two sentences. -->

## Why?

Fixes #<!-- issue number, if any -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation / chore
- [ ] Performance change (include a `vipera bench run` before/after)

## How was it tested?

```bash
vipera system doctor
pytest -q
```

## Checklist

- [ ] Tests added or updated (or a note explaining why not).
- [ ] `pre-commit run --all-files` passes.
- [ ] `pytest -q` passes.
- [ ] `README.md` or `docs/` updated for user-visible behavior.
- [ ] `CHANGELOG.md` updated for public CLI or API changes.
- [ ] No `.gguf` / `.safetensors` / model artifacts committed.
- [ ] No secrets, tokens, or personal paths in the diff.
