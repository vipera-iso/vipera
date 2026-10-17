#!/usr/bin/env bash
set -euo pipefail

REPO="vipera-iso/vipera"
DESC="Modern local AI gateway — OpenAI-compatible, GGUF-native"

# ─── 1. Verify local FIRST ────────────────────────────
echo "→ local verification"
if [ ! -d .venv ]; then
  python -m venv .venv
fi
.venv/bin/pip install -U pip 2>&1 | tail -1
.venv/bin/pip install -e ".[dev]" 2>&1 | tail -1

echo "→ pytest"
.venv/bin/pytest -q

echo "→ CLI smoke test"
.venv/bin/vipera --version
.venv/bin/vipera help > /dev/null

# ─── 2. Branch setup ────────────────────────────────
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "master")
if [ "$CURRENT_BRANCH" != "main" ]; then
  git checkout -b main 2>/dev/null || git branch -m main
fi

# Clean any existing git state and re-init
if [ -d .git ]; then
  echo "→ moving old .git aside"
  mv .git ".git.backup.$(date +%s)"
fi

git init -b main
git add .
git commit -m "feat: Vipera AI Gateway v1.0.0 — OpenAI-compatible local LLM server"

# ─── 3. Delete old repo & create new one ──────────────────────────
echo "→ deleting old repo (if present)"
gh repo delete "$REPO" --yes 2>/dev/null || echo "  (no existing repo or not owner)"

echo "→ creating repo and pushing"
gh repo create "$REPO" --public --source=. --remote=origin --push --description "$DESC"

# ─── 4. Polish ────────────────────────────────────────
gh repo edit "$REPO" \
  --add-topic llm --add-topic gguf --add-topic llama-cpp \
  --add-topic openai-api --add-topic local-ai --add-topic gateway \
  --add-topic python --add-topic inference --add-topic moe

git tag v1.0.0
git push origin v1.0.0

gh release create v1.0.0 \
  --title "Vipera v1.0.0 — first release" \
  --notes "First public release of the Vipera AI Gateway.

- OpenAI-compatible HTTP server with real SSE streaming
- Hardware-aware planner (VRAM/RAM/disk placement)
- Engine fallback: GPU → CPU-safe plan, mlock retry
- Model manager with HuggingFace search
- CLI chat, benchmark, and diagnostics

Install: \`pip install vipera\`"

echo
echo "✅ https://github.com/$REPO"
