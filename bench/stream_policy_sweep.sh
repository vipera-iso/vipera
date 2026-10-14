#!/usr/bin/env bash
# Compare KV eviction policies at an identical KV budget (stage E, docs/REWRITE.md).
#
#   bash bench/stream_policy_sweep.sh              # both experiments, all three policies
#   bash bench/stream_policy_sweep.sh A surprise   # just that cell (appends)
#
# Every arm scores the SAME tokens with the SAME number of tokens dropped; only the choice
# of which tokens to drop differs, so a difference in PPL is attributable to the policy.
#
#   A  ctx 512 (window 768), keep 4, stride 256, 195 chunks   - the row where recency was
#      measured to cost +4.05% against the full window, so there is headroom to recover
#   B  ctx 256 (window 320), keep 4, stride 128, 391 chunks   - higher eviction pressure
#
# Sequential on purpose: FINDINGS.md §2.
set -uo pipefail
cd "$HOME/dev/vipera"
PY=./.venv/bin/python
BIN=$HOME/dev/llama.cpp/build/bin/llama-perplexity
M=models/Qwen3-0.6B-Q4_K_M.gguf
F=$HOME/dev/llama.cpp/wikitext-2-raw/wiki.test.raw
OUT=bench/results/ppl-policy-summary.txt
mkdir -p bench/results

WANT=("$@")
want() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

if [ ${#WANT[@]} -eq 0 ]; then
  : > "$OUT"
else
  echo "=== partial run: $* ===" >> "$OUT"
fi
log() { echo "$*" >> "$OUT"; }

# experiment A: window 768, keep 4
if want A; then
  for P in recency recency-gap surprise; do
    want "A-$P" || want "A" || continue
    log "=== A keep=4 ctx=512 stride=256 chunks=195 policy=$P ==="
    $PY -m bench.perplexity --bin "$BIN" --model "$M" --file "$F" \
        --context 512 --stride 256 --chunks 195 --keep 4 --context-shift --policy "$P" \
        --threads 4 --json "bench/results/ppl-policy-A-$P.json" >> "$OUT" 2>&1
    log "exit=$?"
  done
  # The surprise-local hybrid is a pair (policy, margin), so it gets its own loop rather
  # than a third element in the list above. Both margins it was measured at are here: the
  # margin trades the surprise ordering against recency, and reporting only the better one
  # would be picking the winner after the fact (docs/REWRITE.md §E).
  for MG in 128 64; do
    want "A-surprise-local" || want A || continue
    log "=== A keep=4 ctx=512 stride=256 chunks=195 policy=surprise-local margin=$MG ==="
    $PY -m bench.perplexity --bin "$BIN" --model "$M" --file "$F" \
        --context 512 --stride 256 --chunks 195 --keep 4 --context-shift \
        --policy surprise-local --margin "$MG" --threads 4 \
        --json "bench/results/ppl-policy-A-surprise-local-m$MG.json" >> "$OUT" 2>&1
    log "exit=$?"
  done
fi

# experiment B: window 320, keep 4
if want B; then
  for P in recency recency-gap surprise; do
    want "B-$P" || want "B" || continue
    log "=== B keep=4 ctx=256 stride=128 chunks=391 policy=$P ==="
    $PY -m bench.perplexity --bin "$BIN" --model "$M" --file "$F" \
        --context 256 --stride 128 --chunks 391 --keep 4 --context-shift --policy "$P" \
        --threads 4 --json "bench/results/ppl-policy-B-$P.json" >> "$OUT" 2>&1
    log "exit=$?"
  done
fi

log "=== done ==="
