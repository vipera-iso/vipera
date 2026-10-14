#!/usr/bin/env bash
# Tight-CI perplexity measurement for the streaming (sink + sliding window) attention plan.
# Sequential on purpose: FINDINGS.md §2 forbids two benchmarks at once on this machine.
#
#   bash bench/ppl_stream_sweep.sh              # every configuration (replaces the record)
#   bash bench/ppl_stream_sweep.sh 32 stock256  # just those (appends to the record)
#
# Runs, all on wikitext-2 test / Qwen3-0.6B-Q4_K_M / -t 4:
#   4, 1, 32   sink + sliding vs the full window, ctx 512 -> 768-token window, stride 256,
#              195 chunks (50 304-token span, 49 921 scored tokens). Same scored set on both
#              sides, so the pair isolates the attention pattern.
#   stock256   the memory-constrained alternative a user would otherwise pick - a plain
#              contiguous window of 320 tokens (ctx 256, stride 128, 391 chunks, 50 240-token
#              span). Not span-identical to the runs above: 0.13% shorter, 50 048 scored tokens.
set -uo pipefail
cd "$HOME/dev/vipera"
PY=./.venv/bin/python
BIN=$HOME/dev/llama.cpp/build/bin/llama-perplexity
M=models/Qwen3-0.6B-Q4_K_M.gguf
F=$HOME/dev/llama.cpp/wikitext-2-raw/wiki.test.raw
OUT=bench/results/ppl-stream-summary.txt
mkdir -p bench/results

WANT=("$@")
want() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

if [ ${#WANT[@]} -eq 0 ]; then
  : > "$OUT"                                  # a full run replaces the record
else
  echo "=== partial run: $* ===" >> "$OUT"     # keep what earlier runs already measured
fi

log() { echo "$*" >> "$OUT"; }

# 1-3: identical evaluation (ctx 512 -> window 768, stride 256, 195 chunks), different sinks
for K in 4 1 32; do
  want "$K" || continue
  log "=== compare-stream keep=$K ctx=512 stride=256 chunks=195 ==="
  $PY -m bench.perplexity --bin "$BIN" --model "$M" --file "$F" \
      --context 512 --stride 256 --chunks 195 --keep "$K" --compare-stream \
      --threads 4 --json "bench/results/ppl-stream-k$K.json" >> "$OUT" 2>&1
  log "exit=$?"
done

# 4: the smaller-window control
if want stock256; then
  log "=== stock ctx=256 stride=128 chunks=391 (smaller window) ==="
  $PY -m bench.perplexity --bin "$BIN" --model "$M" --file "$F" \
      --context 256 --stride 128 --chunks 391 \
      --threads 4 --json bench/results/ppl-stock-c256.json >> "$OUT" 2>&1
  log "exit=$?"
fi

log "=== done ==="
