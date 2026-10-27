#!/usr/bin/env bash
# L1a v2 - identical plan to l1a_run.sh, with two fixes from review:
#  1) every PPL run gets its OWN timeout (20 min) and its own JSON, and a
#     failed/timed-out run is reported but does NOT abort the remaining stages
#     (blast radius = 1 run, not the whole script);
#  2) stage 0 pins the llama-perplexity binary identity (sha256 + version) so
#     the L5a baseline and these runs are provably the same tool.
# Stages 1-3 are resumable: existing non-empty outputs are skipped.
set -euo pipefail
cd /home/user/vipera
export VIPERA_LLAMA_BIN="$HOME/dev/llama.cpp/build/bin"
V=.venv/bin/vipera
EVAL=tools/eval/wikitext_sample.txt
M="$HOME/.vipera/models"
HERMES="$M/NousResearch_Hermes-4-14B-Q4_K_M.gguf"
QWEN="$M/qwen15b.gguf"

banner() { echo; echo "===== $* ====="; date +%T; }
need() { [ -s "$1" ]; }
FAILED_RUNS=0
# PPL <gguf> <json> - one run, own timeout, own JSON; failures don't kill the script.
PPL() {
  banner "PPL run: $(basename "$1") -> $(basename "$2")"
  timeout -k 20 1200 .venv/bin/python tools/ppl_baseline.py \
    --gguf "$1" --ctx 512 --chunks 40 --ngl 0 \
    --bin-dir "$VIPERA_LLAMA_BIN" --json "$2" \
    || { echo "!!! PPL RUN FAILED rc=$?: $1 (tiep tuc cac run con lai)"; FAILED_RUNS=$((FAILED_RUNS+1)); }
}

banner "STAGE 0: binary identity"
sha256sum "$VIPERA_LLAMA_BIN/llama-perplexity"
ls -la "$VIPERA_LLAMA_BIN/llama-perplexity"
"$VIPERA_LLAMA_BIN/llama-perplexity" --version 2>&1 | head -2

banner "STAGE 1a: imatrix Hermes"
if need "$M/hermes-q4km.imat"; then echo "skip - da co"; else
  timeout -k 20 1800 $V model imatrix "$HERMES" -f "$EVAL" --ctx 512 -o "$M/hermes-q4km.imat"; fi

banner "STAGE 1b: imatrix qwen15b"
if need "$M/qwen15b-q4km.imat"; then echo "skip - da co"; else
  timeout -k 20 900 $V model imatrix "$QWEN" -f "$EVAL" --ctx 512 -o "$M/qwen15b-q4km.imat"; fi

banner "STAGE 2a: CONTROL quantize Hermes Q4_K_M -> Q4_K_M (no imatrix)"
if need "$M/ctl-hermes-q4km.gguf"; then echo "skip - da co"; else
  timeout -k 20 1200 $V model quantize "$HERMES" -q q4_k_m --allow-requantize -o "$M/ctl-hermes-q4km.gguf"; fi

banner "STAGE 2b: CONTROL quantize qwen15b Q4_K_M -> Q4_K_M"
if need "$M/ctl-qwen15b-q4km.gguf"; then echo "skip - da co"; else
  timeout -k 20 900 $V model quantize "$QWEN" -q q4_k_m --allow-requantize -o "$M/ctl-qwen15b-q4km.gguf"; fi

banner "STAGE 3a: L1a quantize Hermes Q4_K_M WITH imatrix"
if need "$M/l1a-hermes-q4km.gguf"; then echo "skip - da co"; else
  timeout -k 20 1200 $V model quantize "$HERMES" -q q4_k_m --allow-requantize --imatrix "$M/hermes-q4km.imat" -o "$M/l1a-hermes-q4km.gguf"; fi

banner "STAGE 3b: L1a quantize qwen15b Q4_K_M WITH imatrix"
if need "$M/l1a-qwen15b-q4km.gguf"; then echo "skip - da co"; else
  timeout -k 20 900 $V model quantize "$QWEN" -q q4_k_m --allow-requantize --imatrix "$M/qwen15b-q4km.imat" -o "$M/l1a-qwen15b-q4km.gguf"; fi

banner "STAGE 4: PPL Hermes Q4 group (3 runs rieng biet)"
PPL "$HERMES"                          /tmp/opencode/l1a_ppl_hermes_orig2.json
PPL "$M/ctl-hermes-q4km.gguf"          /tmp/opencode/l1a_ppl_hermes_ctl_q4.json
PPL "$M/l1a-hermes-q4km.gguf"          /tmp/opencode/l1a_ppl_hermes_l1a_q4.json

banner "STAGE 5: PPL qwen15b Q4 group (3 runs rieng biet)"
PPL "$QWEN"                            /tmp/opencode/l1a_ppl_qwen_orig2.json
PPL "$M/ctl-qwen15b-q4km.gguf"         /tmp/opencode/l1a_ppl_qwen_ctl_q4.json
PPL "$M/l1a-qwen15b-q4km.gguf"         /tmp/opencode/l1a_ppl_qwen_l1a_q4.json

banner "STAGE 6a: CONTROL quantize Hermes -> Q3_K_M (harness sanity, khong ship)"
if need "$M/ctl-hermes-q3km.gguf"; then echo "skip - da co"; else
  timeout -k 20 1200 $V model quantize "$HERMES" -q q3_k_m --allow-requantize -o "$M/ctl-hermes-q3km.gguf"; fi

banner "STAGE 6b: L1a quantize Hermes -> Q3_K_M WITH imatrix"
if need "$M/l1a-hermes-q3km.gguf"; then echo "skip - da co"; else
  timeout -k 20 1200 $V model quantize "$HERMES" -q q3_k_m --allow-requantize --imatrix "$M/hermes-q4km.imat" -o "$M/l1a-hermes-q3km.gguf"; fi

banner "STAGE 7: PPL Hermes Q3 sanity group (2 runs)"
PPL "$M/ctl-hermes-q3km.gguf"          /tmp/opencode/l1a_ppl_hermes_ctl_q3.json
PPL "$M/l1a-hermes-q3km.gguf"          /tmp/opencode/l1a_ppl_hermes_l1a_q3.json

banner "KIEM TRA CUOI"
ls -la /tmp/opencode/l1a_ppl_*.json
if [ "$FAILED_RUNS" -eq 0 ]; then
  echo "ALL L1A STAGES DONE"
else
  echo "CO $FAILED_RUNS PPL RUN THAT BAI - xem '!!! PPL RUN FAILED' o tren"
  exit 2
fi
