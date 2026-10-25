#!/usr/bin/env bash
# L1a v3 - rerun the imatrix measurement from a high-resolution source.
#
# Why: L1a v2 quantizes Q4_K_M -> Q4_K_M, but llama-quantize copies bytes when
# the source and target types match (src/llama-quant.cpp: `quantize = cur_type != new_type`).
# So both arms are byte-identical, the importance matrix is never read, and the
# +0.0000 result was a no-op, not a "matrix has no effect" statement. Q3 still
# performs a real requantize (different type), so the Q3 arm stays valid.
#
# New source: Q8_0 (bartowski/NousResearch_Hermes-4-14B-GGUF, 15.7 GB, ~0.01% loss
# vs BF16) -> Q4_K_M / Q3_K_M are real requantizes and the matrix actually reaches
# the weights.
#
# Gate: every quantize run must pass tools/gguf_quant_audit.py (rc=0). A no-op =>
# FAIL and the affected PPL number must not be used.
set -euo pipefail

cd /home/user/vipera
export VIPERA_LLAMA_BIN="$HOME/dev/llama.cpp/build/bin"
V=.venv/bin/vipera
PY=.venv/bin/python
EVAL=tools/eval/wikitext_sample.txt
M="$HOME/.vipera/models"
OUT=tools/l1a2_results
SRC="$M/NousResearch_Hermes-4-14B-Q8_0.gguf"
IM_BF="$M/NousResearch_Hermes-4-14B-imatrix.gguf"   # prebuilt from BF16 (high-resolution)
FAILED_RUNS=0

# Do not build the matrix on this machine: Q8_0 is 15.7 GB and this box has
# ~15.4 GB without swap, so llama-imatrix would thrash (~53 min) and risk OOM.
# The prebuilt BF16 matrix (2 tensors/weight: in_sum2 + counts) is the reference
# resolution ladder, not the lossy Q4_K_M source.
mkdir -p "$OUT"
banner() { echo; echo "===== $* ====="; date +%T; }
need() { [ -s "$1" ]; }
fail() { echo "!!! FAIL: $*"; FAILED_RUNS=$((FAILED_RUNS+1)); }

# PPL <gguf> <json> - one run, own timeout, own JSON; a failure reports but
# does not abort the remaining stages.
PPL() {
  banner "PPL: $(basename "$1")"
  if need "$2" && $PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[0]['rc']==0 else 1)" "$2"; then
    echo "skip - already has $2 (rc=0)"
    return
  fi
  timeout -k 20 1800 $PY tools/ppl_baseline.py \
    --gguf "$1" --ctx 512 --chunks 40 --ngl 0 \
    --bin-dir "$VIPERA_LLAMA_BIN" --json "$2" \
    || fail "PPL run: $1"
}

# QUANT <src> <quant> <out> [imatrix] - quantize, then force the audit gate.
QUANT() {
  local src="$1" quant="$2" out="$3" im="${4:-}"
  banner "QUANTIZE $(basename "$out")"
  if need "$out"; then
    echo "skip - already exists"
  else
    if [ -n "$im" ]; then
      timeout -k 20 1200 $V model quantize "$src" -q "$quant" --allow-requantize \
        --imatrix "$im" -o "$out" || { fail "quantize $out"; return; }
    else
      timeout -k 20 1200 $V model quantize "$src" -q "$quant" --allow-requantize \
        -o "$out" || { fail "quantize $out"; return; }
    fi
  fi
  # Gate: no type change => byte copy, not a real requantize -> the PPL number
  # for this arm is meaningless.
  $PY tools/gguf_quant_audit.py --src "$src" --out "$out" --skip-data || \
    fail "no-op quantize: $(basename "$out")"
}

banner "STAGE 0: binary identity + inputs"
$PY - <<'PY'
import hashlib, sys
from pathlib import Path
p = Path.home() / "dev" / "llama.cpp" / "build" / "bin" / "llama-perplexity"
print("llama-perplexity sha256:", hashlib.sha256(p.read_bytes()).hexdigest())
PY
for f in "$SRC" "$IM_BF" "$EVAL"; do
  if [ -s "$f" ]; then echo "ok   $f ($(stat -c%s "$f") bytes)"; else echo "THIEU $f"; exit 1; fi
done
df -h "$M" | tail -1

# --- Q4 group: requantize from Q8_0 (real, matrix reaches the weights) -------
QUANT "$SRC" q4_k_m "$M/q8src-ctl-q4km.gguf"
QUANT "$SRC" q4_k_m "$M/q8src-imbf-q4km.gguf" "$IM_BF"

# --- Q3 group: real requantize, matrix has its strongest, cleanest effect ----
QUANT "$SRC" q3_k_m "$M/q8src-ctl-q3km.gguf"
QUANT "$SRC" q3_k_m "$M/q8src-imbf-q3km.gguf" "$IM_BF"

banner "STAGE 4: PPL (ctx 512, chunks 40, ngl 0 - same conditions as L5a/L1a v2)"
PPL "$M/q8src-ctl-q4km.gguf"  "$OUT/q4_ctl.json"
PPL "$M/q8src-imbf-q4km.gguf" "$OUT/q4_imbf.json"
PPL "$M/q8src-ctl-q3km.gguf"  "$OUT/q3_ctl.json"
PPL "$M/q8src-imbf-q3km.gguf" "$OUT/q3_imbf.json"

banner "STAGE 5: audit - did the matrix actually change weights?"
$PY tools/gguf_quant_audit.py --a "$M/q8src-ctl-q4km.gguf" --b "$M/q8src-imbf-q4km.gguf" || true
$PY tools/gguf_quant_audit.py --a "$M/q8src-ctl-q3km.gguf" --b "$M/q8src-imbf-q3km.gguf" || true

banner "STAGE 6: numbers"
$PY tools/l1a2_report.py | tee "$OUT/report_numbers.txt"

banner "CHECK LAST"
ls -la "$OUT"
$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob("*.json")):
    r = json.loads(p.read_text())[0]
    print(f"{p.name:<16} ppl={r['ppl']} rc={r['rc']} chunks={r['chunks']} tok={r['scored_tok']}")
PY
if [ "$FAILED_RUNS" -eq 0 ]; then
  echo "ALL L1A v3 STAGES DONE"
else
  echo "SOME STAGES FAILED - see '!!! FAIL' above"
  exit 2
fi
