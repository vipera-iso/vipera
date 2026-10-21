#!/usr/bin/env bash
# L1a v3 - chay lai phep do imatrix tu mot nguon CO DO PHAN GIAI CAO.
#
# Vi sao: L1a v2 do Q4_K_M -> Q4_K_M, ma llama-quantize copy nguyen byte khi
# source type == target type (src/llama-quant.cpp: `quantize = cur_type != new_type`).
# => ca 2 arm Q4 giong nhau tung byte, imatrix khong bao gio duoc doc, +0.0000 la
#    no-op chu khong phai ket qua. Q3 thi co requantize that (type doi) nen arm Q3
#    van hop le.
#
# Nguon moi: Q8_0 (bartowski/NousResearch_Hermes-4-14B-GGUF, 15.7GB, ~0.01% loss
# so voi BF16) -> Q4_K_M / Q3_K_M la requantize that, imatrix moi co tac dung.
#
# Gate: MOI lan quantize phai qua tools/gguf_quant_audit.py (rc=0). No-op => FAIL
# va khong duoc dung so PPL cua arm do.
set -euo pipefail
cd /home/user/vipera
export VIPERA_LLAMA_BIN="$HOME/dev/llama.cpp/build/bin"
V=.venv/bin/vipera
PY=.venv/bin/python
EVAL=tools/eval/wikitext_sample.txt
M="$HOME/.vipera/models"
OUT=tools/l1a2_results
SRC="$M/NousResearch_Hermes-4-14B-Q8_0.gguf"
IM_BF="$M/NousResearch_Hermes-4-14B-imatrix.gguf"   # imatrix co san, build tu BF16
FAILED_RUNS=0

# Khong build imatrix tai cho: nguon Q8_0 la 15.7GB, may 15.4GB khong swap nen
# llama-imatrix phai thrash (~53 phut) va de bi OOM. Imatrix co san tu BF16
# (bartowski, 2 tensor/weight: in_sum2 + counts) la dung thu L1a can: mot
# imatrix tu nguon co do phan giai cao, khong phai tu Q4_K_M.

mkdir -p "$OUT"
banner() { echo; echo "===== $* ====="; date +%T; }
need() { [ -s "$1" ]; }

fail() { echo "!!! FAIL: $*"; FAILED_RUNS=$((FAILED_RUNS+1)); }

# PPL <gguf> <json> - mot run, timeout rieng, JSON rieng; fail khong giet script.
PPL() {
  banner "PPL: $(basename "$1")"
  if need "$2" && $PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[0]['rc']==0 else 1)" "$2"; then
    echo "skip - da co $2 (rc=0)"
    return
  fi
  timeout -k 20 1800 $PY tools/ppl_baseline.py \
    --gguf "$1" --ctx 512 --chunks 40 --ngl 0 \
    --bin-dir "$VIPERA_LLAMA_BIN" --json "$2" \
    || fail "PPL run: $1"
}

# QUANT <src> <quant> <out> [imatrix] - quantize + bat buoc qua audit
QUANT() {
  local src="$1" quant="$2" out="$3" im="${4:-}"
  banner "QUANTIZE $(basename "$out")"
  if need "$out"; then
    echo "skip - da co"
  else
    if [ -n "$im" ]; then
      timeout -k 20 1200 $V model quantize "$src" -q "$quant" --allow-requantize \
        --imatrix "$im" -o "$out" || { fail "quantize $out"; return; }
    else
      timeout -k 20 1200 $V model quantize "$src" -q "$quant" --allow-requantize \
        -o "$out" || { fail "quantize $out"; return; }
    fi
  fi
  # Gate: neu khong co tensor nao doi type thi day la copy byte, khong phai
  # requantize -> so PPL vo nghia.
  $PY tools/gguf_quant_audit.py --src "$src" --out "$out" --skip-data || \
    fail "no-op quantize: $(basename "$out")"
}

banner "STAGE 0: binary identity + inputs"
$PY - <<'PY'
import hashlib, sys
from pathlib import Path
p = Path.home()/"dev/llama.cpp/build/bin/llama-perplexity"
print("llama-perplexity sha256:", hashlib.sha256(p.read_bytes()).hexdigest())
print("(L5a/L1a v2 dung 56c271b8ded0ec4b45e2c329e61ce2b03b46a3a420f9570ffba04a9531f2c398)")
PY
for f in "$SRC" "$IM_BF" "$EVAL"; do
  if [ -s "$f" ]; then echo "ok   $f ($(stat -c%s "$f") bytes)"; else echo "THIEU $f"; exit 1; fi
done
df -h "$M" | tail -1

# --- Q4 group: requantize that tu Q8_0 -------------------------------------
QUANT "$SRC" q4_k_m "$M/q8src-ctl-q4km.gguf"
QUANT "$SRC" q4_k_m "$M/q8src-imbf-q4km.gguf" "$IM_BF"

# --- Q3 group: noi imatrix duoc ky vong co tac dung nhat -------------------
QUANT "$SRC" q3_k_m "$M/q8src-ctl-q3km.gguf"
QUANT "$SRC" q3_k_m "$M/q8src-imbf-q3km.gguf" "$IM_BF"

banner "STAGE 4: PPL (ctx 512, chunks 40, ngl 0 - y het L5a/L1a v2)"
PPL "$M/q8src-ctl-q4km.gguf"  "$OUT/q4_ctl.json"
PPL "$M/q8src-imbf-q4km.gguf" "$OUT/q4_imbf.json"
PPL "$M/q8src-ctl-q3km.gguf"  "$OUT/q3_ctl.json"
PPL "$M/q8src-imbf-q3km.gguf" "$OUT/q3_imbf.json"

banner "STAGE 5: audit - imatrix co doi weight khong?"
$PY tools/gguf_quant_audit.py --a "$M/q8src-ctl-q4km.gguf" --b "$M/q8src-imbf-q4km.gguf" || true
$PY tools/gguf_quant_audit.py --a "$M/q8src-ctl-q3km.gguf" --b "$M/q8src-imbf-q3km.gguf" || true

banner "STAGE 6: so lieu"
$PY tools/l1a2_report.py | tee "$OUT/report_numbers.txt"

banner "KIEM TRA CUOI"
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
  echo "CO $FAILED_RUNS BUOC THAT BAI - xem '!!! FAIL' o tren"
  exit 2
fi
