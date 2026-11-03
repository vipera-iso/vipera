#!/usr/bin/env bash
# gpt-oss-120b: tải một lần, rồi benchmark. Giữ log mỗi lần chạy để sau này so sánh.
#
#   ./tools/gpt-oss-120b.sh            # tải + đo, lưu ~/.vipera/logs/
#   CTX="4096,8192" ./tools/gpt-oss-120b.sh
#
# Model này là MoE 120B (61 GB expert) trên máy 15 GB RAM + card 4 GB, nên con số
# ở đây là baseline "đọc từ đĩa", không phải mục tiêu tốc độ: xem
# OPTIMIZE-gptoss120b-tokps.md §0 (F8: 1.59 t/s decode, TTFT 117.9 s).
set -euo pipefail

REPO=${REPO:-unsloth/gpt-oss-120b-GGUF}
QUANT=${QUANT:-Q2_K}
ALIAS=${ALIAS:-gpt-oss-120b-q2k}
DIR=${DIR:-$HOME/.vipera/models/$ALIAS}
CTX=${CTX:-8192}
TOKENS=${TOKENS:-32}
LOGS=${LOGS:-$HOME/.vipera/logs}

mkdir -p "$DIR" "$LOGS"

# Tải theo shard, resume nếu dở (62 GB nên một lần rớt mạng không phải làm lại).
shards=$(curl -sL "https://huggingface.co/api/models/$REPO" |
  python3 -c "import json,sys;print('\n'.join(f['rfilename'] for f in json.load(sys.stdin)['siblings'] if f['rfilename'].startswith('$QUANT/') and f['rfilename'].endswith('.gguf')))")
[ -n "$shards" ] || { echo "no $QUANT shards in $REPO" >&2; exit 1; }
for path in $shards; do
  f=$(basename "$path")
  [ -s "$DIR/$f" ] || curl -L --fail -C - -o "$DIR/$f" "https://huggingface.co/$REPO/resolve/main/$path"
done

# Đăng ký một lần; analyzer tự hợp nhất các shard (OPTIMIZE-gptoss120b-tokps.md §2).
vipera model list 2>/dev/null | grep -q "$ALIAS" ||
  vipera model import "$DIR/$(basename "$(echo "$shards" | head -1)")" -n "$ALIAS"

stamp=$(date +%Y%m%d-%H%M)
log="$LOGS/$ALIAS-$stamp.log"
vipera bench run "$ALIAS" --ctx "$CTX" --max-tokens "$TOKENS" 2>&1 | tee "$log"
echo "log: $log"
