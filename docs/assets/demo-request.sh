#!/usr/bin/env bash
# Streaming chat completion used by docs/assets/demo.tape.
# Prints only the text deltas so the GIF shows tokens, not SSE framing.
set -uo pipefail

PORT="${1:-8686}"

curl -sN "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to five."}],"stream":true}' \
  | while IFS= read -r line; do
      case "$line" in
        "data: [DONE]") break ;;
        "data: "*)
          printf '%s' "${line#data: }" | jq -j '.choices[0].delta.content // empty'
          ;;
      esac
    done
echo
