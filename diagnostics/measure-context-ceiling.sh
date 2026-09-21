#!/usr/bin/env bash
# Bisect a model's num_ctx to find the highest value that still loads 100%
# into GPU VRAM. Needs curl and python3 (stdlib only) on PATH, and an Ollama
# server already running with the OLLAMA_KV_CACHE_TYPE / OLLAMA_FLASH_ATTENTION
# / OLLAMA_SCHED_SPREAD you actually intend to run with -- the ceiling is only
# valid for the settings the server had loaded when you measured it.
#
# Usage: ./measure-context-ceiling.sh <model> <ctx1> [ctx2 ...]
#
# Ollama's own reported model "size" (from /api/ps) is not simple
# weights-plus-linear-KV-cache math -- it is the size of whatever allocation
# plan the runtime actually committed to. Once a requested num_ctx cannot fit
# in VRAM, the runtime can fall back to a smaller, partially-CPU plan rather
# than a bigger, still-100%-GPU one. That is why the *total* footprint can
# go down at a larger num_ctx: it is a different, worse plan, not more of
# the same plan. Confirm the ceiling by watching offload%, not by assuming
# size grows monotonically with num_ctx.
set -euo pipefail

ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"
MODEL="${1:?usage: measure-context-ceiling.sh <model> <ctx1> [ctx2 ...]}"
shift

PROMPT="Write a Python function that merges two sorted lists into one sorted list, with a docstring."

for ctx in "$@"; do
    curl -s "${ENDPOINT}/api/generate" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"stream\":false,\"options\":{\"num_predict\":50,\"num_ctx\":${ctx}}}" \
        -o /dev/null

    curl -s "${ENDPOINT}/api/ps" | python3 -c "
import json, sys
d = json.load(sys.stdin)
models = d.get('models', [])
if not models:
    print('ctx=${ctx}: nothing resident -- load failed')
    sys.exit(1)
m = models[0]
pct = 100 * m['size_vram'] / m['size']
print(f'ctx=${ctx}: offload {pct:.1f}% ({m[\"size_vram\"]/1e9:.2f} / {m[\"size\"]/1e9:.2f} GB)')
"

    curl -s "${ENDPOINT}/api/generate" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"\",\"keep_alive\":0}" \
        -o /dev/null
    sleep 2
done
