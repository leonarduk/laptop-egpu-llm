#!/usr/bin/env bash
# Sweep a model over the num_ctx values you give it and report, for each,
# how much of the model landed in GPU VRAM -- so you can read off the
# highest value that still loads 100% on GPU. It tries exactly the values
# listed, in order; it does not search between them, so to narrow a
# ceiling down, rerun with values between the last pass and the first
# failure. Needs curl and python3 (stdlib only) on PATH, and an Ollama
# server already running with the OLLAMA_KV_CACHE_TYPE / OLLAMA_FLASH_ATTENTION
# / OLLAMA_SCHED_SPREAD you actually intend to run with -- the ceiling is only
# valid for the settings the server had loaded when you measured it.
#
# Usage: ./measure-context-ceiling.sh <model> <ctx1> [ctx2 ...]
#
# A value that fails to load is reported as FAILED and the sweep moves on
# to the next one; the script exits 1 at the end if any value failed.
#
# Ollama's own reported model "size" (from /api/ps) is not simple
# weights-plus-linear-KV-cache math -- it is the size of whatever allocation
# plan the runtime actually committed to. Once a requested num_ctx cannot fit
# in VRAM, the runtime can fall back to a smaller, partially-CPU plan rather
# than a bigger, still-100%-GPU one. That is why the *total* footprint can
# go down at a larger num_ctx: it is a different, worse plan, not more of
# the same plan. Confirm the ceiling by watching offload%, not by assuming
# size grows monotonically with num_ctx.
set -uo pipefail

ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"
MODEL="${1:?usage: measure-context-ceiling.sh <model> <ctx1> [ctx2 ...]}"
shift

PROMPT="Write a Python function that merges two sorted lists into one sorted list, with a docstring."

failures=0
for ctx in "$@"; do
    # --fail turns an HTTP error (e.g. a 500 when the runner cannot
    # allocate) into a non-zero exit instead of a quietly discarded body.
    if ! error=$(curl -sS --fail "${ENDPOINT}/api/generate" \
            -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"stream\":false,\"options\":{\"num_predict\":50,\"num_ctx\":${ctx}}}" \
            -o /dev/null 2>&1); then
        echo "ctx=${ctx}: FAILED to load -- ${error:-curl returned an error}"
        failures=$((failures + 1))
        continue
    fi

    # /api/ps lists every resident model, not just this one, and not in
    # any promised order -- so select the entry by name. A bare name is
    # listed as name:latest, hence the normalisation.
    if ! curl -sS "${ENDPOINT}/api/ps" | MODEL="${MODEL}" CTX="${ctx}" python3 -c '
import json, os, sys

def norm(name):
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"

want = norm(os.environ["MODEL"])
ctx = os.environ["CTX"]
d = json.load(sys.stdin)
matches = [m for m in d.get("models", []) if norm(m.get("name", "")) == want]
if not matches:
    print(f"ctx={ctx}: FAILED -- {want} is not resident after the load")
    sys.exit(1)
vram, size = matches[0].get("size_vram", 0), matches[0].get("size", 0)
pct = 100 * vram / size if size else 0.0
print(f"ctx={ctx}: offload {pct:.1f}% ({vram / 1e9:.2f} / {size / 1e9:.2f} GB)")
'; then
        failures=$((failures + 1))
    fi

    curl -sS "${ENDPOINT}/api/generate" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"\",\"keep_alive\":0}" \
        -o /dev/null || echo "ctx=${ctx}: warning -- unload request failed"
    sleep 2
done

if [ "${failures}" -gt 0 ]; then
    echo "${failures} of $# value(s) failed."
    exit 1
fi
