#!/usr/bin/env bash
#
# Publish the "provider outage" check run: a completed check with conclusion
# `neutral`, attached to $SHA.
#
# Why this is a script rather than inline YAML. It is the deliverable of the
# outage path -- without it the PR's check list shows a bare green tick for a
# PR nothing reviewed -- and it is the least testable code in the repo,
# running only when a provider is actually down. Inline, it could only ever
# be exercised in production, and a bug in it is invisible until the moment
# it matters. One copy here is callable by both `_ai-pr-review.yml` and
# `outage-check-smoke-test.yml`, so the smoke test proves the real code
# rather than a lookalike.
#
# Retried because a transient failure here costs the whole signal. Failures
# are not classified: a 403 and a 502 both end in the same "could not
# publish" state, the caller's message already names the two likely causes,
# and parsing HTTP status out of `gh api` would add branching to exactly the
# code that most needs to stay readable (leonarduk/laptop-egpu-llm#29).
#
# Environment:
#   GH_TOKEN  token with checks: write on $REPO
#   REPO      owner/name
#   PROVIDER  display name, e.g. DeepSeek
#   SHA       commit to attach the check run to
#   SUMMARY   markdown body for the check run
#
# Exit: 0 published, 1 not published after retries. Prints nothing on
# success; the caller owns the annotations.

set -uo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${REPO:?REPO is required}"
: "${PROVIDER:?PROVIDER is required}"
: "${SHA:?SHA is required}"
: "${SUMMARY:?SUMMARY is required}"

for delay in 0 5 15; do
  if [ "$delay" -gt 0 ]; then
    echo "check-run POST failed, retrying in ${delay}s..." >&2
    sleep "$delay"
  fi
  if gh api -X POST "repos/$REPO/check-runs" \
    -f "name=$PROVIDER review: provider outage" \
    -f "head_sha=$SHA" \
    -f "status=completed" \
    -f "conclusion=neutral" \
    -f "output[title]=No review ran - provider unreachable" \
    -f "output[summary]=$SUMMARY" >/dev/null; then
    exit 0
  fi
done

exit 1
