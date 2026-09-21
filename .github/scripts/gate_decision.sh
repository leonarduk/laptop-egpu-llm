#!/usr/bin/env bash
#
# Decide the final "did the review pass" outcome from the verdict step's
# three states. A script for the same reason publish_outage_check.sh is one:
# outage-check-smoke-test.yml needs to drive this exact logic with a forced
# APPROVED=skipped, and a copy in the smoke test would only ever prove the
# copy agrees with itself.
#
# Environment:
#   APPROVED  true | false | skipped | "" (empty means the verdict step
#             never ran -- something earlier in the job failed)
#   OUTCOME   the verdict step's own outcome, used only in the empty-APPROVED
#             message
#   PROVIDER  display name, e.g. DeepSeek
#
# Exit: 0 on true/skipped (does not block merging), 1 on false or empty.

set -uo pipefail

# APPROVED empty is a legitimate input, not a caller mistake: it is exactly
# what steps.check_approval.outputs.approved is when that step never ran,
# and the *) branch below exists to handle it. So this only checks the
# variable is set (declared, however empty), not that it is non-empty --
# `:?` would have rejected the one case *) is for.
: "${APPROVED?APPROVED must be set (empty string is valid: verdict step did not run)}"
: "${PROVIDER:?PROVIDER is required}"
OUTCOME="${OUTCOME:-}"

case "$APPROVED" in
  true)
    echo "$PROVIDER review: APPROVED."
    ;;
  skipped)
    echo "::warning::$PROVIDER review did not run (provider outage). This job is green because an outage does not block merging - see the neutral '$PROVIDER review: provider outage' check."
    ;;
  false)
    echo "::error::$PROVIDER review: CHANGES REQUESTED - address the review findings before merging."
    exit 1
    ;;
  *)
    # No verdict at all: the step never ran because something earlier in the
    # job failed. Reporting that as CHANGES REQUESTED sent people looking for
    # review findings that were never written.
    echo "::error::$PROVIDER review did not complete (verdict step outcome: $OUTCOME) - there is no review to act on. Check the earlier steps in this job."
    exit 1
    ;;
esac
