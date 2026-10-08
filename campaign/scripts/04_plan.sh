#!/usr/bin/env bash
# Expand config/campaign.json into the run plan, generate every xcslib confsys file and
# check parameter parity between the two implementations (fails on any mismatch).
#
# Usage: scripts/04_plan.sh [--profile smoke|pilot] [--config path] [--root results_root]
# Re-running is safe: existing runs must keep an identical specification (otherwise it stops),
# new runs (e.g. a larger n_runs) are appended.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
xc plan "$@"
xc audit "$@"
CDIR="$(xc where "$@")"
echo "[info] campaign directory: $CDIR"
echo "[info] parity table:       $CDIR/manifest/parity_audit.md"
xc status "$@"
