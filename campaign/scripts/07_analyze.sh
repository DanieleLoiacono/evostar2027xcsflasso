#!/usr/bin/env bash
# Verify raw-result integrity, regenerate derived tables from raw/, run the statistics and
# write figures + report.  Everything under derived/ is rebuilt from scratch every time.
#
# Usage: scripts/07_analyze.sh [--profile smoke|pilot] [--allow-incomplete]
set -euo pipefail
source "$(dirname "$0")/_env.sh"
CARGS=(); INC=()
while [ $# -gt 0 ]; do
  case "$1" in
    --allow-incomplete) INC=(--allow-incomplete); shift;;
    --profile|--config|--root|--campaign-dir) CARGS+=("$1" "$2"); shift 2;;
    *) die "unknown argument $1";;
  esac
done
if [ ${#INC[@]} -eq 0 ]; then
  xc verify --require-complete ${CARGS[@]+"${CARGS[@]}"}
else
  xc verify ${CARGS[@]+"${CARGS[@]}"}
fi
xc collect ${INC[@]+"${INC[@]}"} ${CARGS[@]+"${CARGS[@]}"}
xc analyze ${CARGS[@]+"${CARGS[@]}"}
CDIR="$(xc where ${CARGS[@]+"${CARGS[@]}"})"
echo "[ok] report: $CDIR/derived/report.md"
