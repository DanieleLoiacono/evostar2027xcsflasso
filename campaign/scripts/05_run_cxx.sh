#!/usr/bin/env bash
# Run all pending xcslib (C++) runs of the campaign, in parallel, resumable.
#
# Usage: scripts/05_run_cxx.sh [--profile smoke|pilot] [-j N] [--study S] [--benchmark B] [--arm A] [--limit K]
#
# - completed runs (raw/<key>/DONE.json with the planned spec_hash) are never touched;
# - interrupted attempts (raw/<key>.tmp) are moved to raw/_incomplete/ and re-run;
# - a run is published only after its outputs pass validation (cxx-finalize);
# - any failure leaves the outputs in raw/<key>.tmp and makes the script exit non-zero.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

JOBS="$(( $(ncpu) > 1 ? $(ncpu) - 1 : 1 ))"
CARGS=(); FARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2;;
    --profile|--config|--root|--campaign-dir) CARGS+=("$1" "$2"); shift 2;;
    --study|--benchmark|--arm|--limit) FARGS+=("$1" "$2"); shift 2;;
    *) die "unknown argument $1";;
  esac
done

[ -x "$CXX_BIN" ] || die "missing $CXX_BIN: run scripts/02_build_cxx.sh"
CDIR="$(xc where ${CARGS[@]+"${CARGS[@]}"})"
[ -f "$CDIR/plan/runs.jsonl" ] || die "no plan in $CDIR: run scripts/04_plan.sh first"

LOCK="$CDIR/raw/.lock-cxx"
mkdir -p "$CDIR/raw"
mkdir "$LOCK" 2>/dev/null || die "$LOCK exists: another C++ runner is active (if not, remove that directory)"
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PENDING="$CDIR/raw/.cxx-pending.txt"
xc cxx-pending --campaign-dir "$CDIR" ${FARGS[@]+"${FARGS[@]}"} > "$PENDING"
N="$(wc -l < "$PENDING" | tr -d ' ')"
echo "[info] xcslib runs to execute: $N (jobs=$JOBS) binary=$CXX_BIN"
START=$(date +%s)
set +e
xargs -P "$JOBS" -n 1 bash "$HERE/_run_cxx_one.sh" "$CDIR" < "$PENDING"
RC=$?
set -e
echo "[info] elapsed $(( $(date +%s) - START ))s"
xc status --campaign-dir "$CDIR"
[ $RC -eq 0 ] || die "some xcslib runs failed (see messages above; outputs kept in raw/<key>.tmp)"
echo "[ok] xcslib runs complete"
