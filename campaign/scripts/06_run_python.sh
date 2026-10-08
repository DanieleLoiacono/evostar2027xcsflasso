#!/usr/bin/env bash
# Run all pending xcsf_python runs of the campaign (parity study + Python-only extension),
# in parallel, resumable.
#
# Usage: scripts/06_run_python.sh [--profile smoke|pilot] [-j N] [--study S] [--benchmark B] [--arm A] [--limit K]
#   e.g. only the parity study first:   scripts/06_run_python.sh --study parity
set -euo pipefail
source "$(dirname "$0")/_env.sh"
JOBS="$(( $(ncpu) > 1 ? $(ncpu) - 1 : 1 ))"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2;;
    --break-lock) ARGS+=("$1"); shift;;
    --profile|--config|--root|--campaign-dir|--study|--benchmark|--arm|--limit) ARGS+=("$1" "$2"); shift 2;;
    *) die "unknown argument $1";;
  esac
done
xc run-python --jobs "$JOBS" ${ARGS[@]+"${ARGS[@]}"}
