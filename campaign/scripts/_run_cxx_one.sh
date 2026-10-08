#!/usr/bin/env bash
# One xcslib run, through xcslib's own experiment manager:
#   confsys.xcsf (generated at plan time)  ->  xcsf-rf -f xcsf  ->  standard xcslib result files.
# Called by 05_run_cxx.sh as:  _run_cxx_one.sh <campaign_dir> <run_key>
set -uo pipefail
source "$(dirname "$0")/_env.sh"
CDIR="$1"; KEY="$2"

T="$(xc cxx-prepare "$CDIR" "$KEY")" || { echo "[FAIL] $KEY: prepare" >&2; exit 1; }
(
  cd "$T" || exit 1
  # what xcslib actually parsed (verified against the intended values at finalize time)
  "$CXX_BIN" -f xcsf -p > cxx_params_print.txt 2> cxx_params_print.err
  "$CXX_BIN" -f xcsf > stdout.log 2> stderr.log
  echo $? > exit_code.txt
  gzip -f stdout.log stderr.log
)
xc cxx-finalize "$CDIR" "$KEY"
