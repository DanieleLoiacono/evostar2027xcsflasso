#!/usr/bin/env bash
# Pre-campaign validation. MUST pass before planning/running the campaign.
#   - benchmark unit tests (hand-calculated points, float inputs, finite outputs, grid endpoints)
#   - reference plots of the target functions          -> results/_validation/*.png
#   - predictor-level parity (Python mapping == xcslib update formulas)
#   - xcslib benchmark functions == Python definitions (via xcslib's own execution trace)
#   - determinism of both implementations (same seed -> identical raw outputs)
#   - optional: upstream xcsf_python test-suite (--upstream-tests)
# Usage: scripts/03_validate.sh [--upstream-tests] [--no-cxx]
set -euo pipefail
source "$(dirname "$0")/_env.sh"
UP=0; ARGS=()
for a in "$@"; do
  case "$a" in
    --upstream-tests) UP=1;;
    *) ARGS+=("$a");;
  esac
done
xc validate ${ARGS[@]+"${ARGS[@]}"}
if [ $UP -eq 1 ]; then
  echo "[info] running the upstream xcsf_python test-suite (read-only, from its own folder)"
  ( cd "$PY_LIB_DIR" && "$PYTHON" -m pytest -q -p no:cacheprovider tests )
fi
