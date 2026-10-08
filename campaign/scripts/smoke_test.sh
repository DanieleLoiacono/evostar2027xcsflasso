#!/usr/bin/env bash
# End-to-end dry run on the 'smoke' profile (2 benchmarks, 2 runs, short budget):
# plan -> parity audit -> C++ runs -> Python runs -> verify -> collect -> analyze.
# Results go to results/<campaign_id>-smoke and never mix with the real campaign.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
J="${1:-2}"
bash "$HERE/04_plan.sh" --profile smoke
bash "$HERE/05_run_cxx.sh" --profile smoke -j "$J"
bash "$HERE/06_run_python.sh" --profile smoke -j "$J"
bash "$HERE/07_analyze.sh" --profile smoke
