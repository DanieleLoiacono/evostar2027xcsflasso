#!/usr/bin/env bash
# Create the Python environment used by the campaign tools and by the xcsf_python runs.
# The venv lives OUTSIDE the (Dropbox) project folder by default: $HOME/.venvs/xcsfcamp
# Override with XCSFCAMP_VENV=/path/to/venv.  xcsf_python itself is NOT installed: the
# tools import it from ../xcsf_python-2.0.0/src so the upstream copy is always the one used.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

BASE_PY="${BASE_PYTHON:-python3}"
"$BASE_PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"' \
  || die "Python >= 3.10 required (set BASE_PYTHON=/path/to/python3.x)"

if [ ! -x "$XCSFCAMP_VENV/bin/python" ]; then
  echo "[info] creating venv $XCSFCAMP_VENV"
  "$BASE_PY" -m venv "$XCSFCAMP_VENV"
fi
"$XCSFCAMP_VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$XCSFCAMP_VENV/bin/python" -m pip install -r "$CAMPAIGN_DIR/requirements.txt"
"$XCSFCAMP_VENV/bin/python" - <<'EOF'
import numpy, scipy, sklearn, pandas, matplotlib
print("[ok] numpy", numpy.__version__, "| scipy", scipy.__version__, "| scikit-learn", sklearn.__version__,
      "| pandas", pandas.__version__, "| matplotlib", matplotlib.__version__)
EOF
echo "[ok] python environment ready: $XCSFCAMP_VENV"
