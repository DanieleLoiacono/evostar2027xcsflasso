# Sourced by every campaign script. Not meant to be executed directly.
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(dirname "$CAMPAIGN_DIR")"
CXX_LIB_DIR="$ROOT_DIR/xcslib-1.5-rc1-niches"
PY_LIB_DIR="$ROOT_DIR/xcsf_python-2.0.0"
XCSFCAMP_VENV="${XCSFCAMP_VENV:-$HOME/.venvs/xcsfcamp}"

if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "$XCSFCAMP_VENV/bin/python" ]; then
  PYTHON="$XCSFCAMP_VENV/bin/python"
else
  PYTHON="python3"
fi
export PYTHONPATH="$CAMPAIGN_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C

# Executables live OUTSIDE Dropbox: on macOS, executing a binary inside a synced folder can leave the
# process stuck in uninterruptible wait (state "UE", unkillable). build_info.json stays in campaign/build.
export XCSFCAMP_BIN_DIR="${XCSFCAMP_BIN_DIR:-$HOME/.xcsfcamp/bin}"
CXX_BIN="$XCSFCAMP_BIN_DIR/xcsf-rf"
PF_DRIVER_BIN="$XCSFCAMP_BIN_DIR/pf_driver"

xc() { "$PYTHON" -m xcsfcamp "$@"; }

ncpu() {
  if command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then sysctl -n hw.ncpu
  elif command -v nproc >/dev/null 2>&1; then nproc
  else echo 2; fi
}

die() { echo "[fail] $*" >&2; exit 1; }
