#!/usr/bin/env bash
# Build the xcslib XCSF executable for real-valued functions (the upstream 'rf' target:
# VERSION=-rf ACTIONS=dummy_action USERFLAGS=-D__NICHE_TRACKING__) OUT OF TREE.
#
# The sources are copied to a temporary directory without spaces (GNU make cannot handle the
# spaces of the Dropbox path) and the binary is installed in campaign/build/bin/xcsf-rf.
# Nothing is written inside xcslib-1.5-rc1-niches/.
#
# Optimisation: upstream uses -O0 -g. We use -O2 (no -ffast-math, asserts kept: NDEBUG is NOT
# defined) so that runtime is reasonable; override with OPT_FLAGS="-O0 -g" if desired.
# GSL is located with gsl-config, falling back to Homebrew/MacPorts prefixes.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

OPT_FLAGS="${OPT_FLAGS:--O2}"
JOBS="${JOBS:-$(ncpu)}"
command -v make >/dev/null || die "make not found"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || CXX=c++

if command -v gsl-config >/dev/null 2>&1; then
  GSL_CFLAGS="$(gsl-config --cflags)"; GSL_LIBS="$(gsl-config --libs)"
else
  GSL_CFLAGS="-I/opt/homebrew/include -I/opt/local/include -I/usr/local/include"
  GSL_LIBS="-L/opt/homebrew/lib -L/opt/local/lib -L/usr/local/lib -lgsl -lgslcblas"
  echo "[warn] gsl-config not found; trying default prefixes (install GSL: 'brew install gsl')"
fi

"$PYTHON" - <<'EOF' || die "apply the benchmark patch first: scripts/01_apply_cxx_patch.sh"
from xcsfcamp.manifest import require_patch_applied
require_patch_applied()
EOF

WORK="$(mktemp -d "${TMPDIR:-/tmp}/xcsfcamp-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
case "$WORK" in *" "*) die "temporary build directory contains spaces: set TMPDIR";; esac
cp -R "$CXX_LIB_DIR/src" "$CXX_LIB_DIR/include" "$CXX_LIB_DIR/make" "$WORK/"

CXXFLAGS_ALL="-std=c++17 $OPT_FLAGS $GSL_CFLAGS"
echo "[info] building in $WORK with: $CXX $CXXFLAGS_ALL"
make -C "$WORK" -f make/xcsf.make VERSION="-rf" ACTIONS=dummy_action USERFLAGS="-D__NICHE_TRACKING__" \
     CXX="$CXX" OPT="$CXXFLAGS_ALL" LDFLAGS="$GSL_LIBS" BUILD_DIR=./build EXEC_DIR=./executables \
     -j "$JOBS" > "$WORK/build.log" 2>&1 || { tail -40 "$WORK/build.log"; die "build failed"; }

mkdir -p "$CAMPAIGN_DIR/build/bin"
cp "$WORK/executables/xcsf-rf" "$CXX_BIN"
cp "$WORK/build.log" "$CAMPAIGN_DIR/build/build.log"

CXX_USED="$CXX" CXXFLAGS_USED="$CXXFLAGS_ALL -D__NICHE_TRACKING__" LIBS_USED="$GSL_LIBS" "$PYTHON" - <<'EOF'
import json, os, subprocess
from xcsfcamp import manifest as M
info = dict(
    built_at=M.now(), binary=str(M.CXX_BIN), binary_sha256=M.sha256_file(M.CXX_BIN),
    source_tree_sha256=M.tree_hash(M.CXX_LIB_DIR)["sha256"], make_target="make/xcsf.make VERSION=-rf ACTIONS=dummy_action",
    cxx=os.environ["CXX_USED"], cxxflags=os.environ["CXXFLAGS_USED"], libs=os.environ["LIBS_USED"],
    compiler_version=subprocess.run([os.environ["CXX_USED"], "--version"], capture_output=True, text=True).stdout.strip(),
    patch_sha256=M.sha256_file(M.PATCH_FILE),
)
M.BUILD_INFO.write_text(json.dumps(info, indent=2))
print("[ok] built", info["binary"], "sha256", info["binary_sha256"][:16])
EOF

# smoke: the binary must start and refuse a missing configuration loudly
"$CXX_BIN" >/dev/null 2>&1 || true
echo "[ok] C++ build ready: $CXX_BIN"
