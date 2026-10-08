#!/usr/bin/env bash
# Apply the ONLY modification made to an upstream library:
#   patches/xcslib-benchmark-functions.patch
#   - accept the keys 'min input'/'max input' in <environment::real_functions> (they are the keys
#     set_parameters() reads, but check_parameters() rejected them, so the domain was stuck at [0,1]);
#   - add the benchmark functions sine4, abs, sincos2d, friedman5 (sine and sine3 already exist).
# xcsf_python-2.0.0 is not modified at all.
#
# Usage: scripts/01_apply_cxx_patch.sh            apply (idempotent) and print git instructions
#        scripts/01_apply_cxx_patch.sh --commit   also create the two commits required by AGENTS.md:
#              1) "Import upstream libraries (unmodified)"  (only if the repo has no commit yet)
#              2) "xcslib: benchmark functions + min/max input keys (campaign patch)"
set -euo pipefail
source "$(dirname "$0")/_env.sh"
PATCH="$CAMPAIGN_DIR/patches/xcslib-benchmark-functions.patch"
FILES=(xcslib-1.5-rc1-niches/src/environments/real_functions_env.cpp
       xcslib-1.5-rc1-niches/include/environments/real_functions_env.h)
cd "$ROOT_DIR"
command -v git >/dev/null || die "git not found"

COMMIT=0
[ "${1:-}" = "--commit" ] && COMMIT=1

if git apply --check --reverse "$PATCH" 2>/dev/null; then
  echo "[ok] patch already applied"
  APPLIED_NOW=0
else
  git apply --check "$PATCH" || die "patch does not apply cleanly: the xcslib sources differ from the audited version"
  if [ $COMMIT -eq 1 ] && ! git rev-parse --verify -q HEAD >/dev/null; then
    echo "[info] repository has no commit: committing the unmodified upstream libraries first"
    git add -- xcslib-1.5-rc1-niches xcsf_python-2.0.0 ':(exclude)**/__pycache__/**' ':(exclude)**/*.pyc'
    git commit -q -m "Import upstream libraries (unmodified): xcslib-1.5-rc1-niches, xcsf_python-2.0.0" \
      -- xcslib-1.5-rc1-niches xcsf_python-2.0.0
  fi
  git apply "$PATCH"
  echo "[ok] patch applied"
  APPLIED_NOW=1
fi

echo "[info] library diff introduced by the patch:"
git --no-pager diff --stat -- "${FILES[@]}" || true

if [ $COMMIT -eq 1 ]; then
  if git rev-parse --verify -q HEAD >/dev/null && git diff --quiet HEAD -- "${FILES[@]}"; then
    echo "[ok] patch already committed"
  else
    git add -- "${FILES[@]}"
    git commit -q -m "xcslib: benchmark functions (sine4, abs, sincos2d, friedman5) + accept 'min/max input' keys

Only library change of the campaign; see campaign/patches/xcslib-benchmark-functions.patch
and campaign/docs/PARAMETER_PARITY.md (section 'Library modification')." -- "${FILES[@]}"
    echo "[ok] committed: $(git log -1 --format='%h %s')"
  fi
else
  if [ "$APPLIED_NOW" = 1 ]; then
    echo "[next] record the modification in its own commit, e.g.:  scripts/01_apply_cxx_patch.sh --commit"
  fi
fi
