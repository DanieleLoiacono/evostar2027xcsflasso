#!/usr/bin/env bash
# Apply the ONLY modifications made to an upstream library (xcslib), in this order:
#   1. patches/xcslib-benchmark-functions.patch
#      - accept the keys 'min input'/'max input' in <environment::real_functions> (they are the keys
#        set_parameters() reads, but check_parameters() rejected them, so the domain was stuck at [0,1]);
#      - add the benchmark functions sine4, abs, sincos2d, friedman5 (sine and sine3 already exist).
#   2. patches/xcslib-rls-delta.patch
#      - add the prediction function 'rls_delta' (<prediction::rls_delta> x0, delta): recursive least
#        squares exactly as in Lanzi et al. (IlliGAL 2005012, Alg. 5): V0 = delta*I, no matrix added
#        after the update. The existing 'rls' (V0 = 0, V += I every update) is left untouched.
# xcsf_python-2.0.0 is not modified at all.
#
# Usage: scripts/01_apply_cxx_patch.sh            apply (idempotent) and print git instructions
#        scripts/01_apply_cxx_patch.sh --commit   also create one commit per patch (AGENTS.md rule 3),
#              preceded by "Import upstream libraries (unmodified)" if the repo has no commit yet
set -euo pipefail
source "$(dirname "$0")/_env.sh"
L=xcslib-1.5-rc1-niches
PATCHES=(xcslib-benchmark-functions xcslib-rls-delta)
FILES_xcslib_benchmark_functions=($L/src/environments/real_functions_env.cpp
                                  $L/include/environments/real_functions_env.h)
FILES_xcslib_rls_delta=($L/include/xcsf/pf/base.h $L/include/xcsf/pf/prediction_functions.h
                        $L/src/pf/utility.cpp $L/include/xcsf/pf/rls_delta.h $L/src/pf/rls_delta.cpp)
MSG_xcslib_benchmark_functions="xcslib: benchmark functions (sine4, abs, sincos2d, friedman5) + accept 'min/max input' keys

Campaign patch 1/2; see campaign/patches/xcslib-benchmark-functions.patch
and campaign/docs/PARAMETER_PARITY.md (section 'Library modification')."
MSG_xcslib_rls_delta="xcslib: add prediction function rls_delta (RLS with V0 = delta*I, Lanzi et al. 2005 Alg. 5)

Campaign patch 2/2; see campaign/patches/xcslib-rls-delta.patch
and campaign/docs/PARAMETER_PARITY.md (section 'Library modification')."

cd "$ROOT_DIR"
command -v git >/dev/null || die "git not found"
COMMIT=0
[ "${1:-}" = "--commit" ] && COMMIT=1

for name in "${PATCHES[@]}"; do
  PATCH="$CAMPAIGN_DIR/patches/$name.patch"
  var="${name//-/_}"
  eval "FILES=(\"\${FILES_${var}[@]}\")"
  eval "MSG=\"\${MSG_${var}}\""
  echo "== $name"
  if git apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "[ok] patch already applied"
    APPLIED_NOW=0
  else
    git apply --check "$PATCH" || die "$name does not apply cleanly: the xcslib sources differ from the audited version"
    if [ $COMMIT -eq 1 ] && ! git rev-parse --verify -q HEAD >/dev/null; then
      echo "[info] repository has no commit: committing the unmodified upstream libraries first"
      git add -- $L xcsf_python-2.0.0 ':(exclude)**/__pycache__/**' ':(exclude)**/*.pyc'
      git commit -q -m "Import upstream libraries (unmodified): xcslib-1.5-rc1-niches, xcsf_python-2.0.0" \
        -- $L xcsf_python-2.0.0
    fi
    git apply "$PATCH"
    echo "[ok] patch applied"
    APPLIED_NOW=1
  fi
  echo "[info] library diff introduced by the patch (vs HEAD):"
  git --no-pager diff --stat HEAD -- "${FILES[@]}" 2>/dev/null || true
  git ls-files --others --exclude-standard -- "${FILES[@]}" | sed 's/^/  (new, untracked) /'
  if [ $COMMIT -eq 1 ]; then
    if [ -z "$(git status --porcelain -- "${FILES[@]}")" ]; then
      echo "[ok] patch already committed"
    else
      git add -- "${FILES[@]}"
      git commit -q -m "$MSG" -- "${FILES[@]}"
      echo "[ok] committed: $(git log -1 --format='%h %s')"
    fi
  elif [ -n "$(git status --porcelain -- "${FILES[@]}")" ]; then
    echo "[next] record the modification in its own commit:  scripts/01_apply_cxx_patch.sh --commit"
  fi
done
