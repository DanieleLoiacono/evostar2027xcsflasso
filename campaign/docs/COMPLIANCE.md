# Compliance with AGENTS.md and BENCHMARK_FUNCTIONS.md

Checked against the versions in the project root on 2026-10-09 (AGENTS.md with rule 16;
README listing Constant, NLMS, RLS as common predictors, RLSK no longer required); rows 1, 3 and 6
revised on 2026-10-10 after the xcsf_python refactoring (library version 2.1.0); rows 3, 5 and 7
revised the same day after the two xcslib fixes of the interval conditions (mutation, clipping).

## AGENTS.md

| # | requirement | how it is met | where |
|---|---|---|---|
| 1 | libraries are upstream code | nothing generated inside them by the campaign; C++ built out of tree; Python imported read-only from `xcsf_python-2.0.0/src` (bytecode writing disabled). xcsf_python itself was refactored in place on the owner's request: see row 3 | `scripts/02_build_cxx.sh`, `xcsfcamp/cli.py` |
| 2 | thin adapters outside the libraries | everything lives in `campaign/` | — |
| 3 | library changes only when necessary, documented, minimal, separate commit, diff in manifest | four xcslib patches: (3) the fix of the fixed/gaussian mutation of interval conditions, which assigned the lower bound to the upper one (requested explicitly; one file, 6 lines added, 4 removed); (4) no clipping of conditions to the domain after crossover when `bounded = off` (requested explicitly; three lines added in one function); (1) benchmark functions + the `min/max input` keys without which no benchmark domain can be set; (2) the new prediction function `rls_delta` (paper's RLS, requested explicitly: xcslib `rls` ignores δ and adds I to V at every update), new files plus 3 registration lines, existing predictors untouched. `01_apply_cxx_patch.sh --commit` makes one commit per patch; patch texts and `git diff` stored in `manifest/plan_snapshot.json`. **xcsf_python**: one refactoring requested explicitly by the owner (modular structure, single RLS, recursive online Lasso; library version 2.1.0) — reasons and campaign-level effects documented, one commit touching only the library, exact diff against the upstream import written to `manifest/xcsf_python-vs-upstream.diff` with its hash in every snapshot. It is not minimal by nature (a refactoring); the unchanged predictors and the XCSF cycle are verified bit for bit against 2.0.0 | `patches/`, `docs/PARAMETER_PARITY.md §5` |
| 4 | float inputs, never discretised | uniform float64 sampling in both libraries; checked by `validate` on Python samples and on xcslib's own execution trace | `xcsfcamp/benchmarks.py`, `xcsfcamp/validation.py` |
| 5 | identical non-prediction XCSF parameters | single resolved spec translated by `parity.py`; per-campaign audit table fails on any mismatch; each C++ run checks what xcslib actually parsed (`xcsf-rf -p`) ; the condition mutation operator and the handling of unbounded conditions after crossover, which differed because of two xcslib defects, are now the same in both (patches 3 and 4) | `manifest/parity_audit.md` |
| 6 | matched predictor hyper-parameters (Constant, NLMS, RLS) | η, x0 matched; xcslib RLS semantics (V0 = 0, +I per update) reproduced exactly by the single Python `rls` (δ = 0, Q = 1); paper RLS `rls_delta` (x0, δ matched) = Python `rls` with Q = 0; all verified numerically, with predictors built through the campaign mapping, against the formulas and, through `pf_driver`, against the compiled C++ code | `docs/PARAMETER_PARITY.md §3` |
| 7 | names ≠ semantics | source-level audit; 6 residual differences documented (D2, D4–D8); D1 (xcslib mutation bug) and D3 (clipping with `bounded = off`) removed by patches | `docs/PARAMETER_PARITY.md §4` |
| 8 | no independent tuning | one configuration for both; library defaults scaled to the domain | `config/campaign.json` |
| 9 | same benchmarks, streams, evaluation sets, run ids, seeds when possible | same definitions (checked), bit-identical evaluation grid (checked), same run ids and seeds. **Training streams cannot be shared**: xcslib draws inputs from its internal RNG and its experiment manager has no way to read them from a file without modifying the library | `docs/PARAMETER_PARITY.md` D8 |
| 10 | Lasso separate | `pyext` study (`lasso_online`, `lasso_sgd`, `lasso_batch`), own report section, never pooled; config refuses non-matchable predictors in a study with C++ | `config/campaign.json`, `xcsfcamp/config.py` |
| 11 | reproducible from scripts + config | numbered scripts, plan with per-run spec hashes, determinism checked for both implementations | `scripts/`, `plan/runs.jsonl` |
| 12 | raw immutable, derived regenerated | published runs hashed (`DONE.json`) and read-only; `verify` before analysis; `derived/` rebuilt from scratch | `xcsfcamp/runs.py`, `xcsfcamp/collect.py` |
| 13 | resume without silent overwrite | completed runs skipped, interrupted attempts quarantined to `raw/_incomplete/`, changed specs refused | `xcsfcamp/runs.py` |
| 14 | fail loudly | strict config keys, parity errors, xcslib-parsed-parameter check, NaN/inf checks, output completeness, grid identity, incomplete campaign at analysis | throughout |
| 15 | C++ via confsys + standard xcslib result files, no wrapper adding experiments | one generated `confsys.xcsf` per run, run by `xcsf-rf -f xcsf`, xcslib's own `experiment_mgr2` writes `statistics/avf/population/...`; the shell loop only launches the binary | `scripts/05_run_cxx.sh`, `scripts/_run_cxx_one.sh` |
| 16 | manual execution, scripted and documented | step-by-step scripts and README | `campaign/README.md` |

### Interpretation choice on rule 15 (to confirm)

Each xcslib run is a separate process with `number of experiments = 1` and
`first experiment = run_id`. This gives every run its own seed (`seed_base + run_id`,
shared with Python, rule 9) and per-run resume (rule 13).

The alternative reading of rule 15 is one confsys per (benchmark, predictor) with
`number of experiments = 30`, letting xcslib loop over the replicates itself. In that mode
xcslib seeds its generator once and the 30 experiments consume one continuous stream, so
the per-run seed schedule is lost and an interruption forces the whole cell to restart.
Switching is a small change to `parity.py`/`05_run_cxx.sh` if you prefer it.

## BENCHMARK_FUNCTIONS.md

| item | status |
|---|---|
| F1–F5 core, F6–F7 optional (off by default) | definitions in `benchmarks.py`; xcslib: existing `sine`/`sine3` + patched `sine4`, `abs`, `sincos2d`, `friedman5` |
| real, continuous sampling | yes (see rule 4) |
| ε0 = fraction × (max f − min f), 0.05 primary, 0.025 optional | `epsilon_fraction` in the config; ranges analytic (F1, F2, F5, F6, F7) or dense grid + local refinement (F3, F4); numeric ε0 stored in every resolved run spec (`xcsf.epsilon0`) |
| unit tests: hand-calculated points, float preserved, finite outputs, grid endpoints | `validate` (`test_benchmarks`) |
| reference plots before any XCSF experiment | written by `03_validate.sh`; **`04_plan.sh` refuses to plan without a full validation stamp** matching the current build, libraries and execution code, and copies the plots and log into the campaign manifest |
