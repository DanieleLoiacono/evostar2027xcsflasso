# XCSF campaign: `xcsf_python-2.0.0` vs `xcslib-1.5-rc1-niches`

Scripts and configuration to run, by hand, the controlled comparison of the two XCSF
implementations on continuous single-step function approximation, plus the separate
Python-only Lasso extension.

Everything lives in this `campaign/` folder. The two libraries are used as upstream code:
the only changes are two xcslib patches (`patches/`, see `docs/PARAMETER_PARITY.md §5`):
the benchmark functions, and the prediction function `rls_delta` (the RLS of Lanzi et al. 2005,
Alg. 5, with V0 = δI). xcsf_python is not modified.

| file | content |
|---|---|
| `config/campaign.json` | the whole design: benchmarks, XCSF parameters, predictors, studies, seeds, analysis margins, `smoke`/`pilot` profiles |
| `docs/PARAMETER_PARITY.md` | source-level audit of both libraries, the parameter mapping, and the differences that remain |
| `docs/EXPERIMENT_DESIGN.md` | questions → studies → metrics → statistical tests |
| `docs/COMPLIANCE.md` | requirement-by-requirement check against `AGENTS.md` and `BENCHMARK_FUNCTIONS.md` |
| `scripts/NN_*.sh` | the manual steps, in order |
| `xcsfcamp/` | the Python package behind the scripts (`python -m xcsfcamp --help`) |

## Prerequisites

- Python ≥ 3.10, a C++17 compiler, GNU make, git
- GSL (`brew install gsl` on macOS)
- disk: ≈ 0.3 GB for the full campaign (results default to `campaign/results/`; use
  `--root /elsewhere` on every step to keep them outside Dropbox)

## Running the campaign

All commands from the project root (the folder containing `campaign/`).

```bash
# 0. Python environment (venv in ~/.venvs/xcsfcamp, outside Dropbox)
bash campaign/scripts/00_setup_python.sh

# 1. Apply the two xcslib patches (benchmark functions; rls_delta), one commit each
#    (also commits the unmodified libraries first if the repository has no commit yet)
bash campaign/scripts/01_apply_cxx_patch.sh --commit

# 2. Build xcslib out of tree -> campaign/build/bin/xcsf-rf (+ build_info.json), plus the
#    predictor test driver campaign/build/bin/pf_driver (source: campaign/tools/pf_driver.cpp)
bash campaign/scripts/02_build_cxx.sh

# 3. Validation gate (must end with "[ok] validation complete"; planning is refused
#    until a full validation has passed for the current build, libraries and code)
#    benchmark unit tests, reference plots, predictor-level parity (Python mapping and the
#    compiled C++ prediction functions vs NumPy transcriptions), xcslib functions vs
#    Python definitions, rls_delta end-to-end run, determinism of both implementations
bash campaign/scripts/03_validate.sh --upstream-tests

# 4. Smoke test of the whole pipeline (2 benchmarks, 2 runs, short budget; ~2 min)
bash campaign/scripts/smoke_test.sh 4
#    -> campaign/results/evostar2027-v2-smoke/derived/report.md

# 5. Optional pilot (5 runs per cell) to check run times and between-run spread
bash campaign/scripts/04_plan.sh --profile pilot
bash campaign/scripts/05_run_cxx.sh --profile pilot -j 8
bash campaign/scripts/06_run_python.sh --profile pilot -j 8
bash campaign/scripts/07_analyze.sh --profile pilot

# 6. Full campaign
bash campaign/scripts/04_plan.sh                      # plan + parity audit (fails on any mismatch)
bash campaign/scripts/05_run_cxx.sh -j 8              # 600 xcslib runs (minutes)
bash campaign/scripts/06_run_python.sh --study parity -j 8   # 600 Python parity runs
bash campaign/scripts/06_run_python.sh --study pyext  -j 8   # 1200 Python-only runs (Lasso Batch dominates)
bash campaign/scripts/07_analyze.sh                   # verify -> collect -> statistics -> report

bash campaign/scripts/status.sh                       # progress at any time
```

Indicative cost (one core, 50 000 learning problems, N = 800): xcslib ≈ 0.5 s per run;
Python ≈ 15–20 s (Constant/NLMS/RLS/Lasso Online) and ≈ 80 s (Lasso Batch), measured on a
Linux test machine. Full campaign ≈ 15 CPU-hours of Python, i.e. ≈ 2 h with 8 jobs.

`-j` sets parallel runs; `--study/--benchmark/--arm/--limit` select a subset (useful to
spread the campaign over several sessions).

## Resume, immutability, failures

- A run is *published* only after its outputs are validated: files are hashed into
  `DONE.json`, the directory is renamed atomically from `raw/<key>.tmp` to `raw/<key>`, and
  its files are made read-only. Published runs are never re-run or overwritten.
- Interrupting a step is safe: rerun the same command. Half-finished attempts are moved to
  `raw/_incomplete/` (kept, not deleted) and redone.
- Re-planning with a changed configuration that would alter an already planned run stops
  with an error: use a new `campaign_id`. Increasing `n_runs` simply appends runs.
- Runs refuse to start if either library, the build, or the run-execution code
  (`xcsfcamp/{benchmarks,config,parity,python_runner,cxx,runs}.py`) changed since planning.
- Every step fails loudly: unknown/missing configuration keys, parameters without a
  semantic counterpart, parameter values that xcslib did not actually parse (each run's
  `xcsf-rf -p` output is checked), NaN/inf anywhere, missing or truncated output files,
  evaluation grids that differ from the planned one, incomplete campaigns at analysis time.
- `scripts/07_analyze.sh` first runs `verify` (all published files still match their
  hashes), then rebuilds `derived/` from scratch.

## Results layout

```
campaign/results/<campaign_id>/
  manifest/   config.resolved.json, plan_snapshot.json (environment, library hashes, patch diff),
              session-*.json (one per run session), parity_audit.{csv,md}
  plan/       runs.jsonl (every run spec + hash), cxx/<key>/confsys.xcsf
  raw/        parity/cxx/<bench>/<predictor>/run_XXXX/  xcslib standard files (statistics, avf,
                                                         population, timing, logs) + DONE.json
              parity|pyext/py/<bench>/<arm>/run_XXXX/     result.json, curve.csv, checkpoints.csv,
                                                         grid_predictions.csv.gz, population.csv.gz
  derived/    runs.csv (one row per run, all metrics), curves.csv.gz, checkpoints.csv.gz,
              grid_profiles.csv.gz, tables/*.csv, figures/*.png, report.md
```

`derived/report.md` answers Q1–Q5 with tables; `derived/tables/*.csv` hold the full
statistics (HL shifts, bootstrap CIs, Holm-adjusted p-values, A12, verdicts).

## How the two implementations are run

- **xcslib**: through its own experiment manager. For every run the plan step writes a
  `confsys.xcsf`; the runner executes `xcsf-rf -f xcsf` in the run directory and keeps the
  standard files xcslib produces (`statistics.*`, `avf.*`, `population.*`, …). No harness
  replaces or extends the experiment loop.
- **xcsf_python**: through its public scikit-learn API (`XCSFRegressor.partial_fit`,
  `predict`, `match`, `get_rules`), imported from `../xcsf_python-2.0.0/src` (not an
  installed copy), one online pass over N fresh samples.

## Changing the design

Edit `config/campaign.json` and give it a new `campaign_id`. Useful knobs: `n_runs`,
`benchmarks` (add `sin_cos_surface_2d`, `friedman5d` for the optional extensions),
`epsilon_fraction` (0.025 for the strict setting), `xcsf.*`, `benchmark_overrides`
(per-benchmark XCSF parameters, e.g. a larger population for `sinus4_1d`), predictor
variants (lists expand into separate arms, e.g. `lasso_alpha`), analysis margins.
Configuration keys are validated strictly; parameters that cannot be matched across
implementations are rejected for the parity study.

## Troubleshooting

### `02_build_cxx.sh` fails with errors in `<iostream>` / `<__locale>` / `_LIBCPP_BEGIN_NAMESPACE_STD` (macOS)

**Symptom.** Dozens of errors such as `C++ requires a type specifier for all declarations`
inside the libc++ headers of the system SDK, ending in `[fail] build failed`. The build log
shows the compiler as `x86_64-apple-darwin13.4.0-clang++`.

**Cause.** An active conda environment (`(base)`) puts an old, x86_64 cross-compiler clang
first on the `PATH`/`CXX`. It is then used with the macOS SDK, whose libc++ headers are too
recent for it. This is a toolchain mismatch, not a bug in the XCSF sources.

**Fix.** `02_build_cxx.sh` honours `$CXX` (default `g++`), so point it to Apple's compiler:

```bash
conda deactivate                 # run twice if needed, to leave (base)
which clang++                    # expected: /usr/bin/clang++
xcode-select -p                  # must print a valid path
unset CXX CC CXXFLAGS CPPFLAGS   # drop anything conda exported
CXX=/usr/bin/clang++ bash campaign/scripts/02_build_cxx.sh
```

On success `campaign/build/build_info.json` records `/usr/bin/clang++` as the compiler.

**GSL.** The warning `gsl-config not found` means GSL is not on the `PATH`. On Apple
Silicon Homebrew lives in `/opt/homebrew`:

```bash
brew install gsl
export PATH="/opt/homebrew/bin:$PATH"
gsl-config --version
```

**Alternative.** If a conda compiler is still picked up, use Homebrew's LLVM:
`brew install llvm`, then `CXX=/opt/homebrew/opt/llvm/bin/clang++ bash campaign/scripts/02_build_cxx.sh`.

**Architecture.** Do not mix the conda x86_64 toolchain with Homebrew's arm64 GSL: linking
would fail with architecture errors. Use the native arm64 `clang++` together with Homebrew GSL.
