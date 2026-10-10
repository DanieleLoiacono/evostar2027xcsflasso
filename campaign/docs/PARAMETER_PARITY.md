# Parameter parity audit — xcslib-1.5-rc1-niches vs xcsf_python-2.0.0 (library version 2.1.0)

This document records what was verified **in the source code** of both libraries
(AGENTS.md rule 7: equal names do not imply equal semantics), how every parameter
is mapped, and which differences remain because removing them would require
modifying a library. The mapping is implemented in `xcsfcamp/parity.py`; the
generated per-campaign table is `results/<id>/manifest/parity_audit.md`
(the plan step fails if any row is not equal).

## 1. Learning protocol

| aspect | xcslib (`experiment_mgr2`, `xcsf_classifier_system`) | xcsf_python (`XCSFRegressor`, `classifier_system.py`) | campaign setting |
|---|---|---|---|
| problem type | single-step, one dummy action; reward = f(x) | supervised regression, single implicit action | same |
| training | alternates 1 learning problem (update + GA) and 1 test problem | every sample is an update (+GA) | N learning problems ↔ N samples, one pass, `shuffle=False` |
| learning during test problems | `update during test` (**default on!**) | `predict` never learns | **off** in xcslib |
| time used by θ_GA | `total_steps` (learning problems only) | `time` (updates only) | identical meaning |
| online error | rolling mean of \|P − f(x)\| on test problems (covering allowed, no update), `statistics rolling window size` | `performance_history_`: \|y − ŷ\| of the current model on each incoming sample *before* it is learned (after covering), windows of `history_interval` | window = 100 in both; same estimator, different random points |
| final evaluation | `avf.*` file: prediction on the grid `min, min+res, …` (float accumulation, last coordinate fastest) | `predict` on the same grid (replicated bit-for-bit in `benchmarks.Benchmark.grid`) | identical points, checked at publication time |
| uncovered evaluation points | covering inside `predict` (new classifier) | `unmatched="nearest"` (no covering) | reported: `unmatched_grid_frac` (Python) |
| inputs | `xcs_random::random()` (mt19937_64) inside the environment | `numpy` PCG64 stream, `SeedSequence([seed, dim])` | same distribution, same seed integer, **different streams** (sharing the stream would need a library change) |
| seed | `<random> seed` (0 = clock! never used) | `random_state` | `seed = seed_base + run_id` in both |
| runs | one process per run, `first experiment = run_id`, `number of experiments = 1` | one estimator per run | same run ids |

## 2. XCSF parameters (must be semantically identical)

| semantic parameter | xcslib key | xcsf_python | check |
|---|---|---|---|
| N (micro-classifiers) | `population size` | `population_size` | equal |
| ε0 | `epsilon zero` | `epsilon_0` | equal; ε0 = 5 % of the output range |
| β (error, fitness, niche size) | `learning rate` (classifier_system) | `learning_rate` | equal |
| α, ν | `alpha`, `vi` | `alpha`, `nu` | equal |
| θ_GA | `theta GA` | `theta_ga` | equal (numerosity-weighted mean timestamp in both) |
| χ, μ | `crossover probability`, `mutation probability` | same names | equal (μ per endpoint in both, `fixed` mutation) |
| r0 | `r0` (condition) | `cover_radius` (`normalize=False`) | equal, raw units: cover `[x−U(0,r0), x+U(0,r0)]` in both |
| m0 | `m0` | `mutation_scale` | equal, `U(−m0, m0)` per endpoint in both; endpoints mutate independently and are swapped if they cross (xcslib: since patch §5.4) |
| crossover | `crossover = one-point` | `crossover="one_point"` | same operator (whole interval or upper endpoint swapped) |
| θ_del, δ | `theta delete`, δ **hard-coded 0.1** | `theta_delete`, `delta` | equal; config is rejected if δ ≠ 0.1 |
| θ_sub, GA subsumption | `theta GA sub`, `GA subsumption` + `GA subsumption on [A]` | `theta_subsume`, `ga_subsumption` (parents, then niche) | equal |
| action-set subsumption | `AS subsumption = off` (implemented, never called by `step()`) | `match_subsumption=False` | equal (off) |
| selection | `offspring selection for GA = roulette-wheel` | `selection="roulette"` | equal |
| F_I, ε_I, niche size init | `fitness init`, `error init`, `set size init` | `initial_fitness`, `initial_error`, (fixed 1.0) | equal |
| MAM | `use MAM` read but **not settable** (rejected by `check_parameters`), default on | `use_mam=True` | equal; config rejected if `use_mam` is false |
| error before predictor update | `update error first = on` | `error_before_prediction=True` | equal |
| bounded conditions | `bounded = off` | `bounded=False` | equal: conditions are never clipped to the domain (xcslib: since patch §5.5) |
| condensation | 0 problems | 0 epochs | not used (no 1:1 schedule) |

## 3. Common predictors (predictor-specific hyper-parameters)

| predictor | xcslib | xcsf_python | mapping and evidence |
|---|---|---|---|
| Constant | `prediction function = value`, `<prediction::value> learning rate = η`: `w ← w + η (y − w)` | `prediction="constant"`, `prediction_learning_rate=η`, `initial_prediction=0` | identical update (no MAM on the predictor in either) |
| NLMS | `nlms`, `learning rate = η`, `x0`: `w ← w + η e φ / (x0² + Σx²)` with φ = [x0, x] | `prediction="nlms"`, `prediction_learning_rate=η`, `x0` | identical update |
| RLS | `rls`, `x0`. **`delta` is a static never read from confsys → V0 = 0·I**, and the update adds **I** every time: V ← V − K φᵀ V + I | `prediction="rls"`, `rls_delta=0`, `process_noise=1`, `forgetting_factor=1`, `kalman_noise=False` | identical recursion. V0 = 0 is exact in both: the gain of the first update is zero (the first observation is ignored), then V = I |
| RLS (paper) | `rls_delta` (added by `patches/xcslib-rls-delta.patch`), `x0`, `delta`: V0 = δI, e = y − wᵀφ, β = 1 + φᵀVφ, V ← V − (Vφ)(Vφ)ᵀ/β, w ← w + (Vφ/β)·e — Lanzi et al. 2005, Alg. 5 | `prediction="rls"`, `rls_delta=δ`, `process_noise=0`, `forgetting_factor=1`, `kalman_noise=False` | identical recursion, different numerical form (see below). Offspring: weights inherited, V restarted from δI in both |
| RLSK | `rlsk` exists but `init_prediction_functions` registers it as `PREDICTION_RLS` → selecting it aborts ("function not available"); moreover `x0` is never read and Q multiplies V instead of being added | `rls` with `process_noise`/`kalman_noise` | excluded from the parity study (not fixed: `rls_delta` covers the paper's RLS) |

**One Python RLS.** Since library version 2.1.0 xcsf_python has a single RLS predictor
(`prediction="rls"`): both rows above are *settings* of it, not different implementations. It
propagates a triangular factor R of the inverse covariance (V⁻¹ = RᵀR) with orthogonal
transformations instead of updating V by subtraction; in exact arithmetic it computes the
recursions written in the xcslib column (theory: `xcsf_python-2.0.0/docs/prediction-updates.md`, §5).
The former Python `rls_standard` arm (QR form of the paper's RLS, Python-only) is therefore the
Python `rls_delta` arm and no longer exists as a separate predictor.

`validate` checks these mappings numerically. The predictor is built exactly as in a run
(`parity.py_params` → `xcsf.prediction.PredictorFactory`) and must reproduce a NumPy
transcription of `value.cpp`, `nlms.cpp`, `rls.cpp` and `rls_delta.cpp` to ≤ 1e-6 relative
error on 500-sample sequences, also across an offspring/clone (measured: ≤ 2e-15 for Constant,
NLMS and RLS; ≤ 7e-9 for `rls_delta`, where the difference is the rounding of the covariance-form
transcription). The compiled C++ prediction functions (`nlms`, `rls`, `rls_delta`) are
driven sample by sample by `~/.xcsfcamp/bin/pf_driver` and must match the same
transcriptions to ≤ 1e-9; a short `xcsf-rf` run checks that `rls_delta` and its `delta` are
actually selected and parsed (every published `rls_delta` run is checked the same way).

**Numerical form of RLS.** On raw inputs x ≈ 1000 with x0 = 1 the autocorrelation matrix of a
narrow classifier is very ill-conditioned (condition number ≈ 1e10 and above). xcslib updates
the covariance by subtraction; Python works on the triangular factor, whose conditioning is the
square root. Against an exact rational-arithmetic reference (x uniform in [1000, 1001],
3000 samples, δ = 1e6) the Python weights are correct to a relative error ≤ 1e-10 (library test
`test_rls_is_accurate_on_shifted_narrow_inputs_where_covariance_updates_degrade`). This is a
difference of rounding between two forms of the same estimator, not of semantics; its effect,
if any, is part of what the parity study measures on `sine_shifted_1d` and `abs_mix_1d`.

## 3b. Python-only predictors (study `pyext`, never compared with xcslib)

| arm | xcsf_python | parameters in `campaign.json` |
|---|---|---|
| `lasso_online` | `prediction="lasso_online"`: exact Lasso on the RLS statistics of the rule (recursive, no stored samples, no learning rate) | `lasso_alpha` → `lasso_alpha`, `delta` → `rls_delta`, `forgetting_factor`, `max_iter` → `lasso_max_iter`, `tol` → `lasso_tol`, `x0` |
| `lasso_sgd` | `prediction="lasso_sgd"`: proximal stochastic gradient (the `lasso_online` of library 2.0), first-order reference | `eta` → `prediction_learning_rate`, `lasso_alpha`, `learning_rate_decay` → `lasso_learning_rate_decay`, `x0` |
| `lasso_batch` | `prediction="lasso_batch"`: coordinate descent on a FIFO window | `lasso_alpha`, `window` → `lasso_window`, `max_iter`, `tol`, `x0` |

`lasso_online` shares δ and the forgetting factor with `rls_delta` (δ = 1000, λ = 1), so with
`lasso_alpha → 0` the two arms coincide. `validate` builds each configured Lasso arm through the
same mapping and checks that it satisfies the optimality (KKT) conditions of its documented
objective (`lasso_online`, `lasso_batch`) or reproduces its update rule (`lasso_sgd`).

In every xcslib run the `<prediction::nlms>` section must be present (the binary
refuses to start otherwise); for Constant and RLS runs it is inert and marked as such in
the generated confsys.

## 4. Remaining semantic differences (measured, not removed)

They are properties of the implementations being compared; the campaign quantifies
their effect instead of hiding it. None can be aligned through configuration.

- **D1 – xcslib `fixed`/`gaussian` mutation bug: removed** by
  `patches/xcslib-interval-mutation-fix.patch` (§5.4) from campaign v3 on. Until v2 the
  upper endpoint was set with `set_upper_bound(lower)`: whenever the upper bound was
  mutated it became the (possibly mutated) *lower* bound, i.e. a zero-width interval
  `(l, l]` that matches nothing. `n_degenerate_rules` is still reported per run and is
  now expected to be 0 in both implementations.
- **D2 – interval semantics.** xcslib matches `lower < x ≤ upper` (unbounded); Python
  uses the closed interval. Measure-zero for continuous inputs, but grid points equal to
  a rule bound (e.g. the domain minimum) may be treated differently.
- **D3 – clipping after crossover with `bounded = off`: removed** by
  `patches/xcslib-crossover-clipping-fix.patch` (§5.5) from campaign v4 on. Until v3 xcslib
  clipped a condition to `[min input, max input]` after a crossover that swapped single
  bounds (`check()`), whatever the value of `bounded`; Python never clips with
  `bounded=False`.
- **D4 – covering when the population is full.** xcslib inserts then deletes (the new
  rule may be deleted immediately and covering repeats); Python deletes first.
- **D5 – offspring bookkeeping.** Same values (error, fitness·0.1, niche size, averaged on
  crossover) but deletion happens after both children (Python: in one call, xcslib: two
  calls); probabilistically equivalent.
- **D6 – uncovered evaluation points.** xcslib covers during evaluation (prediction of a
  fresh rule = 0 / x0-weighted zero weights); Python extrapolates from the nearest rule.
  `unmatched_grid_frac` is ≈ 0 after training on 1-D problems.
- **D7 – xcslib `value_pf` initial value is uninitialised** (undefined behaviour; in
  practice 0). Published runs are rejected if a constant prediction is absurd.
- **D8 – random streams.** Same seeds, different generators and different consumption
  order: runs are *replicates*, not paired trajectories. Cross-implementation tests are
  therefore unpaired.

## 5. Library modifications

xcslib: four patches, applied in order by `scripts/01_apply_cxx_patch.sh` and committed separately
(one commit each) with `--commit` (§5.1, §5.2, §5.4, §5.5). xcsf_python: one refactoring, committed
separately (§5.3).

### 5.1 `campaign/patches/xcslib-benchmark-functions.patch`

1. `real_functions_env.cpp`: add `"min input"`, `"max input"` to
   `configuration_parameters`. **Reason:** `set_parameters()` reads exactly these keys,
   but `check_parameters()` rejected them (and silently ignored `min value`/`max value`),
   so the input domain was fixed to [0, 1] and none of the benchmark domains could be set.
2. `real_functions_env.{h,cpp}`: add the functions `sine4`, `abs` (F4, F5, paper-derived)
   and `sincos2d`, `friedman5` (optional F6, F7). `sine` and `sine3` were already present
   (`scale factor = 100` gives F1–F3 exactly).

### 5.2 `campaign/patches/xcslib-rls-delta.patch`

**Reason:** the paper's RLS (V0 = δI, no matrix added after the update) is not available in
xcslib: `rls` never reads `delta` (V0 = 0) and adds I to V at every update (a Kalman filter with
process noise I, whose gain does not decay and depends on the input scale), and `rlsk` cannot be
selected (registered as `PREDICTION_RLS`) and never reads `x0`. Rather than changing the
behaviour of existing predictors, the patch adds a new one:

1. new files `include/xcsf/pf/rls_delta.h`, `src/pf/rls_delta.cpp` (`rls_delta_pf`, section
   `<prediction::rls_delta>` with required keys `x0` and `delta > 0`; logs the parsed values on
   stderr; supports `degree`; same print format and clone policy as `rls_pf`);
2. registration: enum value + name `rls_delta` in `pf/base.h`, include in
   `pf/prediction_functions.h`, two blocks in `pf/utility.cpp`.

The exact diffs are stored in every campaign manifest (`manifest/plan_snapshot.json →
patch.patch_text`) together with the hash of both library trees; runs refuse to start if
either tree changes after planning.

### 5.3 xcsf_python refactoring (library version 2.0.0 → 2.1.0)

Requested explicitly by the project owner (2026-10-10); it is not required by the parity study.

**Reasons.** (1) Modularity: conditions, actions, environments, prediction functions, classifier,
classifier system and experiment loop were separated as in xcslib. (2) The online Lasso of 2.0
(proximal stochastic gradient) converged at the speed of LMS: in the v2 pilot it reached ε0 in
≈ 5000–8000 learning steps against ≈ 700–1300 for RLS and Lasso Batch, so a recursive Lasso that
solves the L1 problem exactly on RLS statistics was added. (3) The two RLS implementations
(`rls`, QR form; `rlsk`, covariance form) were merged into one numerically stable predictor that
covers both.

**What changed for the campaign.**

| | library 2.0.0 (campaign v2) | library 2.1.0 (campaign v3) |
|---|---|---|
| Constant, NLMS | `constant`, `nlms` | unchanged: same populations, error curves and predictions, bit for bit, for the same seed |
| RLS (xcslib semantics) | `rlsk`, `rls_delta=1e-12`, `process_noise=1` | `rls`, `rls_delta=0`, `process_noise=1` (exact V0 = 0) |
| RLS (paper) | `rlsk`, `rls_delta=δ`, `process_noise=0` (Joseph covariance form) | `rls`, `rls_delta=δ`, `process_noise=0` (square-root information form) |
| `rls_standard` (pyext) | `rls` (QR) | removed: identical to `rls_delta` |
| `lasso_online` (pyext) | proximal stochastic gradient | recursive Lasso on RLS statistics |
| `lasso_sgd` (pyext) | – | the proximal stochastic gradient of 2.0, kept as reference |
| `lasso_batch` (pyext) | windowed coordinate descent | unchanged, bit for bit |

The XCSF learning cycle (matching, covering, update order, fitness, GA, subsumption, deletion,
random-number consumption) is unchanged: twelve configurations run with both versions give
identical populations and statistics (`xcsf_python-2.0.0/docs/validation.md`).

**Record.** The folder keeps its upstream name; `xcsf.__version__` is 2.1.0 and is checked before
planning and running. The modification is one commit touching only `xcsf_python-2.0.0/`. The exact
diff against the upstream import (commit `5e04df7`, `manifest.PY_LIB_UPSTREAM_COMMIT`) is written
to `manifest/xcsf_python-vs-upstream.diff` when a campaign is planned, and its SHA-256 is stored
in `plan_snapshot.json → xcsf_python_changes` and in every session snapshot.
Results of campaign v2 were produced with library 2.0.0 and are not comparable run by run for the
RLS and Lasso arms: this configuration has a new `campaign_id` (`evostar2027-v3`).

### 5.4 `campaign/patches/xcslib-interval-mutation-fix.patch`

Requested explicitly by the project owner (2026-10-10).

**Reason.** In `real_interval_condition.cpp`, `fixed_mutation` and `gaussian_mutation` ended the
mutation of the upper bound with `value[i].set_upper_bound(lower)`: the upper bound received the
value of the *lower* one, so every such mutation produced a zero-width interval `(l, l]` that
can never match. These rules are never updated, so they keep their inherited statistics and
occupy part of the population: in the v2 pilot the final xcslib populations contained a median
of 16–19 zero-width rules out of ≈ 70 macroclassifiers for Constant, NLMS and RLS (5 of 32 for
`rls_delta`), against 0 in xcsf_python. This was the difference D1 of §4.

**Change.** The two bounds are mutated independently, as before with the same random draws in
the same order, and the interval is then assigned as `[min(lower, upper), max(lower, upper)]`:
if the mutated bounds cross they are swapped, the repair that `check()` documents for the other
operators and that xcsf_python applies. (Assigning the bounds one at a time through
`interval::set_*_bound` would instead collapse a crossed interval to zero width.) No clipping is
added. `gaussian_mutation` had the same line and gets the same correction, although
`mutate()` does not dispatch it. One file, 6 lines added and 4 removed.

**Effect.** Short `xcsf-rf` runs (3000 learning problems, 3 seeds, 2 benchmarks) ended with 2–4
zero-width rules out of ≈ 20 before the patch and with none after it; `validate` repeats this
check on the binary. xcslib results are not comparable with campaign v2, where the bug was part
of what the parity study measured.

### 5.5 `campaign/patches/xcslib-crossover-clipping-fix.patch`

Requested explicitly by the project owner (2026-10-10).

**Reason.** `real_interval_condition::check()` is the repair called after the crossovers that
swap a single bound (always in the uniform crossover; at the crossover points of the one- and
two-point crossovers when the lower or the upper bound alone is exchanged) and after the
proportional mutation. Besides sorting the bounds it clipped the interval to
`[min input, max input]` unconditionally, i.e. also with `bounded = off`, where covering and the
fixed mutation are free to place bounds outside the domain. With `bounded = off` this was

- inconsistent within xcslib: the same condition was clipped or not depending on which operator
  had produced it;
- harmful at the lower limit: unbounded matching is `lower < x ≤ upper`, so a rule whose lower
  bound has been clipped to exactly `min input` no longer matches `x = min input` (the special
  case for the domain minimum exists only in the bounded matching), which is the first point of
  the evaluation grid;
- different from xcsf_python, which never clips with `bounded=False` (difference D3 of §4).

In the v3 pilot (20 xcslib runs per benchmark) the final populations had a median of 1–2 rules
per run with the lower bound exactly at `min input` and 1–2 with the upper bound exactly at
`max input` (up to 5), out of ≈ 55–80 macroclassifiers.

**Change.** `check()` returns after sorting when `bounded` is off: three lines added, none
removed, in one function. With `bounded = on` nothing changes. Random draws are untouched.

**Effect.** Eight short `xcsf-rf` runs (3000 learning problems, `bounded = off`) ended with 22
bounds clipped to the domain limits out of 180 rules before the patch and with none after it;
`validate` repeats this check on the binary together with the one of §5.4. xcslib results are
not comparable with campaign v3 (pilot only), hence the new `campaign_id` `evostar2027-v4`.

## 6. Build

`scripts/02_build_cxx.sh` builds the upstream `rf` target (`make/xcsf.make VERSION=-rf
ACTIONS=dummy_action USERFLAGS=-D__NICHE_TRACKING__`) out of tree with `-O2` instead of
`-O0 -g` (no `-ffast-math`, assertions kept). Flags, compiler and binary hash are in
`campaign/build/build_info.json` and in each published run's `DONE.json`. The same object
files (all but `xcsf_main`) are linked with `campaign/tools/pf_driver.cpp` into
`~/.xcsfcamp/bin/pf_driver`, the predictor-level test driver used by `validate`.
