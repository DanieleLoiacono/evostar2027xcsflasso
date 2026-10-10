# Experiment design

## Questions → studies → measurements

| question | study | primary measurement | analysis |
|---|---|---|---|
| Q1 same configuration + same predictor ⇒ comparable approximation quality? | `parity` (C++ and Python) | grid MAE / ε0 | Hodges–Lehmann shift (Python − C++) + bootstrap CI, Mann–Whitney (Holm across cells), A12 |
| Q2 systematic differences in convergence, final error, model complexity, stability? | `parity` | online-MAE AUC, steps to ε0, final online MAE, grid RMSE / max error, #macroclassifiers, #coefficients, rule width, dispersion across seeds | same tests per metric; direction counts; Brown–Forsythe for dispersion; median curves |
| Q3 Constant vs NLMS vs RLS under the same protocol | `parity`, within each implementation | grid MAE / ε0, #macroclassifiers, online-MAE AUC | Friedman + Wilcoxon (Python, paired by seed), Kruskal–Wallis + Mann–Whitney (C++), Holm; ranking agreement (Kendall τ) |
| Q4 Lasso Batch / Online accuracy–compactness trade-offs | `pyext` (Python only, separate) | grid MAE / ε0 vs #non-zero coefficients | paired Wilcoxon vs each baseline, HL shift + CI, equivalence margins, Pareto front |
| Q5 practically important, not just detectable? | all | — | every comparison gets a verdict combining the test with a TOST-style equivalence check (90% CI inside a pre-registered margin) |

## Factors

- **Implementation**: xcslib (C++) vs xcsf_python — parity study only.
- **Benchmark** (core, 1-D, real-valued): `sine_low_1d` [0,100], `sine_shifted_1d`
  [1000,1100], `sinus3_1d`, `sinus4_1d`, `abs_mix_1d` [950,1050]. Optional F6/F7 exist
  (patched into xcslib too) but are off by default.
- **Benchmarks per study**: parity uses all five; pyext omits `sine_shifted_1d`, which after
  domain scaling is the same problem as `sine_low_1d` (same training stream, same results).
- **Predictor**: parity = Constant, NLMS, RLS (xcslib semantics: V0 = 0, V += I per update) and
  `rls_delta` (RLS of Lanzi et al. 2005, Alg. 5: V0 = δI, δ = 1000; xcslib `rls_delta` vs the
  Python `rls` with the same δ and Q = 0); pyext = Constant, NLMS, RLS, `rls_delta` + three
  Lasso predictors, each at penalty `lasso_alpha` ∈ {0.001, 0.01, 0.1}:
  `lasso_online` (recursive Lasso: exact L1 solution on the RLS statistics of the rule, same
  δ = 1000 and no forgetting as `rls_delta`), `lasso_batch` (coordinate descent on a window of
  256 samples) and `lasso_sgd` (first-order proximal gradient, η = 0.2: the online Lasso of the
  previous library version, kept as a reference for convergence speed).
- **Replicates**: 30 runs per cell, seeds `seed_base + run_id`.

Fixed: N = 800, 50 000 learning problems, ε0 = 0.05 × output range, β = η = 0.2, α = 0.1,
ν = 5, θ_GA = 25, χ = 0.8, μ = 0.04, r0 = m0 = 0.2 × domain width, θ_del = 20, δ = 0.1,
θ_sub = 20, GA subsumption on, MAM on, x0 = 1. Not tuned for either implementation
(AGENTS.md rule 8): defaults of both libraries' documentation, scaled to the domain.
Change them in `config/campaign.json` **before** planning; a different configuration
needs a new `campaign_id`.

## Input representation

xcslib has no input scaling, so the parity study uses raw inputs in both libraries
(`normalize=False`); r0 and m0 are expressed in raw units. The shifted-domain benchmarks
deliberately stress NLMS/RLS with large inputs (x ≈ 1000, x0 = 1).

The Python-only study uses z = (x − min)/(max − min) for *all* its arms, because the
first-order Lasso (`lasso_sgd`, an LMS-type step) diverges on x ≈ 1000 and because the L1
penalty is scale-dependent. Its baselines are re-run in that representation; Constant is representation-invariant and acts as a built-in check
(identical results in both studies for the same seed).

## Training stream, evaluation set, seeds

- Training inputs: x = min + U(0,1)·width, float64, U from `SeedSequence([seed, dim])`.
  For a given seed all Python arms see the same stream (paired comparisons); F1 and F2 see
  the same relative positions. xcslib draws from its own seeded mt19937_64 (D8 in
  PARAMETER_PARITY.md) — same distribution, different stream.
- Evaluation: the deterministic xcslib grid (step = width/1000 → 1001 points in 1-D),
  replicated bit-for-bit in Python; C++ outputs are checked point by point.
- Online curve: window of 100; Python's pre-update error ≡ xcslib's test-problem error.
- Python-only: grid error at 10 checkpoints during training.

## Metrics (per run, identical code for both implementations: `xcsfcamp/metrics.py`)

`grid_mae_eps`, `grid_rmse_eps`, `grid_maxae_eps`, `frac_within_eps`,
`final_online_mae_eps` (last 10 % of windows), `auc_online_mae_eps` (mean over the run),
`steps_to_eps` (first time the online MAE stays ≤ ε0 for 5 windows; censored at N, with
`converged` flag), `late_online_std_eps`, `n_macro`, `n_micro`, `n_params`
(non-zero coefficients), `frac_zero_slopes`, `mean_width_frac`, `n_degenerate_rules`,
`runtime_s` (descriptive only).

## Practical-importance margins (pre-registered in `config/campaign.json → analysis`)

- error metrics (in ε0 units): ±0.1 ε0
- size / speed metrics: ±10 % of the C++ (parity) or baseline (pyext) median
- fractions: ±0.05

Verdicts: *equivalent* (90 % CI inside the margin, no significant difference), *different
but negligible*, *practically different* (significant and 95 % CI beyond the margin),
*different, relevance uncertain*, *inconclusive*.

## Sample size

30 runs per cell gives ≈ 0.8 power for a Mann–Whitney test to detect A12 ≈ 0.71
(a "medium" effect) at α = 0.05 before multiplicity correction; equivalence at ±0.1 ε0
requires the between-run spread of grid MAE to be well below ε0, which the pilot profile
(5 runs) lets you check before committing to the full campaign.

## Note on the Lasso penalty scale

`lasso_alpha` penalises slopes in the units of (input × target). With domain-scaled inputs and
targets of amplitude ≈ 100, a slope of a rule of half-width h is set to zero when its
least-squares value is below 3·`lasso_alpha`/h² (see `xcsf_python-2.0.0/docs/prediction-updates.md`,
§6.1): for the configured values this happens only for very narrow rules or near the extrema of
the target. On these 1-D benchmarks, with a single relevant input, the Lasso arms are therefore
expected to behave like their unpenalised counterparts (`lasso_online` ≈ `rls_delta`); the v2
pilot shows a median `frac_zero_slopes` of at most 0.06 for every Lasso arm. This is a property of the design, kept
unchanged here; larger penalties or the optional multi-input benchmarks (F6/F7) are where
sparsity can appear.
