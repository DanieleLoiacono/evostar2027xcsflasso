# AGENTS.md

## Mission

Build a reproducible experimental campaign that compares the Python and C++ XCSF implementations on continuous real-valued function approximation.

## Non-negotiable requirements

1. Treat the two existing library folders as upstream codebases:
   - `xcsf_python-2.0.0`
   - `xcslib-1.5-rc1-niches`
2. Prefer thin adapters, wrappers, configuration files, and scripts outside the libraries.
3. Do not modify either library unless absolutely necessary. The only exception is adding the benchmark functions. If a modification is required:
   - document the reason;
   - keep it minimal;
   - place it in a separate commit;
   - record the exact diff in the experiment manifest.   
4. The function inputs MUST be floating-point values. Do not discretize the benchmark domains to integers.
5. For cross-language comparisons, all non-prediction XCSF parameters must be semantically identical.
6. For the common predictors (Constant, NLMS, RLS), predictor-specific hyperparameters must also be matched semantically whenever both libraries expose the same mathematical parameter.
7. Do not assume that equal parameter names imply equal semantics. Inspect the code and documentation.
8. Do not independently tune the Python and C++ implementations.
9. Use the same benchmark definitions, training sample streams, evaluation sets, run IDs, and seed schedule whenever technically possible.
10. Keep Python-only Lasso experiments separate from the main Python-vs-C++ parity analysis.
11. Every experiment must be reproducible from scripts and configuration files.
12. Raw results are immutable. Derived results must be regenerated from raw results.
13. The full campaign must support resume/restart without silently overwriting completed runs.
14. Fail loudly on configuration mismatches, unsupported parameters, NaNs, missing outputs, or incomplete runs.
15. For C++ implementation, please relies on the internal experimental method:
- create corresponding confsys files
- let xcsf genrate its standard results file 
- do not write any wrapper to add run experiments
16.  I WANT TO RUN EXPERIMENTS MANUALLY, SO CREATE A SCRIPT TO RUN EXPERIMENTS MANUALLY AND DOCUMENT IT. 