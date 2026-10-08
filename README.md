# XCSF Python vs C++ Experimental Campaign

This folder contains planning and coordination files for Codex agents that will build and run a reproducible comparison between:

- `xcsf_python-2.0.0` - Python implementation
- `xcslib-1.5-rc1-niches` - C++ implementation
- `BENCHMARK_FUNCTIONS.md`- List of benchmark functions 

## Primary goal

Compare the two implementations under matched experimental conditions, using the prediction functions available in both libraries:

- Constant
- NLMS
- RLS

Then run a Python-only extension with:

- Lasso Batch
- Lasso Online

The Python-only Lasso results must be clearly separated from the cross-language parity comparison.


