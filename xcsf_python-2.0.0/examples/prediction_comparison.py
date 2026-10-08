"""Compare v2 local predictors on a sparse linear problem with held-out data.

This isolates prediction updates; it is not a benchmark of evolved populations.
Run from an installed checkout: python examples/prediction_comparison.py
"""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sklearn

from xcsf import LocalPredictor, __version__
from xcsf.prediction import design_matrix


def array_bytes(predictor):
    """Array payload only: excludes Python objects, targets and solver workspace."""
    arrays = [predictor.weights, predictor._rls_factor, predictor._rls_rhs,
              predictor._covariance]
    if predictor._samples is not None:
        arrays.extend(row for row, _ in predictor._samples)
    return sum(array.nbytes for array in arrays if array is not None)


def compare(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (600, 6))
    T = rng.uniform(-1, 1, (400, 6))
    slopes = np.array([2., 0., -1., 0., .5, 0.])
    y = 1.5 + X @ slopes + rng.normal(0, .05, len(X))
    truth = 1.5 + T @ slopes
    phi, test_phi = design_matrix(X), design_matrix(T)
    # Identical presentation order and budget for every method.
    order = np.concatenate([rng.permutation(len(X)) for _ in range(5)])
    results = []
    for method in ("lms", "nlms", "rls", "lasso_online", "lasso_batch"):
        predictor = LocalPredictor(len(slopes) + 1, method=method, learning_rate=.03,
                                   lasso_alpha=.01, lasso_window=256,
                                   lasso_max_iter=1000, lasso_tol=1e-7)
        start = time.perf_counter()
        for index in order:
            predictor.update(phi[index], y[index])
        elapsed = time.perf_counter() - start
        residual = predictor.predict(test_phi) - truth
        results.append(dict(
            method=method, test_rmse=float(np.sqrt(np.mean(residual**2))),
            training_seconds=elapsed,
            nonzero_slopes=int(np.count_nonzero(np.abs(predictor.weights[1:]) > 1e-8)),
            retained_samples=predictor.n_samples_, array_payload_bytes=array_bytes(predictor),
            weights=predictor.weights.tolist(),
            converged=predictor.converged_, kkt_violation=predictor.kkt_violation_,
        ))
    return dict(
        versions=dict(xcsf=__version__, python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__),
        protocol=dict(seed=seed, train_samples=600, test_samples=400, epochs=5,
                      n_features=6, true_nonzero_slopes=3, noise_std=.05,
                      test_targets="noiseless function", learning_rate=.03,
                      lasso_alpha=.01, lasso_window=256, lasso_tol=1e-7,
                      lasso_learning_rate_decay=0., rls_delta=1000., forgetting_factor=1.,
                      memory="array payload only; excludes Python overhead and temporary workspace"),
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/prediction-benchmark-v2.json"))
    args = parser.parse_args()
    report = compare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'method':<15} {'test RMSE':>10} {'seconds':>10} {'nonzero':>8}")
    for row in report["results"]:
        print(f"{row['method']:<15} {row['test_rmse']:>10.6f} "
              f"{row['training_seconds']:>10.3f} {row['nonzero_slopes']:>8}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
