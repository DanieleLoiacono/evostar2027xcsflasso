"""Compare the local predictors in isolation, without evolution.

Two experiments, both with identical samples and order for every predictor:

* sparse linear problem with held-out data (accuracy, sparsity, cost, memory);
* convergence inside one narrow rule of a nonlinear target, the situation a
  predictor meets in XCSF (online error while the rule gains experience).

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

from xcsf import __version__, make_predictor
from xcsf.prediction import design_matrix

SETTINGS = {
    "constant": dict(learning_rate=.2),
    "lms": dict(learning_rate=.03),
    "nlms": dict(learning_rate=.2),
    "rls": dict(delta=1000.),
    "lasso_online": dict(alpha=.01, delta=1000., max_iter=1000, tol=1e-7),
    "lasso_sgd": dict(alpha=.01, learning_rate=.03),
    "lasso_batch": dict(alpha=.01, window=256, max_iter=1000, tol=1e-7),
}


def array_bytes(predictor):
    """Array payload only: excludes Python objects, targets and solver workspace."""
    arrays = [predictor.weights, getattr(predictor, "_factor", None),
              getattr(predictor, "_rhs", None)]
    arrays.extend(row for row, _ in getattr(predictor, "_samples", ()))
    return sum(array.nbytes for array in arrays if array is not None)


def sparse_linear(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (600, 6))
    T = rng.uniform(-1, 1, (400, 6))
    slopes = np.array([2., 0., -1., 0., .5, 0.])
    y = 1.5 + X @ slopes + rng.normal(0, .05, len(X))
    truth = 1.5 + T @ slopes
    phi, test_phi = design_matrix(X), design_matrix(T)
    order = np.concatenate([rng.permutation(len(X)) for _ in range(5)])
    results = []
    for method, settings in SETTINGS.items():
        predictor = make_predictor(method, len(slopes) + 1, **settings)
        start = time.perf_counter()
        for index in order:
            predictor.update(phi[index], y[index])
        elapsed = time.perf_counter() - start
        residual = predictor.predict(test_phi) - truth
        diagnostics = predictor.diagnostics()
        results.append(dict(
            method=method, test_rmse=float(np.sqrt(np.mean(residual**2))),
            training_seconds=elapsed,
            nonzero_slopes=int(np.count_nonzero(np.abs(predictor.weights[1:]) > 1e-8)),
            retained_samples=diagnostics["n_samples"], array_payload_bytes=array_bytes(predictor),
            weights=predictor.weights.tolist(),
            converged=diagnostics["converged"], kkt_violation=diagnostics["kkt_violation"],
        ))
    protocol = dict(seed=seed, train_samples=600, test_samples=400, epochs=5, n_features=6,
                    true_nonzero_slopes=3, noise_std=.05, test_targets="noiseless function",
                    memory="array payload only; excludes Python overhead and temporary workspace")
    return dict(protocol=protocol, results=results)


def local_convergence(seed=0, lower=.30, upper=.42, n_samples=400):
    """Online error of a newborn predictor on one rule of sin(2 pi x), x scaled to [0, 1]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(lower, upper, n_samples)
    phi, y = design_matrix(x[:, None]), np.sin(2 * np.pi * x)
    best = float(np.mean(np.abs(np.polyval(np.polyfit(x, y, 1), x) - y)))
    windows = ((0, 20), (20, 50), (50, 100), (300, 400))
    results = []
    for method, settings in SETTINGS.items():
        if "alpha" in settings:
            settings = settings | dict(alpha=.001)
        if method in ("lms", "lasso_sgd"):
            settings = settings | dict(learning_rate=.2)
        predictor = make_predictor(method, 2, **settings)
        errors = []
        for row, target in zip(phi, y):
            errors.append(abs(target - float(predictor.predict(row))))
            predictor.update(row, target)
        results.append(dict(method=method, weights=predictor.weights.tolist(), online_mae={
            f"{a + 1}-{b}": float(np.mean(errors[a:b])) for a, b in windows}))
    protocol = dict(seed=seed, interval=[lower, upper], n_samples=n_samples,
                    target="sin(2 pi x)", best_linear_fit_mae=best, lasso_alpha=.001,
                    learning_rate=.2, error="absolute error before each update")
    return dict(protocol=protocol, results=results)


def compare():
    return dict(
        versions=dict(xcsf=__version__, python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__),
        settings=SETTINGS, sparse_linear=sparse_linear(), local_convergence=local_convergence(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/prediction-benchmark.json"))
    args = parser.parse_args()
    report = compare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'method':<15} {'test RMSE':>10} {'seconds':>10} {'nonzero':>8} {'bytes':>8}")
    for row in report["sparse_linear"]["results"]:
        print(f"{row['method']:<15} {row['test_rmse']:>10.6f} {row['training_seconds']:>10.3f} "
              f"{row['nonzero_slopes']:>8} {row['array_payload_bytes']:>8}")
    local = report["local_convergence"]
    windows = list(local["results"][0]["online_mae"])
    print(f"\nOnline MAE in one rule (best linear fit {local['protocol']['best_linear_fit_mae']:.4f})")
    print(f"{'method':<15}" + "".join(f"{w:>10}" for w in windows))
    for row in local["results"]:
        print(f"{row['method']:<15}" + "".join(f"{row['online_mae'][w]:>10.4f}" for w in windows))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
