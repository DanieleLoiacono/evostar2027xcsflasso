"""Reproducible held-out synthetic regression benchmarks, with multiple seeds."""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score

from xcsf import XCSFRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = {
        "linear_2d": (2, lambda X: 1 + 2*X[:, 0] - X[:, 1], 1),
        "quadratic_2d": (2, lambda X: 1 + 2*X[:, 0]**2 - X[:, 1]**2, 2),
        "sine_1d": (1, lambda X: np.sin(2*np.pi*X[:, 0]), 1),
        "piecewise_1d": (1, lambda X: np.where(X[:, 0] < 0.5, X[:, 0], 2-X[:, 0]), 1),
    }
    rows = []
    for name, (dimension, function, degree) in cases.items():
        for prediction in ("nlms", "rls"):
            for seed in (0, 1, 2):
                rng = np.random.RandomState(seed)
                X_train = rng.uniform(0, 1, (600, dimension))
                X_test = rng.uniform(0, 1, (300, dimension))
                started = time.perf_counter()
                model = XCSFRegressor(prediction=prediction, degree=degree,
                                      n_epochs=20, random_state=seed).fit(X_train, function(X_train))
                prediction_test = model.predict(X_test)
                row = dict(case=name, prediction=prediction, seed=seed,
                           r2=r2_score(function(X_test), prediction_test),
                           rmse=float(np.sqrt(mean_squared_error(function(X_test), prediction_test))),
                           coverage=float(model.match(X_test).any(axis=1).mean()),
                           macroclassifiers=model.n_macroclassifiers_,
                           microclassifiers=model.n_microclassifiers_,
                           seconds=time.perf_counter()-started)
                rows.append(row)
            group = rows[-3:]
            print(f"{name:16} {prediction:4} R²={np.mean([r['r2'] for r in group]):.5f} "
                  f"RMSE={np.mean([r['rmse'] for r in group]):.5f}", flush=True)
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  sklearn=sklearn.__version__, n_train=600, n_test=300,
                  n_epochs=20, population_size=400, results=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

