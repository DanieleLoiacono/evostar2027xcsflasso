"""One xcsf_python run, through the public scikit-learn API only.

Protocol mirrored from xcslib's experiment manager (single-step, supervised):

* training: N fresh uniform samples, one online pass (n_epochs=1, shuffle=False),
  i.e. N learning problems; covering/GA/deletion inside ``partial_fit``.
* online test curve: ``performance_history_`` = mean |y - prediction| of the
  current model on each *incoming* sample before it is learned (after covering),
  averaged over windows of W samples.  This is the same estimator xcslib reports
  in its rolling 'Testing' statistics (|error| of the current model on a fresh
  uniform point, covering allowed, no update) - see docs/EXPERIMENT_DESIGN.md.
* final evaluation: ``predict`` on the deterministic grid shared with xcslib.
* Python-only extra: grid error at K evenly spaced checkpoints.

Raw outputs (immutable once the run is published):
  result.json, curve.csv, checkpoints.csv, grid_predictions.csv.gz, population.csv.gz
"""

from __future__ import annotations

import json
import os
import platform
import time
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from . import benchmarks as B
from .parity import py_params


def training_stream(spec: Dict[str, Any], n: int) -> np.ndarray:
    """Training inputs in RAW units: x = lower + U(0,1) * width.

    The uniform draws depend only on (seed, input dimension), so for a given seed every
    predictor, study and benchmark with the same dimension sees the same relative positions
    (as in xcslib, whose draws come from the same seeded stream): e.g. sine_low_1d and
    sine_shifted_1d differ only by the input offset.
    """
    bench = B.get(spec["benchmark"])
    ss = np.random.SeedSequence([int(spec["seed"]), int(bench.dim)])
    return bench.sample(np.random.Generator(np.random.PCG64(ss)), n)


def to_representation(X: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    if spec["input_representation"] == "raw":
        return X
    bench = B.get(spec["benchmark"])
    lo, hi = np.asarray(bench.lower), np.asarray(bench.upper)
    return (X - lo) / (hi - lo)


def _population_frame(model, spec) -> pd.DataFrame:
    bench = B.get(spec["benchmark"])
    rules = model.get_rules()
    recs = []
    lo, hi = np.asarray(bench.lower), np.asarray(bench.upper)
    for r in rules:
        lower, upper = np.asarray(r["lower"], float), np.asarray(r["upper"], float)
        if spec["input_representation"] == "unit":       # report conditions in raw units for comparability
            lower, upper = lo + lower * (hi - lo), lo + upper * (hi - lo)
        weights = [float(r["value"])] if r["prediction"] == "constant" else [float(w) for w in r["weights"]]
        rec = dict(id=int(r["id"]), numerosity=int(r["numerosity"]), error=float(r["error"]),
                   fitness=float(r["fitness"]), set_size=float(r["set_size"]), experience=int(r["experience"]),
                   weights=json.dumps(weights))
        for i in range(bench.dim):
            rec[f"lower{i}"], rec[f"upper{i}"] = float(lower[i]), float(upper[i])
        recs.append(rec)
    return pd.DataFrame(recs)


def _grid_metrics(pred, y):
    e = pred - y
    return dict(grid_mae=float(np.mean(np.abs(e))), grid_rmse=float(np.sqrt(np.mean(e ** 2))),
                grid_maxae=float(np.max(np.abs(e))))


def run(spec: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    import sklearn
    import xcsf
    from xcsf import XCSFRegressor

    out_dir.mkdir(parents=True, exist_ok=False)
    bench = B.get(spec["benchmark"])
    N = int(spec["xcsf"]["n_learning_problems"])
    W = int(spec["monitoring"]["window"])
    K = int(spec["monitoring"]["py_grid_checkpoints"])

    X_raw = training_stream(spec, N)
    y = bench(X_raw)
    X = to_representation(X_raw, spec)
    G_raw = bench.grid()
    yG = bench(G_raw)
    G = to_representation(G_raw, spec)

    params = py_params(spec)
    model = XCSFRegressor(**params)
    # fail loudly if the estimator does not hold exactly the requested parameters
    got = model.get_params()
    diff = {k: (v, got.get(k)) for k, v in params.items() if got.get(k) != v}
    if diff:
        raise RuntimeError(f"XCSFRegressor parameters differ from the requested ones: {diff}")

    bounds = np.linspace(0, N, K + 1).astype(int) if K > 0 else np.array([0, N])
    checkpoints = []
    train_time = 0.0
    n_warn = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for a, b in zip(bounds[:-1], bounds[1:]):
            t0 = time.perf_counter()
            model.partial_fit(X[a:b], y[a:b])
            train_time += time.perf_counter() - t0
            if K > 0:
                pred = model.predict(G)
                if not np.all(np.isfinite(pred)):
                    raise FloatingPointError("non-finite grid predictions")
                unmatched = float(np.mean(~model.match(G).any(axis=1)))
                checkpoints.append(dict(step=int(b), **_grid_metrics(pred, yG), macro=int(model.n_macroclassifiers_),
                                        micro=int(model.n_microclassifiers_), unmatched_frac=unmatched))
        n_warn = len(caught)
        warn_msgs = sorted({f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught})

    if model.n_samples_seen_ != N:
        raise RuntimeError(f"model saw {model.n_samples_seen_} samples, expected {N}")
    curve = pd.DataFrame(model.performance_history_)
    if len(curve) != N // W or not curve["complete"].all() or int(curve["step"].iloc[-1]) != N:
        raise RuntimeError("incomplete performance history (N must be a multiple of the window)")
    if not np.all(np.isfinite(curve[["mae", "mse"]].to_numpy())):
        raise FloatingPointError("NaN/inf in the online error curve")

    pred = model.predict(G)
    if not np.all(np.isfinite(pred)):
        raise FloatingPointError("non-finite final grid predictions")
    unmatched = float(np.mean(~model.match(G).any(axis=1)))
    grid_df = pd.DataFrame(G_raw, columns=[f"x{i}" for i in range(bench.dim)])
    grid_df["pred"] = pred

    pop = _population_frame(model, spec)
    if pop.empty or int(pop["numerosity"].sum()) != model.n_microclassifiers_:
        raise RuntimeError("population export inconsistent")

    curve.rename(columns={"macroclassifiers": "macro", "microclassifiers": "micro"}).to_csv(out_dir / "curve.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(out_dir / "checkpoints.csv", index=False)
    grid_df.to_csv(out_dir / "grid_predictions.csv.gz", index=False, float_format="%.17g")
    pop.to_csv(out_dir / "population.csv.gz", index=False, float_format="%.17g")

    result = dict(
        spec=spec,
        xcsf_regressor_params=params,
        n_samples_seen=int(model.n_samples_seen_),
        train_time_s=train_time,
        final_macro=int(model.n_macroclassifiers_),
        final_micro=int(model.n_microclassifiers_),
        unmatched_grid_frac=unmatched,
        stats=dict(model.stats_),
        warnings_count=int(n_warn),
        warnings=warn_msgs[:20],
        training_inputs_sha256=__import__("hashlib").sha256(X_raw.tobytes()).hexdigest(),
        versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                      xcsf=getattr(xcsf, "__version__", "?"), xcsf_path=os.path.dirname(xcsf.__file__)),
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
