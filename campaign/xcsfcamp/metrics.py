"""Per-run metrics, computed with the SAME code for both implementations from
(i) the online error curve, (ii) predictions on the shared evaluation grid and
(iii) the final population."""

from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np
import pandas as pd

from . import benchmarks as B

# metric -> (family, description). Families drive equivalence margins in the analysis.
METRICS = {
    "grid_mae_eps": ("error_eps", "MAE on the evaluation grid / epsilon0"),
    "grid_rmse_eps": ("error_eps", "RMSE on the evaluation grid / epsilon0"),
    "grid_maxae_eps": ("error_eps", "max |error| on the evaluation grid / epsilon0"),
    "frac_within_eps": ("fraction", "fraction of grid points with |error| <= epsilon0"),
    "final_online_mae_eps": ("error_eps", "online test MAE over the last 10% of training / epsilon0"),
    "auc_online_mae_eps": ("error_eps", "mean online test MAE over the whole run / epsilon0 (convergence speed)"),
    "steps_to_eps": ("size", "learning steps until the online MAE stays <= epsilon0 for k windows (censored at N)"),
    "late_online_std_eps": ("error_eps", "std of the online MAE over the last 10% of training / epsilon0 (within-run stability)"),
    "n_macro": ("size", "final number of macroclassifiers"),
    "n_params": ("size", "final number of non-zero local-model coefficients (sum over macroclassifiers)"),
    "mean_width_frac": ("size", "numerosity-weighted mean rule width / domain width (generality)"),
    "frac_zero_slopes": ("fraction", "fraction of slope coefficients exactly zero (sparsity)"),
    "runtime_s": ("descriptive", "training wall time (s); not comparable across languages"),
}


def grid_metrics(pred: np.ndarray, y: np.ndarray, eps0: float, out_range: float) -> Dict[str, float]:
    e = np.asarray(pred, float) - np.asarray(y, float)
    mae, rmse, mx = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))
    return dict(grid_mae=mae, grid_rmse=rmse, grid_maxae=mx, grid_mae_eps=mae / eps0, grid_rmse_eps=rmse / eps0,
                grid_maxae_eps=mx / eps0, frac_within_eps=float(np.mean(np.abs(e) <= eps0)), grid_nrmse=rmse / out_range)


def curve_metrics(curve: pd.DataFrame, eps0: float, N: int, last_fraction: float, k: int) -> Dict[str, float]:
    mae = curve["mae"].to_numpy(float)
    steps = curve["step"].to_numpy(int)
    n_last = max(1, int(math.ceil(last_fraction * len(mae))))
    below = mae <= eps0
    steps_to = float("nan")
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= k:
            steps_to = float(steps[i - k + 1])
            break
    return dict(final_online_mae_eps=float(mae[-n_last:].mean() / eps0), auc_online_mae_eps=float(mae.mean() / eps0),
                late_online_std_eps=float(mae[-n_last:].std(ddof=1) / eps0) if n_last > 1 else float("nan"),
                converged=float(not math.isnan(steps_to)),
                steps_to_eps=steps_to if not math.isnan(steps_to) else float(N),  # censored at the budget
                final_online_macro=float(curve["macro"].to_numpy(float)[-n_last:].mean()))


def population_metrics(pop: pd.DataFrame, bench: B.Benchmark, predictor_type: str) -> Dict[str, float]:
    num = pop["numerosity"].to_numpy(float)
    lo, hi = np.asarray(bench.lower), np.asarray(bench.upper)
    widths = []
    degenerate = 0
    for i in range(bench.dim):
        l = np.clip(pop[f"lower{i}"].to_numpy(float), lo[i], hi[i])
        u = np.clip(pop[f"upper{i}"].to_numpy(float), lo[i], hi[i])
        # zero/negative width: xcslib's mutation can set upper := lower (see PARAMETER_PARITY.md)
        degenerate += int(np.sum(pop[f"upper{i}"].to_numpy(float) <= pop[f"lower{i}"].to_numpy(float)))
        widths.append(np.maximum(u - l, 0.0) / (hi[i] - lo[i]))
    width = np.mean(widths, axis=0)
    W = [json.loads(w) if isinstance(w, str) else list(w) for w in pop["weights"]]
    if predictor_type == "constant":
        n_params = float(len(W))
        frac_zero = float("nan")
    else:
        n_params = float(sum(int(np.count_nonzero(w)) for w in W))
        slopes = np.concatenate([np.asarray(w[1:], float) for w in W]) if W else np.array([])
        frac_zero = float(np.mean(slopes == 0.0)) if slopes.size else float("nan")
    return dict(n_macro=float(len(pop)), n_micro=float(num.sum()), mean_width_frac=float(np.average(width, weights=num)),
                n_degenerate_rules=float(degenerate), n_params=n_params, frac_zero_slopes=frac_zero,
                n_experienced_macro=float(np.sum(pop["experience"].to_numpy() > 0)))
