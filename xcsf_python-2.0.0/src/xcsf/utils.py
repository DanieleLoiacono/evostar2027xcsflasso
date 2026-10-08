"""Read-only population inspection and post-training performance plots.

Matplotlib is imported only when plotting; printing needs no optional packages.
"""

from numbers import Integral

import numpy as np
from sklearn.utils.validation import check_is_fitted

__all__ = ["print_population", "plot_training_history"]


def print_population(model, *, precision=4, file=None):
    """Print all macroclassifiers as a table, without changing the estimator.

    Parameters
    ----------
    model : XCSFRegressor
        Fitted regressor. Conditions are printed in original input units.
    precision : int, default=4
        Significant digits for floating-point values.
    file : text stream or None, default=None
        Destination, e.g. an open text file or io.StringIO. None uses stdout.

    Returns
    -------
    None

    Notes
    -----
    Weights use the fitted polynomial basis in normalized coordinates, printed
    below the table. Constant predictors display their scalar value instead.
    Each row represents one macroclassifier; Num is its microclassifier count.
    """
    rules = model.get_rules()
    if isinstance(precision, (bool, np.bool_)) or not isinstance(precision, Integral) or precision < 1:
        raise ValueError("precision must be a positive integer.")
    number = lambda value: format(value, f".{precision}g")
    vector = lambda values: "[" + ", ".join(number(v) for v in values) + "]"
    names = getattr(model, "feature_names_in_", [f"X[{i}]" for i in range(model.n_features_in_)])
    rows = [["ID", "Condition", "Predictor", "Weights / value", "Fitness", "Error", "Exp", "Num", "Set size"]]
    for rule in rules:
        condition = " & ".join(
            f"{name} in [{number(lo)}, {number(hi)}]"
            for name, lo, hi in zip(names, rule["lower"], rule["upper"])
        )
        prediction = (number(rule["value"]) if rule["prediction"] == "constant"
                      else vector(rule["weights"]))
        rows.append([str(rule["id"]), condition, rule["prediction"], prediction,
                     number(rule["fitness"]), number(rule["error"]), str(rule["experience"]),
                     str(rule["numerosity"]), number(rule["set_size"])])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [f"Population: {len(rules)} macroclassifiers, "
             f"{sum(rule['numerosity'] for rule in rules)} microclassifiers"]
    for i, row in enumerate(rows):
        lines.append(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
        if i == 0:
            lines.append("-+-".join("-" * width for width in widths))
    if any(rule["prediction"] != "constant" for rule in rules):
        config = model._core.config
        basis = [f"x0={number(config.x0)}"] + [
            f"z[{i}]" + (f"**{power}" if power > 1 else "")
            for power in range(1, config.degree + 1) for i in range(model.n_features_in_)
        ]
        lines.extend(["", "Prediction = dot(weights, [" + ", ".join(basis) + "])",
                      "z = (X - offset) / scale (input feature order)",
                      f"offset = {vector(model.feature_offset_)}",
                      f"scale  = {vector(model.feature_scale_)}"])
    print("\n".join(lines), file=file)


def plot_training_history(model, *, metric="mae", history="auto", axes=None):
    """Plot online prediction error and macroclassifier count in two panels.

    Parameters
    ----------
    model : XCSFRegressor
        Fitted regressor with recorded training history.
    metric : {'mae', 'mse', 'rmse'}, default='mae'
        System prediction error before local updates, averaged over the recorded
        window. This is not rule.error, test error, or error of the final model.
    history : {'auto', 'steps', 'epochs'}, default='auto'
        auto uses performance_history_ when available, otherwise history_.
        steps requires history_interval to have been enabled during training.
        epochs uses one record per fit epoch / partial_fit call. Horizontal
        coordinates are cumulative training steps whenever recorded; older
        epoch histories lacking step counts are plotted against epoch numbers.
    axes : sequence of two matplotlib Axes or None, default=None
        Optional error/population axes belonging to the same figure.

    Returns
    -------
    figure : matplotlib.figure.Figure
    axes : ndarray of shape (2,)
        Error and macroclassifier axes. Call plt.show() or figure.savefig(...).

    Notes
    -----
    No retraining, history reconstruction, or implicit show/save is performed.
    A trailing incomplete window uses its actual sample count. Matplotlib is
    an optional dependency: install xcsf-python[plot].
    """
    check_is_fitted(model, "population_")
    if metric not in ("mae", "mse", "rmse"):
        raise ValueError("metric must be 'mae', 'mse', or 'rmse'.")
    if history not in ("auto", "steps", "epochs"):
        raise ValueError("history must be 'auto', 'steps', or 'epochs'.")
    step_history = getattr(model, "performance_history_", [])
    use_steps = history == "steps" or (history == "auto" and bool(step_history))
    records = step_history if use_steps else getattr(model, "history_", [])
    if not records:
        raise ValueError("No requested training history is available. To record step history, "
                         "set history_interval before training; otherwise use history='epochs'.")
    if any(metric not in row and not (metric == "rmse" and "mse" in row) for row in records):
        raise ValueError(f"{metric.upper()} was not recorded. For older models, use metric='mse' "
                         "or metric='rmse', or retrain to record MAE.")
    errors = [row[metric] if metric in row else np.sqrt(row["mse"]) for row in records]
    has_steps = all("step" in row for row in records)
    x = [row["step" if has_steps else "epoch"] for row in records]
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install it with "
                          "`python -m pip install 'xcsf-python[plot]'` "
                          "or, from the source folder, `python -m pip install -e '.[plot]'`.") from exc
    if axes is None:
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), layout="constrained")
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) != 2 or not all(hasattr(ax, "plot") for ax in axes):
            raise ValueError("axes must contain two matplotlib Axes.")
        figure = axes[0].figure
        if axes[1].figure is not figure:
            raise ValueError("Both axes must belong to the same figure.")
    axes[0].plot(x, errors, color="tab:blue", linewidth=1.5, marker="." if len(x) == 1 else None)
    axes[0].set(title="Online prediction error", ylabel=metric.upper())
    axes[1].plot(x, [row["macroclassifiers"] for row in records], color="tab:orange",
                 linewidth=1.5, marker="." if len(x) == 1 else None)
    axes[1].set(title="Population size", ylabel="Macroclassifiers")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("Training steps" if has_steps else "Epochs")
        ax.grid(alpha=0.25)
    return figure, axes
