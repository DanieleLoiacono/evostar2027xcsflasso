"""Check observable logging semantics against a known online predictor."""

import io
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from sklearn.exceptions import NotFittedError

from xcsf import XCSFRegressor
from xcsf.utils import plot_training_history, print_population


def constant_model(**kwargs):
    return XCSFRegressor(prediction="constant", prediction_learning_rate=0.5,
                          discovery=False, normalize=False, random_state=0,
                          shuffle=False, **kwargs)


def test_step_windows_cross_epochs_and_keep_short_tail():
    # Known pre-update errors for constant target 1: 1, 1/2, 1/4, ...
    model = constant_model(n_epochs=2, history_interval=4).fit(np.zeros((3, 1)), np.ones(3))
    rows = model.performance_history_
    assert [r["step"] for r in rows] == [4, 6]
    assert [r["n_samples"] for r in rows] == [4, 2]
    assert [r["complete"] for r in rows] == [True, False]
    errors = 0.5 ** np.arange(6)
    for row, window in zip(rows, [errors[:4], errors[4:]]):
        assert_allclose(row["mae"], np.mean(window))
        assert_allclose(row["mse"], np.mean(window**2))
        assert_allclose(row["rmse"], np.sqrt(np.mean(window**2)))
        assert row["macroclassifiers"] == row["microclassifiers"] == 1
    assert [r["step"] for r in model.history_] == [3, 6]
    assert_allclose(model.history_[0]["mae"], np.mean(errors[:3]))


def test_partial_fit_replaces_tail_and_pickle_continues_exact_windows():
    model = constant_model(history_interval=4).partial_fit(np.zeros((3, 1)), np.ones(3))
    assert model.performance_history_[0]["step"] == 3
    model = pickle.loads(pickle.dumps(model))
    model.partial_fit(np.zeros((2, 1)), np.ones(2))
    assert [r["step"] for r in model.performance_history_] == [4, 5]
    model.partial_fit(np.zeros((3, 1)), np.ones(3))
    whole = constant_model(history_interval=4).partial_fit(np.zeros((8, 1)), np.ones(8))
    assert model.performance_history_ == whole.performance_history_
    assert [r["step"] for r in model.performance_history_] == [4, 8]
    model.fit([[0]], [1])
    assert model.performance_history_[-1]["step"] == model.n_epochs


def test_logging_does_not_change_learning_and_records_population_after_update():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    kwargs = dict(n_epochs=2, population_size=10, theta_ga=0, random_state=4)
    logged = XCSFRegressor(history_interval=1, **kwargs).fit(X, X[:, 0])
    unlogged = XCSFRegressor(history_interval=None, **kwargs).fit(X, X[:, 0])
    assert_array_equal(logged.predict(X), unlogged.predict(X))
    assert logged.stats_ == unlogged.stats_
    assert unlogged.performance_history_ == []
    assert len(logged.performance_history_) == 60
    last = logged.performance_history_[-1]
    assert last["macroclassifiers"] == logged.n_macroclassifiers_
    assert last["microclassifiers"] == logged.n_microclassifiers_
    assert all(r["microclassifiers"] <= 10 for r in logged.performance_history_)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_history_interval(interval):
    with pytest.raises(ValueError, match="history_interval"):
        constant_model(history_interval=interval).fit([[0]], [1])


def test_print_outputs_original_bounds_and_correct_weight_basis(capsys):
    model = XCSFRegressor(n_epochs=1, bounded=True, x0=2, degree=2,
                          random_state=0).fit([[10], [20]], [1, 2])
    before = pickle.dumps(model)
    print_population(model, precision=6)
    text = capsys.readouterr().out
    assert "Population:" in text and "X[0] in [10," in text
    assert "x0=2, z[0], z[0]**2" in text
    assert "offset = [10]" in text and "scale  = [10]" in text
    assert "Fitness" in text and "Num" in text
    stream = io.StringIO()
    print_population(model, precision=6, file=stream)
    assert stream.getvalue() == text
    assert pickle.dumps(model) == before
    constant = constant_model(n_epochs=1).fit([[0]], [2])
    stream = io.StringIO()
    print_population(constant, file=stream)
    assert "constant" in stream.getvalue() and "dot(weights" not in stream.getvalue()


def test_plot_data_returned_axes_and_read_only():
    model = constant_model(n_epochs=2, history_interval=4).fit(np.zeros((3, 1)), np.ones(3))
    before = pickle.dumps(model)
    fig, axes = plot_training_history(model)
    try:
        assert_array_equal(axes[0].lines[0].get_xdata(), [4, 6])
        assert_allclose(axes[0].lines[0].get_ydata(), [r["mae"] for r in model.performance_history_])
        assert_array_equal(axes[1].lines[0].get_ydata(), [1, 1])
        assert axes[1].get_xlabel() == "Training steps"
        assert axes[0].get_ylabel() == "MAE"
        assert pickle.dumps(model) == before
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(2)
    try:
        returned, _ = plot_training_history(model, history="epochs", metric="rmse", axes=axes)
        assert returned is fig
        assert_array_equal(axes[0].lines[0].get_xdata(), [3, 6])
        assert_allclose(axes[0].lines[0].get_ydata(), [r["rmse"] for r in model.history_])
    finally:
        plt.close(fig)


def test_disabled_and_legacy_history():
    model = constant_model(n_epochs=1, history_interval=None).fit([[0]], [1])
    with pytest.raises(ValueError, match="No requested training history"):
        plot_training_history(model, history="steps")
    fig, axes = plot_training_history(model)
    assert_array_equal(axes[0].lines[0].get_xdata(), [1])
    plt.close(fig)
    # Old fitted models stored only per-epoch MSE/population counts.
    del model.performance_history_
    model.history_ = [{"epoch": 1, "mse": 4., "macroclassifiers": 2}]
    with pytest.raises(ValueError, match="MAE was not recorded"):
        plot_training_history(model)
    fig, axes = plot_training_history(model, metric="rmse")
    assert axes[1].get_xlabel() == "Epochs"
    assert_array_equal(axes[0].lines[0].get_ydata(), [2])
    plt.close(fig)


def test_utilities_require_fitted_model():
    for function in (print_population, plot_training_history):
        with pytest.raises(NotFittedError):
            function(XCSFRegressor())
