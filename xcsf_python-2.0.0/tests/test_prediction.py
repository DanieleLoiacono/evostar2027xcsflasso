import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

from xcsf.prediction import (PREDICTION_METHODS, LocalPredictor, PredictorFactory,
                             design_matrix, make_predictor)


def feed(predictor, phi, y):
    for row, target in zip(phi, y):
        predictor.update(row, target)
    return predictor


def test_polynomial_basis_is_cpp_powers_without_cross_terms():
    assert_allclose(design_matrix([[2, 3]], degree=3, x0=0.5),
                    [[0.5, 2, 3, 4, 9, 8, 27]])


def test_registry_has_one_rls_and_every_predictor_follows_the_interface():
    assert PREDICTION_METHODS == ("constant", "lms", "nlms", "rls", "lasso_online",
                                  "lasso_sgd", "lasso_batch")
    for method in PREDICTION_METHODS:
        predictor = make_predictor(method, 3)
        assert isinstance(predictor, LocalPredictor) and predictor.method == method
        assert set(predictor.diagnostics()) == {"n_updates", "n_samples", "n_iter",
                                                "converged", "kkt_violation"}
    with pytest.raises(ValueError, match="prediction must be one of"):
        make_predictor("rlsk", 3)


def test_factory_selects_only_the_parameters_of_its_predictor():
    parameters = dict(prediction_learning_rate=.3, rls_delta=5., forgetting_factor=.9,
                      process_noise=.1, kalman_noise=True, lasso_alpha=.2, lasso_window=8,
                      lasso_max_iter=7, lasso_tol=1e-3, lasso_learning_rate_decay=.5,
                      initial_prediction=2., x0=.5)
    assert PredictorFactory("nlms", parameters)(3).hyperparameters() == dict(learning_rate=.3)
    rls = PredictorFactory("rls", parameters)(3)
    assert rls.hyperparameters() == dict(delta=5., forgetting_factor=.9, process_noise=.1,
                                         kalman_noise=True)
    assert_allclose(rls.weights, [4., 0., 0.])
    assert PredictorFactory("lasso_online", parameters)(3).hyperparameters() == dict(
        alpha=.2, delta=5., forgetting_factor=.9, max_iter=7, tol=1e-3)
    with pytest.raises(ValueError, match="rls_delta"):
        PredictorFactory("lasso_online", parameters | dict(rls_delta=0.))


def test_nlms_hand_computed_step():
    predictor = make_predictor("nlms", 2, learning_rate=0.2)
    predictor.update(np.array([1., 2.]), 10.)
    assert_allclose(predictor.weights, [0.4, 0.8])
    assert_allclose(predictor.predict(np.array([1., 2.])), 2.)


def test_constant_update_and_initial_value():
    predictor = make_predictor("constant", 2, learning_rate=0.2, initial_prediction=3, x0=.5)
    predictor.update(np.array([.5, 20.]), 8)
    assert_allclose(predictor.predict(np.ones((3, 2))), [4, 4, 4])
    assert_allclose(predictor.value, 4)
    assert_allclose(predictor.weights, [8, 0])  # linear view: x0 * weights[0] == value


def test_lms_is_unnormalized_and_accepts_list_inputs():
    predictor = make_predictor("lms", 2, learning_rate=0.2)
    predictor.update([1., 2.], 10.)
    assert_allclose(predictor.weights, [2., 4.])
    assert_allclose(predictor.predict([1., 2.]), 10.)


@pytest.mark.parametrize("scale", [0., 1e-160, 1e160])
def test_nlms_stable_norm_at_extreme_scales(scale):
    predictor = make_predictor("nlms", 2, learning_rate=0.2)
    phi = np.array([scale, scale])
    predictor.update(phi, 2.)
    assert np.isfinite(predictor.weights).all()
    assert_allclose(predictor.predict(phi), 0. if scale == 0 else 0.4)


# ----------------------------------------------------------------------------------- RLS

def kalman_reference(phi, y, delta, forgetting=1., process_noise=0., noise=None):
    """Covariance-form recursion that the square-root information filter must reproduce."""
    size = phi.shape[1]
    weights, covariance = np.zeros(size), delta * np.eye(size)
    for i, (row, target) in enumerate(zip(phi, y)):
        variance = 1. if noise is None else noise[i]
        prior = covariance / forgetting
        gain = prior @ row / (variance + row @ prior @ row)
        weights = weights + gain * (target - row @ weights)
        covariance = prior - np.outer(gain, row @ prior) + process_noise * np.eye(size)
    return weights, covariance


def test_rls_matches_batch_regularized_least_squares():
    rng = np.random.RandomState(17)
    phi = design_matrix(rng.normal(size=(80, 3)))
    y = phi @ np.array([2., -1., 3., 0.4]) + rng.normal(0, 0.01, 80)
    predictor = feed(make_predictor("rls", 4, delta=20.), phi, y)
    gram = phi.T @ phi + np.eye(4) / 20.
    assert_allclose(predictor.weights, np.linalg.solve(gram, phi.T @ y), atol=1e-12)
    assert_allclose(predictor.covariance, np.linalg.inv(gram), atol=1e-12)


def test_rls_hand_computed_forgetting_process_noise_and_measurement_variance():
    predictor = make_predictor("rls", 2, delta=2, forgetting_factor=0.5,
                               process_noise=0.3, kalman_noise=True)
    phi = np.array([1., 2.])
    prior = np.eye(2) * 4
    gain = prior @ phi / (0.2 + phi @ prior @ phi)
    predictor.update(phi, 3, squared_error=0.2)
    assert_allclose(predictor.weights, gain * 3)
    assert_allclose(predictor.covariance, prior - np.outer(gain, phi @ prior) + 0.3*np.eye(2))


@pytest.mark.parametrize("forgetting,process_noise,kalman", [
    (1., 0., False), (.97, 0., False), (1., .05, False), (.98, .01, True)])
def test_rls_reproduces_the_kalman_recursion_in_every_mode(forgetting, process_noise, kalman):
    rng = np.random.default_rng(5)
    phi = design_matrix(rng.normal(size=(300, 3)))
    y = rng.normal(size=300)
    noise = rng.uniform(.01, 3., 300)
    predictor = make_predictor("rls", 4, delta=50., forgetting_factor=forgetting,
                               process_noise=process_noise, kalman_noise=kalman)
    for row, target, variance in zip(phi, y, noise):
        predictor.update(row, target, squared_error=variance)
    weights, covariance = kalman_reference(phi, y, 50., forgetting, process_noise,
                                           noise if kalman else None)
    assert_allclose(predictor.weights, weights, rtol=1e-9, atol=1e-11)
    assert_allclose(predictor.covariance, covariance, rtol=1e-9, atol=1e-11)


def test_rls_kalman_noise_floors_the_measurement_variance():
    floored = make_predictor("rls", 2, delta=3., kalman_noise=True)
    floored.update([1., 2.], 5., squared_error=0.)
    weights, _ = kalman_reference(np.array([[1., 2.]]), [5.], 3., noise=[1e-4])
    assert_allclose(floored.weights, weights)
    with pytest.raises(ValueError, match="squared_error"):
        floored.update([1., 2.], 5., squared_error=-1.)


@pytest.mark.parametrize("lower", [0., 1000.])
def test_rls_zero_covariance_with_unit_process_noise_is_the_xcslib_rls(lower):
    rng = np.random.default_rng(42)
    X = rng.uniform(lower, lower + 100., (300, 1))
    phi = design_matrix(X)
    y = 100 * np.sin(2 * np.pi * X[:, 0] / 100.)
    predictor = make_predictor("rls", 2, delta=0., process_noise=1.)
    assert_allclose(predictor.covariance, 0.)
    weights, covariance = np.zeros(2), np.zeros((2, 2))  # rls.cpp: delta is never read
    for i, (row, target) in enumerate(zip(phi, y)):
        predictor.update(row, target)
        gain = covariance @ row / (1. + row @ covariance @ row)
        weights = weights + gain * (target - row @ weights)
        covariance = covariance - np.outer(gain, row) @ covariance + np.eye(2)
        assert abs(predictor.predict(row) - row @ weights) <= 1e-9 * max(1., abs(row @ weights))
        if i == 0:
            assert_allclose(predictor.weights, 0.)  # zero covariance: first sample ignored
    assert_allclose(predictor.covariance, covariance, rtol=1e-8)
    with pytest.raises(ValueError, match="process_noise"):
        make_predictor("rls", 2, delta=0.)


@pytest.mark.parametrize("forgetting", [1., 0.97])
def test_rls_matches_weighted_augmented_svd_with_nonzero_prior(forgetting):
    rng = np.random.default_rng(123)
    phi = design_matrix(rng.normal(size=(150, 4)), x0=0.5)
    y = rng.normal(size=len(phi))
    predictor = make_predictor("rls", 5, delta=7., x0=.5, initial_prediction=2.,
                               forgetting_factor=forgetting)
    initial = predictor.weights.copy()
    feed(predictor, phi, y)
    discount = forgetting ** np.arange(len(phi) - 1, -1, -1)
    prior = np.sqrt(forgetting ** len(phi) / predictor.delta)
    augmented = np.vstack((np.sqrt(discount)[:, None] * phi, prior * np.eye(5)))
    targets = np.concatenate((np.sqrt(discount) * y, prior * initial))
    reference = np.linalg.lstsq(augmented, targets, rcond=None)[0]
    assert_allclose(predictor.weights, reference, atol=1e-12)
    assert_allclose(predictor.covariance, np.linalg.inv(augmented.T @ augmented), atol=1e-12)


def test_rls_with_large_nearly_collinear_features():
    rng = np.random.default_rng(81)
    x = rng.normal(size=600)
    phi = np.column_stack((np.ones(600), 1e8*x, 1e8*x + rng.normal(size=600)))
    y = 2 + 0.3 * phi[:, 1] - 0.3 * phi[:, 2]
    predictor = feed(make_predictor("rls", 3, delta=10), phi, y)
    augmented = np.vstack((phi, np.eye(3) / np.sqrt(10)))
    reference = np.linalg.lstsq(augmented, np.r_[y, np.zeros(3)], rcond=None)[0]
    assert_allclose(predictor.weights, reference, atol=2e-7)
    assert_allclose(phi @ predictor.weights, phi @ reference, atol=2e-6)
    assert np.isfinite(predictor.covariance).all()
    assert np.linalg.eigvalsh(predictor.covariance).min() >= -1e-15


def test_rls_is_accurate_on_shifted_narrow_inputs_where_covariance_updates_degrade():
    # Raw inputs near 1000 with x0 = 1: the regressors are almost collinear.
    from fractions import Fraction
    rng = np.random.default_rng(9)
    x = rng.uniform(1000., 1001., 3000)
    y = 100 * np.sin(2 * np.pi * x / 100.)
    phi = design_matrix(x[:, None])
    predictor = feed(make_predictor("rls", 2, delta=1e6), phi, y)
    a00 = a01 = a11 = b0 = b1 = Fraction(0)
    for value, target in zip(x, y):
        value, target = Fraction(float(value)), Fraction(float(target))
        a00, a01, a11 = a00 + 1, a01 + value, a11 + value * value
        b0, b1 = b0 + target, b1 + value * target
    a00, a11 = a00 + Fraction(1, 10**6), a11 + Fraction(1, 10**6)
    determinant = a00 * a11 - a01 * a01
    exact = [float((b0 * a11 - b1 * a01) / determinant), float((a00 * b1 - a01 * b0) / determinant)]
    assert_allclose(predictor.weights, exact, rtol=1e-10)


def test_rls_offspring_uses_inherited_weights_as_fresh_prior():
    parent = make_predictor("rls", 3, delta=4., x0=.5)
    parent.update([.5, 1., 2.], 3.)
    child = parent.offspring()
    initial = child.weights.copy()
    phi = np.array([.5, -1., 2.])
    child.update(phi, -2.)
    expected = np.linalg.lstsq(np.vstack((np.eye(3) / 2, phi)),
                               np.r_[initial / 2, -2.], rcond=None)[0]
    assert_allclose(child.weights, expected, atol=1e-12)
    assert child.n_updates_ == 1
    assert not np.shares_memory(child._factor, parent._factor)


@pytest.mark.parametrize("method", ["nlms", "constant", "rls"])
def test_offspring_inherits_weights_without_aliasing_and_resets_covariance(method):
    keywords = dict(delta=10) if method == "rls" else {}
    predictor = make_predictor(method, 2, **keywords)
    phi = np.array([1., 2.])
    predictor.update(phi, 3)
    child = predictor.offspring()
    assert_allclose(child.predict(phi), predictor.predict(phi))
    if method == "rls":
        assert_allclose(child.covariance, 10 * np.eye(2))
    original = predictor.weights.copy()
    child.update(phi, -20)
    assert_allclose(predictor.weights, original)


# ---------------------------------------------------------------------- recursive Lasso

def lasso_kkt(weights, gradient, penalty):
    """Violation of the optimality conditions; the intercept (index 0) is unpenalized."""
    slopes, g = weights[1:], gradient[1:]
    violations = np.where(slopes != 0, np.abs(g + penalty * np.sign(slopes)),
                          np.maximum(np.abs(g) - penalty, 0.))
    return max(abs(gradient[0]), violations.max(initial=0.))


@pytest.mark.parametrize("n_features", [1, 4])
def test_online_lasso_is_the_exact_lasso_of_all_observations(n_features):
    from sklearn.linear_model import Lasso
    rng = np.random.default_rng(73)
    X = rng.normal(size=(120, n_features)) * .3 + .5
    y = 4 + X @ np.array([2., 0., -.5, 0.])[:n_features] + rng.normal(scale=.05, size=120)
    predictor = make_predictor("lasso_online", n_features + 1, alpha=.02, delta=1e9, x0=.5,
                               max_iter=5000, tol=1e-10)
    for i, (row, target) in enumerate(zip(design_matrix(X, x0=.5), y), 1):
        predictor.update(row, target)
        if i in (10, 40, 120):
            reference = Lasso(alpha=.02, tol=1e-13, max_iter=100000).fit(X[:i], y[:i])
            assert_allclose(predictor.weights[1:], reference.coef_, atol=1e-6)
            assert_allclose(predictor.weights[0] * .5, reference.intercept_, atol=1e-6)
            assert predictor.converged_ and predictor.kkt_violation_ <= predictor.tol
    assert predictor.n_samples_ == 0  # no observation is stored
    if n_features == 4:
        assert predictor.weights[2] == predictor.weights[4] == 0


def test_online_lasso_without_penalty_is_rls_bit_for_bit():
    rng = np.random.default_rng(3)
    phi, y = design_matrix(rng.normal(size=(100, 3))), rng.normal(size=100)
    lasso = feed(make_predictor("lasso_online", 4, alpha=0., delta=7., forgetting_factor=.97), phi, y)
    rls = feed(make_predictor("rls", 4, delta=7., forgetting_factor=.97), phi, y)
    assert np.array_equal(lasso.weights, rls.weights)


def test_online_lasso_minimizes_its_documented_objective_with_prior_and_forgetting():
    rng = np.random.default_rng(31)
    phi = design_matrix(rng.normal(size=(90, 3)) + [1., -2., .5])
    y = phi @ [1., 2., 0., -.3] + rng.normal(scale=.1, size=90)
    alpha, delta, forgetting = .05, 5., .97
    predictor = make_predictor("lasso_online", 4, alpha=alpha, delta=delta, initial_prediction=3.,
                               forgetting_factor=forgetting, max_iter=10000, tol=1e-11)
    initial = predictor.weights.copy()
    feed(predictor, phi, y)
    discount = forgetting ** np.arange(len(phi) - 1, -1, -1)
    w = predictor.weights
    gradient = (phi.T @ (discount * (phi @ w - y))
                + forgetting ** len(phi) / delta * (w - initial))
    assert lasso_kkt(w, gradient, alpha * discount.sum()) <= 1e-9 * discount.sum()
    assert predictor.converged_


def test_online_lasso_converges_like_rls_not_like_a_gradient_step():
    rng = np.random.default_rng(0)
    x = rng.uniform(.30, .42, 200)  # one narrow rule of a sine, inputs scaled to [0, 1]
    phi, y = design_matrix(x[:, None]), np.sin(2 * np.pi * x)
    errors = {}
    for method, keywords in (("lasso_online", dict(alpha=.001)), ("rls", {}),
                             ("lasso_sgd", dict(alpha=.001, learning_rate=.2))):
        predictor = make_predictor(method, 2, **keywords)
        absolute = []
        for row, target in zip(phi, y):
            absolute.append(abs(target - predictor.predict(row)))
            predictor.update(row, target)
        errors[method] = np.mean(absolute[20:60])
    assert errors["lasso_online"] < 2.5 * errors["rls"]
    assert errors["lasso_online"] < errors["lasso_sgd"] / 3


def test_online_lasso_reports_nonconvergence():
    from sklearn.exceptions import ConvergenceWarning
    rng = np.random.default_rng(2)
    x = rng.normal(size=30)
    phi = np.column_stack((np.ones(30), x, x + 1e-3 * rng.normal(size=30), rng.normal(size=30)))
    predictor = make_predictor("lasso_online", 4, alpha=.01, max_iter=1, tol=1e-15)
    with pytest.warns(ConvergenceWarning, match="lasso_tol"):
        feed(predictor, phi, x + phi[:, 3])
    assert not predictor.converged_
    assert predictor.kkt_violation_ > predictor.tol and predictor.n_iter_ == 1


# -------------------------------------------------------------------------- Lasso, SGD

def test_lasso_sgd_proximal_step_does_not_penalize_intercept():
    predictor = make_predictor("lasso_sgd", 3, learning_rate=.2, alpha=.5,
                               learning_rate_decay=1, x0=.5)
    predictor.update([.5, 1., .1], 2.)
    assert_allclose(predictor.weights, [.2, .3, 0.])
    # t=2, eta=.1, residual=1.6; shrinkage=.05 on slopes only.
    predictor.update([.5, 1., .1], 2.)
    assert_allclose(predictor.weights, [.28, .41, 0.])


def test_lasso_sgd_zero_penalty_matches_lms():
    rng = np.random.default_rng(13)
    lms = make_predictor("lms", 4, learning_rate=.05)
    lasso = make_predictor("lasso_sgd", 4, learning_rate=.05, alpha=0)
    for row, target in zip(design_matrix(rng.normal(size=(50, 3))), rng.normal(size=50)):
        lms.update(row, target)
        lasso.update(row, target)
    assert_allclose(lms.weights, lasso.weights, atol=1e-15)


# ------------------------------------------------------------------------ Lasso, batch

@pytest.mark.parametrize("window", [16, None])
def test_batch_lasso_matches_sklearn_for_window_and_full_history(window):
    from sklearn.linear_model import Lasso
    rng = np.random.default_rng(73)
    X = rng.normal(size=(70, 4)) + [2., -1., 3., 0.]
    y = 4 + X @ np.array([2., 0., -.5, 0.]) + rng.normal(scale=.05, size=70)
    predictor = make_predictor("lasso_batch", 5, alpha=.1, x0=.5, window=window,
                               max_iter=5000, tol=1e-9)
    for i, (row, target) in enumerate(zip(design_matrix(X, x0=.5), y), 1):
        predictor.update(row, target)
        if i in (20, 40, 70):
            start = max(0, i - window) if window is not None else 0
            reference = Lasso(alpha=.1, tol=1e-12, max_iter=50000).fit(X[start:i], y[start:i])
            assert_allclose(predictor.weights[1:], reference.coef_, atol=2e-8)
            assert_allclose(predictor.weights[0] * .5, reference.intercept_, atol=1e-7)
            assert predictor.converged_
            assert predictor.kkt_violation_ <= predictor.tol
    assert predictor.n_samples_ == (70 if window is None else window)
    assert predictor.weights[2] == predictor.weights[4] == 0


def test_batch_lasso_unpenalized_intercept_constant_features_and_one_sample():
    predictor = make_predictor("lasso_batch", 3, x0=.25, alpha=100.)
    for target in [3., 5., 4.]:
        predictor.update([.25, 7., 0.], target)
    assert_allclose(predictor.weights, [16., 0., 0.])
    assert predictor.converged_
    assert predictor.kkt_violation_ == 0
    window_one = make_predictor("lasso_batch", 2, window=1)
    window_one.update([1., 2.], 3.)
    window_one.update([1., 5.], 9.)
    assert_allclose(window_one.weights, [9., 0.])
    assert window_one.n_samples_ == 1


def test_batch_lasso_zero_penalty_matches_least_squares():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(60, 3))
    y = 2 + X @ [1., -2., .5]
    predictor = feed(make_predictor("lasso_batch", 4, alpha=0, tol=1e-9, max_iter=5000),
                     design_matrix(X), y)
    expected = np.linalg.lstsq(design_matrix(X), y, rcond=None)[0]
    assert_allclose(predictor.weights, expected, atol=1e-8)


def test_batch_lasso_duplicate_features_matches_reference_predictions():
    from sklearn.linear_model import Lasso
    x = np.linspace(-2, 2, 50)
    X = np.column_stack((x, x, np.ones(50)))
    y = 3 + 2*x
    predictor = feed(make_predictor("lasso_batch", 4, alpha=.2, tol=1e-10), design_matrix(X), y)
    reference = Lasso(alpha=.2, tol=1e-12).fit(X, y)
    assert_allclose(predictor.predict(design_matrix(X)), reference.predict(X), atol=1e-9)
    assert predictor.weights[-1] == 0


def test_batch_lasso_reports_nonconvergence():
    from sklearn.exceptions import ConvergenceWarning
    predictor = make_predictor("lasso_batch", 3, alpha=.01, max_iter=1, tol=1e-15)
    predictor.update([1, 0, 1], 1.)
    predictor.update([1, 1, 2], 2.)
    with pytest.warns(ConvergenceWarning, match="lasso_tol"):
        predictor.update([1, 2, 5], -1.)
    assert not predictor.converged_
    assert predictor.kkt_violation_ > predictor.tol
    assert predictor.n_iter_ == 1


# ------------------------------------------------------------------ common behaviour

@pytest.mark.parametrize("method", PREDICTION_METHODS)
def test_offspring_state_is_independent_and_settings_are_inherited(method):
    settings = {"lasso_online": dict(alpha=.2, delta=3., forgetting_factor=.9, max_iter=42, tol=1e-7),
                "lasso_sgd": dict(alpha=.2, learning_rate=.1, learning_rate_decay=.5),
                "lasso_batch": dict(alpha=.2, window=3, max_iter=42, tol=1e-7),
                "rls": dict(delta=3., forgetting_factor=.9, process_noise=.1, kalman_noise=True),
                }.get(method, dict(learning_rate=.1))
    predictor = make_predictor(method, 3, x0=.5, **settings)
    predictor.update([.5, 1., 2.], 3.)
    child = predictor.offspring()
    assert_allclose(child.weights, predictor.weights)
    assert child.x0 == predictor.x0
    assert child.hyperparameters() == predictor.hyperparameters() == settings
    assert child.diagnostics() == dict(n_updates=0, n_samples=0, n_iter=0, converged=None,
                                       kkt_violation=None)
    before = predictor.weights.copy()
    child.update([.5, 2., -1.], 4.)
    assert_allclose(predictor.weights, before)
    if method == "lasso_batch":
        assert child._samples is not predictor._samples
        assert not np.shares_memory(child._samples[0][0], predictor._samples[0][0])


@pytest.mark.parametrize("method,keywords", [
    ("nlms", dict(size=0)), ("nlms", dict(size=True)), ("nlms", dict(learning_rate=0)),
    ("nlms", dict(x0=0)), ("nlms", dict(initial_prediction=np.inf)),
    ("constant", dict(learning_rate=-1)), ("rls", dict(delta=np.inf)), ("rls", dict(delta=-1)),
    ("rls", dict(forgetting_factor=1.1)), ("rls", dict(forgetting_factor=0)),
    ("rls", dict(process_noise=-1)), ("rls", dict(kalman_noise=1)),
    ("lasso_online", dict(alpha=-1)), ("lasso_online", dict(alpha=np.nan)),
    ("lasso_online", dict(delta=0)), ("lasso_online", dict(max_iter=0)),
    ("lasso_online", dict(tol=0)), ("lasso_online", dict(forgetting_factor=2)),
    ("lasso_sgd", dict(learning_rate_decay=-.1)), ("lasso_sgd", dict(learning_rate_decay=2)),
    ("lasso_batch", dict(window=0)), ("lasso_batch", dict(window=True)),
    ("lasso_batch", dict(window=2.5)), ("lasso_batch", dict(max_iter=0)),
    ("lasso_batch", dict(tol=0))])
def test_predictors_reject_invalid_configuration(method, keywords):
    with pytest.raises(ValueError):
        make_predictor(method, **(dict(size=2) | keywords))


@pytest.mark.parametrize("method", PREDICTION_METHODS)
def test_invalid_sample_does_not_change_state(method):
    predictor = make_predictor(method, 2)
    predictor.update([1, 2], 1)
    before = pickle.dumps(predictor)
    for phi, target in [([1, np.nan], 1), ([1, 2], np.inf), ([1], 2)]:
        with pytest.raises(ValueError):
            predictor.update(phi, target)
        assert pickle.dumps(predictor) == before
    if method.startswith("lasso"):
        with pytest.raises(ValueError, match="bias"):
            predictor.update([.5, 2], 1)
        assert pickle.dumps(predictor) == before
