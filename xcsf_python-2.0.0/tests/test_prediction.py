import numpy as np
from numpy.testing import assert_allclose
import pytest

from xcsf.prediction import LocalPredictor, design_matrix


def test_polynomial_basis_is_cpp_powers_without_cross_terms():
    assert_allclose(design_matrix([[2, 3]], degree=3, x0=0.5),
                    [[0.5, 2, 3, 4, 9, 8, 27]])


def test_nlms_hand_computed_step():
    predictor = LocalPredictor(2, learning_rate=0.2)
    predictor.update(np.array([1., 2.]), 10.)
    assert_allclose(predictor.weights, [0.4, 0.8])
    assert_allclose(predictor.predict(np.array([1., 2.])), 2.)


def test_constant_update_and_initial_value():
    predictor = LocalPredictor(2, method="constant", learning_rate=0.2, initial_prediction=3)
    predictor.update(np.array([1., 20.]), 8)
    assert_allclose(predictor.predict(np.ones((3, 2))), [4, 4, 4])


def test_rls_matches_batch_regularized_least_squares():
    rng = np.random.RandomState(17)
    phi = design_matrix(rng.normal(size=(80, 3)))
    y = phi @ np.array([2., -1., 3., 0.4]) + rng.normal(0, 0.01, 80)
    predictor = LocalPredictor(4, method="rls", delta=20.)
    for row, target in zip(phi, y):
        predictor.update(row, target)
    gram = phi.T @ phi + np.eye(4) / 20.
    assert_allclose(predictor.weights, np.linalg.solve(gram, phi.T @ y), atol=1e-12)
    assert_allclose(predictor.covariance, np.linalg.inv(gram), atol=1e-12)


def test_rlsk_hand_computed_forgetting_and_process_noise():
    predictor = LocalPredictor(2, method="rlsk", delta=2, forgetting_factor=0.5,
                               process_noise=0.3, kalman_noise=True)
    phi = np.array([1., 2.])
    prior = np.eye(2) * 4
    gain = prior @ phi / (0.2 + phi @ prior @ phi)
    predictor.update(phi, 3, squared_error=0.2)
    assert_allclose(predictor.weights, gain * 3)
    assert_allclose(predictor.covariance, prior - np.outer(gain, phi @ prior) + 0.3*np.eye(2))


@pytest.mark.parametrize("method", ["nlms", "constant", "rls", "rlsk"])
def test_offspring_inherits_weights_without_aliasing_and_resets_covariance(method):
    predictor = LocalPredictor(2, method=method, delta=10)
    phi = np.array([1., 2.])
    predictor.update(phi, 3)
    child = predictor.offspring()
    assert_allclose(child.predict(phi), predictor.predict(phi))
    if child.covariance is not None:
        assert_allclose(child.covariance, 10 * np.eye(2))
    original = predictor.weights.copy()
    child.update(phi, -20)
    assert_allclose(predictor.weights, original)



def test_lms_is_unnormalized_and_accepts_list_inputs():
    predictor = LocalPredictor(2, method="lms", learning_rate=0.2)
    predictor.update([1., 2.], 10.)
    assert_allclose(predictor.weights, [2., 4.])
    assert_allclose(predictor.predict([1., 2.]), 10.)


@pytest.mark.parametrize("scale", [0., 1e-160, 1e160])
def test_nlms_stable_norm_at_extreme_scales(scale):
    predictor = LocalPredictor(2, method="nlms", learning_rate=0.2)
    phi = np.array([scale, scale])
    predictor.update(phi, 2.)
    assert np.isfinite(predictor.weights).all()
    assert_allclose(predictor.predict(phi), 0. if scale == 0 else 0.4)


@pytest.mark.parametrize("forgetting", [1., 0.97])
def test_qr_rls_matches_weighted_augmented_svd_with_nonzero_prior(forgetting):
    rng = np.random.default_rng(123)
    phi = design_matrix(rng.normal(size=(150, 4)), x0=0.5)
    y = rng.normal(size=len(phi))
    predictor = LocalPredictor(5, method="rls", delta=7., x0=.5,
                               initial_prediction=2., forgetting_factor=forgetting)
    initial = predictor.weights.copy()
    for row, target in zip(phi, y):
        predictor.update(row, target)
    discount = forgetting ** np.arange(len(phi) - 1, -1, -1)
    prior = np.sqrt(forgetting ** len(phi) / predictor.delta)
    augmented = np.vstack((np.sqrt(discount)[:, None] * phi, prior * np.eye(5)))
    targets = np.concatenate((np.sqrt(discount) * y, prior * initial))
    reference = np.linalg.lstsq(augmented, targets, rcond=None)[0]
    assert_allclose(predictor.weights, reference, atol=1e-12)
    assert_allclose(predictor.covariance, np.linalg.inv(augmented.T @ augmented), atol=1e-12)


def test_qr_rls_with_large_nearly_collinear_features():
    rng = np.random.default_rng(81)
    x = rng.normal(size=600)
    phi = np.column_stack((np.ones(600), 1e8*x, 1e8*x + rng.normal(size=600)))
    y = 2 + 0.3 * phi[:, 1] - 0.3 * phi[:, 2]
    predictor = LocalPredictor(3, method="rls", delta=10)
    for row, target in zip(phi, y):
        predictor.update(row, target)
    augmented = np.vstack((phi, np.eye(3) / np.sqrt(10)))
    reference = np.linalg.lstsq(augmented, np.r_[y, np.zeros(3)], rcond=None)[0]
    assert_allclose(predictor.weights, reference, atol=2e-7)
    assert_allclose(phi @ predictor.weights, phi @ reference, atol=2e-6)
    assert np.isfinite(predictor.covariance).all()
    assert np.linalg.eigvalsh(predictor.covariance).min() >= -1e-15


def test_rls_offspring_uses_inherited_weights_as_fresh_prior():
    parent = LocalPredictor(3, method="rls", delta=4., x0=.5)
    parent.update([.5, 1., 2.], 3.)
    child = parent.offspring()
    initial = child.weights.copy()
    phi = np.array([.5, -1., 2.])
    child.update(phi, -2.)
    expected = np.linalg.lstsq(np.vstack((np.eye(3) / 2, phi)),
                               np.r_[initial / 2, -2.], rcond=None)[0]
    assert_allclose(child.weights, expected, atol=1e-12)
    assert child.n_updates_ == 1
    assert not np.shares_memory(child._rls_factor, parent._rls_factor)


def test_online_lasso_proximal_step_does_not_penalize_intercept():
    predictor = LocalPredictor(3, method="lasso_online", learning_rate=.2,
                               lasso_alpha=.5, lasso_learning_rate_decay=1, x0=.5)
    predictor.update([.5, 1., .1], 2.)
    assert_allclose(predictor.weights, [.2, .3, 0.])
    # t=2, eta=.1, residual=1.6; shrinkage=.05 on slopes only.
    predictor.update([.5, 1., .1], 2.)
    assert_allclose(predictor.weights, [.28, .41, 0.])


def test_online_lasso_zero_penalty_matches_lms():
    rng = np.random.default_rng(13)
    lms = LocalPredictor(4, method="lms", learning_rate=.05)
    lasso = LocalPredictor(4, method="lasso_online", learning_rate=.05, lasso_alpha=0)
    for row, target in zip(design_matrix(rng.normal(size=(50, 3))), rng.normal(size=50)):
        lms.update(row, target)
        lasso.update(row, target)
    assert_allclose(lms.weights, lasso.weights, atol=1e-15)


@pytest.mark.parametrize("window", [16, None])
def test_batch_lasso_matches_sklearn_for_window_and_full_history(window):
    from sklearn.linear_model import Lasso
    rng = np.random.default_rng(73)
    X = rng.normal(size=(70, 4)) + [2., -1., 3., 0.]
    y = 4 + X @ np.array([2., 0., -.5, 0.]) + rng.normal(scale=.05, size=70)
    predictor = LocalPredictor(5, method="lasso_batch", lasso_alpha=.1, x0=.5,
                               lasso_window=window, lasso_max_iter=5000, lasso_tol=1e-9)
    for i, (row, target) in enumerate(zip(design_matrix(X, x0=.5), y), 1):
        predictor.update(row, target)
        if i in (20, 40, 70):
            start = max(0, i - window) if window is not None else 0
            reference = Lasso(alpha=.1, tol=1e-12, max_iter=50000).fit(X[start:i], y[start:i])
            assert_allclose(predictor.weights[1:], reference.coef_, atol=2e-8)
            assert_allclose(predictor.weights[0] * .5, reference.intercept_, atol=1e-7)
            assert predictor.converged_
            assert predictor.kkt_violation_ <= predictor.lasso_tol
    assert predictor.n_samples_ == (70 if window is None else window)
    assert predictor.weights[2] == predictor.weights[4] == 0


def test_batch_lasso_unpenalized_intercept_constant_features_and_one_sample():
    predictor = LocalPredictor(3, method="lasso_batch", x0=.25, lasso_alpha=100.)
    for target in [3., 5., 4.]:
        predictor.update([.25, 7., 0.], target)
    assert_allclose(predictor.weights, [16., 0., 0.])
    assert predictor.converged_
    assert predictor.kkt_violation_ == 0
    window_one = LocalPredictor(2, method="lasso_batch", lasso_window=1)
    window_one.update([1., 2.], 3.)
    window_one.update([1., 5.], 9.)
    assert_allclose(window_one.weights, [9., 0.])
    assert window_one.n_samples_ == 1


def test_batch_lasso_zero_penalty_matches_least_squares():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(60, 3))
    y = 2 + X @ [1., -2., .5]
    predictor = LocalPredictor(4, method="lasso_batch", lasso_alpha=0,
                               lasso_tol=1e-9, lasso_max_iter=5000)
    for row, target in zip(design_matrix(X), y):
        predictor.update(row, target)
    expected = np.linalg.lstsq(design_matrix(X), y, rcond=None)[0]
    assert_allclose(predictor.weights, expected, atol=1e-8)


def test_batch_lasso_duplicate_features_matches_reference_predictions():
    from sklearn.linear_model import Lasso
    x = np.linspace(-2, 2, 50)
    X = np.column_stack((x, x, np.ones(50)))
    y = 3 + 2*x
    predictor = LocalPredictor(4, method="lasso_batch", lasso_alpha=.2, lasso_tol=1e-10)
    for row, target in zip(design_matrix(X), y):
        predictor.update(row, target)
    reference = Lasso(alpha=.2, tol=1e-12).fit(X, y)
    assert_allclose(predictor.predict(design_matrix(X)), reference.predict(X), atol=1e-9)
    assert predictor.weights[-1] == 0


def test_batch_lasso_reports_nonconvergence():
    from sklearn.exceptions import ConvergenceWarning
    predictor = LocalPredictor(3, method="lasso_batch", lasso_alpha=.01,
                               lasso_max_iter=1, lasso_tol=1e-15)
    predictor.update([1, 0, 1], 1.)
    predictor.update([1, 1, 2], 2.)
    with pytest.warns(ConvergenceWarning, match="lasso_tol"):
        predictor.update([1, 2, 5], -1.)
    assert not predictor.converged_
    assert predictor.kkt_violation_ > predictor.lasso_tol
    assert predictor.n_iter_ == 1


@pytest.mark.parametrize("method", ["lms", "nlms", "rls", "rlsk", "lasso_online", "lasso_batch", "constant"])
def test_new_predictor_state_is_independent_in_offspring(method):
    predictor = LocalPredictor(3, method=method, x0=.5, lasso_alpha=.2,
                               lasso_window=3, lasso_max_iter=42, lasso_tol=1e-7,
                               lasso_learning_rate_decay=.5)
    predictor.update([.5, 1., 2.], 3.)
    child = predictor.offspring()
    assert_allclose(child.weights, predictor.weights)
    for name in ("x0", "lasso_alpha", "lasso_window", "lasso_max_iter", "lasso_tol", "lasso_learning_rate_decay"):
        assert getattr(child, name) == getattr(predictor, name)
    assert child.n_updates_ == child.n_samples_ == 0
    assert child.converged_ is None
    before = predictor.weights.copy()
    child.update([.5, 2., -1.], 4.)
    assert_allclose(predictor.weights, before)
    if method == "lasso_batch":
        assert child._samples is not predictor._samples
        assert not np.shares_memory(child._samples[0][0], predictor._samples[0][0])


@pytest.mark.parametrize("kwargs", [dict(size=0), dict(size=True), dict(method="bad"),
    dict(learning_rate=0), dict(delta=np.inf), dict(x0=0), dict(lasso_alpha=-1),
    dict(lasso_alpha=np.nan), dict(lasso_window=0), dict(lasso_window=True),
    dict(lasso_window=2.5), dict(lasso_max_iter=0), dict(lasso_tol=0),
    dict(lasso_learning_rate_decay=-.1), dict(lasso_learning_rate_decay=2),
    dict(forgetting_factor=1.1)])
def test_local_predictor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LocalPredictor(**(dict(size=2) | kwargs))


@pytest.mark.parametrize("method", ["lms", "nlms", "rls", "lasso_online", "lasso_batch"])
def test_invalid_sample_does_not_change_state(method):
    import pickle
    predictor = LocalPredictor(2, method=method)
    before = pickle.dumps(predictor)
    for phi, target in [([1, np.nan], 1), ([1, 2], np.inf), ([1], 2)]:
        with pytest.raises(ValueError):
            predictor.update(phi, target)
        assert pickle.dumps(predictor) == before
    if method.startswith("lasso"):
        with pytest.raises(ValueError, match="bias"):
            predictor.update([.5, 2], 1)
        assert pickle.dumps(predictor) == before
