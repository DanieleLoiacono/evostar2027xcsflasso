import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from sklearn.base import clone, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from xcsf import XCSFRegressor


@parametrize_with_checks([XCSFRegressor(random_state=0),
                          XCSFRegressor(prediction="rls", n_epochs=3, random_state=0)])
def test_sklearn_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def data():
    rng = np.random.RandomState(4)
    X = rng.uniform(-2, 2, (100, 2))
    return X, 1 + 2*X[:, 0] - X[:, 1]


def test_not_fitted():
    with pytest.raises(NotFittedError):
        XCSFRegressor().predict([[1]])


def test_reset_reproducibility_clone_and_pickle(data):
    X, y = data
    model = XCSFRegressor(n_epochs=3, prediction="rls", random_state=12).fit(X, y)
    assert is_regressor(model)
    pred = model.predict(X)
    assert_array_equal(pred, clone(model).fit(X, y).predict(X))
    assert_array_equal(pred, model.fit(X, y).predict(X))
    restored = pickle.loads(pickle.dumps(model))
    assert_array_equal(pred, restored.predict(X))
    model.partial_fit(X, y)
    restored.partial_fit(X, y)
    assert_array_equal(model.predict(X), restored.predict(X))


def test_prediction_is_read_only_batch_independent_and_export_copied(data):
    X, y = data
    model = XCSFRegressor(n_epochs=2, random_state=4).fit(X, y)
    prediction = model.predict(X)
    before = pickle.dumps(model)
    assert_allclose(prediction, np.concatenate([model.predict([row]) for row in X]))
    assert_allclose(prediction[::-1], model.predict(X[::-1]))
    rules = model.get_rules()
    rules[0]["weights"][:] = 999
    model.match(X)
    assert pickle.dumps(model) == before


def test_partial_fit_batch_equivalence_with_fixed_coordinates(data):
    X, y = data
    kwargs = dict(normalize=False, n_epochs=1, shuffle=False, random_state=2)
    a = XCSFRegressor(**kwargs).fit(X, y)
    b = XCSFRegressor(**kwargs)
    for start in range(0, 100, 10):
        b.partial_fit(X[start:start+10], y[start:start+10])
    assert_array_equal(a.predict(X), b.predict(X))
    assert a.n_samples_seen_ == b.n_samples_seen_ == 100
    assert a.stats_ == b.stats_


def test_incremental_normalization_is_frozen(data):
    X, y = data
    model = XCSFRegressor().partial_fit(X[:50], y[:50])
    offset, scale = model.feature_offset_.copy(), model.feature_scale_.copy()
    model.partial_fit(X[50:] * 2, y[50:])
    assert_array_equal(model.feature_offset_, offset)
    assert_array_equal(model.feature_scale_, scale)
    model.set_params(degree=2)
    with pytest.raises(ValueError, match="changed after training"):
        model.partial_fit(X, y)
    model.fit(X[:3], y[:3])  # fit permits changed hyperparameters


@pytest.mark.parametrize("capacity", [1, 2, 15])
def test_population_limit_and_niche_history(capacity):
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    model = XCSFRegressor(population_size=capacity, theta_ga=0, n_epochs=2,
                          niche_history=3, random_state=1).fit(X, X[:, 0])
    assert 1 <= model.n_microclassifiers_ <= capacity
    assert model.n_microclassifiers_ == sum(cl.numerosity for cl in model.population_)
    assert len({cl.identifier for cl in model.population_}) == len(model.population_)
    for cl in model.population_:
        assert cl.numerosity >= 1 and cl.fitness > 0 and cl.error >= 0
        assert len(cl.match_history) <= 3
    assert np.isfinite(model.predict(X)).all()


def test_uncovered_prediction_policies():
    model = XCSFRegressor(prediction="rls", n_epochs=2, random_state=0).fit([[0], [1]], [2, 4])
    assert not model.match([[100]]).any()
    assert np.isfinite(model.predict([[100]])).all()
    model.set_params(unmatched="mean")
    assert_allclose(model.predict([[100]]), [3])
    model.set_params(unmatched="raise")
    with pytest.raises(ValueError, match="No classifier"):
        model.predict([[100]])


def test_bounded_online_rejects_out_of_domain_but_predicts():
    model = XCSFRegressor(bounded=True).partial_fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="frozen bounds"):
        model.partial_fit([[2]], [2])
    assert np.isfinite(model.predict([[2]])).all()


@pytest.mark.parametrize("prediction", ["nlms", "rls", "rlsk", "constant"])
def test_constant_input_target_and_single_sample(prediction):
    model = XCSFRegressor(prediction=prediction, n_epochs=50, random_state=1, bounded=True)
    model.fit([[3., 3.]], [7.])
    assert_allclose(model.predict([[3., 3.]]), [7], atol=0.01)


@pytest.mark.parametrize("name,value", [
    ("population_size", 0), ("n_epochs", 0), ("degree", True), ("degree", 1.2),
    ("learning_rate", 0), ("epsilon_0", -1), ("nu", np.inf), ("prediction", "wrong"),
    ("crossover_probability", 1.1), ("mutation_probability", -0.1),
    ("initial_fitness", 0), ("x0", 0), ("normalize", "yes"),
    ("tournament_fraction", 0), ("forgetting_factor", 0), ("process_noise", -1),
])
def test_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        XCSFRegressor(**{name: value}).fit([[0], [1]], [0, 1])


def test_pipeline_grid_search_and_cross_validation(data):
    X, y = data
    pipeline = make_pipeline(StandardScaler(), XCSFRegressor(prediction="rls", n_epochs=2,
                                                            random_state=1))
    search = GridSearchCV(pipeline, {"xcsfregressor__epsilon_0": [0.01, 0.1]}, cv=2)
    search.fit(X, y)
    assert search.best_score_ > 0.9
    assert np.min(cross_val_score(clone(search.best_estimator_), X, y, cv=2)) > 0.9


def test_rls_generalizes_linear_and_quadratic_without_memorization():
    rng = np.random.RandomState(42)
    X = rng.uniform(-2, 2, (200, 2))
    T = rng.uniform(-2, 2, (150, 2))
    for degree in (1, 2):
        function = lambda x: 1 + 2*x[:, 0]**degree - x[:, 1]**degree
        model = XCSFRegressor(prediction="rls", degree=degree, n_epochs=8,
                              random_state=1).fit(X, function(X))
        assert r2_score(function(T), model.predict(T)) > 0.97


def test_nlms_learns_nonlinear_sine_with_genetic_discovery():
    rng = np.random.RandomState(42)
    X = rng.rand(400, 1)
    T = np.linspace(0, 1, 301).reshape(-1, 1)
    model = XCSFRegressor(n_epochs=15, random_state=42).fit(X, np.sin(2*np.pi*X[:, 0]))
    assert r2_score(np.sin(2*np.pi*T[:, 0]), model.predict(T)) > 0.90
    assert model.stats_["ga_runs"] > 0 and model.stats_["deletions"] > 0


def test_dataframe_names_and_order(data):
    pd = pytest.importorskip("pandas")
    X, y = data
    frame = pd.DataFrame(X, columns=["a", "b"])
    model = XCSFRegressor(prediction="rls", n_epochs=2, random_state=0).fit(frame, y)
    assert_array_equal(model.feature_names_in_, ["a", "b"])
    assert model.score(frame, y) > 0.9
    with pytest.raises(ValueError, match="feature names"):
        model.predict(frame[["b", "a"]])
    with pytest.raises(ValueError, match="feature names"):
        model.partial_fit(frame.rename(columns={"b": "c"}), y)


def test_joblib_and_independent_multioutput_populations(data, tmp_path):
    import joblib
    X, y = data
    multi = MultiOutputRegressor(XCSFRegressor(prediction="rls", n_epochs=3,
                                              random_state=0)).fit(X, np.column_stack((y, 2*y)))
    assert multi.estimators_[0].population_ is not multi.estimators_[1].population_
    assert multi.score(X, np.column_stack((y, 2*y))) > 0.98
    filename = tmp_path / "model.joblib"
    joblib.dump(multi, filename)
    assert_array_equal(joblib.load(filename).predict(X), multi.predict(X))


@pytest.mark.parametrize("mutation", ["fixed", "proportional", "gaussian"])
@pytest.mark.parametrize("selection", ["roulette", "tournament"])
def test_alternative_operators_and_kalman_train_with_condensation(mutation, selection):
    rng = np.random.RandomState(10)
    X = rng.uniform(size=(150, 2))
    y = 0.2 + X[:, 0] - 2 * X[:, 1]
    model = XCSFRegressor(prediction="rlsk", process_noise=0.001,
                          forgetting_factor=0.99, kalman_noise=True,
                          selection=selection, mutation=mutation,
                          match_subsumption=True, theta_match_subsume=5,
                          n_epochs=3, condensation_epochs=2,
                          theta_ga=5, random_state=0).fit(X, y)
    assert model.score(X, y) > 0.99
    assert model.stats_["ga_runs"] > 0 and model.stats_["condensation_runs"] > 0
    assert model.n_samples_seen_ == 5*len(X)
    assert model.history_[-1]["condensation"]
    for cl in model.population_:
        assert np.linalg.eigvalsh(cl.predictor.covariance).min() > -1e-10


@pytest.mark.parametrize("prediction", ["lms", "nlms", "rls", "lasso_online", "lasso_batch"])
def test_v2_predictors_incremental_pickle_clone_and_genetics(prediction):
    rng = np.random.default_rng(12)
    X = rng.uniform(-1, 1, (70, 2))
    y = .7 + X @ [1.5, -.5]
    params = dict(prediction=prediction, normalize=False, n_epochs=1,
                  shuffle=False, random_state=9, population_size=20,
                  theta_ga=2, ga_subsumption=False, prediction_learning_rate=.1,
                  lasso_alpha=.002, lasso_window=16, lasso_max_iter=3000,
                  lasso_tol=1e-5, lasso_learning_rate_decay=.2)
    batch = XCSFRegressor(**params).fit(X, y)
    online = XCSFRegressor(**params).partial_fit(X[:35], y[:35])
    online = pickle.loads(pickle.dumps(online))
    online.partial_fit(X[35:], y[35:])
    assert_array_equal(batch.predict(X), online.predict(X))
    assert_array_equal(batch.predict(X), clone(batch).fit(X, y).predict(X))
    assert batch.stats_["ga_runs"] > 0
    assert batch.n_microclassifiers_ <= 20
    assert np.isfinite(batch.predict(X)).all()
    before = pickle.dumps(batch)
    for rule in batch.get_rules():
        assert rule["prediction"] == prediction
        assert "prediction_diagnostics" in rule
    batch.predict(X)
    batch.match(X)
    assert pickle.dumps(batch) == before
    if prediction == "lasso_batch":
        assert all(cl.predictor.n_samples_ <= 16 for cl in batch.population_)


@pytest.mark.parametrize("prediction", ["lms", "lasso_online", "lasso_batch"])
def test_new_predictors_learn_a_linear_function(prediction):
    rng = np.random.default_rng(32)
    X = rng.uniform(-1, 1, (150, 2))
    T = rng.uniform(-1, 1, (80, 2))
    model = XCSFRegressor(prediction=prediction, prediction_learning_rate=.1,
                          lasso_alpha=.001, n_epochs=8, random_state=1,
                          lasso_max_iter=3000, lasso_tol=1e-5)
    model.fit(X, 1 + X @ [2., -.7])
    assert r2_score(1 + T @ [2., -.7], model.predict(T)) > .95


@pytest.mark.parametrize("name,value", [
    ("lasso_alpha", -1), ("lasso_alpha", np.nan), ("lasso_alpha", True),
    ("lasso_window", 0), ("lasso_window", 1.5), ("lasso_window", True),
    ("lasso_max_iter", 0), ("lasso_max_iter", None), ("lasso_tol", 0),
    ("lasso_tol", np.inf), ("lasso_learning_rate_decay", -1),
    ("lasso_learning_rate_decay", 1.1),
])
def test_v2_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        XCSFRegressor(**{name: value}).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize("prediction", ["lasso_online", "lasso_batch"])
def test_lasso_parameters_frozen_during_partial_fit(prediction):
    model = XCSFRegressor(prediction=prediction).partial_fit([[0], [1]], [0, 1])
    model.set_params(lasso_alpha=.5)
    with pytest.raises(ValueError, match="lasso_alpha changed"):
        model.partial_fit([[.5]], [.5])
    model.fit([[0], [1]], [0, 1])
    assert model.population_[0].predictor.lasso_alpha == .5


def test_lasso_grid_search_uses_public_parameters():
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    search = GridSearchCV(XCSFRegressor(prediction="lasso_batch", n_epochs=2,
                                       random_state=1, cover_radius=2),
                          {"lasso_alpha": [.001, .01], "lasso_window": [16, None]}, cv=2)
    search.fit(X, 2 + X[:, 0])
    assert search.best_score_ > .95
