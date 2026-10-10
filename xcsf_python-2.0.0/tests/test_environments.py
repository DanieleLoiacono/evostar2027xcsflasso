import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from xcsf import (DatasetEnvironment, RealFunctionEnvironment, XCSFClassifierSystem,
                  XCSFRegressor)
from xcsf.experiments import TrainingMonitor, run_problems


def problems(environment, n):
    out = []
    for _ in range(n):
        environment.begin_problem()
        out.append((environment.state().copy(), environment.reward(), environment.features()))
        environment.end_problem()
    return out


def test_dataset_environment_presents_rows_in_order_and_wraps():
    X, y = np.arange(6.).reshape(3, 2), np.array([10., 11., 12.])
    environment = DatasetEnvironment(X, y, features=X + 100)
    assert len(environment) == 3
    seen = problems(environment, 5)
    assert_allclose([reward for _, reward, _ in seen], [10, 11, 12, 10, 11])
    assert_array_equal(seen[1][0], X[1])
    assert_array_equal(seen[1][2], X[1] + 100)
    assert problems(DatasetEnvironment(X, y), 1)[0][2] is None


def test_dataset_environment_draws_one_permutation_per_pass():
    X, y = np.arange(8.).reshape(8, 1), np.arange(8.)
    environment = DatasetEnvironment(X, y, shuffle=True, rng=np.random.RandomState(0))
    first = [reward for _, reward, _ in problems(environment, 8)]
    second = [reward for _, reward, _ in problems(environment, 8)]
    assert sorted(first) == sorted(second) == list(range(8)) and first != second
    reference = np.random.RandomState(0)
    assert first == list(reference.permutation(8)) and second == list(reference.permutation(8))
    with pytest.raises(ValueError, match="shuffle"):
        DatasetEnvironment(X, y, shuffle=True)
    with pytest.raises(ValueError, match="same number of rows"):
        DatasetEnvironment(X, y[:3])


def test_real_function_environment_samples_continuous_inputs_in_the_domain():
    environment = RealFunctionEnvironment(lambda x: x[0] - x[1], [1000., 0.], [1100., 1.],
                                          np.random.RandomState(1))
    seen = problems(environment, 500)
    states = np.array([state for state, _, _ in seen])
    assert np.all(states >= [1000., 0.]) and np.all(states < [1100., 1.])
    assert np.all(states != np.round(states))  # never discretized
    assert_allclose([reward for _, reward, _ in seen], states[:, 0] - states[:, 1])
    with pytest.raises(ValueError, match="lower < upper"):
        RealFunctionEnvironment(np.sum, [0.], [0.], np.random.RandomState(1))


def test_classifier_system_learns_a_function_environment_without_the_estimator():
    rng = np.random.RandomState(7)
    parameters = XCSFRegressor(prediction="rls", population_size=200, epsilon_0=.02).get_params()
    system = XCSFClassifierSystem(parameters, rng)
    environment = RealFunctionEnvironment(lambda x: np.sin(2 * np.pi * x[0]), [0.], [1.], rng)
    errors = []
    run_problems(system, environment, 4000,
                 on_problem=lambda prediction, target: errors.append(abs(target - prediction)))
    assert len(errors) == system.time == 4000
    assert np.mean(errors[-500:]) < .05 < np.mean(errors[:100])
    assert system.stats["ga_runs"] > 0 and system.numerosity <= 200


def test_estimator_is_the_experiment_loop_on_a_dataset_environment():
    rng = np.random.RandomState(4)
    X = rng.uniform(size=(300, 1))
    y = np.sin(2 * np.pi * X[:, 0])
    kwargs = dict(n_epochs=1, shuffle=False, normalize=False, random_state=5)
    model = XCSFRegressor(**kwargs).fit(X, y)
    system = XCSFClassifierSystem(XCSFRegressor(**kwargs).get_params(), np.random.RandomState(5))
    run_problems(system, DatasetEnvironment(X, y), len(X))
    assert system.stats == model.stats_
    assert_array_equal([cl.fitness for cl in system.population],
                       [cl.fitness for cl in model.population_])


def test_training_monitor_windows_and_open_tail():
    class Population:
        population, numerosity = [1, 2], 5

    monitor = TrainingMonitor(2)
    monitor.begin_pass()
    for step, error in enumerate([1., -3., 2.], 1):
        monitor.add(error, step, Population)
    monitor.end_pass(3, Population)
    assert [(r["step"], r["n_samples"], r["mae"], r["complete"]) for r in monitor.records] == [
        (2, 2, 2., True), (3, 1, 2., False)]
    monitor.begin_pass()
    monitor.add(4., 4, Population)
    monitor.end_pass(4, Population)
    assert [(r["step"], r["mae"], r["mse"], r["complete"]) for r in monitor.records] == [
        (2, 2., 5., True), (4, 3., 10., True)]
    assert monitor.records[-1]["macroclassifiers"] == 2 and monitor.records[-1]["microclassifiers"] == 5
    disabled = TrainingMonitor(None)
    disabled.add(1., 1, Population)
    disabled.end_pass(1, Population)
    assert disabled.records == []
