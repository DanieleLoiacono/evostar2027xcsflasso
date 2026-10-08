import numpy as np
from numpy.testing import assert_allclose
import pytest

from xcsf import Classifier, IntervalCondition, LocalPredictor, XCSFRegressor
from xcsf.core import XCSFCore


def core(**kwargs):
    return XCSFCore(XCSFRegressor(**kwargs).get_params(), np.random.RandomState(2))


def rule(lower=0, upper=1, *, fitness=0.01, numerosity=1, error=0, experience=0):
    return Classifier(IntervalCondition([lower], [upper]), LocalPredictor(2),
                      fitness=fitness, numerosity=numerosity, error=error, experience=experience)


def test_error_fitness_and_set_size_hand_calculation():
    engine = core(discovery=False, learning_rate=0.2, epsilon_0=0.5)
    a, b = rule(numerosity=2), rule()
    b.predictor.weights[0] = 1
    engine.population.extend([a, b])
    engine.update(np.array([0.5]), 1, np.array([1., 0.5]))
    assert_allclose([a.error, b.error], [1, 0])
    assert_allclose([a.set_size, b.set_size], [3, 3])
    assert_allclose([a.experience, b.experience], [1, 1])
    raw = np.array([0.1 * (1 / 0.5)**-5 * 2, 1])
    expected = 0.01 + 0.2 * (raw / raw.sum() - 0.01)
    assert_allclose([a.fitness, b.fitness], expected)
    assert_allclose(a.predictor.weights, [0.16, 0.08])


def test_post_update_error_and_non_mam():
    engine = core(discovery=False, use_mam=False, error_before_prediction=False)
    a = rule()
    engine.population.append(a)
    engine.update(np.array([0.5]), 1, np.array([1., 0.5]))
    assert_allclose(a.error, 0.2 * 0.8)
    assert_allclose(a.squared_error, 0.2 * 0.8**2)


def test_prediction_does_not_double_count_numerosity():
    a, b = rule(fitness=0.25, numerosity=20), rule(fitness=0.75)
    a.predictor.weights[0], b.predictor.weights[0] = 2, 6
    assert_allclose(XCSFCore.aggregate([a, b], np.array([1., 0.5])), 5)


def test_deletion_uses_per_micro_fitness_and_experience():
    engine = core(theta_delete=20)
    a, b = rule(fitness=1, numerosity=2), rule(fitness=0.001, experience=21)
    a.set_size, b.set_size = 3, 4
    engine.population.extend([a, b])
    mean = 1.001 / 3
    assert_allclose(engine.deletion_votes(), [6, 4 * mean / 0.001])
    engine._delete_to_size(1)
    assert engine.numerosity == 1
    assert engine.stats["deletions"] == 2


def test_equal_conditions_merge_without_overwriting_predictor():
    engine = core()
    a, b = rule(), rule()
    b.predictor.weights[:] = 5
    engine._insert(a)
    engine._insert(b)
    assert engine.numerosity == 2 and len(engine.population) == 1
    assert_allclose(a.predictor.weights, [0, 0])


def test_match_subsumption_preserves_population_numerosity():
    engine = core(match_subsumption=True, theta_match_subsume=10)
    a, b = rule(experience=11), rule(0.2, 0.8, numerosity=3)
    engine.population.extend([a, b])
    assert engine._subsume_match_set([a, b]) == [a]
    assert engine.population == [a] and a.numerosity == 4


def test_ga_parent_subsumption_and_timestamp():
    engine = core(theta_ga=0, theta_subsume=0, mutation_probability=0)
    a = rule(experience=2, fitness=1)
    engine.population.append(a)
    engine.time = 19
    engine._evolve([a], condensation=False)
    assert a.numerosity == 3 and a.timestamp == 19
    assert engine.stats["subsumptions"] == 2


def test_ga_age_is_numerosity_weighted():
    engine = core(theta_ga=5)
    a, b = rule(numerosity=9), rule()
    a.timestamp, b.timestamp = 10, 0
    engine.population.extend([a, b])
    engine.time = 10
    engine.update(np.array([0.5]), 0, np.array([1., 0.5]))
    assert engine.stats["ga_runs"] == 0  # average age 11 - 9 = 2


def test_condensation_only_reproduces_existing_conditions():
    engine = core(population_size=3)
    engine.population.extend([rule(), rule(-1, 2)])
    before = [(cl.condition.lower.copy(), cl.condition.upper.copy()) for cl in engine.population]
    engine._evolve(list(engine.population), condensation=True)
    assert engine.numerosity == 3
    assert engine.stats["ga_runs"] == 0 and engine.stats["condensation_runs"] == 1
    assert all(any(np.array_equal(cl.condition.lower, lo) and np.array_equal(cl.condition.upper, hi)
                   for lo, hi in before) for cl in engine.population)


@pytest.mark.parametrize("fraction", [1., 0.4, 1e-12])
def test_tournament_handles_small_participation_without_hanging(fraction):
    engine = core(selection="tournament", tournament_fraction=fraction)
    a = rule()
    assert engine._select([a]) is a


@pytest.mark.parametrize("method", ["fixed", "proportional", "gaussian"])
def test_mutation_preserves_valid_bounds(method):
    rng = np.random.RandomState(1)
    condition = IntervalCondition([0.1, 0.3], [0.7, 0.8])
    for _ in range(100):
        condition.mutate(1., 1., method, rng, (np.zeros(2), np.ones(2)))
        assert np.all(condition.lower <= condition.upper)
        assert np.all(condition.lower >= 0) and np.all(condition.upper <= 1)


def test_upper_endpoint_mutates_independently_regression_for_cpp_bug():
    condition = IntervalCondition([0.1], [0.9])
    condition.mutate(1., 0.02, "fixed", np.random.RandomState(1))
    assert condition.upper[0] > 0.85 and condition.lower[0] < 0.15
    assert condition.upper[0] != 0.9


@pytest.mark.parametrize("method", ["uniform", "one_point", "two_point"])
def test_crossover_preserves_matching_parents_common_point(method):
    rng = np.random.RandomState(9)
    for _ in range(30):
        a = IntervalCondition([0, 0], [0.8, 0.9])
        b = IntervalCondition([0.3, 0.2], [1, 1])
        a.crossover(b, method, rng)
        assert a.matches([0.5, 0.5]) and b.matches([0.5, 0.5])


def test_cover_matches_boundaries_and_constant_features():
    x = np.array([0., 1., 0.5])
    bounds = (np.array([0., 0., 0.5]), np.array([1., 1., 0.5]))
    condition = IntervalCondition.cover(x, 0.2, np.random.RandomState(4), bounds)
    assert condition.matches(x)
    assert not condition.matches([0, 1.01, 0.5])

