"""Validation of the XCSFRegressor hyper-parameters, independent of any training state."""

from numbers import Integral, Real

import numpy as np
from sklearn.utils import check_random_state

from .conditions import CROSSOVER_METHODS, MUTATION_METHODS
from .prediction import PREDICTION_METHODS, PredictorFactory

SELECTION_METHODS = ("roulette", "tournament")
UNMATCHED_POLICIES = ("nearest", "mean", "raise")

# name -> smallest admissible value
INTEGERS = {"population_size": 1, "n_epochs": 1, "degree": 1, "condensation_epochs": 0,
            "niche_history": 0, "lasso_max_iter": 1}
OPTIONAL_POSITIVE_INTEGERS = ("history_interval", "lasso_window")
POSITIVE = ("learning_rate", "prediction_learning_rate", "epsilon_0", "alpha", "nu",
            "cover_radius", "delta", "tournament_fraction", "initial_fitness",
            "forgetting_factor", "x0", "lasso_tol")
NONNEGATIVE = ("theta_ga", "crossover_probability", "mutation_probability", "mutation_scale",
               "theta_delete", "theta_subsume", "theta_match_subsume", "initial_error",
               "rls_delta", "process_noise", "lasso_alpha", "lasso_learning_rate_decay")
AT_MOST_ONE = ("learning_rate", "prediction_learning_rate", "alpha", "delta",
               "tournament_fraction", "forgetting_factor", "crossover_probability",
               "mutation_probability", "lasso_learning_rate_decay")
BOOLEANS = ("ga_subsumption", "match_subsumption", "use_mam", "error_before_prediction",
            "kalman_noise", "normalize", "bounded", "discovery", "shuffle")
CHOICES = {"prediction": PREDICTION_METHODS, "mutation": MUTATION_METHODS,
           "crossover": CROSSOVER_METHODS, "selection": SELECTION_METHODS,
           "unmatched": UNMATCHED_POLICIES}


def _is_integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def validate_parameters(parameters):
    """Raise ValueError naming the first XCSFRegressor parameter outside its domain."""
    for name in OPTIONAL_POSITIVE_INTEGERS:
        value = parameters[name]
        if value is not None and (not _is_integer(value) or value < 1):
            raise ValueError(f"{name} must be a positive integer or None.")
    for name, minimum in INTEGERS.items():
        value = parameters[name]
        if not _is_integer(value) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name in POSITIVE + NONNEGATIVE + ("initial_prediction",):
        value = parameters[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real number.")
        if (name in POSITIVE and value <= 0) or (name in NONNEGATIVE and value < 0):
            raise ValueError(f"{name} is outside its valid range.")
        if name in AT_MOST_ONE and value > 1:
            raise ValueError(f"{name} must be <= 1.")
    for name in BOOLEANS:
        if not isinstance(parameters[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean.")
    for name, choices in CHOICES.items():
        if parameters[name] not in choices:
            raise ValueError(f"{name} must be one of {choices}.")
    check_random_state(parameters["random_state"])
    # Constraints that involve the selected predictor (e.g. rls_delta = 0).
    PredictorFactory(parameters["prediction"], parameters)
