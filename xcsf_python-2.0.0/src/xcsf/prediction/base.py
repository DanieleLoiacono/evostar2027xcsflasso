"""Common interface of the local predictors (cf. xcslib ``xcsf/pf/base.h``).

Every classifier owns one predictor: a linear model w of the prediction inputs
phi = [x0, x1, ..., xd, x1**2, ...]. Subclasses differ only in how one
observation (phi, target) changes w; docs/prediction-updates.md derives each rule.
"""

from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np


def design_matrix(X, degree=1, x0=1.0):
    """[x0, x1, ..., xd, x1**2, ..., xd**degree], without cross terms."""
    X = np.asarray(X, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.concatenate(
            [np.full((*X.shape[:-1], 1), x0)]
            + [X ** power for power in range(1, degree + 1)], axis=-1,
        )
    if not np.isfinite(result).all():
        raise ValueError("Polynomial features overflow; scale X or reduce degree.")
    return result


def soft_threshold(value, threshold):
    """Proximal operator of threshold * |.|: shrink towards zero, exactly zero inside."""
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def check_real(name, value, *, positive=True, maximum=None):
    """Validate a finite real hyper-parameter that is positive or nonnegative."""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")


def check_integer(name, value, *, optional=False):
    """Validate a positive integer hyper-parameter; None is accepted if optional."""
    if optional and value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer" + (" or None." if optional else "."))


class LocalPredictor(ABC):
    """One scalar linear predictor per classifier; inputs include the constant bias.

    Subclasses set ``name`` (the value of ``XCSFRegressor(prediction=...)``) and
    ``parameters``, which maps each constructor keyword to the estimator
    parameter that feeds it. Hyper-parameters are stored under the keyword name.
    Offspring inherit the weights and restart every other part of the state.
    """

    name = None
    parameters = {}

    def __init__(self, size, *, initial_prediction=0.0, x0=1.0):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 1:
            raise ValueError("size must be a positive integer.")
        check_real("x0", x0)
        if (isinstance(initial_prediction, (bool, np.bool_))
                or not isinstance(initial_prediction, Real) or not np.isfinite(initial_prediction)):
            raise ValueError("initial_prediction must be a finite real number.")
        self.x0 = x0
        self.weights = np.zeros(size)
        self.weights[0] = initial_prediction / x0
        if not np.isfinite(self.weights[0]):
            raise ValueError("initial_prediction / x0 must be finite.")
        self.n_updates_ = 0

    @property
    def method(self):
        """Name under which the predictor is selected."""
        return self.name

    @property
    def value(self):
        """Scalar output of a constant predictor; None for linear predictors."""
        return None

    def hyperparameters(self):
        """Constructor keywords that reproduce this predictor's settings."""
        return {keyword: getattr(self, keyword) for keyword in self.parameters}

    def predict(self, phi):
        return np.asarray(phi, dtype=float) @ self.weights

    def update(self, phi, target, squared_error=1.0):
        """Learn from one observation. ``squared_error`` is the owning classifier's
        estimate of its squared prediction error; most predictors ignore it."""
        phi = np.asarray(phi, dtype=float)
        if phi.shape != self.weights.shape or not np.isfinite(phi).all():
            raise ValueError("phi must be a finite vector with the predictor's size.")
        if np.ndim(target) != 0 or not np.isfinite(target):
            raise ValueError("target must be a finite scalar.")
        self._check_observation(phi, squared_error)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                self._update(phi, float(target), squared_error)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise FloatingPointError(
                "Local prediction update failed numerically; scale inputs/targets "
                "or reduce prediction_learning_rate."
            ) from exc
        self.n_updates_ += 1

    def _check_observation(self, phi, squared_error):
        """Reject observations a predictor cannot use, before any state changes."""

    @abstractmethod
    def _update(self, phi, target, squared_error):
        """Change the state for one validated observation; all or nothing."""

    def offspring(self):
        """A new predictor with the same settings and a copy of the weights."""
        child = type(self)(len(self.weights), x0=self.x0, **self.hyperparameters())
        child.weights = self.weights.copy()
        return child

    def diagnostics(self):
        """Counters describing the last update, for inspection only."""
        return dict(n_updates=self.n_updates_, n_samples=0, n_iter=0, converged=None,
                    kkt_violation=None)
