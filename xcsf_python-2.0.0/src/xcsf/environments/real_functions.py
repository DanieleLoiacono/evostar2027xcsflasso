"""Environment that samples a real function (cf. xcslib ``real_functions_env``)."""

import numpy as np

from .base import Environment


class RealFunctionEnvironment(Environment):
    """Each problem draws x uniformly in [lower, upper) and rewards f(x).

    ``function`` maps one input vector to a scalar. Inputs are continuous
    floating-point values; the domain is never discretized.
    """

    def __init__(self, function, lower, upper, rng):
        self.function = function
        self.lower = np.atleast_1d(np.asarray(lower, dtype=float))
        self.upper = np.atleast_1d(np.asarray(upper, dtype=float))
        if self.lower.shape != self.upper.shape or np.any(self.lower >= self.upper):
            raise ValueError("lower and upper must have the same shape with lower < upper.")
        self.rng = rng
        self._state = None

    def begin_problem(self):
        self._state = self.lower + self.rng.random_sample(len(self.lower)) * (self.upper - self.lower)

    def state(self):
        return self._state

    def reward(self):
        return float(self.function(self._state))
