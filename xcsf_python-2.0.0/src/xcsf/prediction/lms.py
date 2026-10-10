"""Stochastic-gradient predictors: LMS and normalized LMS (cf. xcslib ``pf/nlms.cpp``)."""

import numpy as np

from .base import LocalPredictor, check_real


class LMSPredictor(LocalPredictor):
    """Widrow-Hoff rule: w <- w + eta * (y - phi @ w) * phi."""

    name = "lms"
    parameters = {"learning_rate": "prediction_learning_rate"}

    def __init__(self, size, *, learning_rate=0.2, initial_prediction=0.0, x0=1.0):
        super().__init__(size, initial_prediction=initial_prediction, x0=x0)
        check_real("learning_rate", learning_rate)
        self.learning_rate = learning_rate

    def _step(self, phi, error):
        return self.learning_rate * error * phi

    def _update(self, phi, target, squared_error):
        step = self._step(phi, target - self.predict(phi))
        if step is None:
            return
        weights = self.weights + step
        if not np.isfinite(weights).all():
            raise FloatingPointError("Nonfinite prediction weights.")
        self.weights = weights


class NLMSPredictor(LMSPredictor):
    """Normalized rule: w <- w + eta * (y - phi @ w) * phi / (phi @ phi)."""

    name = "nlms"

    def _step(self, phi, error):
        # Scaling first avoids overflow/underflow when computing phi @ phi.
        scale = np.max(np.abs(phi))
        if scale == 0:
            return None
        unit = phi / scale
        return (self.learning_rate * error / scale) * unit / (unit @ unit)
