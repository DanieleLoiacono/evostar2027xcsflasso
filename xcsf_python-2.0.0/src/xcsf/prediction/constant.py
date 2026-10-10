"""Constant prediction (cf. xcslib ``pf/value.cpp``)."""

import numpy as np

from .base import LocalPredictor, check_real


class ConstantPredictor(LocalPredictor):
    """Predicts one value for the whole region: value <- value + eta * (y - value)."""

    name = "constant"
    parameters = {"learning_rate": "prediction_learning_rate"}

    def __init__(self, size, *, learning_rate=0.2, initial_prediction=0.0, x0=1.0):
        super().__init__(size, initial_prediction=initial_prediction, x0=x0)
        check_real("learning_rate", learning_rate)
        self.learning_rate = learning_rate
        self._value = float(initial_prediction)

    @property
    def value(self):
        return self._value

    def predict(self, phi):
        return np.full(np.asarray(phi, dtype=float).shape[:-1], self._value)

    def _update(self, phi, target, squared_error):
        value = float(self._value + self.learning_rate * (target - self._value))
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite constant prediction.")
        self._value = value
        # Keep the linear view consistent: phi @ weights == value whenever phi[0] == x0.
        self.weights[0] = value / self.x0

    def offspring(self):
        child = super().offspring()
        child._value = self._value
        return child
