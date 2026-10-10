"""Environment that presents the rows of a dataset, one per problem."""

import numpy as np

from .base import Environment


class DatasetEnvironment(Environment):
    """Presents (X[i], y[i]) in order, or in a new random order at every pass.

    ``len(environment)`` problems make one pass (epoch). With ``shuffle`` the
    permutation of a pass is drawn from ``rng`` when its first problem begins.
    ``features`` optionally holds one precomputed prediction input per row.
    """

    def __init__(self, X, y, *, features=None, shuffle=False, rng=None):
        if len(X) != len(y) or (features is not None and len(features) != len(X)):
            raise ValueError("X, y and features must have the same number of rows.")
        if shuffle and rng is None:
            raise ValueError("shuffle requires a random number generator.")
        self.X = X
        self.y = y
        self._features = features
        self.shuffle = shuffle
        self.rng = rng
        self._order = None
        self._position = len(X)
        self._row = None

    def __len__(self):
        return len(self.X)

    def begin_problem(self):
        if self._position >= len(self.X):
            self._order = (self.rng.permutation(len(self.X)) if self.shuffle
                           else np.arange(len(self.X)))
            self._position = 0
        self._row = self._order[self._position]
        self._position += 1

    def state(self):
        return self.X[self._row]

    def features(self):
        return None if self._features is None else self._features[self._row]

    def reward(self):
        return self.y[self._row]
