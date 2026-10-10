"""Real interval conditions and genetic operators from xcslib.

Intervals are closed. This deliberately includes the lower endpoint, including
for constant features. See docs/algorithm.md for differences from the C++ code.
"""

from dataclasses import dataclass

import numpy as np

from .base import Condition, ConditionRepresentation

MUTATION_METHODS = ("fixed", "proportional", "gaussian")
CROSSOVER_METHODS = ("one_point", "two_point", "uniform")


@dataclass(eq=False)
class IntervalCondition(Condition):
    """Axis-aligned region; lower and upper are independent 1-D float arrays."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        self.lower = np.array(self.lower, dtype=float, copy=True)
        self.upper = np.array(self.upper, dtype=float, copy=True)
        if (self.lower.ndim != 1 or self.lower.size == 0
                or self.lower.shape != self.upper.shape
                or not np.isfinite([self.lower, self.upper]).all()
                or np.any(self.lower > self.upper)):
            raise ValueError("Condition bounds must be finite 1-D arrays with lower <= upper.")

    @classmethod
    def cover(cls, x, radius, rng, bounds=None):
        """Create a region containing x, with independent U(0, radius) sides."""
        lower = x - radius * rng.random_sample(len(x))
        upper = x + radius * rng.random_sample(len(x))
        if bounds is not None:
            lower = np.maximum(lower, bounds[0])
            upper = np.minimum(upper, bounds[1])
        return cls(lower, upper)

    def matches(self, X):
        """Return a bool for one point or a boolean vector for a batch."""
        return np.all((X >= self.lower) & (X <= self.upper), axis=-1)

    def contains(self, other):
        """Whether this region contains the entire other region (equality allowed)."""
        return bool(np.all(self.lower <= other.lower) and np.all(self.upper >= other.upper))

    def same_as(self, other):
        return (np.array_equal(self.lower, other.lower)
                and np.array_equal(self.upper, other.upper))

    def distance(self, X):
        """Euclidean distance to this region; zero for points inside it."""
        return np.linalg.norm(np.maximum(np.maximum(self.lower - X, X - self.upper), 0), axis=-1)

    def copy(self):
        return type(self)(self.lower, self.upper)

    def _repair(self, bounds):
        lower = np.minimum(self.lower, self.upper)
        upper = np.maximum(self.lower, self.upper)
        if bounds is not None:
            lower = np.clip(lower, bounds[0], bounds[1])
            upper = np.clip(upper, bounds[0], bounds[1])
        self.lower, self.upper = lower, upper

    def mutate(self, probability, scale, method, rng, bounds=None):
        """Mutate in place; offspring need not match the generating input."""
        n = len(self.lower)
        if method == "proportional":
            for i in range(n):
                if rng.random_sample() < probability:
                    width = self.upper[i] - self.lower[i]
                    center = self.lower[i] + rng.random_sample() * width
                    factor = (0.5 if rng.random_sample() < 0.5 else 1.0)
                    width *= factor + 0.5 * rng.random_sample()
                    self.lower[i], self.upper[i] = center - width / 2, center + width / 2
        elif method in ("fixed", "gaussian"):
            for values in (self.lower, self.upper):
                mask = rng.random_sample(n) < probability
                changes = (rng.uniform(-scale, scale, n) if method == "fixed"
                           else rng.normal(0, scale, n))
                values += mask * changes
        else:
            raise ValueError("Unknown mutation method.")
        self._repair(bounds)

    def crossover(self, other, method, rng, bounds=None):
        """Exchange interval endpoints in place; predictors are not crossed."""
        a = np.column_stack((self.lower, self.upper)).ravel()
        b = np.column_stack((other.lower, other.upper)).ravel()
        if method == "uniform":
            mask = rng.random_sample(len(a)) < 0.5
        elif method == "one_point":
            mask = np.arange(len(a)) >= rng.randint(len(a))
        elif method == "two_point":
            # The original operator selects two feature positions, then chooses
            # which endpoint to exchange at each boundary (even if they coincide).
            i, j = sorted(rng.randint(len(self.lower), size=2))
            mask = np.zeros(len(a), dtype=bool)
            start = 2 * i + int(rng.random_sample() >= 0.5)
            mask[start:2 * i + 2] ^= True
            mask[2 * i + 2:2 * j] ^= True
            end = 2 * j + (2 if rng.random_sample() < 0.5 else 1)
            mask[2 * j:end] ^= True
        else:
            raise ValueError("Unknown crossover method.")
        a[mask], b[mask] = b[mask].copy(), a[mask].copy()
        self.lower, self.upper = a[::2].copy(), a[1::2].copy()
        other.lower, other.upper = b[::2].copy(), b[1::2].copy()
        self._repair(bounds)
        other._repair(bounds)


class IntervalMatcher:
    """Vectorized match queries over the stacked bounds of many interval conditions."""

    def __init__(self, conditions):
        self.lower = np.array([condition.lower for condition in conditions])
        self.upper = np.array([condition.upper for condition in conditions])

    def matching(self, x):
        return np.all((x >= self.lower) & (x <= self.upper), axis=1)

    def distances(self, x):
        return np.linalg.norm(np.maximum(np.maximum(self.lower - x, x - self.upper), 0), axis=1)


class RealIntervalRepresentation(ConditionRepresentation):
    """Interval conditions with the settings of xcslib's ``<condition::real_interval>``.

    ``cover_radius`` is r0, ``mutation_scale`` is m0. ``bounds`` is an optional
    (lower, upper) pair of arrays to which every condition is clipped.
    """

    def __init__(self, cover_radius=0.2, mutation_scale=0.2, mutation="fixed",
                 crossover="one_point", bounds=None):
        if mutation not in MUTATION_METHODS:
            raise ValueError(f"mutation must be one of {MUTATION_METHODS}.")
        if crossover not in CROSSOVER_METHODS:
            raise ValueError(f"crossover must be one of {CROSSOVER_METHODS}.")
        self.cover_radius = cover_radius
        self.mutation_scale = mutation_scale
        self.mutation = mutation
        self.crossover_method = crossover
        self.bounds = bounds

    def cover(self, x, rng):
        return IntervalCondition.cover(x, self.cover_radius, rng, self.bounds)

    def mutate(self, condition, probability, rng):
        condition.mutate(probability, self.mutation_scale, self.mutation, rng, self.bounds)

    def crossover(self, first, second, rng):
        first.crossover(second, self.crossover_method, rng, self.bounds)

    def matcher(self, conditions):
        return IntervalMatcher(conditions)
