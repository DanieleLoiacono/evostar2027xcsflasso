"""Interfaces of classifier conditions (cf. xcslib ``conditions/condition_base.h``).

A ``Condition`` is one region of the input space. A ``ConditionRepresentation``
owns the settings of a condition type and is the only object the classifier
system talks to when it has to create, mutate, recombine or match conditions,
so a new condition type never requires changes to the classifier system.
"""

from abc import ABC, abstractmethod

import numpy as np


class Condition(ABC):
    """A region of the input space advocated by one classifier."""

    @abstractmethod
    def matches(self, X):
        """Return a bool for one point or a boolean vector for a batch."""

    @abstractmethod
    def contains(self, other):
        """Whether this region is at least as general as ``other`` (equality allowed)."""

    @abstractmethod
    def same_as(self, other):
        """Whether the two conditions describe exactly the same region."""

    @abstractmethod
    def distance(self, X):
        """Distance from this region; zero for the points it matches."""

    @abstractmethod
    def copy(self):
        """Return an independent copy."""


class PopulationMatcher:
    """Matches one input against many conditions; representations may vectorize it."""

    def __init__(self, conditions):
        self.conditions = list(conditions)

    def matching(self, x):
        """Boolean mask of the conditions that match the point ``x``."""
        return np.array([condition.matches(x) for condition in self.conditions], dtype=bool)

    def distances(self, x):
        """Distance of the point ``x`` from every condition."""
        return np.array([condition.distance(x) for condition in self.conditions], dtype=float)


class ConditionRepresentation(ABC):
    """Covering and genetic operators of one condition type, bound to their settings."""

    @abstractmethod
    def cover(self, x, rng):
        """Create a condition that matches ``x``."""

    @abstractmethod
    def mutate(self, condition, probability, rng):
        """Mutate ``condition`` in place."""

    @abstractmethod
    def crossover(self, first, second, rng):
        """Recombine two conditions in place."""

    def matcher(self, conditions):
        """Return an object answering match queries for a fixed list of conditions."""
        return PopulationMatcher(conditions)
