"""Interface of single-step environments (cf. xcslib ``environments/environment_base.h``).

A problem is one input (the state) followed by one scalar reward. In function
approximation the reward is the target value f(x), so learning is supervised.
"""

from abc import ABC, abstractmethod


class Environment(ABC):
    """Source of learning problems for the classifier system."""

    def begin_experiment(self):
        """Called once before the first problem."""

    def end_experiment(self):
        """Called once after the last problem."""

    @abstractmethod
    def begin_problem(self):
        """Select the state of the next problem."""

    def end_problem(self):
        """Called after the reward of the current problem has been used."""

    @abstractmethod
    def state(self):
        """Input vector of the current problem."""

    def features(self):
        """Optional prediction inputs of the current state, precomputed by the
        environment; None lets the classifier system compute them."""
        return None

    def perform(self, action):
        """Execute ``action``; it has no effect in function approximation."""

    @abstractmethod
    def reward(self):
        """Scalar target of the current problem."""
