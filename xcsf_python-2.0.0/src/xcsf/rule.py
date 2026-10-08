"""Classifier state; population entries are macroclassifiers."""

from collections import deque
from dataclasses import dataclass, field

from .conditions import IntervalCondition
from .prediction import LocalPredictor


@dataclass(eq=False)
class Classifier:
    """A region, a local predictor, accuracy statistics, and microclass count.

    Fitness is the aggregate macroclassifier fitness: it is NOT multiplied by
    numerosity a second time when computing predictions or roulette selection.
    """

    condition: IntervalCondition
    predictor: LocalPredictor
    identifier: int = 0
    error: float = 0.0
    squared_error: float = 0.0
    fitness: float = 0.01
    set_size: float = 1.0
    experience: int = 0
    numerosity: int = 1
    timestamp: int = 0
    created_at: int = 0
    last_match: int = 0
    match_history: deque = field(default_factory=deque)

    def can_subsume(self, epsilon_0, threshold):
        return self.experience > threshold and self.error < epsilon_0

    def offspring(self, identifier, timestamp, history_size):
        return type(self)(
            condition=self.condition.copy(), predictor=self.predictor.offspring(),
            identifier=identifier, error=self.error, squared_error=self.squared_error,
            fitness=self.fitness, set_size=self.set_size, timestamp=timestamp,
            created_at=timestamp, match_history=deque(maxlen=history_size),
        )

