"""Interface of classifier actions (cf. xcslib ``actions/action_base.h``)."""

from abc import ABC


class Action(ABC):
    """What a classifier advocates. Actions must be immutable, hashable and comparable:
    two classifiers are the same rule only if both condition and action coincide."""
