"""The single implicit action of function approximation (cf. xcslib ``dummy_action``)."""

from dataclasses import dataclass

from .base import Action


@dataclass(frozen=True)
class DummyAction(Action):
    """An action without content: all instances are equal, so the action set of a
    regression problem is its whole match set."""

    def __str__(self):
        return "#"
