"""Classifier actions (cf. xcslib ``actions/``).

Function approximation needs no decision: every classifier advocates the same
implicit action and its prediction estimates the target. xcslib expresses this
by building XCSF with ``ACTIONS=dummy_action``; ``DummyAction`` is its counterpart.
"""

from .base import Action
from .dummy import DummyAction

__all__ = ["Action", "DummyAction"]
