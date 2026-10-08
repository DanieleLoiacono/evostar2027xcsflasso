"""Accuracy-based evolutionary function approximation, without native XCSF code."""

from .regressor import XCSFRegressor
from .conditions import IntervalCondition
from .prediction import LocalPredictor
from .rule import Classifier

__version__ = "2.0.0"
__all__ = ["XCSFRegressor", "IntervalCondition", "LocalPredictor", "Classifier"]

