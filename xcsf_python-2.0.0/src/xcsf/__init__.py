"""Accuracy-based evolutionary function approximation, without native XCSF code."""

from .actions import DummyAction
from .classifier import Classifier
from .classifier_system import XCSFClassifierSystem
from .conditions import IntervalCondition, RealIntervalRepresentation
from .environments import DatasetEnvironment, Environment, RealFunctionEnvironment
from .prediction import LocalPredictor, PredictorFactory, make_predictor
from .regressor import XCSFRegressor

__version__ = "2.1.0"
__all__ = ["XCSFRegressor", "XCSFClassifierSystem", "Classifier", "IntervalCondition",
           "RealIntervalRepresentation", "DummyAction", "Environment", "DatasetEnvironment",
           "RealFunctionEnvironment", "LocalPredictor", "PredictorFactory", "make_predictor"]
