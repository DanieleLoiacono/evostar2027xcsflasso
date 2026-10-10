"""Local predictors (cf. xcslib ``xcsf/pf/``) and their selection by name."""

from .base import LocalPredictor, design_matrix, soft_threshold
from .constant import ConstantPredictor
from .lasso import LassoBatchPredictor, LassoOnlinePredictor, LassoSGDPredictor
from .lms import LMSPredictor, NLMSPredictor
from .rls import RLSPredictor

PREDICTORS = {cls.name: cls for cls in (
    ConstantPredictor, LMSPredictor, NLMSPredictor, RLSPredictor,
    LassoOnlinePredictor, LassoSGDPredictor, LassoBatchPredictor,
)}
PREDICTION_METHODS = tuple(PREDICTORS)

# Constructor keywords shared by every predictor -> XCSFRegressor parameter.
COMMON_PARAMETERS = {"initial_prediction": "initial_prediction", "x0": "x0"}


def make_predictor(method, size, **keywords):
    """Create the predictor registered as ``method`` with its own constructor keywords."""
    if method not in PREDICTORS:
        raise ValueError(f"prediction must be one of {PREDICTION_METHODS}.")
    return PREDICTORS[method](size, **keywords)


class PredictorFactory:
    """Creates the predictors of one classifier system (cf. xcslib ``get_prediction_function``).

    ``parameters`` uses the names of the XCSFRegressor parameters; each predictor
    class picks the ones it declares, so settings of other predictors are inert.
    The settings are validated once, here, rather than at the first covering.
    """

    def __init__(self, method, parameters):
        if method not in PREDICTORS:
            raise ValueError(f"prediction must be one of {PREDICTION_METHODS}.")
        self.method = method
        self.predictor_class = PREDICTORS[method]
        names = {**COMMON_PARAMETERS, **self.predictor_class.parameters}
        self.keywords = {keyword: parameters[name] for keyword, name in names.items()
                         if name in parameters}
        try:
            self.predictor_class(1, **self.keywords)
        except ValueError as exc:
            renamed = ", ".join(f"{keyword} is {name}" for keyword, name in names.items()
                                if keyword != name)
            raise ValueError(f"prediction='{method}': {exc}"
                             + (f" (here {renamed})" if renamed else "")) from exc

    def __call__(self, size):
        return self.predictor_class(size, **self.keywords)


__all__ = ["LocalPredictor", "ConstantPredictor", "LMSPredictor", "NLMSPredictor",
           "RLSPredictor", "LassoOnlinePredictor", "LassoSGDPredictor", "LassoBatchPredictor",
           "PREDICTORS", "PREDICTION_METHODS", "PredictorFactory", "make_predictor",
           "design_matrix", "soft_threshold"]
