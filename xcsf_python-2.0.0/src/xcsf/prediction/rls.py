"""Recursive least squares (cf. xcslib ``pf/rls.cpp``, ``pf/rlsk.cpp``, ``pf/rls_delta.cpp``)."""

import numpy as np

from . import information
from .base import LocalPredictor, check_real

MINIMUM_NOISE_VARIANCE = 1e-4


class RLSPredictor(LocalPredictor):
    """The single RLS of the library, in square-root information (QR) form.

    It minimizes, over the observations i = 1..t of the classifier,

        sum(lambda**(t-i) / r_i * (y_i - phi_i @ w)**2) + lambda**t / delta * |w - w0|**2

    and optionally lets w drift as a random walk of variance ``process_noise``.
    Equivalent covariance-form recursion (Kalman filter with state w):

        P <- P / lambda;  k = P phi / (r + phi @ P phi);  w <- w + k (y - phi @ w)
        P <- P - k phi.T P + process_noise * I,           P0 = delta * I

    delta : initial covariance scale. Zero means "weights known exactly" and is
        accepted only with process_noise > 0: the first observation is then
        ignored and P becomes process_noise * I (the behaviour of xcslib's rls).
    forgetting_factor : lambda in (0, 1]; it also discounts the prior.
    process_noise : variance added to every diagonal entry of P after an update.
    kalman_noise : use r = max(squared error of the classifier, 1e-4) instead of 1.

    With lambda = 1, process_noise = 0 and kalman_noise = False this is the RLS
    of Lanzi et al. (2005), Algorithm 5. ``covariance`` reconstructs P on demand.
    """

    name = "rls"
    parameters = {"delta": "rls_delta", "forgetting_factor": "forgetting_factor",
                  "process_noise": "process_noise", "kalman_noise": "kalman_noise"}

    def __init__(self, size, *, delta=1000.0, forgetting_factor=1.0, process_noise=0.0,
                 kalman_noise=False, initial_prediction=0.0, x0=1.0):
        super().__init__(size, initial_prediction=initial_prediction, x0=x0)
        check_real("delta", delta, positive=False)
        check_real("forgetting_factor", forgetting_factor, maximum=1)
        check_real("process_noise", process_noise, positive=False)
        if not isinstance(kalman_noise, (bool, np.bool_)):
            raise ValueError("kalman_noise must be boolean.")
        if delta == 0 and process_noise == 0:
            raise ValueError("delta = 0 (rls_delta) requires process_noise > 0: "
                             "with zero covariance and no process noise RLS cannot learn.")
        self.delta = delta
        self.forgetting_factor = forgetting_factor
        self.process_noise = process_noise
        self.kalman_noise = kalman_noise
        # None encodes zero covariance (infinite information) until the first update.
        self._factor = information.prior_factor(size, delta) if delta > 0 else None
        self._rhs = None if self._factor is None else self._factor @ self.weights

    @property
    def covariance(self):
        """Covariance P of the weights. Reconstructed on demand, not mutable state."""
        if self._factor is None:
            return np.zeros((len(self.weights), len(self.weights)))
        return information.covariance(self._factor)

    def _check_observation(self, phi, squared_error):
        if self.kalman_noise and (np.ndim(squared_error) != 0 or not np.isfinite(squared_error)
                                  or squared_error < 0):
            raise ValueError("squared_error must be a finite nonnegative scalar.")

    def _update(self, phi, target, squared_error):
        if self._factor is None:
            weights = self.weights.copy()
            factor = information.prior_factor(len(phi), self.process_noise)
            rhs = factor @ weights
        else:
            scale = np.sqrt(self.forgetting_factor)
            factor = scale * self._factor
            # At birth weights may have been inherited from a parent. Thereafter
            # retain the transformed targets rather than rebuilding them from w.
            rhs = factor @ self.weights if self.n_updates_ == 0 else scale * self._rhs
            if self.kalman_noise:
                deviation = np.sqrt(max(squared_error, MINIMUM_NOISE_VARIANCE))
                information.observe(factor, rhs, phi / deviation, target / deviation)
            else:
                information.observe(factor, rhs, phi.copy(), target)
            weights = information.solve(factor, rhs)
            if self.process_noise > 0:
                factor = information.diffuse(factor, self.process_noise)
                rhs = factor @ weights
        if not np.isfinite(weights).all() or not np.isfinite(factor).all():
            raise FloatingPointError("Nonfinite RLS state.")
        self._factor = factor
        self._rhs = rhs
        self.weights = weights
