"""L1-regularized predictors: recursive online Lasso, proximal SGD and windowed batch.

All three leave weights[0] (the intercept) unpenalized and require phi[0] == x0.
Their common objective, for the observations a method retains, is

    mean squared residual / 2 + alpha * sum(abs(w[1:]))
"""

from collections import deque
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from . import information
from .base import LocalPredictor, check_integer, check_real, soft_threshold


def kkt_violation(gradient, slopes, penalty):
    """Largest violation of the Lasso optimality conditions.

    At the optimum gradient_j = -penalty * sign(slope_j) for nonzero slopes and
    abs(gradient_j) <= penalty for zero slopes.
    """
    violations = np.where(slopes != 0, np.abs(gradient + penalty * np.sign(slopes)),
                          np.maximum(np.abs(gradient) - penalty, 0.0))
    return float(np.max(violations, initial=0.0))


def triangular_lasso(factor, rhs, penalty, start, max_iter, tolerance):
    """Minimize ||factor @ s - rhs||**2 / 2 + penalty * sum(abs(s)), factor upper triangular.

    Cyclic coordinate descent on the p compressed equations, O(p**2) per sweep
    whatever the number of observations they summarize. Returns the slopes, the
    number of sweeps, the final KKT violation and whether it met ``tolerance``.
    """
    slopes = start.copy()
    norms = np.einsum("ij,ij->j", factor, factor)
    residual = rhs - factor @ slopes
    violation = np.inf
    for iteration in range(1, max_iter + 1):
        for j in range(len(slopes)):
            column = factor[:j + 1, j]
            correlation = column @ residual[:j + 1] + norms[j] * slopes[j]
            coefficient = soft_threshold(correlation, penalty) / norms[j]
            residual[:j + 1] += column * (slopes[j] - coefficient)
            slopes[j] = coefficient
        # Recompute residuals to avoid drift; check the actual optimality
        # conditions rather than only the size of the last coefficient step.
        residual = rhs - factor @ slopes
        violation = kkt_violation(-(factor.T @ residual), slopes, penalty)
        if violation <= tolerance:
            return slopes, iteration, violation, True
    return slopes, max_iter, violation, False


class _LassoPredictor(LocalPredictor):
    """Shared checks and diagnostics of the Lasso predictors."""

    def __init__(self, size, *, alpha, initial_prediction, x0):
        super().__init__(size, initial_prediction=initial_prediction, x0=x0)
        check_real("alpha", alpha, positive=False)
        self.alpha = alpha
        self.n_iter_ = 0
        self.converged_ = None
        self.kkt_violation_ = None
        self._convergence_warned = False

    @property
    def n_samples_(self):
        """Number of retained observations (zero for methods without a buffer)."""
        return 0

    def _check_observation(self, phi, squared_error):
        if phi[0] != self.x0:
            raise ValueError("Lasso requires phi[0] == x0 (the constant bias input).")

    def _record_solver(self, n_iter, converged, violation, hint):
        self.n_iter_, self.converged_, self.kkt_violation_ = n_iter, converged, violation
        if not converged and not self._convergence_warned:
            warnings.warn(
                f"Local Lasso did not reach lasso_tol; {hint} "
                "Inspect predictor.kkt_violation_ and converged_.",
                ConvergenceWarning, stacklevel=5,
            )
            self._convergence_warned = True

    def diagnostics(self):
        return dict(n_updates=self.n_updates_, n_samples=self.n_samples_, n_iter=self.n_iter_,
                    converged=self.converged_, kkt_violation=self.kkt_violation_)


class LassoOnlinePredictor(_LassoPredictor):
    """Recursive Lasso: the exact L1 solution on RLS statistics, without stored samples.

    After t observations of the classifier, with n = sum(lambda**(t-i)), w minimizes

        sum(lambda**(t-i) * (y_i - phi_i @ w)**2) / 2
        + lambda**t / (2 * delta) * |w - w0|**2 + alpha * n * sum(abs(w[1:]))

    The least-squares part is carried by the square-root information filter of
    RLS as p equations R w = z. Because the intercept is the first, unpenalized
    unknown, the slopes solve a Lasso on the trailing triangular block of R,
    by coordinate descent warm-started at the previous slopes, and the
    intercept follows by back substitution. Convergence is that of RLS, not of
    a gradient step: no learning rate, and one slope is solved in closed form.

    alpha = 0 reproduces RLS exactly. delta ties a newborn rule to its
    inherited weights w0 until its own observations dominate. ``tol`` is an
    absolute tolerance on the KKT violation of the mean objective, as for the
    batch Lasso; ``max_iter`` bounds the coordinate-descent sweeps per update.
    """

    name = "lasso_online"
    parameters = {"alpha": "lasso_alpha", "delta": "rls_delta",
                  "forgetting_factor": "forgetting_factor", "max_iter": "lasso_max_iter",
                  "tol": "lasso_tol"}

    def __init__(self, size, *, alpha=0.001, delta=1000.0, forgetting_factor=1.0,
                 max_iter=1000, tol=1e-6, initial_prediction=0.0, x0=1.0):
        super().__init__(size, alpha=alpha, initial_prediction=initial_prediction, x0=x0)
        check_real("delta", delta)
        check_real("forgetting_factor", forgetting_factor, maximum=1)
        check_integer("max_iter", max_iter)
        check_real("tol", tol)
        self.delta = delta
        self.forgetting_factor = forgetting_factor
        self.max_iter = max_iter
        self.tol = tol
        self._factor = information.prior_factor(size, delta)
        self._rhs = self._factor @ self.weights
        self._effective_samples = 0.0

    def _update(self, phi, target, squared_error):
        scale = np.sqrt(self.forgetting_factor)
        factor = scale * self._factor
        rhs = factor @ self.weights if self.n_updates_ == 0 else scale * self._rhs
        samples = self.forgetting_factor * self._effective_samples + 1.0
        information.observe(factor, rhs, phi.copy(), target)
        if self.alpha == 0 or len(phi) == 1:
            weights, n_iter, violation, converged = information.solve(factor, rhs), 0, 0.0, True
        else:
            slopes, n_iter, violation, converged = triangular_lasso(
                factor[1:, 1:], rhs[1:], self.alpha * samples, self.weights[1:],
                self.max_iter, self.tol * samples)
            violation /= samples
            intercept = (rhs[0] - factor[0, 1:] @ slopes) / factor[0, 0]
            weights = np.concatenate(([intercept], slopes))
        if not np.isfinite(weights).all() or not np.isfinite(factor).all():
            raise FloatingPointError("Nonfinite Lasso state.")
        self._factor, self._rhs, self._effective_samples = factor, rhs, samples
        self.weights = weights
        self._record_solver(n_iter, converged, violation,
                            "increase lasso_max_iter or scale features.")


class LassoSGDPredictor(_LassoPredictor):
    """Proximal stochastic gradient (first order): an LMS step, then soft thresholding.

        eta_t = learning_rate / t**learning_rate_decay
        v = w + eta_t * (y - phi @ w) * phi;  w[0] = v[0];  w[1:] = soft(v[1:], eta_t * alpha)

    It tracks the Lasso solution only at the speed of LMS and needs well-scaled
    inputs; kept as the first-order reference for ``lasso_online``.
    """

    name = "lasso_sgd"
    parameters = {"learning_rate": "prediction_learning_rate", "alpha": "lasso_alpha",
                  "learning_rate_decay": "lasso_learning_rate_decay"}

    def __init__(self, size, *, learning_rate=0.2, alpha=0.001, learning_rate_decay=0.0,
                 initial_prediction=0.0, x0=1.0):
        super().__init__(size, alpha=alpha, initial_prediction=initial_prediction, x0=x0)
        check_real("learning_rate", learning_rate)
        check_real("learning_rate_decay", learning_rate_decay, positive=False, maximum=1)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

    def _update(self, phi, target, squared_error):
        step = self.learning_rate / (self.n_updates_ + 1) ** self.learning_rate_decay
        weights = self.weights + step * (target - self.predict(phi)) * phi
        weights[1:] = soft_threshold(weights[1:], step * self.alpha)
        if not np.isfinite(weights).all():
            raise FloatingPointError("Nonfinite prediction weights.")
        self.weights = weights


class LassoBatchPredictor(_LassoPredictor):
    """Coordinate-descent Lasso refitted on a FIFO window of the classifier's observations.

    ``window=None`` retains every observation since the birth of the rule. The
    solver centers the retained inputs and targets, cycles over the slopes with
    soft thresholding from the previous solution, and rebuilds the intercept.
    """

    name = "lasso_batch"
    parameters = {"alpha": "lasso_alpha", "window": "lasso_window",
                  "max_iter": "lasso_max_iter", "tol": "lasso_tol"}

    def __init__(self, size, *, alpha=0.001, window=256, max_iter=1000, tol=1e-6,
                 initial_prediction=0.0, x0=1.0):
        super().__init__(size, alpha=alpha, initial_prediction=initial_prediction, x0=x0)
        check_integer("window", window, optional=True)
        check_integer("max_iter", max_iter)
        check_real("tol", tol)
        self.window = window
        self.max_iter = max_iter
        self.tol = tol
        self._samples = deque(maxlen=None if window is None else int(window))

    @property
    def n_samples_(self):
        return len(self._samples)

    def _update(self, phi, target, squared_error):
        # Retain actual rows to avoid squaring the design's condition number in
        # a Gram matrix. Build the prospective window without changing state.
        samples = list(self._samples)
        if self.window is not None and len(samples) == self.window:
            samples = samples[1:]
        samples.append((phi.copy(), target))
        design = np.array([sample[0][1:] for sample in samples])
        targets = np.array([sample[1] for sample in samples])
        mean_x, mean_y = design.mean(axis=0), targets.mean()
        centered = design - mean_x
        response = targets - mean_y
        norms = np.mean(centered * centered, axis=0)
        slopes = self.weights[1:].copy()
        slopes[norms == 0] = 0.0
        converged = False
        for iteration in range(1, self.max_iter + 1):
            residual = response - centered @ slopes
            for j in range(len(slopes)):
                if norms[j] == 0:
                    continue
                column = centered[:, j]
                correlation = column @ residual / len(samples) + norms[j] * slopes[j]
                coefficient = soft_threshold(correlation, self.alpha) / norms[j]
                residual += column * (slopes[j] - coefficient)
                slopes[j] = coefficient
            # Recompute residuals to avoid drift; check the actual optimality
            # conditions rather than only the size of the last coefficient step.
            residual = response - centered @ slopes
            violation = kkt_violation(-(centered.T @ residual) / len(samples), slopes, self.alpha)
            if violation <= self.tol:
                converged = True
                break
        weights = np.concatenate(([(mean_y - mean_x @ slopes) / self.x0], slopes))
        if not np.isfinite(weights).all() or not np.isfinite(violation):
            raise FloatingPointError("Nonfinite Lasso solution.")
        self.weights = weights
        self._samples.append(samples[-1])
        self._record_solver(iteration, converged, violation,
                            "increase lasso_max_iter or scale features.")
