"""Local online predictors and a windowed Lasso solver, implemented in Python.

NumPy handles arrays and SciPy only supplies a triangular linear solve. No
external regression estimator is used for training.
"""

from collections import deque
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.exceptions import ConvergenceWarning


PREDICTION_METHODS = ("lms", "nlms", "rls", "lasso_online", "lasso_batch", "rlsk", "constant")


def design_matrix(X, degree=1, x0=1.0):
    """[x0, x1, ..., xd, x1**2, ..., xd**degree], without cross terms."""
    X = np.asarray(X, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.concatenate(
            [np.full((*X.shape[:-1], 1), x0)]
            + [X ** power for power in range(1, degree + 1)], axis=-1,
        )
    if not np.isfinite(result).all():
        raise ValueError("Polynomial features overflow; scale X or reduce degree.")
    return result


def _soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


class LocalPredictor:
    """One scalar predictor per classifier; inputs include the constant bias.

    Methods: LMS, normalized LMS, QR recursive least squares, proximal online
    Lasso, coordinate-descent batch Lasso, legacy RLSK and constant prediction.
    ``lasso_batch`` fits the mean squared loss / 2 plus ``lasso_alpha * |w|_1``
    on a FIFO window (or all local observations with ``lasso_window=None``).
    Both Lasso methods leave weights[0] unpenalized and require phi[0] == x0.

    RLS stores an upper triangular information factor, not its inverse.
    ``covariance`` reconstructs the inverse information matrix for inspection.
    Offspring inherit weights, resetting information factors, counters and data.
    Batch Lasso exposes ``converged_``, ``kkt_violation_`` and ``n_iter_``.
    """

    def __init__(self, size, *, method="nlms", learning_rate=0.2, delta=1000.0,
                 forgetting_factor=1.0, process_noise=0.0, kalman_noise=False,
                 initial_prediction=0.0, x0=1.0, lasso_alpha=0.001,
                 lasso_window=256, lasso_max_iter=1000, lasso_tol=1e-6,
                 lasso_learning_rate_decay=0.0):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 1:
            raise ValueError("size must be a positive integer.")
        if method not in PREDICTION_METHODS:
            raise ValueError(f"method must be one of {PREDICTION_METHODS}.")
        for name, value, positive in (
            ("learning_rate", learning_rate, True), ("delta", delta, True),
            ("forgetting_factor", forgetting_factor, True), ("x0", x0, True),
            ("process_noise", process_noise, False), ("lasso_alpha", lasso_alpha, False),
            ("lasso_tol", lasso_tol, True),
            ("lasso_learning_rate_decay", lasso_learning_rate_decay, False),
        ):
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
        if lasso_learning_rate_decay > 1:
            raise ValueError("lasso_learning_rate_decay must be <= 1.")
        if forgetting_factor > 1:
            raise ValueError("forgetting_factor must be <= 1.")
        if (isinstance(initial_prediction, (bool, np.bool_))
                or not isinstance(initial_prediction, Real) or not np.isfinite(initial_prediction)):
            raise ValueError("initial_prediction must be a finite real number.")
        if not isinstance(kalman_noise, (bool, np.bool_)):
            raise ValueError("kalman_noise must be boolean.")
        for name, value in (("lasso_window", lasso_window), ("lasso_max_iter", lasso_max_iter)):
            if name == "lasso_window" and value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer" + (" or None." if name == "lasso_window" else "."))
        self.method = method
        self.learning_rate = learning_rate
        self.delta = delta
        self.forgetting_factor = forgetting_factor
        self.process_noise = process_noise
        self.kalman_noise = kalman_noise
        self.x0 = x0
        self.lasso_alpha = lasso_alpha
        self.lasso_window = lasso_window
        self.lasso_max_iter = lasso_max_iter
        self.lasso_tol = lasso_tol
        self.lasso_learning_rate_decay = lasso_learning_rate_decay
        self.weights = np.zeros(size)
        self.weights[0] = initial_prediction / x0
        if not np.isfinite(self.weights[0]):
            raise ValueError("initial_prediction / x0 must be finite.")
        self.value = float(initial_prediction)
        self._rls_factor = np.eye(size) / np.sqrt(delta) if method == "rls" else None
        self._rls_rhs = self.weights / np.sqrt(delta) if method == "rls" else None
        self._covariance = np.eye(size) * delta if method == "rlsk" else None
        self.n_updates_ = 0
        self.n_iter_ = 0
        self.converged_ = None
        self.kkt_violation_ = None
        self._convergence_warned = False
        self._samples = deque(maxlen=None if lasso_window is None else int(lasso_window)) if method == "lasso_batch" else None

    @property
    def covariance(self):
        """Inverse information for RLS, Kalman covariance for RLSK; otherwise None.

        The RLS result is reconstructed on demand and is not mutable model state.
        """
        if self.method == "rls":
            inverse = solve_triangular(self._rls_factor, np.eye(len(self.weights)), check_finite=False)
            return inverse @ inverse.T
        return self._covariance

    @property
    def n_samples_(self):
        """Number of retained batch Lasso observations (zero for online methods)."""
        return len(self._samples) if self._samples is not None else 0

    def predict(self, phi):
        phi = np.asarray(phi, dtype=float)
        if self.method == "constant":
            return np.full(phi.shape[:-1], self.value)
        return phi @ self.weights

    def update(self, phi, target, squared_error=1.0):
        phi = np.asarray(phi, dtype=float)
        if phi.shape != self.weights.shape or not np.isfinite(phi).all():
            raise ValueError("phi must be a finite vector with the predictor's size.")
        if np.ndim(target) != 0 or not np.isfinite(target):
            raise ValueError("target must be a finite scalar.")
        if self.method.startswith("lasso") and phi[0] != self.x0:
            raise ValueError("Lasso requires phi[0] == x0 (the constant bias input).")
        if self.method == "rlsk" and self.kalman_noise:
            if np.ndim(squared_error) != 0 or not np.isfinite(squared_error) or squared_error < 0:
                raise ValueError("squared_error must be a finite nonnegative scalar.")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                self._update(phi, float(target), squared_error)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise FloatingPointError(
                "Local prediction update failed numerically; scale inputs/targets "
                "or reduce prediction_learning_rate."
            ) from exc
        self.n_updates_ += 1

    def _update(self, phi, target, squared_error):
        if self.method == "rls":
            self._update_rls(phi, target)
            return
        if self.method == "lasso_batch":
            self._update_lasso_batch(phi, target)
            return
        error = target - self.predict(phi)
        if self.method == "constant":
            self.value += self.learning_rate * error
            return
        if self.method == "lms":
            weights = self.weights + self.learning_rate * error * phi
        elif self.method == "nlms":
            # Scaling first avoids overflow/underflow when computing phi @ phi.
            scale = np.max(np.abs(phi))
            if scale == 0:
                return
            unit = phi / scale
            weights = self.weights + (self.learning_rate * error / scale) * unit / (unit @ unit)
        elif self.method == "lasso_online":
            step = self.learning_rate / (self.n_updates_ + 1) ** self.lasso_learning_rate_decay
            weights = self.weights + step * error * phi
            weights[1:] = _soft_threshold(weights[1:], step * self.lasso_alpha)
        else:  # Legacy RLSK, with a Joseph covariance update.
            noise = max(squared_error, 1e-4) if self.kalman_noise else 1.0
            prior = self._covariance / self.forgetting_factor
            product = prior @ phi
            gain = product / (noise + phi @ product)
            weights = self.weights + gain * error
            transform = np.eye(len(phi)) - np.outer(gain, phi)
            covariance = transform @ prior @ transform.T + noise * np.outer(gain, gain)
            covariance.flat[::len(phi) + 1] += self.process_noise
            self._covariance = (covariance + covariance.T) / 2
        if not np.isfinite(weights).all():
            raise FloatingPointError("Nonfinite prediction weights.")
        self.weights = weights

    def _update_rls(self, phi, target):
        # Givens QR on [sqrt(lambda) R | sqrt(lambda) R w; phi | y].
        # No normal equations or inverse covariance subtraction. O(p^2).
        factor = np.sqrt(self.forgetting_factor) * self._rls_factor
        # At birth weights may have been inherited from a parent. Thereafter
        # retain the transformed targets rather than rebuilding them from w.
        rhs = factor @ self.weights if self.n_updates_ == 0 else np.sqrt(self.forgetting_factor) * self._rls_rhs
        row = phi.copy()
        remaining = target
        for j in range(len(phi)):
            diagonal = np.hypot(factor[j, j], row[j])
            if diagonal == 0:
                raise FloatingPointError("RLS information factor lost rank.")
            cosine, sine = factor[j, j] / diagonal, row[j] / diagonal
            old = factor[j, j:].copy()
            factor[j, j:] = cosine * old + sine * row[j:]
            row[j:] = -sine * old + cosine * row[j:]
            rhs[j], remaining = cosine * rhs[j] + sine * remaining, -sine * rhs[j] + cosine * remaining
        weights = solve_triangular(factor, rhs, check_finite=False)
        if not np.isfinite(weights).all() or not np.isfinite(factor).all():
            raise FloatingPointError("Nonfinite RLS state.")
        self._rls_factor = factor
        self._rls_rhs = rhs
        self.weights = weights

    def _update_lasso_batch(self, phi, target):
        # Retain actual rows to avoid squaring the design's condition number in
        # a Gram matrix. Build the prospective window without changing state.
        samples = list(self._samples)
        if self.lasso_window is not None and len(samples) == self.lasso_window:
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
        for iteration in range(1, self.lasso_max_iter + 1):
            residual = response - centered @ slopes
            for j in range(len(slopes)):
                if norms[j] == 0:
                    continue
                column = centered[:, j]
                correlation = column @ residual / len(samples) + norms[j] * slopes[j]
                coefficient = _soft_threshold(correlation, self.lasso_alpha) / norms[j]
                residual += column * (slopes[j] - coefficient)
                slopes[j] = coefficient
            # Recompute residuals to avoid drift; check the actual optimality
            # conditions rather than only the size of the last coefficient step.
            residual = response - centered @ slopes
            gradient = -(centered.T @ residual) / len(samples)
            violations = np.where(slopes != 0,
                                  np.abs(gradient + self.lasso_alpha * np.sign(slopes)),
                                  np.maximum(np.abs(gradient) - self.lasso_alpha, 0.0))
            violation = float(np.max(violations, initial=0.0))
            if violation <= self.lasso_tol:
                converged = True
                break
        weights = np.concatenate(([(mean_y - mean_x @ slopes) / self.x0], slopes))
        if not np.isfinite(weights).all() or not np.isfinite(violation):
            raise FloatingPointError("Nonfinite Lasso solution.")
        self.weights = weights
        self._samples.append(samples[-1])
        self.n_iter_, self.converged_, self.kkt_violation_ = iteration, converged, violation
        if not converged and not self._convergence_warned:
            warnings.warn(
                "Local batch Lasso did not reach lasso_tol; increase lasso_max_iter "
                "or scale features. Inspect predictor.kkt_violation_ and converged_.",
                ConvergenceWarning, stacklevel=4,
            )
            self._convergence_warned = True

    def offspring(self):
        child = type(self)(
            len(self.weights), method=self.method, learning_rate=self.learning_rate,
            delta=self.delta, forgetting_factor=self.forgetting_factor,
            process_noise=self.process_noise, kalman_noise=self.kalman_noise,
            x0=self.x0, lasso_alpha=self.lasso_alpha, lasso_window=self.lasso_window,
            lasso_max_iter=self.lasso_max_iter, lasso_tol=self.lasso_tol,
            lasso_learning_rate_decay=self.lasso_learning_rate_decay,
        )
        child.weights = self.weights.copy()
        child.value = self.value
        return child
