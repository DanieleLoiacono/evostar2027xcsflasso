"""Square-root information form of recursive least squares.

The state of the estimator is an upper triangular ``factor`` R and a vector
``rhs`` z such that the accumulated cost is ||R w - z||**2 plus a constant:
R.T @ R is the information matrix (the inverse covariance) and R w = z gives
the estimate. Every operation below is an orthogonal transformation of [R | z];
the information matrix is never formed and no covariance is ever subtracted,
so the conditioning of the data is not squared and R.T @ R stays positive
definite by construction. Used by both RLS and the recursive online Lasso.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular


def prior_factor(size, delta):
    """Factor of the prior covariance delta * I."""
    return np.eye(size) / np.sqrt(delta)


def observe(factor, rhs, row, value):
    """Fold the equation row @ w = value into [factor | rhs], in place. O(p**2).

    Givens rotations triangularize [factor | rhs; row | value]; ``row`` is
    overwritten. Scale the equation by 1 / sqrt(variance) to weight it.
    """
    for j in range(len(row)):
        diagonal = np.hypot(factor[j, j], row[j])
        if diagonal == 0:
            raise FloatingPointError("RLS information factor lost rank.")
        cosine, sine = factor[j, j] / diagonal, row[j] / diagonal
        old = factor[j, j:].copy()
        factor[j, j:] = cosine * old + sine * row[j:]
        row[j:] = -sine * old + cosine * row[j:]
        rhs[j], value = cosine * rhs[j] + sine * value, -sine * rhs[j] + cosine * value


def diffuse(factor, noise):
    """Factor after the covariance grows by noise * I (a random-walk step of w).

    Dyer-McReynolds time update: with w' = w + v, v ~ N(0, noise * I), the
    equations v / sqrt(noise) = 0 and R (w' - v) = z are triangularized in the
    unknowns (v, w') and the block of w' is kept. The estimate is unchanged, so
    the caller recomputes rhs = factor @ estimate. No matrix is inverted.
    """
    size = len(factor)
    array = np.zeros((2 * size, 2 * size))
    array[:size, :size] = np.eye(size) / np.sqrt(noise)
    array[size:, :size] = -factor
    array[size:, size:] = factor
    result = qr(array, mode="r", overwrite_a=True, check_finite=False)[0][size:, size:]
    return np.where(np.diag(result) < 0, -1.0, 1.0)[:, None] * result


def solve(factor, rhs):
    """Estimate w of factor @ w = rhs, by back substitution."""
    return solve_triangular(factor, rhs, check_finite=False)


def covariance(factor):
    """Inverse information matrix, reconstructed for inspection. O(p**3)."""
    inverse = solve_triangular(factor, np.eye(len(factor)), check_finite=False)
    return inverse @ inverse.T
