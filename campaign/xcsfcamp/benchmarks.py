"""Continuous benchmark functions shared by both implementations.

Single source of truth for the campaign (see ../../BENCHMARK_FUNCTIONS.md).
The C++ side evaluates the same formulas inside ``real_functions_env``
(functions added by ``patches/xcslib-benchmark-functions.patch``); the
validation stage checks that both agree on sampled points.

All domains are real and continuous: training inputs are drawn with
``Generator.uniform`` as float64 and are never rounded or discretised.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi


def _sine(x):
    return 100.0 * np.sin(TWO_PI * x[..., 0] / 100.0)


def _sinus3(x):
    a = TWO_PI * x[..., 0] / 100.0
    return 100.0 * (np.sin(a) + np.sin(2 * a) + np.sin(3 * a))


def _sinus4(x):
    a = TWO_PI * x[..., 0] / 100.0
    return 100.0 * (np.sin(a) + np.sin(2 * a) + np.sin(3 * a) + np.sin(4 * a))


def _abs_mix(x):
    a = TWO_PI * x[..., 0] / 100.0
    return 100.0 * np.abs(np.sin(a) + np.abs(np.cos(a)))


def _sin_cos_surface(x):
    return 100.0 * np.sin(TWO_PI * x[..., 0]) * np.cos(TWO_PI * x[..., 1])


def _friedman(x):
    return (10.0 * np.sin(math.pi * x[..., 0] * x[..., 1]) + 20.0 * (x[..., 2] - 0.5) ** 2
            + 10.0 * x[..., 3] + 5.0 * x[..., 4])


@dataclass(frozen=True)
class Benchmark:
    name: str
    code: int                      # stable integer used in seed derivation
    function: Callable[[np.ndarray], np.ndarray]
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    cxx_function: str              # value of `function =` in <environment::real_functions>
    cxx_scale_factor: float        # value of `scale factor =` (amplitude/period 100 for the paper family)
    grid_resolution: float         # step of the deterministic evaluation grid (C++ `sampling resolution`)
    lipschitz: float               # upper bound of |df/dx_i| used for print-precision tolerances
    core: bool = True
    known_points: Tuple[Tuple[Tuple[float, ...], float], ...] = field(default_factory=tuple)

    @property
    def dim(self) -> int:
        return len(self.lower)

    @property
    def width(self) -> float:
        widths = {u - l for l, u in zip(self.lower, self.upper)}
        if len(widths) != 1:
            raise ValueError(f"{self.name}: xcslib requires one common [min,max] for all inputs")
        return widths.pop()

    def __call__(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, self.dim)
        if X.shape[-1] != self.dim:
            raise ValueError(f"{self.name}: expected {self.dim} inputs, got {X.shape[-1]}")
        y = self.function(X)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"{self.name}: non-finite target values")
        return y

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform real-valued samples in the domain (float64, never discretised)."""
        X = rng.uniform(np.asarray(self.lower), np.asarray(self.upper), size=(n, self.dim))
        assert X.dtype == np.float64
        return X

    def grid(self) -> np.ndarray:
        """Deterministic evaluation grid, bit-identical to xcslib's reset_input/next_input.

        xcslib starts every coordinate at `min input` and repeatedly adds
        `sampling resolution` (float64 accumulation) while the value is
        <= `max input`; the last coordinate varies fastest.  Replicating the
        accumulation (instead of linspace) guarantees both implementations are
        evaluated on exactly the same points.
        """
        axes = []
        for lo, hi in zip(self.lower, self.upper):
            values, v = [], float(lo)
            while v <= hi:
                values.append(v)
                v += self.grid_resolution
            axes.append(values)
        return np.array(list(itertools.product(*axes)), dtype=np.float64)

    def output_range(self) -> Tuple[float, float]:
        """Min/max of f over the domain: analytic for 1-D families, dense grid otherwise."""
        return _RANGES[self.name]

    def epsilon0(self, fraction: float) -> float:
        lo, hi = self.output_range()
        return float(fraction * (hi - lo))


def _dense_range_1d(f, lo, hi, n=2_000_001):
    from scipy.optimize import minimize_scalar
    x = np.linspace(lo, hi, n)
    y = f(x[:, None])
    out = []
    for sign in (+1, -1):
        i = int(np.argmin(sign * y))
        a, b = x[max(i - 1, 0)], x[min(i + 1, n - 1)]
        res = minimize_scalar(lambda t: sign * float(f(np.array([[t]]))[0]), bounds=(a, b), method="bounded",
                              options=dict(xatol=1e-12))
        out.append(min(sign * y[i], res.fun) * sign)
    return float(out[0]), float(out[1])


def _dense_range_nd(f, lower, upper, per_dim):
    axes = [np.linspace(l, u, per_dim) for l, u in zip(lower, upper)]
    pts = np.array(list(itertools.product(*axes)))
    y = f(pts)
    return float(y.min()), float(y.max())


BENCHMARKS: Dict[str, Benchmark] = {b.name: b for b in [
    Benchmark("sine_low_1d", 1, _sine, (0.0,), (100.0,), "sine", 100.0, 0.1, TWO_PI,
              known_points=(((0.0,), 0.0), ((25.0,), 100.0), ((50.0,), 0.0), ((75.0,), -100.0), ((12.5,), 100 * math.sqrt(0.5)))),
    Benchmark("sine_shifted_1d", 2, _sine, (1000.0,), (1100.0,), "sine", 100.0, 0.1, TWO_PI,
              known_points=(((1000.0,), 0.0), ((1025.0,), 100.0), ((1075.0,), -100.0))),
    Benchmark("sinus3_1d", 3, _sinus3, (950.0,), (1050.0,), "sine3", 100.0, 0.1, 6 * TWO_PI,
              known_points=(((1000.0,), 0.0), ((1025.0,), 0.0), ((962.5,), 100.0 * (1 - math.sqrt(2.0))), ((1012.5,), 100.0 * (1 + math.sqrt(2.0))))),
    Benchmark("sinus4_1d", 4, _sinus4, (950.0,), (1050.0,), "sine4", 100.0, 0.1, 10 * TWO_PI,
              known_points=(((1000.0,), 0.0), ((1025.0,), 0.0), ((962.5,), 100.0 * (1 - math.sqrt(2.0))), ((1012.5,), 100.0 * (1 + math.sqrt(2.0))))),
    Benchmark("abs_mix_1d", 5, _abs_mix, (950.0,), (1050.0,), "abs", 100.0, 0.1, 2 * TWO_PI,
              known_points=(((1000.0,), 100.0), ((1025.0,), 100.0), ((962.5,), 0.0), ((1037.5,), 100.0 * math.sqrt(2.0)), ((1050.0,), 100.0))),
    Benchmark("sin_cos_surface_2d", 6, _sin_cos_surface, (0.0, 0.0), (1.0, 1.0), "sincos2d", 100.0, 0.03125, 100 * TWO_PI,
              core=False, known_points=(((0.25, 0.0), 100.0), ((0.25, 0.5), -100.0), ((0.5, 0.3), 0.0))),
    Benchmark("friedman5d", 7, _friedman, (0.0,) * 5, (1.0,) * 5, "friedman5", 1.0, 0.25, 40.0,
              core=False, known_points=(((0.0,) * 5, 5.0), ((1.0,) * 5, 0.0 + 5.0 + 10.0 + 5.0), ((0.5, 1.0, 0.5, 0.0, 0.0), 10.0))),
]}

# Output ranges used for epsilon0 = fraction * (max f - min f).
_RANGES: Dict[str, Tuple[float, float]] = {
    "sine_low_1d": (-100.0, 100.0),
    "sine_shifted_1d": (-100.0, 100.0),
    "abs_mix_1d": (0.0, 100.0 * math.sqrt(2.0)),
}


def _fill_ranges():
    for name, b in BENCHMARKS.items():
        if name in _RANGES:
            continue
        if b.dim == 1:
            _RANGES[name] = _dense_range_1d(b.function, b.lower[0], b.upper[0])
        elif name == "sin_cos_surface_2d":
            _RANGES[name] = (-100.0, 100.0)
        elif name == "friedman5d":
            # analytic: min 0 at x1*x2=0, x3=0.5, x4=x5=0; max 30 at x1*x2=0.5, x3 in {0,1}, x4=x5=1.
            _RANGES[name] = (0.0, 30.0)
        else:  # pragma: no cover
            _RANGES[name] = _dense_range_nd(b.function, b.lower, b.upper, 11)


_fill_ranges()


def get(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise KeyError(f"unknown benchmark '{name}'; known: {sorted(BENCHMARKS)}") from None
