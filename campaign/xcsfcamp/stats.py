"""Non-parametric effect sizes, tests, bootstrap CIs and multiplicity control."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
from scipy import stats as ss


def hl_shift(a, b) -> float:
    """Hodges-Lehmann shift: median of all pairwise differences a_i - b_j (independent samples)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.median(a[:, None] - b[None, :]))


def hl_paired(d) -> float:
    """Hodges-Lehmann estimator for paired differences: median of the Walsh averages."""
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def a12(a, b) -> float:
    """Vargha-Delaney A12 = P(A > B) + 0.5 P(A = B)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)


def a12_magnitude(v: float) -> str:
    d = abs(v - 0.5) * 2  # = |Cliff's delta|
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def bootstrap_ci(stat, samples: Sequence[np.ndarray], n_boot: int, rng: np.random.Generator,
                 levels=(0.90, 0.95), paired=False) -> Dict[float, tuple]:
    """Percentile bootstrap CI.  Independent resampling of each sample, or of pairs if paired."""
    samples = [np.asarray(s, float) for s in samples]
    vals = np.empty(n_boot)
    for k in range(n_boot):
        if paired:
            idx = rng.integers(0, len(samples[0]), len(samples[0]))
            vals[k] = stat(*[s[idx] for s in samples])
        else:
            vals[k] = stat(*[s[rng.integers(0, len(s), len(s))] for s in samples])
    out = {}
    for lv in levels:
        lo, hi = np.quantile(vals, [(1 - lv) / 2, 1 - (1 - lv) / 2])
        out[lv] = (float(lo), float(hi))
    return out


def mannwhitney_p(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    return float(ss.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def wilcoxon_p(d) -> float:
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    return float(ss.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def brown_forsythe_p(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 and np.ptp(b) == 0:
        return 1.0
    return float(ss.levene(a, b, center="median").pvalue)


def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def iqr(a) -> float:
    q1, q3 = np.quantile(np.asarray(a, float), [.25, .75])
    return float(q3 - q1)


def verdict(p_adj: float, ci90: tuple, ci95: tuple, shift: float, margin: float, alpha: float) -> str:
    """Combine a difference test (Holm-adjusted) with an equivalence test (90% CI inside +-margin, TOST)."""
    if margin is None or not np.isfinite(margin) or margin <= 0:
        return "different" if p_adj < alpha else "no detectable difference"
    equivalent = (-margin < ci90[0]) and (ci90[1] < margin)
    different = p_adj < alpha
    if equivalent and not different:
        return "equivalent"
    if equivalent and different:
        return "different but negligible"
    if different and (ci95[0] >= margin or ci95[1] <= -margin):
        return "practically different"
    if different:
        return "different, relevance uncertain"
    return "inconclusive"
