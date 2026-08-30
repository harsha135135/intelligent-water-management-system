"""Paired significance tests for forecast comparisons.

Two tests, reported together, because they make different assumptions and a claim that only one
of them supports is a weaker claim:

* **Paired bootstrap** over forecast origins — assumes only that the origins are exchangeable.
* **Diebold-Mariano** with the Harvey-Leybourne-Newbold small-sample correction — assumes a
  covariance structure that the 23-hour origin stride only approximates, but is the test a
  forecasting reviewer will look for by name.

Both operate on a paired difference ``d = base - opponent`` computed on identical rows, which is
only meaningful because the backtest guarantees row parity.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

N_BOOT = 10_000
SEED = 20260830
RNG = np.random.default_rng(SEED)


def boot_ci(d: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float, float]:
    """Percentile bootstrap CI on the mean of a paired difference, resampling origins.

    Returns ``(mean, ci_lo, ci_hi, p_two_sided)``. The p-value is the fraction of resampled means
    that cross zero, doubled — so it is bounded below by ``2/n_boot`` and reported as "< 0.0001"
    rather than as an exact zero.
    """
    n = len(d)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means >= 0).mean(), (means <= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def dm_test(d: np.ndarray, horizon: int, *, stride_hours: int = 23) -> tuple[float, float]:
    """Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction.

    ``n = 24`` origins, so the correction matters. The autocovariance lag is derived from how many
    origins a forecast of this horizon can overlap, given the origin stride.
    """
    n = len(d)
    dbar = d.mean()
    h_lag = max(1, int(np.ceil(horizon / stride_hours)))
    gamma0 = np.sum((d - dbar) ** 2) / n
    var = gamma0
    for lag in range(1, h_lag):
        var += 2 * np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / n)
    corr = np.sqrt((n + 1 - 2 * h_lag + h_lag * (h_lag - 1) / n) / n)
    dm_hln = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)
