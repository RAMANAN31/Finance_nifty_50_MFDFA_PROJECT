"""A transparent MFDFA implementation for exploratory scaling analysis."""

from __future__ import annotations

import numpy as np


def mfdfa(returns, q_values=None, scales=None, order: int = 1) -> dict[str, np.ndarray]:
    """Compute Fq(s), H(q), tau(q), alpha(q), f(alpha), and spectrum width.

    Follows the standard profile/detrending construction: demean the input,
    integrate it, detrend each non-overlapping segment from both ends, and fit
    log Fq against log scale. The output is an estimate, sensitive to all these
    choices and to the finite sample.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max(32, 2 * (order + 2)):
        raise ValueError("MFDFA needs at least 32 finite observations")
    if order < 0 or order > 5:
        raise ValueError("Polynomial order must be between 0 and 5")
    q = np.asarray(q_values if q_values is not None else np.arange(-5, 6, 1), dtype=float)
    if q.ndim != 1 or q.size < 3 or not np.isfinite(q).all() or np.unique(q).size != q.size:
        raise ValueError("q_values must contain at least three unique finite values")
    if np.count_nonzero(np.isclose(q, 0.0)) > 1:
        raise ValueError("q_values may contain zero only once")
    profile = np.cumsum(x - x.mean())
    if scales is None:
        lo, hi = max(order + 2, 10), x.size // 4
        if hi <= lo:
            raise ValueError("Sample is too short for the requested detrending order")
        scale_arr = np.unique(np.floor(np.geomspace(lo, hi, min(20, hi - lo + 1))).astype(int))
    else:
        scale_arr = np.asarray(scales, dtype=int)
    scale_arr = np.unique(scale_arr)
    if scale_arr.size < 3 or (scale_arr <= order + 1).any() or (scale_arr > x.size // 2).any():
        raise ValueError("Provide at least three scales greater than order+1 and no larger than half the sample")
    if np.any(scale_arr <= 0):
        raise ValueError("Scales must be positive")

    fq = np.empty((q.size, scale_arr.size), dtype=float)
    for j, scale in enumerate(scale_arr):
        nseg = x.size // scale
        variances = []
        t = np.arange(scale, dtype=float)
        for start in list(range(nseg)) + list(range(nseg, 2 * nseg)):
            offset = start * scale if start < nseg else x.size - (start - nseg + 1) * scale
            segment = profile[offset:offset + scale]
            coeff = np.polyfit(t, segment, order)
            residual = segment - np.polyval(coeff, t)
            variances.append(float(np.mean(residual ** 2)))
        v = np.maximum(np.asarray(variances), np.finfo(float).tiny)
        for i, qi in enumerate(q):
            fq[i, j] = np.exp(0.5 * np.mean(np.log(v))) if np.isclose(qi, 0.0) else np.mean(v ** (qi / 2.0)) ** (1.0 / qi)
    if not np.isfinite(fq).all() or (fq <= 0).any():
        raise ValueError("MFDFA produced non-finite or non-positive fluctuation values")
    hq = np.array([np.polyfit(np.log(scale_arr), np.log(row), 1)[0] for row in fq])
    tau = q * hq - 1.0
    alpha = np.gradient(tau, q)
    spectrum = q * alpha - tau
    return {"q": q, "scales": scale_arr, "Fq": fq, "Hq": hq, "tau": tau,
            "alpha": alpha, "f_alpha": spectrum, "spectrum_width": np.array(float(alpha.max() - alpha.min())),
            "order": np.array(order, dtype=int)}
