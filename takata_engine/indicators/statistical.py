"""Statistical features for ML signal quality.

Three features grounded in the multifractal/long-range-dependence
literature on financial returns (Bloch 2016 §2.1.4–2.1.5,
Mandelbrot 1963, Hurst 1951):

- Hurst exponent (long-range dependence). H ≈ 0.5 = random walk;
  H > 0.5 = persistent / trending; H < 0.5 = anti-persistent /
  mean-reverting. Discriminates regimes where momentum signals
  should work versus where fades should.

- Rolling skewness of returns. Asymmetric tails — when the recent
  return distribution is left-skewed, downside trades have richer
  payoff than the symmetric assumption would suggest, and vice versa.

- Rolling excess kurtosis of returns. Heavy-tail awareness — fixed
  stop sizing under-prices tail risk in high-kurtosis regimes; the
  ML model can learn to discount signals when current kurtosis is
  far above normal.

These are pure-python (numpy/pandas) and inexpensive to compute on
each scan cycle, so the live scanner can attach them to every
signal's indicator snapshot for the ML logger.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    """Estimate the Hurst exponent via rescaled-range (R/S) analysis.

    Parameters
    ----------
    series : pd.Series
        Price (not returns) series. R/S works on the cumulative
        deviation of returns, so we compute log-returns internally.
    min_lag, max_lag : int
        Range of window sizes used to fit the log-log line. The slope
        of log(R/S) vs log(lag) ≈ Hurst exponent. ``max_lag`` should
        be ≤ len(series) // 2 — we clamp to that.

    Returns
    -------
    float
        Estimated Hurst exponent. Returns 0.5 (random-walk default)
        on insufficient data, NaN inputs, or numerical failure —
        downstream consumers should treat 0.5 as "no edge known"
        rather than special-case errors.

    Notes
    -----
    R/S is a classical estimator (Hurst 1951, Mandelbrot 1972) and
    has bias on short windows; it's adequate for the 30–100 bar
    intraday windows we use, where DFA's extra complexity isn't
    warranted. The bias direction is documented and consistent, so
    the ML model can learn to compensate.
    """
    if series is None or len(series) < min_lag * 4:
        return 0.5
    s = pd.Series(series).dropna().astype(float)
    if len(s) < min_lag * 4:
        return 0.5

    # Log-returns, then center.
    rets = np.log(s).diff().dropna().to_numpy()
    if rets.size < min_lag * 2 or not np.isfinite(rets).all():
        return 0.5

    upper_lag = min(max_lag, max(min_lag + 1, rets.size // 2))
    lags = np.arange(min_lag, upper_lag + 1)

    rs_values = []
    for lag in lags:
        # Split returns into non-overlapping chunks of length `lag`.
        n_chunks = rets.size // lag
        if n_chunks < 1:
            continue
        chunks = rets[: n_chunks * lag].reshape(n_chunks, lag)
        # Per-chunk rescaled range.
        means = chunks.mean(axis=1, keepdims=True)
        deviations = chunks - means
        cum = deviations.cumsum(axis=1)
        ranges = cum.max(axis=1) - cum.min(axis=1)
        stds = chunks.std(axis=1, ddof=0)
        # Avoid div-by-zero on flat chunks.
        valid = stds > 1e-12
        if not valid.any():
            continue
        rs = (ranges[valid] / stds[valid]).mean()
        if rs > 0 and np.isfinite(rs):
            rs_values.append((lag, rs))

    if len(rs_values) < 3:
        return 0.5

    log_lags = np.log(np.array([x[0] for x in rs_values], dtype=float))
    log_rs = np.log(np.array([x[1] for x in rs_values], dtype=float))
    if not (np.isfinite(log_lags).all() and np.isfinite(log_rs).all()):
        return 0.5

    # Slope of log(R/S) vs log(lag) is the Hurst estimate.
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    if not np.isfinite(slope):
        return 0.5
    return float(np.clip(slope, 0.0, 1.0))


def rolling_skewness(returns: pd.Series, window: int = 30) -> pd.Series:
    """Rolling skewness of the return distribution.

    Returns NaN-tolerant: bars before the window fills get NaN.
    Consumers should `.fillna(0)` if the signal logger needs a
    scalar at log time.
    """
    if returns is None or len(returns) == 0:
        return pd.Series(dtype=float)
    return pd.Series(returns).rolling(window, min_periods=max(5, window // 3)).skew()


def rolling_kurtosis(returns: pd.Series, window: int = 30) -> pd.Series:
    """Rolling excess kurtosis (Fisher definition: normal = 0)."""
    if returns is None or len(returns) == 0:
        return pd.Series(dtype=float)
    return pd.Series(returns).rolling(window, min_periods=max(5, window // 3)).kurt()


def latest_statistical_features(
    closes: pd.Series,
    hurst_window: int = 50,
    moments_window: int = 30,
) -> dict:
    """Compute the three features on the most recent bars and return
    a dict suitable for splatting into the live scanner's
    `indicators` snapshot.

    Returns
    -------
    dict
        ``hurst_50``, ``return_skew_30``, ``return_kurt_30`` — all
        float, never NaN. Each falls back to a neutral value
        (0.5 / 0.0 / 0.0) when there isn't enough history.

    Use this rather than calling each function separately — it
    centralises the NaN handling and naming convention so the ML
    feature store and the signal logger stay in sync.
    """
    if closes is None or len(closes) == 0:
        return {"hurst_50": 0.5, "return_skew_30": 0.0, "return_kurt_30": 0.0}

    s = pd.Series(closes).dropna().astype(float)
    if len(s) < 5:
        return {"hurst_50": 0.5, "return_skew_30": 0.0, "return_kurt_30": 0.0}

    # Hurst takes the price series.
    h_window = s.tail(max(hurst_window, 30))
    hurst = hurst_exponent(h_window)

    # Skew / kurt are on log-returns over the trailing window.
    rets = np.log(s).diff().dropna()
    if rets.size < 5:
        return {"hurst_50": hurst, "return_skew_30": 0.0, "return_kurt_30": 0.0}

    tail = rets.tail(moments_window)
    skew_val = float(tail.skew()) if tail.size >= 5 else 0.0
    kurt_val = float(tail.kurt()) if tail.size >= 5 else 0.0
    if not np.isfinite(skew_val):
        skew_val = 0.0
    if not np.isfinite(kurt_val):
        kurt_val = 0.0

    return {
        "hurst_50": hurst,
        "return_skew_30": skew_val,
        "return_kurt_30": kurt_val,
    }
