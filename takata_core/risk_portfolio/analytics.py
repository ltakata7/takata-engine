"""Portfolio risk analytics — volatility, beta, VaR, drawdown, correlation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualized volatility from daily returns."""
    return float(returns.std() * np.sqrt(trading_days))


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Portfolio beta vs benchmark."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 1.0
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    return float(cov / var) if var > 0 else 1.0


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (positive number = potential loss)."""
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) — average loss beyond VaR."""
    threshold = np.percentile(returns.dropna(), (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else var_historical(returns, confidence)


def max_drawdown_analysis(returns: pd.Series) -> Dict[str, Any]:
    """Compute max drawdown with dates."""
    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    dd = (equity - peak) / peak

    max_dd = float(abs(dd.min()))
    max_dd_idx = dd.idxmin()

    # Find peak before max drawdown
    peak_idx = equity.loc[:max_dd_idx].idxmax()

    # Find recovery (if any)
    post_dd = equity.loc[max_dd_idx:]
    peak_val = equity.loc[peak_idx]
    recovered = post_dd[post_dd >= peak_val]
    recovery_idx = recovered.index[0] if len(recovered) > 0 else None

    return {
        "max_drawdown": max_dd,
        "peak_date": str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx),
        "trough_date": str(max_dd_idx.date()) if hasattr(max_dd_idx, "date") else str(max_dd_idx),
        "recovery_date": str(recovery_idx.date()) if recovery_idx is not None and hasattr(recovery_idx, "date") else None,
        "drawdown_days": (max_dd_idx - peak_idx).days if hasattr(max_dd_idx, "date") else 0,
    }


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix between holdings."""
    return returns_df.corr()


def rolling_beta(
    returns: pd.Series, benchmark: pd.Series, window: int = 60
) -> pd.Series:
    """Rolling beta vs benchmark."""
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    port = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    cov = port.rolling(window).cov(bench)
    var = bench.rolling(window).var()
    result = cov / var.replace(0, np.nan)
    result.name = "rolling_beta"
    return result


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05, trading_days: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(trading_days))


def compute_risk_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, Any]:
    """Compute all risk metrics."""
    return {
        "annualized_volatility": annualized_volatility(portfolio_returns),
        "beta": beta(portfolio_returns, benchmark_returns),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
        "var_95": var_historical(portfolio_returns, 0.95),
        "cvar_95": cvar(portfolio_returns, 0.95),
        "max_drawdown": max_drawdown_analysis(portfolio_returns),
        "annualized_return": float(portfolio_returns.mean() * 252),
    }
