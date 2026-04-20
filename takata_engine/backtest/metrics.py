"""Performance metrics for backtesting — trade-based and equity-curve based."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from takata_engine.signals.position import Position


# ── Trade-based metrics ──────────────────────────────────────────────


def win_rate(positions: List[Position]) -> float:
    """Fraction of winning trades (P&L > 0)."""
    if not positions:
        return 0.0
    return sum(1 for p in positions if p.pnl > 0) / len(positions)


def profit_factor(positions: List[Position]) -> float:
    """Gross profit / gross loss. >1 is profitable."""
    gross_profit = sum(p.pnl for p in positions if p.pnl > 0)
    gross_loss = abs(sum(p.pnl for p in positions if p.pnl <= 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def expectancy(positions: List[Position]) -> float:
    """Average P&L per trade."""
    if not positions:
        return 0.0
    return sum(p.pnl for p in positions) / len(positions)


def avg_winner(positions: List[Position]) -> float:
    """Average P&L of winning trades."""
    winners = [p.pnl for p in positions if p.pnl > 0]
    return float(np.mean(winners)) if winners else 0.0


def avg_loser(positions: List[Position]) -> float:
    """Average P&L of losing trades (negative value)."""
    losers = [p.pnl for p in positions if p.pnl <= 0]
    return float(np.mean(losers)) if losers else 0.0


def max_consecutive_losses(positions: List[Position]) -> int:
    """Longest streak of consecutive losing trades."""
    max_streak = 0
    current = 0
    for p in positions:
        if p.pnl <= 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def trade_duration_stats(positions: List[Position]) -> Dict[str, float]:
    """Duration stats in minutes for closed trades."""
    durations = []
    for p in positions:
        if p.exit_time and p.entry_time:
            delta = (p.exit_time - p.entry_time).total_seconds() / 60.0
            durations.append(delta)
    if not durations:
        return {"avg_minutes": 0, "median_minutes": 0, "max_minutes": 0}
    return {
        "avg_minutes": float(np.mean(durations)),
        "median_minutes": float(np.median(durations)),
        "max_minutes": float(np.max(durations)),
    }


# ── Equity-curve metrics ─────────────────────────────────────────────


def equity_curve(positions: List[Position], initial_capital: float = 10000.0) -> pd.Series:
    """Build cumulative equity curve from closed positions (ordered by exit time)."""
    if not positions:
        return pd.Series([initial_capital], name="equity")

    sorted_pos = sorted(positions, key=lambda p: p.exit_time or p.entry_time)
    pnls = [p.pnl for p in sorted_pos]
    cum_pnl = np.cumsum(pnls)
    equity = initial_capital + cum_pnl

    times = [p.exit_time or p.entry_time for p in sorted_pos]
    return pd.Series(equity, index=times, name="equity")


def returns_series(positions: List[Position], initial_capital: float = 10000.0) -> pd.Series:
    """Compute per-trade returns as fraction of equity at trade entry."""
    if not positions:
        return pd.Series(dtype=float, name="returns")

    sorted_pos = sorted(positions, key=lambda p: p.exit_time or p.entry_time)
    equity = initial_capital
    rets = []
    times = []
    for p in sorted_pos:
        r = p.pnl / equity if equity > 0 else 0.0
        rets.append(r)
        times.append(p.exit_time or p.entry_time)
        equity += p.pnl

    return pd.Series(rets, index=times, name="returns")


def sharpe_ratio(returns: pd.Series, trades_per_year: float = 252.0) -> float:
    """Annualized Sharpe ratio from per-trade returns.

    Parameters
    ----------
    returns : pd.Series
        Per-trade returns (fractions, not percentages).
    trades_per_year : float
        Estimated number of trades per year for annualization.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(trades_per_year))


def sortino_ratio(returns: pd.Series, trades_per_year: float = 252.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std() * np.sqrt(trades_per_year))


def max_drawdown(eq: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.15 = 15%)."""
    if len(eq) < 2:
        return 0.0
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    return float(abs(dd.min()))


def max_drawdown_duration(eq: pd.Series) -> int:
    """Number of trades spent in the longest drawdown period."""
    if len(eq) < 2:
        return 0
    peak = eq.expanding().max()
    in_dd = eq < peak

    max_dur = 0
    current = 0
    for v in in_dd:
        if v:
            current += 1
            max_dur = max(max_dur, current)
        else:
            current = 0
    return max_dur


def calmar_ratio(returns: pd.Series, max_dd: float, trades_per_year: float = 252.0) -> float:
    """Annualized return / max drawdown."""
    if max_dd == 0 or len(returns) == 0:
        return 0.0
    annual_return = returns.mean() * trades_per_year
    return float(annual_return / max_dd)


# ── Summary ──────────────────────────────────────────────────────────


def compute_all(
    positions: List[Position],
    initial_capital: float = 10000.0,
    trades_per_year: float = 252.0,
) -> Dict[str, Any]:
    """Compute all performance metrics."""
    eq = equity_curve(positions, initial_capital)
    rets = returns_series(positions, initial_capital)
    mdd = max_drawdown(eq)

    return {
        "total_trades": len(positions),
        "win_rate": win_rate(positions),
        "profit_factor": profit_factor(positions),
        "expectancy": expectancy(positions),
        "avg_winner": avg_winner(positions),
        "avg_loser": avg_loser(positions),
        "max_consecutive_losses": max_consecutive_losses(positions),
        "total_pnl": sum(p.pnl for p in positions),
        "final_equity": float(eq.iloc[-1]) if len(eq) > 0 else initial_capital,
        "sharpe_ratio": sharpe_ratio(rets, trades_per_year),
        "sortino_ratio": sortino_ratio(rets, trades_per_year),
        "max_drawdown": mdd,
        "max_drawdown_duration": max_drawdown_duration(eq),
        "calmar_ratio": calmar_ratio(rets, mdd, trades_per_year),
        "trade_duration": trade_duration_stats(positions),
    }


def format_report(metrics: Dict[str, Any], title: str = "Backtest Report") -> str:
    """Format metrics dict as a printable report."""
    lines = [
        title,
        "=" * 55,
        f"  Total trades:          {metrics['total_trades']}",
        f"  Win rate:              {metrics['win_rate']:.1%}",
        f"  Profit factor:         {metrics['profit_factor']:.2f}",
        f"  Expectancy:            {metrics['expectancy']:,.2f}",
        f"  Avg winner:            {metrics['avg_winner']:,.2f}",
        f"  Avg loser:             {metrics['avg_loser']:,.2f}",
        f"  Max consec. losses:    {metrics['max_consecutive_losses']}",
        "",
        f"  Total P&L:             {metrics['total_pnl']:,.2f}",
        f"  Final equity:          {metrics['final_equity']:,.2f}",
        "",
        f"  Sharpe ratio:          {metrics['sharpe_ratio']:.2f}",
        f"  Sortino ratio:         {metrics['sortino_ratio']:.2f}",
        f"  Max drawdown:          {metrics['max_drawdown']:.1%}",
        f"  Max DD duration:       {metrics['max_drawdown_duration']} trades",
        f"  Calmar ratio:          {metrics['calmar_ratio']:.2f}",
        "",
        f"  Avg trade duration:    {metrics['trade_duration']['avg_minutes']:.0f} min",
    ]
    return "\n".join(lines)
