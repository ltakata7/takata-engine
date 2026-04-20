"""Multi-factor stock/ETF screener — Renaissance Technologies style."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from takata_core.screener.factors import compute_all_factors

logger = logging.getLogger(__name__)

# Default screening universes
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "UNH", "XOM", "PG", "MA", "HD", "COST", "ABBV",
    "KO", "PEP", "MRK", "LLY", "AVGO", "CRM", "ORCL", "NFLX", "AMD",
    "INTC", "DIS", "CSCO", "WMT", "BAC", "GS", "MS", "PFE", "TMO",
    "NEE", "LOW", "UPS", "CAT", "DE", "MMM", "GE",
]

ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE",
    "EFA", "EEM", "VWO", "VXUS",
    "TLT", "IEF", "AGG", "BND", "LQD", "HYG", "SHY",
    "GLD", "SLV", "IAU",
    "VNQ", "SCHH",
    "SCHD", "VIG", "VYM", "DVY", "HDV",
    "ARKK", "SOXX", "SMH", "XBI",
]


class FactorScreener:
    """Multi-factor screener for stocks and ETFs.

    Parameters
    ----------
    universe : list of str, optional
        Ticker list to screen. Defaults to combined stock + ETF universe.
    universe_type : str
        ``"stocks"``, ``"etfs"``, ``"all"``, or ``"custom"``.
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        universe_type: str = "all",
    ):
        if universe:
            self.universe = [t.upper() for t in universe]
        elif universe_type == "stocks":
            self.universe = STOCK_UNIVERSE
        elif universe_type == "etfs":
            self.universe = ETF_UNIVERSE
        else:
            self.universe = STOCK_UNIVERSE + ETF_UNIVERSE

    def screen(self, top_n: int = 10) -> pd.DataFrame:
        """Run the screener across the universe.

        Parameters
        ----------
        top_n : int
            Number of top-ranked tickers to highlight.

        Returns
        -------
        pd.DataFrame
            All tickers with factor scores, sorted by composite descending.
        """
        results = []
        total = len(self.universe)

        for i, ticker in enumerate(self.universe):
            logger.info("[%d/%d] Screening %s", i + 1, total, ticker)
            try:
                scores = compute_all_factors(ticker)
                results.append(scores)
            except Exception as e:
                logger.warning("Failed to screen %s: %s", ticker, e)

        if not results:
            return pd.DataFrame()

        # Build results DataFrame
        rows = []
        for r in results:
            rows.append({
                "ticker": r["ticker"],
                "name": r["name"],
                "sector": r["sector"],
                "value": r["value"]["score"],
                "quality": r["quality"]["score"],
                "momentum": r["momentum"]["score"],
                "growth": r["growth"]["score"],
                "sentiment": r["sentiment"]["score"],
                "composite": r["composite"],
            })

        df = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # 1-based rank
        df.index.name = "rank"

        self._results = results
        self._df = df
        return df

    def format_report(self, top_n: int = 10) -> str:
        """Format screener results as institutional report."""
        df = self._df if hasattr(self, "_df") else self.screen(top_n)
        results = self._results if hasattr(self, "_results") else []

        w = 90
        lines = [
            "=" * w,
            "  MULTI-FACTOR SCREENER — RENAISSANCE TECHNOLOGIES STYLE",
            f"  Universe: {len(df)} tickers  |  Factors: Value, Quality, Momentum, Growth, Sentiment",
            "=" * w,
            "",
            f"TOP {top_n} RANKED",
            "-" * w,
            f"  {'Rank':>4s}  {'Ticker':8s} {'Name':20s} {'Composite':>9s} "
            f"{'Value':>6s} {'Qual':>6s} {'Mom':>6s} {'Grow':>6s} {'Sent':>6s}",
        ]

        for rank, row in df.head(top_n).iterrows():
            lines.append(
                f"  {rank:>4d}  {row['ticker']:8s} {row['name'][:20]:20s} "
                f"{row['composite']:>8.1f}  "
                f"{row['value']:>5.0f}  {row['quality']:>5.0f}  "
                f"{row['momentum']:>5.0f}  {row['growth']:>5.0f}  {row['sentiment']:>5.0f}"
            )

        # Factor breakdown for top 3
        lines.extend(["", "FACTOR BREAKDOWN — TOP 3", "-" * w])
        top_results = [r for r in results if r["ticker"] in df.head(3)["ticker"].values]
        top_results.sort(key=lambda r: -r["composite"])

        for r in top_results[:3]:
            lines.append(f"\n  {r['ticker']} — {r['name']} (Composite: {r['composite']})")
            for factor_name in ["value", "quality", "momentum", "growth", "sentiment"]:
                f = r[factor_name]
                detail_str = ", ".join(f"{k}: {v}" for k, v in f["details"].items())
                bar = "#" * int(f["score"] / 5)
                lines.append(f"    {factor_name.capitalize():12s} {f['score']:>5.0f}  {bar}")
                if detail_str:
                    lines.append(f"      {detail_str}")

        # Sector diversification
        lines.extend(["", "SECTOR DIVERSIFICATION — TOP 10", "-" * w])
        top10 = df.head(top_n)
        sector_counts = top10["sector"].value_counts()
        for sector, count in sector_counts.items():
            lines.append(f"  {sector:25s}  {count} picks")

        # Watchlist (ranks 11-20)
        if len(df) > top_n:
            lines.extend(["", "WATCHLIST (next 10)", "-" * w])
            for rank, row in df.iloc[top_n:top_n + 10].iterrows():
                lines.append(
                    f"  {rank:>4d}  {row['ticker']:8s} {row['name'][:20]:20s}  "
                    f"Composite: {row['composite']:.0f}"
                )

        lines.append("")
        lines.append("=" * w)
        return "\n".join(lines)
