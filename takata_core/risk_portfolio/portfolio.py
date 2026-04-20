"""Portfolio data model — loads holdings from YAML, fetches prices."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from takata_core.data.feed_market import MarketDataFeed

logger = logging.getLogger(__name__)

PORTFOLIOS_DIR = Path(__file__).resolve().parent.parent.parent / "portfolios"


@dataclass
class Holding:
    """A single portfolio position."""

    ticker: str
    shares: float
    cost_basis: float
    current_price: float = 0.0
    market_value: float = 0.0
    weight: float = 0.0
    gain_loss: float = 0.0
    gain_loss_pct: float = 0.0


class Portfolio:
    """Portfolio loaded from YAML with live prices.

    Parameters
    ----------
    name : str
        Account name.
    account_type : str
        ``"personal"`` or ``"business"``.
    benchmark : str
        Benchmark ticker (e.g. ``"SPY"``).
    holdings : list of Holding
        Portfolio positions.
    currency : str
        Base currency.
    """

    def __init__(
        self,
        name: str,
        account_type: str,
        benchmark: str,
        holdings: List[Holding],
        currency: str = "USD",
    ):
        self.name = name
        self.account_type = account_type
        self.benchmark = benchmark
        self.holdings = holdings
        self.currency = currency

    @classmethod
    def load(cls, path: str | Path) -> "Portfolio":
        """Load portfolio from a YAML file and fetch current prices."""
        path = Path(path)
        if not path.exists():
            # Try portfolios directory
            alt = PORTFOLIOS_DIR / f"{path}.yaml"
            if alt.exists():
                path = alt
            else:
                alt2 = PORTFOLIOS_DIR / path
                if alt2.exists():
                    path = alt2
                else:
                    raise FileNotFoundError(f"Portfolio file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        account = data.get("account", {})
        raw_holdings = data.get("holdings", [])

        holdings = []
        for h in raw_holdings:
            holdings.append(Holding(
                ticker=h["ticker"],
                shares=float(h["shares"]),
                cost_basis=float(h["cost_basis"]),
            ))

        portfolio = cls(
            name=account.get("name", path.stem),
            account_type=account.get("type", "personal"),
            benchmark=account.get("benchmark", "SPY"),
            holdings=holdings,
            currency=account.get("currency", "USD"),
        )
        portfolio.refresh_prices()
        return portfolio

    def refresh_prices(self) -> None:
        """Fetch current prices and compute market values + weights."""
        total = 0.0
        for h in self.holdings:
            try:
                feed = MarketDataFeed(h.ticker, period="5d", interval="1d")
                df = feed.load()
                h.current_price = float(df["close"].iloc[-1])
            except Exception as e:
                logger.warning("Could not fetch price for %s: %s", h.ticker, e)
                h.current_price = h.cost_basis  # fallback

            h.market_value = h.shares * h.current_price
            h.gain_loss = (h.current_price - h.cost_basis) * h.shares
            h.gain_loss_pct = (h.current_price / h.cost_basis - 1) * 100 if h.cost_basis > 0 else 0
            total += h.market_value

        # Compute weights
        for h in self.holdings:
            h.weight = h.market_value / total if total > 0 else 0

    @property
    def total_value(self) -> float:
        return sum(h.market_value for h in self.holdings)

    @property
    def total_cost(self) -> float:
        return sum(h.shares * h.cost_basis for h in self.holdings)

    @property
    def total_gain_loss(self) -> float:
        return self.total_value - self.total_cost

    @property
    def tickers(self) -> List[str]:
        return [h.ticker for h in self.holdings]

    def holdings_df(self) -> pd.DataFrame:
        """Return holdings as a DataFrame."""
        records = []
        for h in self.holdings:
            records.append({
                "ticker": h.ticker,
                "shares": h.shares,
                "cost_basis": h.cost_basis,
                "current_price": h.current_price,
                "market_value": h.market_value,
                "weight": h.weight,
                "gain_loss": h.gain_loss,
                "gain_loss_pct": h.gain_loss_pct,
            })
        return pd.DataFrame(records)

    def returns_history(self, period: str = "1y") -> pd.DataFrame:
        """Fetch daily returns for all holdings.

        Returns
        -------
        pd.DataFrame
            Columns = tickers, index = dates, values = daily returns.
        """
        all_returns = {}
        for h in self.holdings:
            try:
                feed = MarketDataFeed(h.ticker, period=period, interval="1d")
                df = feed.load()
                all_returns[h.ticker] = df["close"].pct_change().dropna()
            except Exception as e:
                logger.warning("Could not fetch history for %s: %s", h.ticker, e)

        return pd.DataFrame(all_returns).dropna()

    def portfolio_returns(self, period: str = "1y") -> pd.Series:
        """Compute weighted portfolio daily returns."""
        ret_df = self.returns_history(period)
        weights = {h.ticker: h.weight for h in self.holdings}

        # Align weights with available tickers
        available = [t for t in ret_df.columns if t in weights]
        w = pd.Series({t: weights[t] for t in available})
        w = w / w.sum()  # renormalize

        port_ret = (ret_df[available] * w).sum(axis=1)
        port_ret.name = "portfolio"
        return port_ret
