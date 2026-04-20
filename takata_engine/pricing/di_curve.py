"""DI Curve — Brazilian interest rate curve bootstrapping and interpolation.

Ports the institutional VBA logic (BNDF5 exponential interpolation) from
Lauro's NDF pricing spreadsheet into Python using FinancePy conventions.

The DI curve is the backbone of BRL derivatives pricing:
- DI futures (DI1FUT) traded on B3
- Used to derive: forward rates, cupom cambial, NDF fair value
- Day count: bus/252 (Brazilian business days)
- Compounding: discrete, annual

Key relationship for WDO:
- DI curve up → Selic hike expectations → BRL strengthens → WDO falls
- DI curve down → rate cut expectations → BRL weakens → WDO rises
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# DI futures month codes (B3 convention)
MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}


@dataclass
class DIPoint:
    """A single point on the DI curve."""
    ticker: str          # e.g. "N26" (July 2026)
    maturity: date       # first business day of expiry month
    business_days: int   # business days from today to maturity (DU)
    calendar_days: int   # calendar days (DC)
    rate: float          # annualized DI rate (e.g. 13.63 = 13.63%)
    fra: float = 0.0     # forward rate agreement between this and next tenor


class DICurve:
    """Brazilian DI interest rate curve.

    Parameters
    ----------
    valuation_date : date
        Today's date.
    points : list of DIPoint
        DI futures curve points.
    cdi : float
        Current CDI overnight rate (e.g. 0.019 = 1.9% p.a. or daily rate).
    """

    def __init__(self, valuation_date: date, points: List[DIPoint], cdi: float = 0.0):
        self.valuation_date = valuation_date
        self.points = sorted(points, key=lambda p: p.business_days)
        self.cdi = cdi
        self._compute_fras()

    @classmethod
    def from_bridge_data(cls, bridge_url: str = "http://127.0.0.1:8002") -> Optional["DICurve"]:
        """Build DI curve from live Profit Ultra bridge data."""
        import requests

        try:
            r = requests.get(f"{bridge_url}/quote/DI1FUT", timeout=3)
            di_data = r.json()
            if "error" in di_data:
                return None

            # For now, create a single-point curve from the front DI contract
            today = date.today()
            rate = float(di_data.get("last", 0))
            if rate <= 0:
                return None

            # Front DI contract typically expires ~6 months out
            # TODO: Expand to full curve when we have all DI tenors from bridge
            point = DIPoint(
                ticker="DI1",
                maturity=date(today.year, today.month + 6 if today.month <= 6 else today.month - 6, 1),
                business_days=126,  # ~6 months of business days
                calendar_days=180,
                rate=rate,
            )

            return cls(today, [point], cdi=rate)
        except Exception as e:
            logger.warning("Could not build DI curve from bridge: %s", e)
            return None

    def _compute_fras(self) -> None:
        """Compute Forward Rate Agreements between consecutive tenors.

        FRA = forward rate between tenor i and tenor i+1
        """
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]

            # Compound factor to each maturity
            cf1 = (1 + p1.rate / 100) ** (p1.business_days / 252)
            cf2 = (1 + p2.rate / 100) ** (p2.business_days / 252)

            # Forward compound factor
            du_fwd = p2.business_days - p1.business_days
            if du_fwd > 0:
                cf_fwd = cf2 / cf1
                p2.fra = (cf_fwd ** (252 / du_fwd) - 1) * 100
            else:
                p2.fra = p2.rate

    def interpolate(self, target_du: int) -> float:
        """Interpolate the DI rate at a given number of business days.

        Uses exponential interpolation (BNDF5 from institutional spreadsheet):
        rate(t) = (1 + rate_i)^d * (1 + rate_{i+1})^(1-d) - 1

        Where d = (DU_{i+1} - target) / (DU_{i+1} - DU_i)

        Parameters
        ----------
        target_du : int
            Target business days from today.

        Returns
        -------
        float
            Interpolated annualized DI rate (percentage).
        """
        if not self.points:
            return 0.0

        # Extrapolation: flat
        if target_du <= self.points[0].business_days:
            return self.points[0].rate
        if target_du >= self.points[-1].business_days:
            return self.points[-1].rate

        # Find bracketing points
        for i in range(len(self.points) - 1):
            if self.points[i].business_days <= target_du <= self.points[i + 1].business_days:
                break

        p1 = self.points[i]
        p2 = self.points[i + 1]

        # Exponential interpolation (BNDF5)
        d = (p2.business_days - target_du) / (p2.business_days - p1.business_days)
        rate = (1 + p1.rate / 100) ** d * (1 + p2.rate / 100) ** (1 - d) - 1

        return rate * 100

    def discount_factor(self, du: int) -> float:
        """Compute discount factor for a given number of business days.

        DF = 1 / (1 + rate)^(DU/252)
        """
        rate = self.interpolate(du)
        return 1.0 / (1 + rate / 100) ** (du / 252)

    def forward_rate(self, du1: int, du2: int) -> float:
        """Compute forward rate between two business day dates.

        FRA = ((DF1 / DF2)^(252 / (DU2 - DU1)) - 1) * 100
        """
        if du2 <= du1:
            return 0.0
        df1 = self.discount_factor(du1)
        df2 = self.discount_factor(du2)
        return ((df1 / df2) ** (252 / (du2 - du1)) - 1) * 100

    def dv01(self, du: int, notional: float = 100000) -> float:
        """Dollar value of a basis point for a given tenor.

        BNDF2 from institutional spreadsheet:
        DV01 = notional * (DF(rate-0.01%) - DF(rate+0.01%))
        """
        rate = self.interpolate(du)
        df_down = 1.0 / (1 + (rate - 0.01) / 100) ** (du / 252)
        df_up = 1.0 / (1 + (rate + 0.01) / 100) ** (du / 252)
        return notional * (df_down - df_up)

    def theoretical_dollar_future(self, du_short: int, du_long: int, spot_usd: float, fra: Optional[float] = None) -> float:
        """Compute theoretical dollar futures price from DI curve.

        BNDF3 from institutional spreadsheet:
        DolFwd = Spot * (DF_short / DF_long) / (1 + FRA * DC/360)

        This is the fair value that WDO should converge to.
        Divergence from actual WDO price = arbitrage/flow signal.

        Parameters
        ----------
        du_short : int
            Business days to near maturity.
        du_long : int
            Business days to far maturity.
        spot_usd : float
            Current spot USDBRL rate (e.g. 5.19).
        fra : float, optional
            Offshore forward rate. If None, uses curve-implied rate.

        Returns
        -------
        float
            Theoretical dollar futures price (in B3 quotation: BRL per USD 1,000).
        """
        df_short = self.discount_factor(du_short)
        df_long = self.discount_factor(du_long)

        if fra is None:
            fra = self.forward_rate(du_short, du_long)

        dc = (du_long - du_short) * 365 / 252  # approximate calendar days
        fwd_factor = 1 + (fra / 100) * dc / 360

        return spot_usd * 1000 * (df_short / df_long) / fwd_factor

    def cupom_cambial(self, du: int, spot_usd: float, dol_fut: float) -> float:
        """Compute cupom cambial (offshore USD implied rate).

        BNDF1 from institutional spreadsheet:
        Linha = (Juroslongo * Juroscurto * Pronto / DolFut - 1) * 360 / diaslinha

        Parameters
        ----------
        du : int
            Business days to maturity.
        spot_usd : float
            Current spot USDBRL.
        dol_fut : float
            Dollar futures price (B3 quotation).

        Returns
        -------
        float
            Implied offshore USD rate (annualized, ACT/360).
        """
        di_rate = self.interpolate(du)
        df_onshore = (1 + di_rate / 100) ** (du / 252)
        dc = du * 365 / 252  # approximate calendar days

        if dol_fut <= 0:
            return 0.0

        pronto = spot_usd * 1000  # convert to B3 quotation
        cupom = (df_onshore * pronto / dol_fut - 1) * 360 / dc

        return cupom * 100  # as percentage

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "valuation_date": str(self.valuation_date),
            "cdi": self.cdi,
            "points": [
                {
                    "ticker": p.ticker,
                    "maturity": str(p.maturity),
                    "business_days": p.business_days,
                    "rate": p.rate,
                    "fra": round(p.fra, 4),
                }
                for p in self.points
            ],
        }
