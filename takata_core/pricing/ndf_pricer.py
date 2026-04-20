"""NDF Pricer — BRL Non-Deliverable Forward pricing.

Ports the institutional Roll and on/off spread calculations from
Lauro's VBA spreadsheet into Python.

Key concepts:
- Onshore roll: B3 futures spread between maturities
- Offshore roll: NDF spread between maturities
- On/Off spread: offshore - onshore (intermediary margin for foreign access)
- Casado: futures - spot (the basis)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class NDFQuote:
    """An NDF quote for a specific tenor."""
    tenor: str          # e.g. "1M", "3M", "6M"
    bid: float
    ask: float
    mid: float
    fixing_date: str
    value_date: str


class NDFPricer:
    """Prices BRL NDFs and computes on/off spreads.

    The on/off spread is the key institutional signal:
    - Widening on/off = foreign stress, risk-off → WDO bullish (BRL weakening)
    - Tightening on/off = normalization → WDO bearish (BRL strengthening)
    """

    def __init__(self, spot_usd: float = 5.19, ptax: float = 5.19):
        self.spot_usd = spot_usd
        self.ptax = ptax

    def roll_onshore(self, fut_near: float, fut_far: float) -> float:
        """Onshore roll: spread between B3 futures maturities.

        From VBA Roll function, tipo=1:
        Roll = 10 * (BMF_fut(vcto2) - BMF_fut(vcto1))

        Parameters
        ----------
        fut_near : float
            Near-month B3 futures price.
        fut_far : float
            Far-month B3 futures price.

        Returns
        -------
        float
            Onshore roll in points (× R$10 per contract).
        """
        return 10 * (fut_far - fut_near)

    def roll_offshore(self, ndf_near: float, ndf_far: float) -> float:
        """Offshore roll: spread between NDF maturities.

        From VBA Roll function, tipo=2:
        Roll = 10 * (NDF(vcto2) - NDF(vcto1))
        """
        return 10 * (ndf_far - ndf_near)

    def onoff_spread(self, fut_near: float, fut_far: float, ndf_near: float, ndf_far: float) -> float:
        """On/Off spread: offshore roll minus onshore roll.

        From VBA Roll function, tipo=3:
        OnOff = 10 * (NDF_roll - BMF_roll)

        This is the institutional cost for offshore access.
        Widening = stress, tightening = normalization.
        """
        offshore = self.roll_offshore(ndf_near, ndf_far)
        onshore = self.roll_onshore(fut_near, fut_far)
        return offshore - onshore

    def casado(self, futures_price: float) -> float:
        """Casado: futures price minus PTAX × 1000.

        The basis between futures and spot.
        Abnormal casado = positioning pressure signal.
        """
        return futures_price - self.ptax * 1000

    def forward_points(self, futures_price: float) -> float:
        """Forward points: futures - PTAX × 1000.

        FRP0 formula: Ft = (PTAX × 1,000) + f
        Therefore: f = Ft - (PTAX × 1,000)
        """
        return futures_price - self.ptax * 1000

    def ndf_from_forward_points(self, fwd_pts: float) -> float:
        """Derive NDF rate from forward points.

        NDF = (PTAX × 1,000 + fwd_pts) / 1,000
        """
        return (self.ptax * 1000 + fwd_pts) / 1000

    def theoretical_wdo(self, di_rate: float, cupom: float, du: int) -> float:
        """Compute theoretical WDO fair value from interest rate differential.

        WDO_fair = PTAX × 1000 × (1 + DI)^(DU/252) / (1 + cupom × DC/360)

        Parameters
        ----------
        di_rate : float
            DI rate (percentage, e.g. 13.63).
        cupom : float
            Cupom cambial / offshore rate (percentage).
        du : int
            Business days to maturity.

        Returns
        -------
        float
            Theoretical WDO price.
        """
        dc = du * 365 / 252  # approximate calendar days
        onshore_factor = (1 + di_rate / 100) ** (du / 252)
        offshore_factor = 1 + (cupom / 100) * dc / 360
        return self.ptax * 1000 * onshore_factor / offshore_factor

    def fair_value_signal(self, actual_wdo: float, di_rate: float, cupom: float, du: int) -> Dict[str, Any]:
        """Compare actual WDO price to theoretical fair value.

        Divergence = potential mean-reversion signal.

        Returns
        -------
        dict
            ``theoretical``: fair value price
            ``actual``: current WDO price
            ``divergence_pts``: actual - theoretical (positive = WDO overvalued)
            ``signal``: "overvalued" / "undervalued" / "fair"
        """
        theo = self.theoretical_wdo(di_rate, cupom, du)
        divergence = actual_wdo - theo

        if divergence > 5:
            signal = "overvalued"  # WDO too high → expect mean reversion down
        elif divergence < -5:
            signal = "undervalued"  # WDO too low → expect mean reversion up
        else:
            signal = "fair"

        return {
            "theoretical": round(theo, 2),
            "actual": actual_wdo,
            "divergence_pts": round(divergence, 2),
            "signal": signal,
        }
