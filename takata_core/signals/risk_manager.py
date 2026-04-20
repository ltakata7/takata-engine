"""Risk management — position sizing, stop/target computation, circuit breakers."""

from __future__ import annotations

import math
from typing import Any, Dict


class RiskManager:
    """Computes position sizes and stop/target levels.

    Parameters
    ----------
    config : dict
        Risk section of the instrument config. Expected keys:
        ``max_position_size``, ``stop_atr_mult``, ``target_atr_mult``,
        ``max_daily_loss``.
    point_value : float
        Dollar/BRL value per point for the instrument.
    """

    def __init__(self, config: Dict[str, Any], point_value: float = 1.0):
        self.max_position_size = config.get("max_position_size", 1.0)
        self.stop_atr_mult = config.get("stop_atr_mult", 2.0)
        self.target_atr_mult = config.get("target_atr_mult", 3.0)
        self.max_daily_loss = config.get("max_daily_loss", 500.0)
        self.point_value = point_value

    def update_from_regime(self, risk_config: Dict[str, Any]) -> None:
        """Update risk params from regime-adapted config."""
        self.max_position_size = risk_config.get("max_position_size", self.max_position_size)
        self.stop_atr_mult = risk_config.get("stop_atr_mult", self.stop_atr_mult)
        self.target_atr_mult = risk_config.get("target_atr_mult", self.target_atr_mult)

    def compute_stop(self, price: float, atr_value: float, direction: str) -> float:
        """Compute stop loss price.

        Parameters
        ----------
        price : float
            Entry price.
        atr_value : float
            Current ATR value.
        direction : str
            ``"long"`` or ``"short"``.
        """
        distance = atr_value * self.stop_atr_mult
        if direction == "long":
            return price - distance
        return price + distance

    def compute_target(self, price: float, atr_value: float, direction: str) -> float:
        """Compute profit target price."""
        distance = atr_value * self.target_atr_mult
        if direction == "long":
            return price + distance
        return price - distance

    def compute_size(
        self,
        price: float,
        stop: float,
        account_risk: float,
    ) -> float:
        """Compute position size based on fixed-risk model.

        Sizes the position so that if the stop is hit, the loss equals
        ``account_risk`` (in currency units).

        Parameters
        ----------
        price : float
            Entry price.
        stop : float
            Stop price.
        account_risk : float
            Maximum loss in currency if stop is hit.

        Returns
        -------
        float
            Number of contracts/shares, capped by ``max_position_size``.
        """
        risk_per_unit = abs(price - stop) * self.point_value
        if risk_per_unit <= 0:
            return 0.0

        raw_size = account_risk / risk_per_unit
        capped = min(raw_size, self.max_position_size)
        return max(math.floor(capped), 0)

    def check_daily_limit(self, realized_pnl: float) -> bool:
        """Check if daily loss limit has been breached.

        Returns
        -------
        bool
            ``True`` if trading should continue, ``False`` if limit breached.
        """
        return realized_pnl > -self.max_daily_loss
