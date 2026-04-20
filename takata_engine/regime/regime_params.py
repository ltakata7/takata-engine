"""Regime-adaptive parameter profile loading."""

from __future__ import annotations

from typing import Any, Dict, Optional

from takata_engine.config.loader import _deep_merge


class RegimeParams:
    """Loads and applies regime-specific parameter overrides.

    Given a base config and a ``regimes`` section mapping regime labels
    to parameter overrides, produces a merged config for any detected regime.

    Parameters
    ----------
    base_config : dict
        Full instrument config (from ``load_config``).
    """

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.regime_profiles: Dict[str, Dict[str, Any]] = base_config.get("regimes", {})

    def get_params(self, regime_label: str) -> Dict[str, Any]:
        """Return merged config for a given regime.

        Parameters
        ----------
        regime_label : str
            Detected regime (e.g. ``"low_vol_trending"``).

        Returns
        -------
        dict
            Base config with regime-specific overrides merged in.
        """
        overrides = self.regime_profiles.get(regime_label, {})
        if not overrides:
            return self.base_config
        return _deep_merge(self.base_config, overrides)

    @property
    def available_regimes(self) -> list:
        """List of regime labels that have parameter profiles."""
        return list(self.regime_profiles.keys())
