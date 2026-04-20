"""Configuration loader with defaults and override merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent
CONFIGS = {
    "MES": CONFIG_DIR / "mes_config.yaml",
    "WDO": CONFIG_DIR / "wdo_config.yaml",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(instrument: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load instrument config, optionally merging *overrides* on top.

    Parameters
    ----------
    instrument : str
        Instrument key (e.g. ``"MES"`` or ``"WDO"``).
    overrides : dict, optional
        Nested dict of values to override (e.g. regime-specific params).

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    instrument = instrument.upper()
    path = CONFIGS.get(instrument)
    if path is None or not path.exists():
        raise FileNotFoundError(f"No config found for instrument '{instrument}'")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
