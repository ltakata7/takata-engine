"""Unit tests for configuration loader."""

import pytest

from takata_engine.config import load_config


class TestLoadConfig:
    def test_load_mes(self):
        cfg = load_config("MES")
        assert cfg["instrument"]["symbol"] == "MES"
        assert cfg["instrument"]["tick_size"] == 0.25

    def test_load_wdo(self):
        cfg = load_config("WDO")
        assert cfg["instrument"]["symbol"] == "WDO"
        assert cfg["instrument"]["exchange"] == "B3"

    def test_case_insensitive(self):
        cfg = load_config("mes")
        assert cfg["instrument"]["symbol"] == "MES"

    def test_unknown_instrument(self):
        with pytest.raises(FileNotFoundError):
            load_config("UNKNOWN")

    def test_overrides(self):
        cfg = load_config("MES", overrides={"indicators": {"rsi": {"period": 7}}})
        assert cfg["indicators"]["rsi"]["period"] == 7
        # Other values unchanged
        assert cfg["indicators"]["ema"]["fast"] == 9
