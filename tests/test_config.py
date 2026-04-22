"""Unit tests for configuration loader."""

import pathlib

import pytest

import takata_engine.config as _cfg_mod
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


class TestPackaging:
    """Regression tests for pre-v0.1.4 bug: setuptools shipped loader.py
    without the YAML configs, so every consumer's load_config("MES") hit
    FileNotFoundError on a fresh install. Root cause was missing
    [tool.setuptools.package-data] in pyproject.toml. These tests assert
    the data files live alongside the installed module and not just in
    the source tree — so running the suite against an installed wheel
    (as CI and consumer apps do) will catch a regression.
    """

    def _config_dir(self) -> pathlib.Path:
        return pathlib.Path(_cfg_mod.__file__).parent

    def test_mes_yaml_is_packaged(self):
        p = self._config_dir() / "mes_config.yaml"
        assert p.exists(), (
            f"mes_config.yaml missing from installed package dir {p.parent}. "
            "Check [tool.setuptools.package-data] in pyproject.toml."
        )

    def test_wdo_yaml_is_packaged(self):
        p = self._config_dir() / "wdo_config.yaml"
        assert p.exists(), (
            f"wdo_config.yaml missing from installed package dir {p.parent}."
        )
