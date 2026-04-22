"""Tests for LiveSignalLogger path resolution.

Regression against the pre-v0.1.5 bug where LOG_DIR was
`site-packages/logs/ml/` — invisible to every consumer app that expected
logs to land in their project directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from takata_engine.ml.signal_logger import LiveSignalLogger, _default_log_dir


def _fake_signal() -> dict:
    return {
        "direction": "long",
        "price": 100.0,
        "stop": 99.0,
        "target": 102.0,
        "strength": 0.7,
        "reasons": ["test"],
    }


class TestLogDirResolution:
    def test_constructor_kwarg_wins(self, tmp_path: Path):
        """Explicit log_dir= overrides env var and default."""
        logger = LiveSignalLogger(log_dir=tmp_path)
        assert logger.log_dir == tmp_path.resolve()
        assert logger.signal_log.parent == tmp_path.resolve()

    def test_env_var_used_when_no_kwarg(self, tmp_path: Path, monkeypatch):
        """TAKATA_ML_LOG_DIR env var is read when no kwarg provided."""
        monkeypatch.setenv("TAKATA_ML_LOG_DIR", str(tmp_path))
        logger = LiveSignalLogger()
        assert logger.log_dir == tmp_path.resolve()

    def test_home_fallback_when_no_env_no_kwarg(self, monkeypatch, tmp_path: Path):
        """With no kwarg and no env var, falls back to ~/.takata-engine/logs/ml.
        We spoof $HOME to tmp_path so we don't actually create files in the
        real user home during test runs.
        """
        monkeypatch.delenv("TAKATA_ML_LOG_DIR", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        # Windows uses USERPROFILE for Path.home(); cover both.
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        expected = (tmp_path / ".takata-engine" / "logs" / "ml").resolve()
        logger = LiveSignalLogger()
        assert logger.log_dir == expected

    def test_log_dir_is_created(self, tmp_path: Path):
        target = tmp_path / "nested" / "does" / "not" / "exist"
        assert not target.exists()
        logger = LiveSignalLogger(log_dir=target)
        assert target.exists() and target.is_dir()
        assert logger.log_dir == target.resolve()

    def test_not_site_packages_relative(self):
        """Regression: the default must NOT end up inside site-packages.

        Pre-v0.1.5 bug: LOG_DIR was
        `Path(__file__).parent.parent.parent / "logs" / "ml"`, which only
        worked when takata_engine was in-tree. Once extracted to its own
        pip-installable package, `__file__.parent.parent.parent` became
        `site-packages/` — invisible to consumer apps and wiped by any
        reinstall.
        """
        default = _default_log_dir()
        parts = default.parts
        assert "site-packages" not in parts, (
            f"Default log dir resolves inside site-packages: {default}. "
            "This is the pre-v0.1.5 regression — log_dir must be under "
            "$HOME or an explicit env/kwarg override, never site-packages."
        )


class TestLogWriteRoundtrip:
    def test_log_and_stats_see_same_dir(self, tmp_path: Path):
        """Writing a signal and reading stats() must agree on the dir."""
        logger = LiveSignalLogger(log_dir=tmp_path)
        logger.log_signal(
            instrument="MES", signal=_fake_signal(),
            indicators={"close": 100, "rsi": 50}, regime="trending",
        )
        s = logger.stats()
        assert s["total_signals_logged"] == 1
        assert s["log_dir"] == str(tmp_path.resolve())

    def test_stats_reads_pre_existing_log(self, tmp_path: Path):
        """A new logger pointed at a dir with existing JSONL must count them.
        This mirrors the real-world case of a restart picking up prior
        session's data.
        """
        # Seed the dir with a pre-existing log line.
        (tmp_path / "live_signals.jsonl").write_text(
            json.dumps({"signal_id": "x", "instrument": "MES"}) + "\n"
        )
        logger = LiveSignalLogger(log_dir=tmp_path)
        assert logger.total_logged == 1
