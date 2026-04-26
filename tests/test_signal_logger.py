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


class TestTimeoutClassification:
    """v0.1.6: timeouts are split into timeout_favorable / timeout_adverse
    by tracking MFE/MAE during pending. This unlocks ~99% of historical
    signals that previously had no usable label.
    """

    def _setup(self, tmp_path: Path):
        logger = LiveSignalLogger(log_dir=tmp_path)
        logger.log_signal(
            instrument="MES",
            signal={"direction": "long", "price": 7000.0,
                    "stop": 6990.0, "target": 7020.0, "strength": 1.5,
                    "reasons": ["test"]},
            indicators={"close": 7000.0},
            regime="trending",
        )
        sig_id = next(iter(logger._pending))
        return logger, sig_id

    def test_target_hit_is_win(self, tmp_path: Path):
        logger, _ = self._setup(tmp_path)
        out = logger.check_outcomes("MES", current_price=7020.0)
        assert len(out) == 1
        assert out[0]["outcome"] == "win"

    def test_stop_hit_is_loss(self, tmp_path: Path):
        logger, _ = self._setup(tmp_path)
        out = logger.check_outcomes("MES", current_price=6990.0)
        assert out[0]["outcome"] == "loss"

    def test_mfe_mae_track_running_max(self, tmp_path: Path):
        # Breakage: if MFE/MAE only reflect the latest tick, the
        # timeout classifier looks at the noisy current price not the
        # session's actual extremes. Many setups bounce both ways
        # before settling — without running max we mislabel them.
        logger, sig_id = self._setup(tmp_path)
        # Long entry @ 7000. Walk: 7005 → 6993 → 7012 → 7008.
        for px in (7005.0, 6993.0, 7012.0, 7008.0):
            logger.check_outcomes("MES", current_price=px)
        rec = logger._pending[sig_id]
        assert rec["mfe_pts"] == 12.0   # 7012 − 7000
        assert rec["mae_pts"] == 7.0    # 7000 − 6993

    def _run_until_timeout(self, logger, prices: list) -> dict:
        """Drive check_outcomes until the timeout fires. The MES signal
        in _setup() has bars_counted=0; the loop increments before the
        target/stop/timeout check, so the 61st call with no target/stop
        hit triggers timeout. Caller supplies a price sequence at least
        61 long; we return the resolved record."""
        for px in prices:
            out = logger.check_outcomes("MES", current_price=px)
            if out:
                return out[0]
        raise AssertionError("timeout never fired in test driver")

    def test_timeout_favorable_when_mfe_dominates(self, tmp_path: Path):
        logger, _ = self._setup(tmp_path)
        # 30 favorable bars (MFE rises to 15), then 31 neutral. The
        # 61st call triggers the timeout block; price never crossed
        # the stop or target so bin = timeout_favorable.
        prices = [7015.0] * 30 + [7000.0] * 31
        rec = self._run_until_timeout(logger, prices)
        assert rec["outcome"] == "timeout_favorable"
        assert rec["mfe_pts"] == pytest.approx(15.0)
        assert rec["mae_pts"] == 0.0

    def test_timeout_adverse_when_mae_dominates(self, tmp_path: Path):
        logger, _ = self._setup(tmp_path)
        # MAE rises to 5 in the first 30 bars (price below entry but
        # above stop=6990), then flat for 31 more = timeout fires.
        prices = [6995.0] * 30 + [7000.0] * 31
        rec = self._run_until_timeout(logger, prices)
        assert rec["outcome"] == "timeout_adverse"
        assert rec["mae_pts"] == pytest.approx(5.0)
        assert rec["mfe_pts"] == 0.0

    def test_short_direction_excursions_inverted(self, tmp_path: Path):
        # Breakage: forgetting to flip the sign for shorts makes every
        # short signal look adverse-on-MFE — the model gets a corrupted
        # label set for half its training data.
        logger = LiveSignalLogger(log_dir=tmp_path)
        logger.log_signal(
            instrument="MES",
            signal={"direction": "short", "price": 7000.0,
                    "stop": 7010.0, "target": 6980.0, "strength": 1.5,
                    "reasons": ["test"]},
            indicators={"close": 7000.0},
            regime="trending",
        )
        sig_id = next(iter(logger._pending))
        # Price drops (favorable for short).
        logger.check_outcomes("MES", current_price=6995.0)
        rec = logger._pending[sig_id]
        assert rec["mfe_pts"] == 5.0  # 7000 − 6995, favorable for short
        assert rec["mae_pts"] == 0.0


class TestStatsBuckets:
    def test_stats_surfaces_new_outcome_buckets(self, tmp_path: Path):
        # Breakage: if stats() doesn't expose timeout_favorable, the
        # /api/signals/ml/stats endpoint silently shows the same 0.9%
        # trainable share as the broken pre-v0.1.6 pipeline, masking
        # the fix.
        logger = LiveSignalLogger(log_dir=tmp_path)
        # Pre-seed outcomes log directly with a mix of label types.
        records = [
            {"outcome": "win",                "bars_to_exit": 10},
            {"outcome": "loss",               "bars_to_exit": 8},
            {"outcome": "timeout_favorable",  "bars_to_exit": 60},
            {"outcome": "timeout_favorable",  "bars_to_exit": 60},
            {"outcome": "timeout_adverse",    "bars_to_exit": 60},
            {"outcome": "timeout",            "bars_to_exit": 60},
        ]
        with open(logger.outcome_log, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        s = logger.stats()
        assert s["wins"] == 1
        assert s["losses"] == 1
        assert s["timeout_favorable"] == 2
        assert s["timeout_adverse"] == 1
        assert s["timeouts_legacy"] == 1
        # Trainable = wins + losses + timeout_favorable + timeout_adverse
        assert s["trainable"] == 5
        # Strict win rate = wins / (wins+losses)
        assert s["win_rate"] == 50.0
        # Directional rate = (wins + timeout_favorable) / trainable
        assert s["directional_rate"] == 60.0
