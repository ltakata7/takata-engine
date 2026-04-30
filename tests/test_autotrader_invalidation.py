"""Tests for Phase 7 dynamic-stop invalidation in MESAutotrader."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from takata_engine.execution import autotrader as at_module
from takata_engine.execution.autotrader import (
    AutotraderConfig,
    AutotraderState,
    MESAutotrader,
)


@pytest.fixture
def at(tmp_path, monkeypatch):
    """Fresh autotrader instance — its config + state files redirected to tmp."""
    monkeypatch.setattr(at_module, "CONFIG_PATH", tmp_path / "autotrader_config.json")
    monkeypatch.setattr(at_module, "STATE_PATH", tmp_path / "autotrader_state.json")
    autotrader = MESAutotrader()
    # Stub the executor so flatten() doesn't try to talk to IBKR.
    class _StubExecutor:
        def get_positions(self):
            return []
        def flatten(self, *a, **kw):
            return {"status": "stubbed"}
    autotrader._executor = _StubExecutor()
    return autotrader


def _open_long(at, entry=7195.0, stop=7190.0, target=7205.0, contracts=1):
    """Inject a fake open position into autotrader state."""
    at.state.open_position = {
        "direction": "long",
        "entry_price": entry,
        "stop": stop,
        "target": target,
        "contracts": contracts,
        "order_ids": [],
        "opened_at": "2026-04-29T18:00:00",
        "opened_at_ts": time.time() - 60,   # 60s ago
        "signal_strength": 0.8,
        "reasons": ["test"],
        "best_favorable_pts": 0.0,
        "cvd_at_entry": None,
    }


def _open_short(at, entry=7195.0, stop=7200.0, target=7185.0, contracts=1):
    at.state.open_position = {
        "direction": "short", "entry_price": entry, "stop": stop, "target": target,
        "contracts": contracts, "order_ids": [],
        "opened_at": "2026-04-29T18:00:00",
        "opened_at_ts": time.time() - 60,
        "signal_strength": 0.8, "reasons": ["test"],
        "best_favorable_pts": 0.0, "cvd_at_entry": None,
    }


# ── Master switch ──

def test_invalidation_silent_when_master_disabled(at):
    _open_long(at)
    at.config.invalidation_enabled = False
    at.config.invalidation_time_enabled = True
    result = at.evaluate_invalidation({"current_price": 7195.0, "atr": 2.0})
    assert result is None


def test_invalidation_silent_when_no_position(at):
    at.config.invalidation_enabled = True
    at.config.invalidation_time_enabled = True
    at.state.open_position = None
    result = at.evaluate_invalidation({"current_price": 7195.0, "atr": 2.0})
    assert result is None


# ── Time-based ──

def test_time_based_fires_after_no_progress(at, monkeypatch):
    """Position open ≥ N bars with progress < threshold → flatten."""
    _open_long(at, entry=7195.0)
    # Pretend the position opened 5 bars ago (5 * 300s = 1500s).
    at.state.open_position["opened_at_ts"] = time.time() - 1500
    at.config.invalidation_enabled = True
    at.config.invalidation_time_enabled = True
    at.config.invalidation_time_bars = 4
    at.config.invalidation_time_progress_atr = 0.5

    # Price barely moved — current=7195.5 vs entry=7195, ATR=2 → progress=0.25 ATR.
    result = at.evaluate_invalidation({"current_price": 7195.5, "atr": 2.0})
    assert result is not None
    assert result["triggered"]
    assert result["reason"] == "invalidation_time"
    assert result["details"]["bars_open"] >= 4


def test_time_based_silent_with_progress(at):
    """Same time elapsed but progress is good → no trigger."""
    _open_long(at, entry=7195.0)
    at.state.open_position["opened_at_ts"] = time.time() - 1500
    at.config.invalidation_enabled = True
    at.config.invalidation_time_enabled = True
    at.config.invalidation_time_bars = 4
    at.config.invalidation_time_progress_atr = 0.5

    # Price moved 2 ATR favorable — well past threshold.
    result = at.evaluate_invalidation({"current_price": 7199.0, "atr": 2.0})
    assert result is None


def test_time_based_silent_before_bar_threshold(at):
    """Not enough bars elapsed → no trigger even with no progress."""
    _open_long(at, entry=7195.0)
    at.state.open_position["opened_at_ts"] = time.time() - 60   # 1 min ago
    at.config.invalidation_enabled = True
    at.config.invalidation_time_enabled = True
    at.config.invalidation_time_bars = 4

    result = at.evaluate_invalidation({"current_price": 7195.0, "atr": 2.0})
    assert result is None


# ── Structural reversal ──

def test_structural_fires_when_long_breaks_down(at):
    _open_long(at, entry=7195.0)
    at.config.invalidation_enabled = True
    at.config.invalidation_structural_enabled = True
    at.config.invalidation_structural_atr_threshold = 0.6

    # Adverse move: 7193 vs 7195 = -2pts, ATR=2 → 1.0 ATR adverse → triggers.
    result = at.evaluate_invalidation({"current_price": 7193.0, "atr": 2.0})
    assert result is not None
    assert result["reason"] == "invalidation_structural"


def test_structural_silent_within_threshold(at):
    _open_long(at, entry=7195.0)
    at.config.invalidation_enabled = True
    at.config.invalidation_structural_enabled = True
    at.config.invalidation_structural_atr_threshold = 0.6

    # 0.5 ATR adverse = below 0.6 threshold.
    result = at.evaluate_invalidation({"current_price": 7194.0, "atr": 2.0})
    assert result is None


def test_structural_handles_short(at):
    _open_short(at, entry=7195.0)
    at.config.invalidation_enabled = True
    at.config.invalidation_structural_enabled = True
    at.config.invalidation_structural_atr_threshold = 0.6
    # Short: adverse = price up. 7197 - 7195 = 2pts / ATR 2 = 1 ATR.
    result = at.evaluate_invalidation({"current_price": 7197.0, "atr": 2.0})
    assert result is not None
    assert result["reason"] == "invalidation_structural"


# ── CVD divergence ──

def test_cvd_divergence_long_after_n_ticks(at):
    _open_long(at)
    at.config.invalidation_enabled = True
    at.config.invalidation_cvd_enabled = True
    at.config.invalidation_cvd_min_ticks = 3

    # First call: snapshot CVD at entry.
    at.evaluate_invalidation({"current_price": 7195.0, "atr": 2.0, "cum_delta": 100})
    # Now CVD drops on 3 consecutive evaluations — should fire.
    at.evaluate_invalidation({"current_price": 7195.0, "atr": 2.0, "cum_delta": 80})
    at.evaluate_invalidation({"current_price": 7195.0, "atr": 2.0, "cum_delta": 70})
    result = at.evaluate_invalidation({"current_price": 7195.0, "atr": 2.0, "cum_delta": 60})
    assert result is not None
    assert result["reason"] == "invalidation_cvd_divergence"


def test_cvd_divergence_resets_when_recovers_above_entry(at):
    """If CVD ever climbs back above the entry snapshot, the against-counter
    resets. The trigger only fires after `min_ticks` consecutive ticks
    *below* the entry CVD value."""
    _open_long(at)
    at.config.invalidation_enabled = True
    at.config.invalidation_cvd_enabled = True
    at.config.invalidation_cvd_min_ticks = 3

    # Entry snapshot at CVD=100.
    at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 100})
    # Two against ticks (below 100).
    at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 80})
    at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 70})
    # Recovery — CVD pops back above entry → counter resets.
    at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 110})
    # Two new against ticks — should NOT fire (need 3 in a row).
    at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 90})
    result = at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 80})
    assert result is None


def test_cvd_divergence_short_inverts_correctly(at):
    """Short positions expect CVD to FALL; rising CVD = against."""
    _open_short(at)
    at.config.invalidation_enabled = True
    at.config.invalidation_cvd_enabled = True
    at.config.invalidation_cvd_min_ticks = 2

    at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 100})
    at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 110})
    result = at.evaluate_invalidation({"current_price": 7195, "atr": 2, "cum_delta": 120})
    assert result is not None
    assert result["reason"] == "invalidation_cvd_divergence"


# ── Defaults are off ──

def test_default_config_has_invalidation_off():
    cfg = AutotraderConfig()
    assert not cfg.invalidation_enabled
    assert not cfg.invalidation_time_enabled
    assert not cfg.invalidation_structural_enabled
    assert not cfg.invalidation_cvd_enabled


# ── Best-favorable tracking ──

def test_best_favorable_excursion_persists(at):
    """Even if price retraces, the best-ever favorable move is remembered."""
    _open_long(at, entry=7195.0)
    at.config.invalidation_enabled = True
    # Price spikes up 3 pts.
    at.evaluate_invalidation({"current_price": 7198.0, "atr": 2.0})
    assert at.state.open_position["best_favorable_pts"] == 3.0
    # Price retraces, but best is still tracked.
    at.evaluate_invalidation({"current_price": 7196.0, "atr": 2.0})
    assert at.state.open_position["best_favorable_pts"] == 3.0
