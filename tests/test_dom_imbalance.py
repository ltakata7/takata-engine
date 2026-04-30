"""Tests for the Phase 8 DOM imbalance tracker."""

from __future__ import annotations

import pytest

from takata_engine.pricing.dom_imbalance import (
    DOMImbalanceTracker,
    PERSISTENCE_TICKS,
    THIN_BOOK_TOTAL,
    get_dom_tracker,
    reset_all_dom_trackers,
)


@pytest.fixture(autouse=True)
def _isolate():
    reset_all_dom_trackers()
    yield
    reset_all_dom_trackers()


# ── Imbalance detection ──

def test_balanced_book_no_signal():
    t = DOMImbalanceTracker()
    bids = [(7195.0, 50), (7194.75, 40), (7194.5, 35)]
    asks = [(7195.25, 50), (7195.5, 40), (7195.75, 35)]
    t.replace_book(bids, asks)
    sig = t.signal()
    assert not sig["active"]
    assert sig["direction"] == "none"


def test_bull_imbalance_after_persistence():
    """Heavy bid stack persisting for ≥3 ticks → bull signal."""
    t = DOMImbalanceTracker()
    bids = [(7195.0, 200), (7194.75, 150), (7194.5, 100)]   # 450 total
    asks = [(7195.25, 50), (7195.5, 40), (7195.75, 30)]      # 120 total
    # Same book pushed multiple times — simulates persistent state.
    for _ in range(PERSISTENCE_TICKS):
        t.replace_book(bids, asks)
    sig = t.signal()
    assert sig["active"]
    assert sig["direction"] == "bull"
    assert sig["persistence"] >= PERSISTENCE_TICKS
    assert any("dom_imbalance_bull" in r for r in sig["reasons"])


def test_bear_imbalance_after_persistence():
    t = DOMImbalanceTracker()
    bids = [(7195.0, 50), (7194.75, 40), (7194.5, 30)]       # 120
    asks = [(7195.25, 200), (7195.5, 150), (7195.75, 100)]   # 450
    for _ in range(PERSISTENCE_TICKS):
        t.replace_book(bids, asks)
    sig = t.signal()
    assert sig["active"]
    assert sig["direction"] == "bear"


def test_imbalance_under_threshold_no_signal():
    """Bid stack only 1.3× the ask stack → below threshold (default 1.8×) → no signal."""
    t = DOMImbalanceTracker()
    bids = [(7195.0, 130), (7194.75, 0), (7194.5, 0)]
    asks = [(7195.25, 100), (7195.5, 0), (7195.75, 0)]
    for _ in range(PERSISTENCE_TICKS):
        t.replace_book(bids, asks)
    sig = t.signal()
    assert not sig["active"]


def test_imbalance_resets_on_direction_flip():
    """Persistence counter resets when direction flips."""
    t = DOMImbalanceTracker()
    bull_book = ([(7195.0, 200)], [(7195.25, 50)])
    bear_book = ([(7195.0, 50)], [(7195.25, 200)])
    # Build up bull persistence.
    for _ in range(2):
        t.replace_book(*bull_book)
    # Single bear flip — should reset, not yet reach persistence threshold.
    t.replace_book(*bear_book)
    sig = t.signal()
    assert not sig["active"]
    assert sig["persistence"] == 1


# ── Thin book ──

def test_thin_book_flag_overrides_active():
    """Even a heavy ratio doesn't fire if total resting size is too low."""
    t = DOMImbalanceTracker()
    # Total size 15 < THIN_BOOK_TOTAL (30); ratio is 4× but book is empty.
    bids = [(7195.0, 12)]
    asks = [(7195.25, 3)]
    for _ in range(PERSISTENCE_TICKS):
        t.replace_book(bids, asks)
    sig = t.signal()
    assert sig["thin"]
    assert not sig["active"]
    assert "dom_thin_book" in sig["reasons"]


# ── Score adjustment ──

def test_score_caps_at_0_12():
    t = DOMImbalanceTracker()
    bids = [(7195.0, 500), (7194.75, 400)]
    asks = [(7195.25, 50), (7195.5, 40)]
    # Build up a long streak.
    for _ in range(20):
        t.replace_book(bids, asks)
    sig = t.signal()
    assert sig["score_adjustment"] <= 0.12


# ── update_level vs replace_book parity ──

def test_update_level_matches_replace_book():
    """Setting levels one at a time should produce the same state as replace."""
    a = DOMImbalanceTracker()
    a.replace_book(
        [(7195.0, 200), (7194.75, 150)],
        [(7195.25, 50), (7195.5, 40)],
    )
    sig_a = a.signal()

    b = DOMImbalanceTracker()
    b.update_level("bid", 0, 7195.0, 200)
    b.update_level("bid", 1, 7194.75, 150)
    b.update_level("ask", 0, 7195.25, 50)
    b.update_level("ask", 1, 7195.5, 40)
    sig_b = b.signal()

    assert sig_a["bid_top_n_size"] == sig_b["bid_top_n_size"]
    assert sig_a["ask_top_n_size"] == sig_b["ask_top_n_size"]


# ── Singleton ──

def test_per_instrument_singletons():
    mes = get_dom_tracker("MES")
    wdo = get_dom_tracker("WDO")
    assert mes is not wdo

    mes.replace_book([(7195.0, 200)], [(7195.25, 50)])
    assert wdo.signal()["bid_top_n_size"] == 0


def test_get_dom_tracker_returns_same_instance():
    a = get_dom_tracker("MES")
    b = get_dom_tracker("MES")
    assert a is b


# ── Reset ──

def test_reset_clears_state():
    t = DOMImbalanceTracker()
    t.replace_book([(7195.0, 200)], [(7195.25, 50)])
    t.reset()
    sig = t.signal()
    assert sig["bid_top_n_size"] == 0
    assert sig["ask_top_n_size"] == 0
    assert sig["persistence"] == 0


# ── Schema ──

def test_signal_has_required_fields():
    t = DOMImbalanceTracker()
    sig = t.signal()
    for k in ("active", "direction", "persistence",
              "bid_top_n_size", "ask_top_n_size",
              "ratio_bid_over_ask", "thin", "reasons", "score_adjustment"):
        assert k in sig
