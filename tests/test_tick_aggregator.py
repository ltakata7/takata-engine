"""Tests for the IBKRTickAggregator (Phase 6.5 — MES tick subscription).

All tests use the mock-tick interface (`feed_last`, `feed_bidask`) — no IBKR
Gateway needed. Verifies aggressor inference, bar binning, delta polling,
and singleton-per-instrument isolation.
"""

from __future__ import annotations

import time

import pytest

from takata_engine.data.tick_aggregator import (
    IBKRTickAggregator,
    TickBar,
    get_tick_aggregator,
    reset_all_aggregators,
)


@pytest.fixture(autouse=True)
def _isolate_aggregators():
    reset_all_aggregators()
    yield
    reset_all_aggregators()


# ── Aggressor inference ──

def test_buyer_aggression_when_print_at_or_above_ask():
    """Trade ≥ ask price → buyer-aggressed (positive agg_balance)."""
    a = IBKRTickAggregator("MES")
    a.feed_bidask(bid=5900.0, ask=5900.25, ts=100)
    a.feed_last(price=5900.25, size=2, ts=101)        # AT the ask
    a.feed_last(price=5900.50, size=1, ts=102)        # ABOVE the ask (rare but possible)
    snap = a.poll()
    assert snap["agg_balance_delta"] == 3   # +2 + +1
    assert snap["volume_delta"] == 3


def test_seller_aggression_when_print_at_or_below_bid():
    a = IBKRTickAggregator("MES")
    a.feed_bidask(bid=5900.00, ask=5900.25, ts=100)
    a.feed_last(price=5900.00, size=4, ts=101)        # AT the bid
    a.feed_last(price=5899.75, size=2, ts=102)        # BELOW the bid
    snap = a.poll()
    assert snap["agg_balance_delta"] == -6


def test_midpoint_trade_does_not_count_toward_agg_balance():
    """Print between bid and ask is ambiguous — counted as volume but not agg."""
    a = IBKRTickAggregator("MES")
    a.feed_bidask(bid=5900.00, ask=5900.50, ts=100)
    a.feed_last(price=5900.25, size=5, ts=101)        # midpoint
    snap = a.poll()
    assert snap["agg_balance_delta"] == 0
    assert snap["volume_delta"] == 5


def test_first_tick_with_no_bidask_yet_no_aggressor():
    """If a Last tick arrives before any BidAsk, aggressor can't be inferred —
    print should still count as volume, but agg_balance stays 0."""
    a = IBKRTickAggregator("MES")
    a.feed_last(price=5900.00, size=3, ts=100)
    snap = a.poll()
    assert snap["agg_balance_delta"] == 0
    assert snap["volume_delta"] == 3


def test_invalid_bidask_ignored():
    """Garbage bid/ask values should be dropped silently."""
    a = IBKRTickAggregator("MES")
    a.feed_bidask(bid=0, ask=5900.25, ts=100)         # 0 bid
    a.feed_bidask(bid=5901, ask=5900, ts=101)         # crossed
    a.feed_bidask(bid=-5, ask=-3, ts=102)             # negative
    a.feed_last(price=5900, size=1, ts=103)
    snap = a.poll()
    # No bid/ask was ever set → agg_balance stays 0.
    assert snap["agg_balance_delta"] == 0


# ── Bar binning ──

def test_ticks_within_a_bar_accumulate_into_one_bar():
    a = IBKRTickAggregator("MES", bar_seconds=300)
    a.feed_bidask(bid=5900, ask=5900.25, ts=950)
    # All within the same 5-min bin [900, 1200).
    for t in (950, 1050, 1150):
        a.feed_last(price=5900.25, size=1, ts=t)
    snap = a.poll()
    cur = snap["current_bar"]
    assert cur is not None
    assert cur["volume"] == 3
    assert cur["tick_count"] == 3
    assert cur["bin_start_ts"] == 900
    assert snap["bars_in_history"] == 0


def test_bar_advances_at_5min_boundary():
    a = IBKRTickAggregator("MES", bar_seconds=300)
    a.feed_bidask(bid=5900, ask=5900.25, ts=100)
    # Bar 1: ts in [0, 300)
    a.feed_last(price=5900.25, size=2, ts=200)
    # Bar 2: ts in [300, 600)
    a.feed_last(price=5901.00, size=3, ts=350)
    snap = a.poll()
    assert snap["bars_in_history"] == 1
    history = a.get_recent_bars()
    assert history[0]["volume"] == 2     # finalized first bar


def test_ohlc_built_correctly_within_bar():
    a = IBKRTickAggregator("MES", bar_seconds=300)
    a.feed_bidask(bid=5900, ask=5900.25, ts=100)
    a.feed_last(price=5900.25, size=1, ts=101)   # open=high=low=close
    a.feed_last(price=5905.00, size=1, ts=102)   # new high
    a.feed_last(price=5895.00, size=1, ts=103)   # new low
    a.feed_last(price=5901.00, size=1, ts=104)   # close
    snap = a.poll()
    cur = snap["current_bar"]
    assert cur["open"] == 5900.25
    assert cur["high"] == 5905.00
    assert cur["low"] == 5895.00
    assert cur["close"] == 5901.00


# ── Polling deltas ──

def test_poll_returns_delta_since_last_call():
    a = IBKRTickAggregator("MES")
    a.feed_bidask(bid=5900, ask=5900.25, ts=100)

    a.feed_last(price=5900.25, size=2, ts=101)
    snap1 = a.poll()
    assert snap1["agg_balance_delta"] == 2

    # Subsequent ticks; next poll() should return only delta from snap1.
    a.feed_last(price=5900.25, size=3, ts=102)
    a.feed_last(price=5900.00, size=1, ts=103)   # seller-aggressed
    snap2 = a.poll()
    assert snap2["agg_balance_delta"] == 2     # +3 - 1 = +2
    assert snap2["volume_delta"] == 4


def test_poll_with_no_new_ticks_returns_zero_delta():
    a = IBKRTickAggregator("MES")
    a.feed_bidask(bid=5900, ask=5900.25, ts=100)
    a.feed_last(price=5900.25, size=5, ts=101)
    a.poll()
    snap = a.poll()
    assert snap["agg_balance_delta"] == 0
    assert snap["volume_delta"] == 0


def test_cumulative_aggregates_persist_across_polls():
    a = IBKRTickAggregator("MES")
    a.feed_bidask(bid=5900, ask=5900.25, ts=100)
    a.feed_last(price=5900.25, size=10, ts=101)
    a.poll()
    a.feed_last(price=5900.25, size=15, ts=102)
    snap = a.poll()
    assert snap["cum_agg_balance"] == 25
    assert snap["cum_volume"] == 25


# ── Singleton registry ──

def test_per_instrument_singletons_isolated():
    """MES + WDO aggregators share no state."""
    mes = get_tick_aggregator("MES")
    wdo = get_tick_aggregator("WDO")
    assert mes is not wdo

    mes.feed_bidask(bid=5900, ask=5900.25)
    mes.feed_last(price=5900.25, size=10)

    wdo_snap = wdo.poll()
    assert wdo_snap["volume_delta"] == 0
    assert wdo_snap["agg_balance_delta"] == 0


def test_get_tick_aggregator_returns_same_instance():
    a = get_tick_aggregator("MES")
    b = get_tick_aggregator("MES")
    assert a is b


# ── Reset ──

def test_reset_drops_all_state():
    a = IBKRTickAggregator("MES")
    a.feed_bidask(bid=5900, ask=5900.25, ts=100)
    a.feed_last(price=5900.25, size=10, ts=101)
    a.reset()
    snap = a.poll()
    assert snap["agg_balance_delta"] == 0
    assert snap["volume_delta"] == 0
    assert snap["cum_volume"] == 0
    assert snap["current_bar"] is None
    assert snap["bars_in_history"] == 0


# ── Snapshot shape ──

def test_snapshot_has_all_required_fields():
    """Scanner consumes specific keys — make sure they're all present."""
    a = IBKRTickAggregator("MES")
    snap = a.poll()
    for k in ("instrument", "source_mode",
              "agg_balance_delta", "volume_delta",
              "cum_agg_balance", "cum_volume",
              "current_bar", "last_finalized_bar",
              "current_bid", "current_ask", "ticker_age_s",
              "bars_in_history"):
        assert k in snap, f"snapshot missing {k}"


def test_source_mode_is_none_before_subscription():
    a = IBKRTickAggregator("MES")
    assert a.poll()["source_mode"] is None


def test_fallback_handler_recognizes_permission_errors():
    """The error handler must recognize 10189 / 10168 / 10089 as 'no perms'."""
    a = IBKRTickAggregator("MES")
    a._fallback_armed = True   # would-be set by subscribe()
    # We can't actually call _on_ib_error without an IB instance, but we can
    # verify the codes it watches for by inspecting the implementation.
    import inspect
    src = inspect.getsource(a._on_ib_error)
    for code in ("10189", "10168", "10089"):
        assert code in src, f"fallback handler missing error code {code}"
