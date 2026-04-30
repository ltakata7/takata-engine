"""IBKR tick-by-tick aggregator — gives MES the per-print aggressor data
that WDO already gets from Profit Ultra's RTD.

Subscribes two tick streams via ib_insync:
- ``Last`` (or ``AllLast``)  — every executed trade with price + size + time
- ``BidAsk``                 — best bid/ask updates

For each Last tick, infers the aggressor by comparing trade price to the
most recent BidAsk:
- price ≥ ask  →  buyer-aggressed (+size)
- price ≤ bid  →  seller-aggressed (-size)
- between     →  midpoint trade (0)

Maintains per-(5-min)-bin running aggregates and exposes:
- ``agg_balance_delta`` — net buyer-vs-seller volume since last poll
- ``volume_delta``      — total volume since last poll
- ``current_bar``       — running OHLC + agg + volume of the current 5-min bar
- ``last_finalized_bar``— most recently completed 5-min bar

This module has NO dependency on ib_insync at import time — the tick
subscription is set up via the optional ``subscribe(ib, contract)`` method
which lazily imports ``ib_insync``. That keeps the aggregator unit-testable
with mock ticks (`feed_test_tick`) on Mac, without needing IBKR Gateway.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bar size in seconds — matches scanner's 5-min historical bars.
DEFAULT_BAR_SECONDS = 300

# Minimum tick price/size sanity bounds.
_MIN_PRICE = 0.01
_MIN_SIZE = 0.0


@dataclass
class TickBar:
    """One bin's accumulated tick state (running until finalized)."""
    bin_start_ts: float
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0
    agg_balance: float = 0.0   # buyer-aggressed minus seller-aggressed volume
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    tick_count: int = 0
    started: bool = False

    def update(self, price: float, size: float, agg_signed: float) -> None:
        if not self.started:
            self.open = price
            self.high = price
            self.low = price
            self.started = True
        else:
            if price > self.high:
                self.high = price
            if price < self.low:
                self.low = price
        self.close = price
        self.volume += size
        self.agg_balance += agg_signed
        if agg_signed > 0:
            self.buy_volume += size
        elif agg_signed < 0:
            self.sell_volume += size
        self.tick_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_start_ts": self.bin_start_ts,
            "bin_start_iso": datetime.fromtimestamp(self.bin_start_ts).isoformat() if self.bin_start_ts else None,
            "open": round(self.open, 4),
            "high": round(self.high, 4) if self.high != float("inf") else 0.0,
            "low": round(self.low, 4) if self.low != float("inf") else 0.0,
            "close": round(self.close, 4),
            "volume": round(self.volume, 0),
            "agg_balance": round(self.agg_balance, 0),
            "buy_volume": round(self.buy_volume, 0),
            "sell_volume": round(self.sell_volume, 0),
            "tick_count": self.tick_count,
        }


class IBKRTickAggregator:
    """Receives raw IBKR tick streams and emits bar-level aggregates."""

    def __init__(self, instrument: str = "MES",
                 bar_seconds: int = DEFAULT_BAR_SECONDS) -> None:
        self.instrument = instrument
        self.bar_seconds = bar_seconds

        # Inside-market tracking from BidAsk ticks.
        self._current_bid: float = 0.0
        self._current_ask: float = 0.0
        self._bidask_ts: float = 0.0

        # Bar accumulator — the *current* bin always exists; finalized bins
        # roll into history (max 200 = ~16h on 5min bars).
        self._current: Optional[TickBar] = None
        self._history: Deque[TickBar] = deque(maxlen=200)

        # Cumulative agg + volume from session start (used for delta calc).
        self._cum_agg_balance: float = 0.0
        self._cum_volume: float = 0.0
        # The values at the last `poll()` so the scanner gets *delta* numbers.
        self._last_polled_agg: float = 0.0
        self._last_polled_volume: float = 0.0

        # Thread-safe under a single lock — ib_insync events arrive on the
        # event-loop thread; the scanner reads from its own thread.
        self._lock = threading.RLock()

        # Subscription handles, populated by subscribe().
        self._ib = None
        self._last_ticker = None
        self._bidask_ticker = None
        # Mkt-data fallback state — when tick-by-tick is blocked by IBKR
        # entitlements (error 10189), we fall back to reqMktData. This gives
        # us streaming bid/ask/last with size, just lower granularity than
        # true tick-by-tick (multiple prints between updates can be lumped).
        self._mkt_ticker = None
        self._fallback_armed: bool = False
        self._mkt_last_print_key: Any = None  # (ts, price, size) of most recent print
        self._mkt_last_volume: float = 0.0    # for volume-delta math
        # Mode: "tick_by_tick" | "mkt_data" | None
        self._source_mode: Optional[str] = None
        # DOM (Phase 8) — only set when subscribe_dom() called. Holds the
        # depth ticker so we can drain DOM updates in _on_pending_tickers.
        self._dom_ticker = None
        self._dom_tracker = None

    # ── Bin math ──

    def _bin_start(self, ts: float) -> float:
        """Floor ts to the nearest bar boundary (in epoch seconds)."""
        return ts - (ts % self.bar_seconds)

    def _maybe_advance_bin(self, ts: float) -> None:
        """If ts crosses into a new bin, finalize the current bar and open a fresh one."""
        bin_start = self._bin_start(ts)
        if self._current is None:
            self._current = TickBar(bin_start_ts=bin_start)
            return
        if bin_start > self._current.bin_start_ts:
            # Push finalized bar to history, start fresh.
            self._history.append(self._current)
            self._current = TickBar(bin_start_ts=bin_start)

    # ── Tick ingestion ──

    def feed_bidask(self, bid: float, ask: float, ts: Optional[float] = None) -> None:
        """Update inside market from a BidAsk tick."""
        if bid <= _MIN_PRICE or ask <= _MIN_PRICE:
            return
        if ask < bid:
            return  # crossed/garbage tick
        with self._lock:
            self._current_bid = bid
            self._current_ask = ask
            self._bidask_ts = ts if ts is not None else datetime.now().timestamp()

    def feed_last(self, price: float, size: float,
                  ts: Optional[float] = None) -> None:
        """Process a single Last/AllLast trade tick.

        Aggressor inference uses the most recent BidAsk:
          price ≥ ask  →  buyer-aggressed (+size)
          price ≤ bid  →  seller-aggressed (-size)
          else         →  midpoint (0)
        """
        if price <= _MIN_PRICE or size <= _MIN_SIZE:
            return
        ts = ts if ts is not None else datetime.now().timestamp()
        with self._lock:
            self._maybe_advance_bin(ts)

            agg_signed = 0.0
            if self._current_ask > 0 and price >= self._current_ask:
                agg_signed = +size
            elif self._current_bid > 0 and price <= self._current_bid:
                agg_signed = -size
            # else: midpoint trade — counts toward volume but not agg_balance

            self._current.update(price=price, size=size, agg_signed=agg_signed)
            self._cum_agg_balance += agg_signed
            self._cum_volume += size

    # ── Scanner-facing API ──

    def poll(self) -> Dict[str, Any]:
        """Snapshot for the scanner. Returns deltas since the last poll() call,
        plus current-bar OHLC / agg state. Each call advances the polling
        cursor — the scanner uses this delta to feed engine trackers."""
        with self._lock:
            cur_agg = self._cum_agg_balance
            cur_vol = self._cum_volume
            agg_delta = cur_agg - self._last_polled_agg
            vol_delta = cur_vol - self._last_polled_volume
            self._last_polled_agg = cur_agg
            self._last_polled_volume = cur_vol

            current_bar = self._current.to_dict() if self._current and self._current.started else None
            last_finalized = self._history[-1].to_dict() if self._history else None

            return {
                "instrument": self.instrument,
                "source_mode": self._source_mode,    # tick_by_tick | mkt_data | None
                "agg_balance_delta": round(agg_delta, 2),
                "volume_delta": round(vol_delta, 2),
                "cum_agg_balance": round(cur_agg, 2),
                "cum_volume": round(cur_vol, 2),
                "current_bar": current_bar,
                "last_finalized_bar": last_finalized,
                "current_bid": round(self._current_bid, 4),
                "current_ask": round(self._current_ask, 4),
                "ticker_age_s": round(datetime.now().timestamp() - self._bidask_ts, 1)
                                if self._bidask_ts else None,
                "bars_in_history": len(self._history),
            }

    def get_recent_bars(self, n: int = 60) -> List[Dict[str, Any]]:
        """Return the N most recent finalized bars (oldest first), excluding
        the still-running current bar."""
        with self._lock:
            return [b.to_dict() for b in list(self._history)[-n:]]

    def reset(self) -> None:
        """Drop all state — useful for tests + new sessions."""
        with self._lock:
            self._current = None
            self._history.clear()
            self._cum_agg_balance = 0.0
            self._cum_volume = 0.0
            self._last_polled_agg = 0.0
            self._last_polled_volume = 0.0
            self._current_bid = 0.0
            self._current_ask = 0.0
            self._bidask_ts = 0.0

    # ── ib_insync wiring (lazy import) ──

    def subscribe(self, ib, contract, last_mode: str = "Last",
                  fallback_to_mkt_data: bool = True) -> bool:
        """Subscribe to the Last + BidAsk tick streams on the given IBKR
        connection. Idempotent — calling twice on the same instance is safe.

        ``last_mode``: "Last" (regular hours) or "AllLast" (RTH + ETH).
        ``fallback_to_mkt_data``: if True (default), watch for IBKR
        permission errors (10189 / 10168 / 10089) and automatically switch to
        ``reqMktData`` — works with basic streaming L1 even without the
        tick-by-tick entitlement. Lower granularity but same downstream API.
        """
        if self._source_mode is not None:
            return True
        try:
            self._ib = ib
            self._fallback_armed = fallback_to_mkt_data
            self._last_ticker = ib.reqTickByTickData(contract, last_mode, 0, False)
            self._bidask_ticker = ib.reqTickByTickData(contract, "BidAsk", 0, False)
            # Also subscribe reqMktData — some Gateway builds don't deliver
            # tick-by-tick BidAsk events reliably, but reqMktData's streaming
            # bid/ask works on the same L1 entitlement and gives us the
            # inside market that Last-tick aggressor inference needs.
            self._mkt_ticker = ib.reqMktData(contract, "", False, False)
            ib.pendingTickersEvent += self._on_pending_tickers
            ib.errorEvent += self._on_ib_error
            self._source_mode = "tick_by_tick"
            logger.info("Tick subscription active for %s (%s + BidAsk + mktData)",
                        self.instrument, last_mode)
            return True
        except Exception as e:
            logger.warning("Tick subscription failed for %s: %s", self.instrument, e)
            return False

    def subscribe_mkt_data(self, ib, contract) -> bool:
        """Subscribe via ``reqMktData`` (basic streaming L1). Used as a
        fallback when tick-by-tick is unavailable.
        """
        if self._mkt_ticker is not None:
            return True
        try:
            self._ib = ib
            self._mkt_ticker = ib.reqMktData(contract, "", False, False)
            ib.pendingTickersEvent += self._on_pending_tickers
            self._source_mode = "mkt_data"
            logger.info("MktData subscription active for %s (fallback path)", self.instrument)
            return True
        except Exception as e:
            logger.warning("MktData subscription failed for %s: %s", self.instrument, e)
            return False

    def subscribe_dom(self, ib, contract, num_rows: int = 5,
                      smart_depth: bool = False) -> bool:
        """Subscribe to Level 2 / DOM via ``reqMktDepth``. Requires the
        exchange's L2 entitlement (e.g. CME L2 Booktrader). Updates flow
        through ``Ticker.domBids`` / ``Ticker.domAsks`` lists; we forward
        them to the per-instrument :class:`DOMImbalanceTracker`.
        """
        if self._dom_ticker is not None:
            return True
        try:
            from takata_engine.pricing.dom_imbalance import get_dom_tracker
            self._ib = ib
            self._dom_tracker = get_dom_tracker(self.instrument)
            self._dom_ticker = ib.reqMktDepth(contract, numRows=num_rows,
                                              isSmartDepth=smart_depth)
            # _on_pending_tickers handler is already registered if subscribe()
            # was called; if not, register it now.
            try:
                ib.pendingTickersEvent += self._on_pending_tickers
            except Exception:
                pass
            logger.info("DOM subscription active for %s (top-%d)", self.instrument, num_rows)
            return True
        except Exception as e:
            logger.warning("DOM subscription failed for %s: %s", self.instrument, e)
            return False

    def _on_ib_error(self, reqId, errorCode, errorString, contract) -> None:
        """Watch for IBKR permission errors that mean we can't use
        ``reqTickByTickData``. Falls back to ``reqMktData`` if armed."""
        # 10189 = no market data permissions for tick-by-tick
        # 10168 = requested market data is not subscribed
        # 10089 = requested market data requires additional subscription
        if errorCode in (10189, 10168, 10089) and self._fallback_armed:
            self._fallback_armed = False
            logger.warning(
                "Tick-by-tick permission error %d for %s — falling back to reqMktData",
                errorCode, self.instrument,
            )
            try:
                self._cancel_tick_by_tick()
            except Exception:
                pass
            # Resolve the contract: ib_insync error events sometimes don't
            # pass the full Contract back. Use the one from our Ticker.
            target_contract = (
                getattr(self._last_ticker, "contract", None)
                or getattr(self._bidask_ticker, "contract", None)
                or contract
            )
            if target_contract is not None and self._ib is not None:
                self.subscribe_mkt_data(self._ib, target_contract)

    def _cancel_tick_by_tick(self) -> None:
        if self._ib is None:
            return
        try:
            if self._last_ticker is not None:
                self._ib.cancelTickByTickData(self._last_ticker.contract, "Last")
            if self._bidask_ticker is not None:
                self._ib.cancelTickByTickData(self._bidask_ticker.contract, "BidAsk")
        except Exception as e:
            logger.debug("cancelTickByTickData failed: %s", e)
        finally:
            self._last_ticker = None
            self._bidask_ticker = None

    def unsubscribe(self) -> None:
        """Cancel any active subscriptions and detach event handlers."""
        if self._ib is None:
            return
        try:
            self._cancel_tick_by_tick()
            if self._mkt_ticker is not None:
                self._ib.cancelMktData(self._mkt_ticker.contract)
            if self._dom_ticker is not None:
                self._ib.cancelMktDepth(self._dom_ticker.contract)
            self._ib.pendingTickersEvent -= self._on_pending_tickers
            try:
                self._ib.errorEvent -= self._on_ib_error
            except Exception:
                pass
        except Exception as e:
            logger.debug("Tick unsubscribe failed: %s", e)
        finally:
            self._mkt_ticker = None
            self._dom_ticker = None
            self._dom_tracker = None
            self._ib = None
            self._source_mode = None

    def _on_pending_tickers(self, tickers) -> None:
        """ib_insync event handler. ib_insync returns the SAME Ticker object
        for all subscription types on the same contract — so a single ticker
        update can carry tick-by-tick events AND streaming bid/ask AND last
        snapshot. We process all three on every event."""
        for ticker in tickers:
            # Identify whether this ticker is one we subscribed to. Object
            # identity is unreliable (ib_insync may return the same Ticker for
            # different reqs on the same contract), so any of our references
            # matching is enough.
            is_ours = (
                (self._last_ticker is not None and ticker is self._last_ticker)
                or (self._bidask_ticker is not None and ticker is self._bidask_ticker)
                or (self._mkt_ticker is not None and ticker is self._mkt_ticker)
                or (self._dom_ticker is not None and ticker is self._dom_ticker)
            )
            if not is_ours:
                continue

            # 0. DOM ladder updates — replace book snapshot from the Ticker's
            #    domBids / domAsks lists. Each entry is a DOMLevel(price, size).
            if self._dom_tracker is not None:
                try:
                    bids = [
                        (float(lvl.price), float(lvl.size))
                        for lvl in (getattr(ticker, "domBids", []) or [])
                        if lvl.size and lvl.size > 0
                    ]
                    asks = [
                        (float(lvl.price), float(lvl.size))
                        for lvl in (getattr(ticker, "domAsks", []) or [])
                        if lvl.size and lvl.size > 0
                    ]
                    if bids or asks:
                        self._dom_tracker.replace_book(bids, asks)
                except Exception as e:
                    logger.debug("DOM update parse failed: %s", e)

            # 1. Drain tick-by-tick events. They land in tickByTicks regardless
            #    of which req method spawned them. Last and BidAsk events are
            #    distinguished by their attribute set:
            #      - Last:    has .price, .size
            #      - BidAsk:  has .bidPrice, .askPrice, .bidSize, .askSize
            try:
                for t in list(ticker.tickByTicks):
                    try:
                        ts = t.time.timestamp() if hasattr(t.time, "timestamp") else float(t.time)
                        if hasattr(t, "bidPrice") and hasattr(t, "askPrice"):
                            self.feed_bidask(bid=float(t.bidPrice), ask=float(t.askPrice), ts=ts)
                        elif hasattr(t, "price") and hasattr(t, "size"):
                            self.feed_last(price=float(t.price), size=float(t.size), ts=ts)
                    except Exception as e:
                        logger.debug("tickByTick parse failed: %s", e)
                ticker.tickByTicks.clear()
            except Exception as e:
                logger.debug("tickByTicks drain failed: %s", e)

            # 2. Always read the streaming bid/ask snapshot — gives us inside
            #    market even when tick-by-tick BidAsk events aren't flowing.
            try:
                bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
                ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
                if bid > 0 and ask > 0:
                    self.feed_bidask(bid=bid, ask=ask)
            except Exception as e:
                logger.debug("snapshot bid/ask read failed: %s", e)

            # 3. Last-print snapshot — only used in pure mkt_data fallback mode.
            #    When tick-by-tick is providing prints we'd double-count.
            if self._source_mode == "mkt_data":
                try:
                    last_price = float(ticker.last) if ticker.last and ticker.last > 0 else 0.0
                    last_size = float(ticker.lastSize) if ticker.lastSize and ticker.lastSize > 0 else 0.0
                    last_time = ticker.time.timestamp() if hasattr(ticker.time, "timestamp") else None
                    print_key = (last_time, last_price, last_size)
                    if last_price > 0 and last_size > 0 and print_key != self._mkt_last_print_key:
                        self._mkt_last_print_key = print_key
                        vol = float(ticker.volume) if ticker.volume and ticker.volume > 0 else 0.0
                        size = max(last_size, vol - self._mkt_last_volume) if vol > self._mkt_last_volume else last_size
                        self._mkt_last_volume = vol if vol > 0 else self._mkt_last_volume
                        self.feed_last(price=last_price, size=size, ts=last_time)
                except Exception as e:
                    logger.debug("mkt_data last snapshot parse failed: %s", e)


# ── Singleton registry per instrument ──

_aggregators: Dict[str, IBKRTickAggregator] = {}
_aggregators_lock = threading.Lock()


def get_tick_aggregator(instrument: str = "MES",
                        bar_seconds: int = DEFAULT_BAR_SECONDS) -> IBKRTickAggregator:
    """Return the singleton aggregator for an instrument. Construct on first call."""
    with _aggregators_lock:
        agg = _aggregators.get(instrument)
        if agg is None:
            agg = IBKRTickAggregator(instrument=instrument, bar_seconds=bar_seconds)
            _aggregators[instrument] = agg
        return agg


def reset_all_aggregators() -> None:
    """Drop all aggregator state — for tests."""
    with _aggregators_lock:
        for agg in _aggregators.values():
            agg.reset()
        _aggregators.clear()
