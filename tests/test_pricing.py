"""Tests for DI curve pricing formulas and the Linha (cupom) tracker.

Covers:
- DICurve.interpolate (BNDF5 exponential)
- DICurve.theoretical_dollar_future (BNDF3)
- DICurve.cupom_cambial (BNDF1) round-trip
- LinhaTracker.signal: linha_spike / linha_compression firing logic

No live data; all unit tests with synthetic curves.
"""

from __future__ import annotations

from datetime import date

import pytest

from takata_core.pricing.di_curve import DICurve, DIPoint
from takata_core.pricing.linha_tracker import LinhaTracker
import takata_core.pricing.linha_tracker as lt_mod


@pytest.fixture(autouse=True)
def _reset_linha_module_state():
    """Ensure the module-level deque is clean for every test."""
    lt_mod._linha_history.clear()
    lt_mod._linha_daily_reset = ""
    yield
    lt_mod._linha_history.clear()
    lt_mod._linha_daily_reset = ""


def _sample_curve() -> DICurve:
    today = date.today()
    pts = [
        DIPoint(ticker="DI1F26", maturity=today, business_days=21,  calendar_days=30,  rate=10.50),
        DIPoint(ticker="DI1J26", maturity=today, business_days=63,  calendar_days=90,  rate=11.00),
        DIPoint(ticker="DI1N26", maturity=today, business_days=126, calendar_days=180, rate=11.50),
        DIPoint(ticker="DI1F27", maturity=today, business_days=252, calendar_days=365, rate=12.00),
    ]
    return DICurve(valuation_date=today, points=pts)


def test_interpolate_within_bracket():
    c = _sample_curve()
    r = c.interpolate(42)  # mid-way between 21 (10.50) and 63 (11.00)
    assert 10.50 <= r <= 11.00, f"Interpolated rate {r} out of expected bracket"


def test_interpolate_flat_extrapolation():
    c = _sample_curve()
    assert c.interpolate(1) == pytest.approx(10.50)
    assert c.interpolate(500) == pytest.approx(12.00)


def test_theoretical_dollar_future_sanity():
    """Theoretical WDO from a reasonable spot/curve must be in a plausible band."""
    c = _sample_curve()
    theo = c.theoretical_dollar_future(du_short=1, du_long=21, spot_usd=5.20)
    # Spot×1000 = 5200. With DI short-end ~10.5% and curve normal, theo should
    # be above spot×1000 but within a few percent.
    assert 5100 < theo < 5400, f"Theoretical WDO {theo} outside plausible band"


def test_cupom_roundtrip():
    """Feed a known cupom → compute implied DolFut → recover cupom via BNDF1.

    This verifies our port of the institutional VBA cupom_cambial formula is
    self-consistent with theoretical_dollar_future.
    """
    c = _sample_curve()
    spot, du = 5.20, 21
    cupom_target_pct = 8.50  # percentage

    onshore = (1 + c.interpolate(du) / 100) ** (du / 252)
    dc = du * 365 / 252
    dol_fut = spot * 1000 * onshore / (1 + (cupom_target_pct / 100) * dc / 360)

    recovered = c.cupom_cambial(du, spot, dol_fut)  # already returns %
    assert abs(recovered - cupom_target_pct) < 0.10, (
        f"Cupom round-trip failed: input={cupom_target_pct}, recovered={recovered}"
    )


def test_linha_tracker_spike(monkeypatch):
    """A +50 bps move within the 5-minute window fires `linha_spike`."""
    tracker = LinhaTracker()
    base = 1_700_000_000.0

    # Seed reading 4.5 min ago (inside the 5-min window)
    monkeypatch.setattr(lt_mod.time, "time", lambda: base - 270)
    tracker.update(8.00)

    # Current reading: +0.50% later = +50 bps
    monkeypatch.setattr(lt_mod.time, "time", lambda: base)
    out = tracker.signal(8.50)

    assert out["roc_5m"] is not None, "roc_5m should find the 4.5-min-old seed"
    assert out["roc_5m"] > 30, f"roc_5m={out['roc_5m']} should exceed 30 bps"
    assert "linha_spike" in out["reasons"]
    assert out["score_adjustment"] > 0


def test_linha_tracker_compression(monkeypatch):
    """A -50 bps move within the 5-minute window fires `linha_compression`."""
    tracker = LinhaTracker()
    base = 1_700_000_000.0

    monkeypatch.setattr(lt_mod.time, "time", lambda: base - 270)
    tracker.update(8.50)

    monkeypatch.setattr(lt_mod.time, "time", lambda: base)
    out = tracker.signal(8.00)

    assert out["roc_5m"] is not None
    assert out["roc_5m"] < -30
    assert "linha_compression" in out["reasons"]


def test_linha_tracker_on_off_widening(monkeypatch):
    """A +60 bps move within the 30-minute window fires `on_off_widening`."""
    tracker = LinhaTracker()
    base = 1_700_000_000.0

    # Seed 25 min ago (inside the 30-min window); roc_5m window won't see it
    monkeypatch.setattr(lt_mod.time, "time", lambda: base - 1500)
    tracker.update(8.00)

    monkeypatch.setattr(lt_mod.time, "time", lambda: base)
    out = tracker.signal(8.60)

    # roc_5m should be None (no reading in 5-min window) or not spike
    assert out["roc_30m"] is not None
    assert out["roc_30m"] > 50
    assert "on_off_widening" in out["reasons"]
    assert "linha_spike" not in out["reasons"]


def test_linha_tracker_no_history_returns_nones():
    """First call with no prior history returns None ROCs and no reasons."""
    tracker = LinhaTracker()
    out = tracker.signal(8.00)
    assert out["roc_5m"] is None
    assert out["roc_30m"] is None
    assert out["reasons"] == []
    assert out["score_adjustment"] == 0.0
