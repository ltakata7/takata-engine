"""Tests for cross_asset confluence scoring."""

from __future__ import annotations

import pytest

from takata_engine.signals import cross_asset_confluence, ConfluenceResult


def test_wdo_long_strong_agreement_scores_high():
    """WDO long expected when VIX↑, ES↓, IND↓, NQ↓ — full risk-off picture."""
    r = cross_asset_confluence(
        "WDO", "long",
        {"VIX": +2.5, "ESFUT": -0.8, "INDFUT": -0.6, "NQFUT": -0.5, "VXBR": +1.5,
         "WINFUT": -0.7, "IBOV": -0.6},
    )
    assert r.score >= 80, f"expected strong (>=80), got {r.score}"
    assert r.label == "strong"
    assert any(e["peer"] == "VIX" for e in r.agreed), "VIX must be in agreed list"
    assert not r.disagreed


def test_wdo_long_strong_disagreement_scores_low():
    """WDO long but VIX collapsing, ES rallying, IND green — disagrees."""
    r = cross_asset_confluence(
        "WDO", "long",
        {"VIX": -2.0, "ESFUT": +1.0, "INDFUT": +0.8, "NQFUT": +1.2, "VXBR": -1.5,
         "WINFUT": +0.7, "IBOV": +0.7},
    )
    assert r.score <= 25, f"expected disagree (<=25), got {r.score}"
    assert r.label == "disagree"
    assert any(e["peer"] == "VIX" for e in r.disagreed), "VIX must be in disagreed list"
    assert not r.agreed


def test_vix_is_primary_lever_for_mes():
    """VIX should weight ±2 in MES correlation table — equal to NQ/ES."""
    from takata_engine.signals.cross_asset import _CORRELATIONS
    assert _CORRELATIONS[("MES", "long")]["VIX"] == -2
    assert _CORRELATIONS[("MES", "short")]["VIX"] == +2
    # VXBR demoted to weak secondary — magnitude must be smaller.
    assert abs(_CORRELATIONS[("MES", "long")]["VXBR"]) < abs(_CORRELATIONS[("MES", "long")]["VIX"])
    assert abs(_CORRELATIONS[("WDO", "long")]["VXBR"]) < abs(_CORRELATIONS[("WDO", "long")]["VIX"])


def test_vix_alone_can_flip_mes_score():
    """A clean VIX signal with no other peers should already push score off neutral.

    VIX alone (weight ±2 out of total |weight| = 10) yields score = 60 for long
    when VIX falls, 40 for short. That's modest but directionally correct —
    the point is VIX *moves the needle* without any other peer.
    """
    long_score = cross_asset_confluence("MES", "long", {"VIX": -2.0}).score
    short_score = cross_asset_confluence("MES", "short", {"VIX": -2.0}).score
    assert long_score >= 60   # VIX falling → MES long agrees
    assert short_score <= 40  # VIX falling → MES short disagrees
    # And the two should be mirror images.
    assert long_score + short_score == 100


def test_wdo_short_mirrors_long():
    """Same data, opposite direction — score should flip."""
    deltas = {"ESFUT": +1.0, "INDFUT": +0.8, "NQFUT": +1.2, "VXBR": -2.0}
    long_score = cross_asset_confluence("WDO", "long", deltas).score
    short_score = cross_asset_confluence("WDO", "short", deltas).score
    # Long with this data is bad, short is good. Sum should be ~100 (mirror).
    assert abs((long_score + short_score) - 100) <= 5


def test_mes_long_with_nq_leadership():
    """MES long + VIX↓, NQ↑, ES↑ → strong confluence."""
    r = cross_asset_confluence(
        "MES", "long",
        {"VIX": -1.5, "NQFUT": +1.0, "ESFUT": +0.7, "VXBR": -1.0, "INDFUT": +0.4},
    )
    assert r.score >= 70


def test_mes_short_with_tech_breakdown():
    r = cross_asset_confluence(
        "MES", "short",
        {"VIX": +2.0, "NQFUT": -1.5, "ESFUT": -0.9, "VXBR": +0.8, "INDFUT": -0.5},
    )
    assert r.score >= 70


def test_flat_peers_dont_count():
    """Tiny moves below threshold don't push score either way."""
    r = cross_asset_confluence(
        "WDO", "long",
        {"ESFUT": +0.001, "INDFUT": -0.001, "VXBR": +0.001},   # all sub-threshold
        threshold=0.05,
    )
    # No peers contributed → score is 50 (neutral).
    assert r.score == 50
    assert not r.agreed
    assert not r.disagreed


def test_missing_peers_listed():
    """Peers in the correlation table but absent from `deltas_pct` are reported as missing."""
    r = cross_asset_confluence("WDO", "long", {"ESFUT": -0.5})
    # Most expected peers (NQ, IND, VXBR, etc.) are missing.
    assert "VXBR" in r.missing
    assert "INDFUT" in r.missing
    assert "NQFUT" in r.missing


def test_unknown_instrument_returns_neutral():
    r = cross_asset_confluence("UNKNOWN", "long", {"ESFUT": +1.0})
    assert r.score == 50
    assert r.label == "moderate"


def test_to_dict_shape():
    r = cross_asset_confluence("WDO", "long", {"ESFUT": -0.5, "VXBR": +1.0})
    d = r.to_dict()
    assert {"score", "label", "agreed", "disagreed", "missing", "raw_score", "max_possible"} <= d.keys()
    assert isinstance(d["score"], int)


def test_label_thresholds_are_correct():
    """Construct synthetic results to verify label boundaries (70/50/30)."""
    # All peers agreeing maximally → score = 100
    r_strong = cross_asset_confluence(
        "WDO", "long",
        {"VIX": +1, "ESFUT": -1, "NQFUT": -1, "INDFUT": -1, "WINFUT": -1, "IBOV": -1,
         "VXBR": +1, "DI1FUT": +1, "BITFUT": -1},
    )
    assert r_strong.label == "strong"
    # All peers disagreeing maximally → score = 0
    r_disagree = cross_asset_confluence(
        "WDO", "long",
        {"VIX": -1, "ESFUT": +1, "NQFUT": +1, "INDFUT": +1, "WINFUT": +1, "IBOV": +1,
         "VXBR": -1, "DI1FUT": -1, "BITFUT": +1},
    )
    assert r_disagree.label == "disagree"
