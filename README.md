# takata-engine

Shared analytical library for Lauro Takata's trading ecosystem. Consumed by three apps:

- **takata-trading** — WDO/MES day trading (Windows-only, Profit Ultra bridge)
- **takata-wealth** — Takata Holdings personal wealth management (swing trades, research)
- **sympatheia-os** — Sympatheia Advisory client business (covered calls, portfolios, Railway)

## Contents

| Module | Purpose |
|---|---|
| `indicators/` | EMA, MACD, RSI, ATR, ADX, Bollinger, VWAP |
| `pricing/` | DI curve, NDF pricer, implied vol, 9 microstructure trackers (ORB, VWAP bands, cum delta, spread, RV/IV, vol burst, cross-corr, lead-lag, risk regime), PTAX + Linha trackers |
| `regime/` | Wasserstein k-means regime detection (configurable per timeframe) |
| `signals/` | Generic signal generator, filters, risk manager, trade setup, position |
| `ml/` | XGBoost framework: feature store, signal learner, signal logger |
| `backtest/` | Backtester, metrics, walk-forward optimization |
| `macro/` | Cycle classifier, outlook, sizing, data feeds (FRED, BCB, yield curve) |
| `research/` | Fundamentals, equity research |
| `screener/` | Multi-factor scoring |
| `technical/` | Dashboard, levels, patterns, trend classification |
| `sector_rotation/` | Citadel-style ETF rotation model |
| `portfolio/` | Portfolio construction (value/growth/income) |
| `risk_portfolio/` | Portfolio risk (Greeks, concentration, stress, hedging) |
| `data/` | Feeds: CSV, IBKR, yfinance, bar builder, multi-timeframe |
| `execution/` | IBKR executor, order manager, autotrader, live runner |
| `agents/` | Claude LLM agents: premarket briefing, signal explainer, research analyst, trade analyst, risk narrator, session debrief, macro flash |
| `utils/` | Session, calendar (static + investing.com live), logger, config loader |
| `blotter/` | BTG Pactual trade note parser |
| `config/` | Instrument configs: WDO, MES |

## Install

```bash
# From a consumer app:
pip install git+ssh://git@github.com/ltakata7/takata-engine.git@v0.1.0

# For local development (editable):
pip install -e /path/to/takata-engine
```

## Development

```bash
cd takata-engine
pip install -e ".[dev]"
pytest tests/
```

## Release

```bash
git commit -am "Release notes..."
git tag v0.X.0
git push --tags
```

Then in consumer apps, bump the pin in `pyproject.toml`:

```toml
dependencies = [
    "takata-engine @ git+ssh://git@github.com/ltakata7/takata-engine.git@v0.X.0",
]
```
