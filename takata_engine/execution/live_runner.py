"""Live signal runner — connects the signal engine to real-time IBKR execution.

Runs in a loop: fetch latest bars → detect regime → generate signals → execute.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from takata_engine.config import load_config
from takata_engine.data.feed_ibkr import IBKRDataFeed
from takata_engine.execution.executor_ibkr import IBKRExecutor
from takata_engine.execution.order_manager import OrderManager
from takata_engine.ml.signal_learner import SignalLearner
from takata_engine.regime import RegimeDetector
from takata_engine.signals import SignalGenerator
from takata_engine.utils.logger import TradeLogger
from takata_engine.utils.session import get_session

logger = logging.getLogger(__name__)


class LiveRunner:
    """Runs the signal engine live against IBKR.

    Parameters
    ----------
    instrument : str
        ``"MES"`` or other supported instrument.
    host : str
        IBKR Gateway/TWS host.
    port : int
        API port (4002 for paper, 4001 for live).
    paper : bool
        If True, use paper trading.
    poll_interval : int
        Seconds between each signal check.
    use_ml : bool
        If True, filter signals through the ML quality model.
    min_ml_prob : float
        Minimum ML probability to execute (if use_ml=True).
    """

    def __init__(
        self,
        instrument: str = "MES",
        host: str = "127.0.0.1",
        port: int = 4002,
        paper: bool = True,
        poll_interval: int = 60,
        use_ml: bool = False,
        min_ml_prob: float = 0.55,
    ):
        self.instrument = instrument
        self.host = host
        self.port = port
        self.paper = paper
        self.poll_interval = poll_interval
        self.use_ml = use_ml
        self.min_ml_prob = min_ml_prob

        # Components
        self.config = load_config(instrument)
        self.session = get_session(instrument)
        self.executor = IBKRExecutor(host=host, port=port, paper=paper)
        self.order_mgr = OrderManager()
        self.trade_logger = TradeLogger(
            trade_log=f"logs/{instrument.lower()}_live_trades.jsonl",
            signal_log=f"logs/{instrument.lower()}_live_signals.jsonl",
        )
        self.ml_model = SignalLearner() if use_ml else None
        self._running = False

    def run(self) -> None:
        """Main loop: fetch data → analyze → signal → execute."""
        logger.info(
            "Starting LiveRunner for %s on %s:%d (paper=%s)",
            self.instrument, self.host, self.port, self.paper,
        )
        logger.info("Poll interval: %ds | ML filter: %s", self.poll_interval, self.use_ml)

        # Fit regime detector on recent historical data
        feed = IBKRDataFeed(self.instrument, self.host, self.port, duration="5 D", bar_size="5 mins")
        df = feed.load()

        regime_cfg = self.config.get("regime", {})
        detector = RegimeDetector(
            k=regime_cfg.get("k", 3),
            window_size=regime_cfg.get("window_size", 30),
            stride=regime_cfg.get("stride", 5),
            random_state=42,
        )
        detector.fit(df)
        logger.info("Regime detector fitted on %d bars", len(df))

        # Set up signal generator
        generator = SignalGenerator(
            config=self.config,
            regime_detector=detector,
            trade_logger=self.trade_logger,
        )

        self._running = True
        while self._running:
            try:
                self._tick(feed, generator, detector)
            except KeyboardInterrupt:
                logger.info("Stopping LiveRunner...")
                self._running = False
            except Exception as e:
                logger.error("LiveRunner error: %s", e, exc_info=True)
                time.sleep(10)

            time.sleep(self.poll_interval)

        logger.info("LiveRunner stopped. Total executions: %d", len(self.order_mgr.records))

    def _tick(self, feed: IBKRDataFeed, generator: SignalGenerator, detector: RegimeDetector) -> None:
        """Single iteration of the live loop."""
        now = datetime.now()

        # Check session
        if not self.session.is_active(now):
            logger.debug("Outside session hours, skipping")
            return

        # Fetch latest bars
        feed._df = None  # Force refetch
        try:
            df = feed.load()
        except Exception as e:
            logger.warning("Could not fetch data: %s", e)
            return

        if len(df) < 50:
            logger.warning("Not enough bars (%d), need 50+", len(df))
            return

        # Detect current regime
        regime = detector.detect(df)

        # Run signal generation on latest data
        signals = generator.run(df, min_strength=0.5)

        if not signals:
            logger.debug("No signals at %s (regime: %s)", now, regime)
            return

        latest = signals[-1]
        logger.info(
            "SIGNAL: %s %s @ %.2f [stop=%.2f target=%.2f] strength=%.2f regime=%s",
            latest.direction, latest.instrument, latest.price,
            latest.stop, latest.target, latest.strength, latest.regime,
        )

        # ML filter
        if self.use_ml and self.ml_model:
            prediction = self.ml_model.predict(latest)
            prob = prediction["probability"]
            rec = prediction["recommendation"]
            logger.info("ML prediction: prob=%.3f recommendation=%s", prob, rec)

            if prob < self.min_ml_prob:
                logger.info("Signal rejected by ML (prob=%.3f < %.3f)", prob, self.min_ml_prob)
                return

        # Execute
        logger.info("Executing signal...")
        results = self.executor.execute_signal(latest, quantity=1)

        # Log execution
        self.order_mgr.record_execution(
            instrument=latest.instrument,
            side=latest.direction,
            order_type="BRACKET",
            quantity=1,
            entry_price=latest.price,
            stop_price=latest.stop,
            target_price=latest.target,
            status=results[0].status if results else "error",
            regime=latest.regime,
            signal_strength=latest.strength,
            order_ids=[r.order_id for r in results],
        )

    def stop(self) -> None:
        """Stop the live runner."""
        self._running = False
