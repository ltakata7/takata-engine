"""Structured trade and signal logging."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with structured format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class TradeLogger:
    """Append-only JSONL logger for signals and trade outcomes.

    Parameters
    ----------
    trade_log : str | Path
        Path to the trades JSONL file.
    signal_log : str | Path | None
        Path to the signals JSONL file. If ``None``, signals go to trade_log.
    """

    def __init__(
        self,
        trade_log: str | Path = "logs/trades.jsonl",
        signal_log: str | Path | None = None,
    ):
        self.trade_log = Path(trade_log)
        self.signal_log = Path(signal_log) if signal_log else self.trade_log
        self.trade_log.parent.mkdir(parents=True, exist_ok=True)
        self.signal_log.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("trade_logger")

    def _write(self, path: Path, record: dict[str, Any]) -> None:
        record["logged_at"] = datetime.utcnow().isoformat()
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_signal(
        self,
        instrument: str,
        signal: str,
        price: float,
        regime: str | None = None,
        indicators: dict[str, float] | None = None,
        **extra: Any,
    ) -> None:
        """Log a generated signal."""
        record = {
            "type": "signal",
            "instrument": instrument,
            "signal": signal,
            "price": price,
            "regime": regime,
            "indicators": indicators or {},
            **extra,
        }
        self._write(self.signal_log, record)
        self._logger.info(
            "SIGNAL %s %s @ %.2f [regime=%s]", instrument, signal, price, regime
        )

    def log_entry(
        self,
        instrument: str,
        side: str,
        price: float,
        size: float,
        stop: float | None = None,
        target: float | None = None,
        regime: str | None = None,
        **extra: Any,
    ) -> None:
        """Log a trade entry."""
        record = {
            "type": "entry",
            "instrument": instrument,
            "side": side,
            "price": price,
            "size": size,
            "stop": stop,
            "target": target,
            "regime": regime,
            **extra,
        }
        self._write(self.trade_log, record)
        self._logger.info(
            "ENTRY %s %s %.0f @ %.2f stop=%.2f target=%.2f [regime=%s]",
            instrument, side, size, price, stop or 0, target or 0, regime,
        )

    def log_exit(
        self,
        instrument: str,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float,
        reason: str = "",
        regime: str | None = None,
        **extra: Any,
    ) -> None:
        """Log a trade exit with P&L calculation."""
        pnl_per_unit = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
        pnl = pnl_per_unit * size

        record = {
            "type": "exit",
            "instrument": instrument,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl,
            "reason": reason,
            "regime": regime,
            **extra,
        }
        self._write(self.trade_log, record)
        self._logger.info(
            "EXIT %s %s %.0f @ %.2f (entry=%.2f) P&L=%.2f [%s]",
            instrument, side, size, exit_price, entry_price, pnl, reason,
        )
