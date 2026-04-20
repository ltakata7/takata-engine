"""Order state management — tracks orders, fills, and execution status."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ORDER_LOG = Path(__file__).resolve().parent.parent.parent / "logs" / "orders.jsonl"


@dataclass
class ExecutionRecord:
    """Record of an executed order."""
    timestamp: str
    instrument: str
    side: str
    order_type: str
    quantity: float
    entry_price: float
    stop_price: Optional[float]
    target_price: Optional[float]
    status: str
    regime: str
    signal_strength: float
    order_ids: List[int] = field(default_factory=list)


class OrderManager:
    """Manages execution state and logging.

    Tracks what was sent to the broker, matches fills to signals,
    and logs everything for post-trade analysis.
    """

    def __init__(self):
        self._records: List[ExecutionRecord] = []
        ORDER_LOG.parent.mkdir(parents=True, exist_ok=True)

    def record_execution(
        self,
        instrument: str,
        side: str,
        order_type: str,
        quantity: float,
        entry_price: float,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        status: str = "submitted",
        regime: str = "",
        signal_strength: float = 0.0,
        order_ids: Optional[List[int]] = None,
    ) -> ExecutionRecord:
        """Log an execution."""
        record = ExecutionRecord(
            timestamp=datetime.utcnow().isoformat(),
            instrument=instrument,
            side=side,
            order_type=order_type,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            status=status,
            regime=regime,
            signal_strength=signal_strength,
            order_ids=order_ids or [],
        )
        self._records.append(record)

        # Append to JSONL log
        with open(ORDER_LOG, "a") as f:
            f.write(json.dumps(record.__dict__, default=str) + "\n")

        logger.info(
            "EXEC %s %s %s %.0f @ %.2f [stop=%.2f target=%.2f] regime=%s",
            record.status, instrument, side, quantity, entry_price,
            stop_price or 0, target_price or 0, regime,
        )
        return record

    @property
    def records(self) -> List[ExecutionRecord]:
        return list(self._records)
