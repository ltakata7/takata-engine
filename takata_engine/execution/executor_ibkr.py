"""IBKR order execution for MES/ES futures via ib_insync."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ib_insync import IB, LimitOrder, MarketOrder, Order, StopOrder, Trade

from takata_engine.data.feed_ibkr import IBKRConnection, get_contract
from takata_engine.signals.signal_generator import Signal

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order submission."""
    order_id: int
    instrument: str
    side: str
    order_type: str
    quantity: float
    limit_price: Optional[float]
    status: str
    fill_price: Optional[float] = None
    commission: float = 0.0
    error: str = ""


class IBKRExecutor:
    """Executes orders on Interactive Brokers.

    Parameters
    ----------
    host : str
        TWS/Gateway hostname.
    port : int
        API port.
    paper : bool
        If True, use paper trading port (4002 for Gateway, 7497 for TWS).
    account : str, optional
        IBKR account number (for multi-account setups).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,      # default to paper trading
        paper: bool = True,
        account: Optional[str] = None,
    ):
        self.host = host
        self.port = port if not paper else (4002 if port in (4001, 4002) else 7497)
        self.account = account
        self._conn: Optional[IBKRConnection] = None
        self._pending_orders: List[Trade] = []

    @property
    def ib(self) -> IB:
        if self._conn is None or not self._conn.connected:
            self._conn = IBKRConnection.get(self.host, self.port)
        return self._conn.ib

    def submit_market_order(
        self,
        instrument: str,
        side: str,
        quantity: float,
    ) -> OrderResult:
        """Submit a market order.

        Parameters
        ----------
        instrument : str
            e.g. ``"MES"``.
        side : str
            ``"long"`` or ``"short"`` (mapped to BUY/SELL).
        quantity : float
            Number of contracts.
        """
        contract = get_contract(instrument)
        self.ib.qualifyContracts(contract)

        action = "BUY" if side == "long" else "SELL"
        order = MarketOrder(action, quantity)
        if self.account:
            order.account = self.account

        trade = self.ib.placeOrder(contract, order)
        self._pending_orders.append(trade)

        logger.info("MARKET %s %s %d contracts", action, instrument, quantity)

        # Wait briefly for fill
        self.ib.sleep(2)

        return self._trade_to_result(trade, instrument, side, "MARKET")

    def submit_bracket_order(
        self,
        instrument: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_price: float,
        target_price: float,
    ) -> List[OrderResult]:
        """Submit a bracket order (entry + stop + target).

        Parameters
        ----------
        instrument : str
            e.g. ``"MES"``.
        side : str
            ``"long"`` or ``"short"``.
        quantity : float
            Number of contracts.
        entry_price : float
            Limit price for entry.
        stop_price : float
            Stop loss price.
        target_price : float
            Take profit price.
        """
        contract = get_contract(instrument)
        self.ib.qualifyContracts(contract)

        action = "BUY" if side == "long" else "SELL"
        reverse_action = "SELL" if side == "long" else "BUY"

        bracket = self.ib.bracketOrder(
            action=action,
            quantity=quantity,
            limitPrice=entry_price,
            takeProfitPrice=target_price,
            stopLossPrice=stop_price,
        )

        if self.account:
            for o in bracket:
                o.account = self.account

        results = []
        for order in bracket:
            trade = self.ib.placeOrder(contract, order)
            self._pending_orders.append(trade)
            results.append(self._trade_to_result(trade, instrument, side, order.orderType))

        logger.info(
            "BRACKET %s %s %d @ %.2f [stop=%.2f target=%.2f]",
            action, instrument, quantity, entry_price, stop_price, target_price,
        )
        return results

    def execute_signal(self, signal: Signal, quantity: float = 1) -> List[OrderResult]:
        """Execute a Signal as a bracket order.

        Parameters
        ----------
        signal : Signal
            Generated signal with price, stop, target.
        quantity : float
            Number of contracts.

        Returns
        -------
        list of OrderResult
            Entry + stop + target order results.
        """
        return self.submit_bracket_order(
            instrument=signal.instrument,
            side=signal.direction,
            quantity=quantity,
            entry_price=signal.price,
            stop_price=signal.stop,
            target_price=signal.target,
        )

    def cancel_all(self, instrument: Optional[str] = None) -> int:
        """Cancel all open orders, optionally filtered by instrument."""
        cancelled = 0
        for trade in self.ib.openTrades():
            if instrument and trade.contract.symbol != instrument.upper():
                continue
            self.ib.cancelOrder(trade.order)
            cancelled += 1
            logger.info("Cancelled order %d", trade.order.orderId)
        return cancelled

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        positions = []
        for pos in self.ib.positions():
            if self.account and pos.account != self.account:
                continue
            positions.append({
                "account": pos.account,
                "symbol": pos.contract.symbol,
                "exchange": pos.contract.exchange,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost,
            })
        return positions

    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary (NAV, cash, margin, etc.)."""
        summaries = self.ib.accountSummary()
        result = {}
        for s in summaries:
            if self.account and s.account != self.account:
                continue
            result[s.tag] = s.value
        return result

    def flatten_position(self, instrument: str) -> Optional[OrderResult]:
        """Close all positions for an instrument."""
        for pos in self.ib.positions():
            if pos.contract.symbol == instrument.upper() and pos.position != 0:
                side = "short" if pos.position > 0 else "long"
                qty = abs(pos.position)
                return self.submit_market_order(instrument, side, qty)
        return None

    def _trade_to_result(self, trade: Trade, instrument: str, side: str, order_type: str) -> OrderResult:
        return OrderResult(
            order_id=trade.order.orderId,
            instrument=instrument,
            side=side,
            order_type=order_type,
            quantity=trade.order.totalQuantity,
            limit_price=getattr(trade.order, "lmtPrice", None),
            status=trade.orderStatus.status,
            fill_price=trade.orderStatus.avgFillPrice if trade.orderStatus.avgFillPrice else None,
            commission=0.0,
        )
