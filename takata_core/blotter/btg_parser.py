"""BTG Pactual Nota de Corretagem PDF Parser.

Parses brokerage notes from BTG Pactual into structured trade data.
Extracts individual trades, costs, fees, and net P&L for:
- Historical P&L tracking
- ML training data (real trade outcomes)
- Cost analysis (slippage, fees impact)
- Tax reporting (IRRF, day trade tax)

PDF structure (BTG standard):
- Page 1+: trade rows (C/V, instrument, expiry, qty, price, type, value, D/C, fee)
- Last page: summary section (fees, adjustments, totals)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BrokerTrade:
    """A single trade from the brokerage note."""
    side: str              # "C" (compra/buy) or "V" (venda/sell)
    instrument: str        # e.g., "WDOK26"
    expiry: str            # e.g., "04/05/2026"
    quantity: int
    price: float           # e.g., 5195.0
    trade_type: str        # "DAY TRADE" or "NORMAL"
    value: float           # operation value in R$
    debit_credit: str      # "D" or "C"
    operational_fee: float


@dataclass
class BrokerNoteSummary:
    """Summary totals from the brokerage note."""
    trade_value: float = 0         # Valor dos negócios
    position_adjustment: float = 0  # Ajuste de posição
    day_trade_adjustment: float = 0 # Ajuste day trade
    operational_fee: float = 0      # Taxa operacional
    registration_fee: float = 0     # Taxa registro BM&F
    exchange_fees: float = 0        # Taxas BM&F (emol+f.gar)
    iss: float = 0                  # ISS tax
    other_costs: float = 0          # Outros custos
    irrf: float = 0                 # IRRF withholding
    irrf_day_trade: float = 0       # IRRF Day Trade (projected)
    brokerage: float = 0            # Corretagem
    total_expenses: float = 0       # Total das despesas
    net_total: float = 0            # Total líquido da nota
    net_dc: str = ""                # D or C


@dataclass
class ParsedBrokerNote:
    """Complete parsed brokerage note."""
    # Header
    note_number: int = 0
    trade_date: str = ""       # YYYY-MM-DD
    broker: str = "BTG Pactual"
    client_name: str = ""
    client_code: str = ""
    cpf: str = ""

    # Trades
    trades: List[BrokerTrade] = field(default_factory=list)

    # Summary
    summary: BrokerNoteSummary = field(default_factory=BrokerNoteSummary)

    # Computed
    total_bought: int = 0
    total_sold: int = 0
    total_contracts: int = 0
    instruments: List[str] = field(default_factory=list)
    gross_pnl: float = 0
    total_fees: float = 0
    net_pnl: float = 0

    # Grouped trades (matched buys/sells)
    round_trips: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note_number": self.note_number,
            "trade_date": self.trade_date,
            "broker": self.broker,
            "client_name": self.client_name,
            "client_code": self.client_code,
            "num_trades": len(self.trades),
            "total_bought": self.total_bought,
            "total_sold": self.total_sold,
            "total_contracts": self.total_contracts,
            "instruments": self.instruments,
            "gross_pnl": round(self.gross_pnl, 2),
            "total_fees": round(self.total_fees, 2),
            "net_pnl": round(self.net_pnl, 2),
            "summary": {
                "trade_value": self.summary.trade_value,
                "position_adjustment": self.summary.position_adjustment,
                "day_trade_adjustment": self.summary.day_trade_adjustment,
                "operational_fee": self.summary.operational_fee,
                "registration_fee": self.summary.registration_fee,
                "exchange_fees": self.summary.exchange_fees,
                "iss": self.summary.iss,
                "irrf": self.summary.irrf,
                "irrf_day_trade": self.summary.irrf_day_trade,
                "brokerage": self.summary.brokerage,
                "total_expenses": self.summary.total_expenses,
                "net_total": self.summary.net_total,
                "net_dc": self.summary.net_dc,
            },
            "trades": [
                {
                    "side": t.side,
                    "instrument": t.instrument,
                    "expiry": t.expiry,
                    "quantity": t.quantity,
                    "price": t.price,
                    "trade_type": t.trade_type,
                    "value": t.value,
                    "debit_credit": t.debit_credit,
                    "operational_fee": t.operational_fee,
                }
                for t in self.trades
            ],
            "round_trips": self.round_trips,
        }


def _parse_br_number(s: str) -> float:
    """Parse Brazilian number format: 1.234,56 → 1234.56"""
    if not s or s.strip() in ("", "-", "|"):
        return 0.0
    s = s.strip().replace(" ", "")
    # Remove thousands separator (.) and convert decimal comma
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_date_br(s: str) -> str:
    """Convert DD/MM/YYYY to YYYY-MM-DD."""
    parts = s.strip().split("/")
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return s


def parse_btg_nota(pdf_path: str) -> ParsedBrokerNote:
    """Parse a BTG Pactual Nota de Corretagem PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    ParsedBrokerNote
        Structured trade data with computed P&L and fees.
    """
    try:
        import pdfplumber
    except ImportError:
        # Fallback to basic text extraction
        return _parse_with_basic_extraction(pdf_path)

    note = ParsedBrokerNote()

    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text += text + "\n"

    lines = all_text.split("\n")
    _parse_header(lines, note)
    _parse_trades(lines, note)
    _parse_summary(lines, note)
    _compute_metrics(note)

    return note


def _parse_with_basic_extraction(pdf_path: str) -> ParsedBrokerNote:
    """Fallback parser using PyPDF2 or similar."""
    note = ParsedBrokerNote()
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        all_text = ""
        for page in reader.pages:
            all_text += (page.extract_text() or "") + "\n"

        lines = all_text.split("\n")
        _parse_header(lines, note)
        _parse_trades(lines, note)
        _parse_summary(lines, note)
        _compute_metrics(note)
    except Exception as e:
        logger.error("PDF parse failed: %s", e)

    return note


def _parse_header(lines: List[str], note: ParsedBrokerNote) -> None:
    """Extract header info: note number, date, client."""
    for line in lines[:20]:
        # Note number
        m = re.search(r'Nr\.\s*nota\s*(\d+)', line)
        if not m:
            m = re.search(r'(\d{6})', line)
        if m and not note.note_number:
            try:
                note.note_number = int(m.group(1))
            except ValueError:
                pass

        # Trade date
        m = re.search(r'Data pregão\s*(\d{2}/\d{2}/\d{4})', line)
        if m:
            note.trade_date = _parse_date_br(m.group(1))
        elif not note.trade_date:
            m = re.search(r'(\d{2}/\d{2}/\d{4})', line)
            if m:
                note.trade_date = _parse_date_br(m.group(1))

        # Client name
        if "LAURO" in line.upper() or "TAKATA" in line.upper():
            note.client_name = line.strip().split("  ")[0].strip()

        # Client code
        m = re.search(r'Código cliente\s*(\d+)', line)
        if m:
            note.client_code = m.group(1)
        elif re.match(r'^\d{7}$', line.strip()):
            note.client_code = line.strip()

        # CPF
        m = re.search(r'(\d{3}\.\d{3}\.\d{3}-\d{2})', line)
        if m:
            note.cpf = m.group(1)


def _parse_trades(lines: List[str], note: ParsedBrokerNote) -> None:
    """Extract individual trade rows."""
    # Trade line pattern: C/V  INSTRUMENT  DATE  QTY  PRICE  TYPE  VALUE  D/C  FEE
    # Example: C WDOK26 04/05/2026 10 5.195,0000 DAY TRADE 512,20 D 0,00
    trade_pattern = re.compile(
        r'^([CV])\s+'                            # C or V
        r'(\w+)\s+'                              # instrument
        r'(\d{2}/\d{2}/\d{4})\s+'               # expiry date
        r'(\d+)\s+'                              # quantity
        r'([\d.,]+)\s+'                          # price
        r'(DAY TRADE|NORMAL)\s+'                 # trade type
        r'([\d.,]+)\s+'                          # operation value
        r'([DC])\s+'                             # debit/credit
        r'([\d.,]+)'                             # operational fee
    )

    for line in lines:
        line = line.strip()
        m = trade_pattern.match(line)
        if m:
            trade = BrokerTrade(
                side=m.group(1),
                instrument=m.group(2),
                expiry=m.group(3),
                quantity=int(m.group(4)),
                price=_parse_br_number(m.group(5)),
                trade_type=m.group(6),
                value=_parse_br_number(m.group(7)),
                debit_credit=m.group(8),
                operational_fee=_parse_br_number(m.group(9)),
            )
            note.trades.append(trade)


def _parse_summary(lines: List[str], note: ParsedBrokerNote) -> None:
    """Extract summary financials from BTG nota.

    BTG PDFs have a positional table layout where header rows and value rows
    are on separate lines. The pattern is:
      Header line: "Label1 Label2 Label3 Label4 Label5"
      Value line:  "0,00 0,00 0,00 172,80 91,80 | D"

    We match header lines and then parse the next line for values.
    """
    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        # Row: Venda disponível ... Valor dos negócios
        # Next: 0,00 0,00 0,00 0,00 2.780,00 | D
        if "Valor dos neg" in line and next_line:
            vals = re.findall(r'([\d.,]+)', next_line)
            if len(vals) >= 5:
                note.summary.trade_value = _parse_br_number(vals[4])
            # D/C flag
            m = re.search(r'\|\s*([DC])', next_line)
            if m:
                note.summary.net_dc = m.group(1)  # preliminary — overridden by net total line

        # Row: IRRF ... Taxa operacional ... Taxa registro BM&F ... Taxas BM&F
        # Next: 0,00 0,00 0,00 172,80 91,80 | D
        if "Taxa operacional" in line and "Taxa registro" in line and next_line:
            vals = re.findall(r'([\d.,]+)', next_line)
            if len(vals) >= 5:
                note.summary.irrf = _parse_br_number(vals[0])
                note.summary.irrf_day_trade = _parse_br_number(vals[1])
                note.summary.operational_fee = _parse_br_number(vals[2])
                note.summary.registration_fee = _parse_br_number(vals[3])
                note.summary.exchange_fees = _parse_br_number(vals[4])

        # Row: +Outros Custos ... I.S.S ... Ajuste de posição ... Ajuste day trade ... Total das despesas
        # Next: 0,00 0,00 0,00 | 2.780,00 | D 264,60 | D
        if "Ajuste day trade" in line and "Total das despesas" in line and next_line:
            # Extract all number|DC pairs
            pairs = re.findall(r'([\d.,]+)\s*\|?\s*([DC])?', next_line)
            vals = re.findall(r'([\d.,]+)', next_line)
            if len(vals) >= 5:
                note.summary.other_costs = _parse_br_number(vals[0])
                note.summary.iss = _parse_br_number(vals[1])
                note.summary.position_adjustment = _parse_br_number(vals[2])
                note.summary.day_trade_adjustment = _parse_br_number(vals[3])
                note.summary.total_expenses = _parse_br_number(vals[4])

        # Row: Outros ... IRRF Corretagem ... Total Conta Investimento ... Total líquido da nota
        # Next: 0,00 0,00 0,00 | 3.044,60 | D 3.044,60 | D 3.044,60 | D
        if "Total l" in line and "quido da nota" in line and next_line:
            vals = re.findall(r'([\d.,]+)', next_line)
            # Last value is the net total
            if vals:
                note.summary.net_total = _parse_br_number(vals[-1])
                # Find D/C — last occurrence
                dcs = re.findall(r'\|\s*([DC])', next_line)
                if dcs:
                    note.summary.net_dc = dcs[-1]
                # Brokerage is typically vals[2] if present
                if len(vals) >= 3:
                    note.summary.brokerage = _parse_br_number(vals[2])


def _compute_metrics(note: ParsedBrokerNote) -> None:
    """Compute derived metrics from parsed trades."""
    buys = [t for t in note.trades if t.side == "C"]
    sells = [t for t in note.trades if t.side == "V"]

    note.total_bought = sum(t.quantity for t in buys)
    note.total_sold = sum(t.quantity for t in sells)
    note.total_contracts = note.total_bought + note.total_sold
    note.instruments = list(set(t.instrument for t in note.trades))

    # Compute VWAP for buys and sells
    if buys:
        buy_value = sum(t.quantity * t.price for t in buys)
        buy_qty = sum(t.quantity for t in buys)
        avg_buy = buy_value / buy_qty if buy_qty > 0 else 0
    else:
        avg_buy = 0

    if sells:
        sell_value = sum(t.quantity * t.price for t in sells)
        sell_qty = sum(t.quantity for t in sells)
        avg_sell = sell_value / sell_qty if sell_qty > 0 else 0
    else:
        avg_sell = 0

    # For day trades, matched quantity
    matched_qty = min(note.total_bought, note.total_sold)

    point_value = 10.0  # WDO default
    if any("WIN" in t.instrument for t in note.trades):
        point_value = 0.20  # WIN mini index

    # ── P&L: ALWAYS use official BTG summary as authoritative ──
    # The ajuste values are the official settlement — more accurate than FIFO matching
    if note.summary.day_trade_adjustment > 0 or note.summary.position_adjustment > 0:
        note.gross_pnl = note.summary.day_trade_adjustment + note.summary.position_adjustment
    else:
        # Fallback: compute from round trips if summary not parsed
        note.gross_pnl = (avg_sell - avg_buy) * matched_qty * point_value

    # D/C flag: "D" on a nota = debit = you owe = LOSS
    # The ajuste values are always positive in the PDF — the D/C tells you the sign
    if note.summary.net_dc == "D":
        note.gross_pnl = -abs(note.gross_pnl)

    # Total fees
    note.total_fees = (
        note.summary.operational_fee +
        note.summary.registration_fee +
        note.summary.exchange_fees +
        note.summary.iss +
        note.summary.other_costs +
        note.summary.brokerage
    )

    # Net P&L: use official net total from BTG
    if note.summary.net_total > 0:
        # Apply D/C sign
        note.net_pnl = -note.summary.net_total if note.summary.net_dc == "D" else note.summary.net_total
    else:
        note.net_pnl = note.gross_pnl - note.total_fees

    # Build round trips (simplified: match by FIFO within same instrument)
    _build_round_trips(note, point_value)


def _build_round_trips(note: ParsedBrokerNote, point_value: float) -> None:
    """Match buys and sells into round trips for ML training."""
    # Group by instrument
    by_inst: Dict[str, Dict[str, list]] = {}
    for t in note.trades:
        if t.instrument not in by_inst:
            by_inst[t.instrument] = {"buys": [], "sells": []}
        if t.side == "C":
            by_inst[t.instrument]["buys"].append(t)
        else:
            by_inst[t.instrument]["sells"].append(t)

    for inst, sides in by_inst.items():
        buy_queue = list(sides["buys"])
        sell_queue = list(sides["sells"])

        # FIFO matching
        bi, si = 0, 0
        buy_remaining = buy_queue[0].quantity if buy_queue else 0
        sell_remaining = sell_queue[0].quantity if sell_queue else 0

        while bi < len(buy_queue) and si < len(sell_queue):
            matched = min(buy_remaining, sell_remaining)
            if matched > 0:
                buy_price = buy_queue[bi].price
                sell_price = sell_queue[si].price
                pnl_pts = sell_price - buy_price
                pnl_brl = pnl_pts * matched * point_value

                note.round_trips.append({
                    "instrument": inst,
                    "quantity": matched,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "pnl_pts": round(pnl_pts, 2),
                    "pnl_brl": round(pnl_brl, 2),
                    "outcome": "win" if pnl_pts > 0 else "loss" if pnl_pts < 0 else "breakeven",
                })

            buy_remaining -= matched
            sell_remaining -= matched

            if buy_remaining <= 0:
                bi += 1
                if bi < len(buy_queue):
                    buy_remaining = buy_queue[bi].quantity
            if sell_remaining <= 0:
                si += 1
                if si < len(sell_queue):
                    sell_remaining = sell_queue[si].quantity


def parse_btg_folder(folder_path: str) -> List[ParsedBrokerNote]:
    """Parse all BTG PDF notes in a folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing BTG nota PDFs.

    Returns
    -------
    list of ParsedBrokerNote
        All parsed notes sorted by date.
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning("Folder not found: %s", folder_path)
        return []

    notes = []
    for pdf_file in sorted(folder.glob("*.pdf")):
        try:
            note = parse_btg_nota(str(pdf_file))
            if note.trades:
                notes.append(note)
                logger.info("Parsed %s: %d trades, net R$%.2f",
                            pdf_file.name, len(note.trades), note.net_pnl)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", pdf_file.name, e)

    notes.sort(key=lambda n: n.trade_date)
    return notes
