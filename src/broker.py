"""Alpaca paper-trading connector with circuit breakers.

Uses alpaca-py. Pointed at the PAPER endpoint by default (config.py).
The exact same code trades live later by changing ALPACA_BASE_URL and keys
in .env — which is why every order passes through the risk gate here.

SAFETY: circuit breakers run before every order. Never bypass them, even
in paper mode — paper trading is your rehearsal of the live system,
including its safety rails.
"""
import logging
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import config

log = logging.getLogger("broker")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


class RiskGateError(Exception):
    """Raised when a circuit breaker blocks an order."""


class PaperBroker:
    def __init__(self):
        if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
            raise RuntimeError(
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file. "
                "Get free paper keys at https://alpaca.markets"
            )
        self.client = TradingClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
            paper="paper" in config.ALPACA_BASE_URL,
        )

    # ---------- account state ----------
    def account(self):
        return self.client.get_account()

    def positions(self):
        return self.client.get_all_positions()

    def position_qty(self, symbol: str) -> float:
        for p in self.positions():
            if p.symbol == symbol:
                return float(p.qty)
        return 0.0

    # ---------- circuit breakers ----------
    def check_risk_gate(self, symbol: str, side: str, qty: int):
        acct = self.account()
        equity = float(acct.equity)
        last_equity = float(acct.last_equity)

        # 1) Daily loss halt
        daily_pnl_pct = (equity - last_equity) / last_equity * 100
        if daily_pnl_pct < -config.MAX_DAILY_LOSS_PCT:
            raise RiskGateError(
                f"HALT: account down {daily_pnl_pct:.2f}% today "
                f"(limit {config.MAX_DAILY_LOSS_PCT}%). No new orders."
            )

        # 2) Position size cap
        if qty > config.MAX_POSITION_QTY:
            raise RiskGateError(
                f"BLOCKED: qty {qty} exceeds MAX_POSITION_QTY "
                f"({config.MAX_POSITION_QTY})."
            )

        # 3) Max open positions
        if side == "buy" and self.position_qty(symbol) == 0:
            if len(self.positions()) >= config.MAX_OPEN_POSITIONS:
                raise RiskGateError(
                    f"BLOCKED: already at MAX_OPEN_POSITIONS "
                    f"({config.MAX_OPEN_POSITIONS})."
                )

    # ---------- orders ----------
    def submit(self, symbol: str, side: str, qty: int) -> dict:
        """Submit a market order after passing the risk gate."""
        side = side.lower()
        assert side in ("buy", "sell")
        self.check_risk_gate(symbol, side, qty)

        # Selling more than we hold = don't (no accidental shorts in paper)
        if side == "sell":
            held = self.position_qty(symbol)
            if held <= 0:
                raise RiskGateError(f"BLOCKED: no {symbol} position to sell.")
            qty = min(qty, int(held))

        order = self.client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))
        log.info("Order submitted: %s %s x%s id=%s", side.upper(), symbol, qty, order.id)
        return {
            "id": str(order.id), "symbol": symbol, "side": side, "qty": qty,
            "status": str(order.status),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def flatten_all(self):
        """KILL SWITCH: close every open position immediately."""
        log.warning("KILL SWITCH: closing all positions.")
        self.client.close_all_positions(cancel_orders=True)
