from datetime import datetime

class PaperTrader:
    def __init__(self, initial_balance: float = 1_000_000, fee: float = 0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = None
        self.entry_notional = None
        self.fee = fee
        self.history = []

    def can_buy(self):
        return self.position == 0 and self.balance > 0

    def can_sell(self):
        return self.position > 0

    def buy(self, price: float, timestamp: datetime, spend_krw: float | None = None):
        if not self.can_buy():
            return
        if spend_krw is None:
            spend_krw = self.balance
        spend_krw = max(0.0, min(float(spend_krw), self.balance))
        if spend_krw <= 0:
            return

        fee_paid = spend_krw * self.fee
        cost = spend_krw - fee_paid
        qty = cost / price
        self.position = qty
        self.entry_price = price
        self.entry_notional = spend_krw
        self.balance -= spend_krw
        self.history.append(
            {
                "time": timestamp,
                "type": "BUY",
                "price": price,
                "qty": qty,
                "spend_krw": spend_krw,
                "fee": fee_paid,
                "balance": self.balance,
            }
        )
        print(f"[PAPER BUY] price={price:.2f} qty={qty:.6f}")

    def sell(self, price: float, timestamp: datetime, reason: str = "EXIT"):
        if not self.can_sell():
            return
        proceeds = self.position * price * (1 - self.fee)
        entry_notional = self.entry_notional if self.entry_notional is not None else (self.position * self.entry_price)
        pnl = proceeds - entry_notional
        self.balance += proceeds
        self.position = 0.0
        self.entry_price = None
        self.entry_notional = None
        self.history.append({"time": timestamp, "type": "SELL", "price": price, "pnl": pnl, "balance": self.balance, "reason": reason})
        print(f"[PAPER SELL] price={price:.2f} pnl={pnl:.2f} balance={self.balance:.2f} reason={reason}")

    def check_tp_sl(self, high: float, low: float, timestamp: datetime, tp: float, sl: float):
        if not self.can_sell():
            return
        tp_price = self.entry_price * (1 + tp)
        sl_price = self.entry_price * (1 - sl)
        tp_hit = high >= tp_price
        sl_hit = low <= sl_price
        if tp_hit and sl_hit:
            # Intrabar order is unknown for OHLC. Use conservative SL-first tie-break.
            self.sell(sl_price, timestamp, "SL_TIE")
        elif tp_hit:
            self.sell(tp_price, timestamp, "TP")
        elif sl_hit:
            self.sell(sl_price, timestamp, "SL")
