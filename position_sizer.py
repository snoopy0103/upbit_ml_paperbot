from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class PositionSizeResult:
    krw_to_spend: float
    qty: Optional[float]
    stop_price: Optional[float]
    risk_krw: float

class PositionSizer:
    def __init__(self, risk_per_trade_pct: float = 0.30, max_allocation_pct: float = 10.0, min_order_krw: float = 5000.0, fee_roundtrip_pct: float = 0.10):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_allocation_pct = max_allocation_pct
        self.min_order_krw = min_order_krw
        self.fee_roundtrip_pct = fee_roundtrip_pct

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def size_from_stop_pct(self, equity_krw: float, stop_pct: float, price: Optional[float] = None) -> PositionSizeResult:
        stop_pct = float(stop_pct)
        if stop_pct <= 0:
            return PositionSizeResult(0.0, None, None, 0.0)

        risk_budget_krw = equity_krw * (self.risk_per_trade_pct / 100.0)
        fee_component = (self.fee_roundtrip_pct / 100.0)
        risk_per_invested = stop_pct + fee_component

        krw_to_spend = risk_budget_krw / max(risk_per_invested, 1e-9)
        cap = equity_krw * (self.max_allocation_pct / 100.0)
        krw_to_spend = self._clamp(krw_to_spend, 0.0, cap)

        if krw_to_spend < self.min_order_krw:
            return PositionSizeResult(0.0, None, None, 0.0)

        if price is None:
            return PositionSizeResult(krw_to_spend, None, None, risk_budget_krw)

        qty = (krw_to_spend * (1 - fee_component)) / price
        stop_price = price * (1 - stop_pct)
        risk_krw = krw_to_spend * risk_per_invested
        return PositionSizeResult(krw_to_spend, qty, stop_price, risk_krw)

    def size_from_atr_pct(self, equity_krw: float, atr_pct: float, stop_atr_mult: float = 1.0, price: Optional[float] = None) -> PositionSizeResult:
        stop_pct = float(atr_pct) * float(stop_atr_mult)
        return self.size_from_stop_pct(equity_krw=equity_krw, stop_pct=stop_pct, price=price)
