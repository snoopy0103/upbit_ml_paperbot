from datetime import datetime
from typing import Optional

from risk_engine import RiskEngine
from position_sizer import PositionSizer, PositionSizeResult

class TradeDecision:
    def __init__(self, allowed: bool, reason: str, sizing: Optional[PositionSizeResult] = None):
        self.allowed = allowed
        self.reason = reason
        self.sizing = sizing

class TradeGuard:
    def __init__(self, risk_engine: RiskEngine, sizer: PositionSizer):
        self.risk_engine = risk_engine
        self.sizer = sizer

    def evaluate_entry(
        self,
        *,
        equity_krw: float,
        now: datetime,
        price: float,
        stop_pct: Optional[float] = None,
        atr_pct: Optional[float] = None,
        stop_atr_mult: float = 1.0,
    ) -> TradeDecision:
        if not self.risk_engine.can_trade(equity_krw, now):
            return TradeDecision(False, "risk_gate_blocked")

        if stop_pct is not None:
            sizing = self.sizer.size_from_stop_pct(equity_krw=equity_krw, stop_pct=stop_pct, price=price)
        elif atr_pct is not None:
            sizing = self.sizer.size_from_atr_pct(equity_krw=equity_krw, atr_pct=atr_pct, stop_atr_mult=stop_atr_mult, price=price)
        else:
            return TradeDecision(False, "no_stop_defined")

        if sizing.krw_to_spend <= 0:
            return TradeDecision(False, "size_zero_or_below_min")

        return TradeDecision(True, "ok", sizing)
