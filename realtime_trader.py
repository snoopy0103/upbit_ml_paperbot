import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Optional, Tuple

import joblib
import pandas as pd
import websockets

from feature_engineering import generate_features
from paper_trader import PaperTrader
from position_sizer import PositionSizer
from risk_engine import RiskEngine
from trade_guard import TradeGuard

WS_URL = "wss://api.upbit.com/websocket/v1"
MARKETS = ["KRW-BTC"]  # TODO: daily scanner -> top 10 markets
ENTRY_THRESHOLD = 0.60
TP_PCT = 0.015
SL_PCT = 0.009
MODEL_PATH = "model_champion.pkl"
HISTORY_LEN = 600

@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class CandleBuilder5m:
    def __init__(self):
        self.current: Dict[str, Optional[Candle]] = defaultdict(lambda: None)

    @staticmethod
    def _bucket(dt: datetime) -> datetime:
        m = (dt.minute // 5) * 5
        return dt.replace(minute=m, second=0, microsecond=0)

    def update_trade(self, market: str, price: float, vol: float, tms_ms: int) -> Tuple[Optional[Candle], Optional[Candle]]:
        dt = datetime.fromtimestamp(tms_ms / 1000, tz=timezone.utc)
        bucket = self._bucket(dt)

        cur = self.current[market]
        if cur is None or cur.time != bucket:
            closed = cur
            self.current[market] = Candle(time=bucket, open=price, high=price, low=price, close=price, volume=vol)
            return closed, self.current[market]

        cur.high = max(cur.high, price)
        cur.low = min(cur.low, price)
        cur.close = price
        cur.volume += vol
        return None, cur

class RealtimePaperBot:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        self.risk = RiskEngine(max_daily_loss_pct=3.0, max_consecutive_losses=3, cooldown_minutes=60)
        self.sizer = PositionSizer(risk_per_trade_pct=0.30, max_allocation_pct=10.0)
        self.guard = TradeGuard(self.risk, self.sizer)

        self.paper = PaperTrader(initial_balance=1_000_000, fee=0.001)

        self.builder = CandleBuilder5m()
        self.history: Dict[str, Deque[Candle]] = {m: deque(maxlen=HISTORY_LEN) for m in MARKETS}

        self.open_market: Optional[str] = None
        self.open_tp: Optional[float] = None
        self.open_sl: Optional[float] = None

        # align columns if available
        self.feature_names = getattr(self.model, "feature_name", lambda: None)()

    def _candles_to_df(self, market: str) -> pd.DataFrame:
        rows = [{"datetime": c.time, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in self.history[market]]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("datetime").reset_index(drop=True)

    def _make_realtime_features(self, market: str) -> Optional[pd.DataFrame]:
        df = self._candles_to_df(market)
        if len(df) < 200:
            return None

        feat = generate_features(df)
        if feat.empty:
            return None

        X = feat.drop(columns=["datetime"], errors="ignore").tail(1)

        if self.feature_names:
            for c in self.feature_names:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[self.feature_names]
        return X

    def _predict_proba(self, X: pd.DataFrame) -> float:
        return float(self.model.predict(X)[0])

    def on_candle_close(self, market: str, candle: Candle):
        self.history[market].append(candle)

        # Manage open position
        if self.open_market == market and self.paper.can_sell():
            self.paper.check_tp_sl(high=candle.high, low=candle.low, timestamp=candle.time, tp=self.open_tp or TP_PCT, sl=self.open_sl or SL_PCT)
            if not self.paper.can_sell():
                last_sell = next((h for h in reversed(self.paper.history) if h.get("type") == "SELL"), None)
                if last_sell:
                    self.risk.record_trade_result(float(last_sell.get("pnl", 0.0)), candle.time)
                self.open_market = None
                self.open_tp = None
                self.open_sl = None

        # Entry if flat
        if not self.paper.can_buy():
            return

        X = self._make_realtime_features(market)
        if X is None:
            return

        proba = self._predict_proba(X)
        print(f"[{market}] {candle.time} TP_proba={proba:.3f} balance={self.paper.balance:.0f}")

        if proba < ENTRY_THRESHOLD:
            return

        decision = self.guard.evaluate_entry(equity_krw=self.paper.balance, now=candle.time, price=candle.close, stop_pct=SL_PCT)
        if not decision.allowed or decision.sizing is None:
            print(f"[{market}] entry blocked: {decision.reason}")
            return

        self.paper.buy(price=candle.close, timestamp=candle.time, spend_krw=decision.sizing.krw_to_spend)
        self.open_market = market
        self.open_tp = TP_PCT
        self.open_sl = SL_PCT

    def handle_trade_message(self, msg: dict):
        market = msg["cd"]
        price = float(msg["tp"])
        vol = float(msg["tv"])
        tms = int(msg["tms"])

        closed, _ = self.builder.update_trade(market, price, vol, tms)
        if closed is not None:
            self.on_candle_close(market, closed)

async def websocket_run(bot: RealtimePaperBot):
    async with websockets.connect(WS_URL, ping_interval=60) as ws:
        subscribe_msg = [
            {"ticket": "paper-bot"},
            {"type": "trade", "codes": MARKETS},
            {"format": "SIMPLE"},
        ]
        await ws.send(json.dumps(subscribe_msg))
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            msg = json.loads(raw)
            bot.handle_trade_message(msg)

if __name__ == "__main__":
    bot = RealtimePaperBot()
    asyncio.run(websocket_run(bot))
