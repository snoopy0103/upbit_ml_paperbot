import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import joblib
import pandas as pd

from feature_engineering import generate_features
from paper_trader import PaperTrader
from position_sizer import PositionSizer
from risk_engine import RiskEngine
from trade_guard import TradeGuard


@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def build_feature_row(history_df: pd.DataFrame, feature_names: List[str]) -> Optional[pd.DataFrame]:
    if len(history_df) < 200:
        return None
    feat = generate_features(history_df)
    if feat.empty:
        return None
    x = feat.drop(columns=["datetime"], errors="ignore").tail(1).copy()
    for col in feature_names:
        if col not in x.columns:
            x[col] = 0.0
    return x[feature_names]


def max_drawdown_pct(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100.0 if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="OHLCV CSV with datetime/open/high/low/close/volume")
    ap.add_argument("--model", default="model_champion.pkl")
    ap.add_argument("--entry-threshold", type=float, default=0.60)
    ap.add_argument("--tp", type=float, default=0.015)
    ap.add_argument("--sl", type=float, default=0.009)
    ap.add_argument("--initial-balance", type=float, default=1_000_000)
    ap.add_argument("--history-len", type=int, default=600)
    ap.add_argument("--export-trades", default="", help="Optional output CSV path for trade history")
    args = ap.parse_args()

    model = joblib.load(args.model)
    feature_names = list(getattr(model, "feature_name", lambda: [])() or [])
    if not feature_names:
        raise ValueError("Model has no feature_name metadata. Train with current pipeline first.")

    df = load_ohlcv(args.data)
    risk = RiskEngine(max_daily_loss_pct=3.0, max_consecutive_losses=3, cooldown_minutes=60)
    sizer = PositionSizer(risk_per_trade_pct=0.30, max_allocation_pct=10.0)
    guard = TradeGuard(risk, sizer)
    paper = PaperTrader(initial_balance=args.initial_balance, fee=0.001)

    history: List[Candle] = []
    equity_curve: List[float] = []

    for row in df.itertuples(index=False):
        candle = Candle(
            time=row.datetime,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        history.append(candle)
        if len(history) > args.history_len:
            history.pop(0)

        if paper.can_sell():
            paper.check_tp_sl(high=candle.high, low=candle.low, timestamp=candle.time, tp=args.tp, sl=args.sl)
            if not paper.can_sell():
                last_sell = next((h for h in reversed(paper.history) if h.get("type") == "SELL"), None)
                if last_sell:
                    risk.record_trade_result(float(last_sell.get("pnl", 0.0)), candle.time)

        if paper.can_buy():
            hist_df = pd.DataFrame(
                [
                    {
                        "datetime": c.time,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in history
                ]
            )
            x = build_feature_row(hist_df, feature_names)
            if x is not None:
                proba = float(model.predict(x)[0])
                if proba >= args.entry_threshold:
                    decision = guard.evaluate_entry(
                        equity_krw=paper.balance,
                        now=candle.time,
                        price=candle.close,
                        stop_pct=args.sl,
                    )
                    if decision.allowed and decision.sizing is not None:
                        paper.buy(price=candle.close, timestamp=candle.time, spend_krw=decision.sizing.krw_to_spend)

        marked_equity = paper.balance
        if paper.can_sell():
            marked_equity += paper.position * candle.close * (1 - paper.fee)
        equity_curve.append(marked_equity)

    sells = [h for h in paper.history if h.get("type") == "SELL"]
    total_pnl = sum(float(h.get("pnl", 0.0)) for h in sells)
    wins = sum(1 for h in sells if float(h.get("pnl", 0.0)) > 0)
    win_rate = (wins / len(sells) * 100.0) if sells else 0.0
    final_equity = equity_curve[-1] if equity_curve else args.initial_balance
    ret_pct = (final_equity - args.initial_balance) / args.initial_balance * 100.0
    mdd = max_drawdown_pct(equity_curve)

    print("===== REPLAY RESULT =====")
    print(f"Rows: {len(df)}")
    print(f"Trades (SELL): {len(sells)}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Return: {ret_pct:.2f}%")
    print(f"Max Drawdown: {mdd:.2f}%")

    if args.export_trades:
        pd.DataFrame(paper.history).to_csv(args.export_trades, index=False)
        print(f"Saved trades: {args.export_trades}")


if __name__ == "__main__":
    main()
