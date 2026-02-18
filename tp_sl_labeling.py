import argparse
import pandas as pd
import numpy as np

def create_tp_sl_labels(
    df: pd.DataFrame,
    tp: float = 0.015,
    sl: float = 0.009,
    max_holding: int = 60,
) -> pd.DataFrame:
    df = df.copy()

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    labels = np.full(len(df), np.nan)

    for i in range(len(df) - max_holding - 1):
        entry = closes[i]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)

        future_highs = highs[i + 1 : i + 1 + max_holding]
        future_lows = lows[i + 1 : i + 1 + max_holding]

        tp_hit = np.where(future_highs >= tp_price)[0]
        sl_hit = np.where(future_lows <= sl_price)[0]

        tp_first = int(tp_hit[0]) if len(tp_hit) else None
        sl_first = int(sl_hit[0]) if len(sl_hit) else None

        if tp_first is None and sl_first is None:
            continue

        # Tie-break rule: if both occur on the same future candle, treat as SL first.
        if sl_first is None or (tp_first is not None and tp_first < sl_first):
            labels[i] = 1
        else:
            labels[i] = 0

    df["label"] = labels
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--tp", type=float, default=0.015)
    ap.add_argument("--sl", type=float, default=0.009)
    ap.add_argument("--max_holding", type=int, default=60)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df["datetime"] = pd.to_datetime(df["datetime"])
    labeled = create_tp_sl_labels(df, tp=args.tp, sl=args.sl, max_holding=args.max_holding)
    labeled.to_csv(args.out, index=False)
    print(f"Saved: {args.out} | rows={len(labeled)}")

if __name__ == "__main__":
    main()
