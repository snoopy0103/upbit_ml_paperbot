import argparse
import pandas as pd
import numpy as np

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in [5, 10, 20, 60, 120]:
        df[f"ma_{window}"] = df["close"].rolling(window).mean()
        df[f"ma_slope_{window}"] = df[f"ma_{window}"].diff()

    # strict alignment (binary)
    df["ma_alignment"] = (
        (df["ma_20"] > df["ma_60"]) &
        (df["ma_60"] > df["ma_120"])
    ).astype(int)

    # softer alignment score (0, 0.5, 1.0)
    df["ma_alignment_score"] = (
        (df["ma_20"] > df["ma_60"]).astype(int) +
        (df["ma_60"] > df["ma_120"]).astype(int)
    ) / 2.0

    # distances
    df["dist_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["dist_ma60"] = (df["close"] - df["ma_60"]) / df["ma_60"]
    return df

def add_pullback_features(df: pd.DataFrame) -> pd.DataFrame:
    rolling_high20 = df["high"].rolling(20).max()
    rolling_low20 = df["low"].rolling(20).min()
    df["pullback_depth20"] = (rolling_high20 - df["close"]) / rolling_high20
    df["range_pos20"] = (df["close"] - rolling_low20) / (rolling_high20 - rolling_low20 + 1e-9)

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_10"] = df["close"].pct_change(10)

    df["rebound_strength3"] = df["ret_3"]
    df["rebound_strength10"] = df["ret_10"]

    return df

def add_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    high20 = df["high"].rolling(20).max()
    low20 = df["low"].rolling(20).min()
    df["breakout_up20"] = (df["close"] > high20.shift(1)).astype(int)
    df["breakout_down20"] = (df["close"] < low20.shift(1)).astype(int)

    df["range_width20"] = (high20 - low20) / df["close"]
    df["range_width20_chg"] = df["range_width20"].diff()
    df["volatility20"] = df["ret_1"].rolling(20).std()
    df["volatility20_chg"] = df["volatility20"].diff()
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ma60"] = df["volume"].rolling(60).mean()
    df["vol_ratio20"] = df["volume"] / (df["vol_ma20"] + 1e-9)
    df["vol_ratio60"] = df["volume"] / (df["vol_ma60"] + 1e-9)
    df["vol_z20"] = (df["volume"] - df["vol_ma20"]) / (df["volume"].rolling(20).std() + 1e-9)
    return df

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_trend_features(df)
    df = add_pullback_features(df)
    df = add_breakout_features(df)
    df = add_volume_features(df)
    df = add_momentum_features(df)

    # drop rows with NA from rolling calcs
    df = df.dropna().reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df["datetime"] = pd.to_datetime(df["datetime"])
    feat = generate_features(df)
    feat.to_csv(args.out, index=False)
    print(f"Saved: {args.out} | rows={len(feat)} | cols={len(feat.columns)}")

if __name__ == "__main__":
    main()
