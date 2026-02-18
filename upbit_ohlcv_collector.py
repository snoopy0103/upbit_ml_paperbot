import argparse
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

BASE_URL = "https://api.upbit.com/v1/candles/minutes/5"

def fetch_ohlcv(market: str, to: str | None = None, count: int = 200) -> pd.DataFrame:
    params = {"market": market, "count": count}
    if to:
        params["to"] = to

    r = requests.get(BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = df[[
        "candle_date_time_kst",
        "opening_price",
        "high_price",
        "low_price",
        "trade_price",
        "candle_acc_trade_volume",
    ]]
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    # Upbit candle_date_time_kst is KST naive text. Convert to UTC-aware timestamps.
    df["datetime"] = pd.to_datetime(df["datetime"] + "+09:00", utc=True)
    return df.sort_values("datetime")

def collect_last_year(market: str, days: int = 365) -> pd.DataFrame:
    all_dfs = []
    to_time = None
    end_time = datetime.now(timezone.utc)
    start_limit = end_time - timedelta(days=days)

    while True:
        df = fetch_ohlcv(market, to=to_time, count=200)
        if df.empty:
            break

        all_dfs.append(df)

        oldest_time = df["datetime"].min()
        if oldest_time < start_limit:
            break

        # Upbit expects `to` in KST text.
        oldest_kst = oldest_time.tz_convert(timezone(timedelta(hours=9)))
        to_time = oldest_kst.strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(0.12)  # basic rate-limit safety

    full_df = pd.concat(all_dfs).drop_duplicates().sort_values("datetime")
    full_df = full_df[full_df["datetime"] >= start_limit]
    return full_df.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="KRW-BTC")
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()

    print(f"Collecting ~{args.days} days of 5m OHLCV for {args.market} ...")
    df = collect_last_year(args.market, days=args.days)
    out = f"data_{args.market.replace('-', '_')}_5m.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out} | rows={len(df)}")

if __name__ == "__main__":
    main()
