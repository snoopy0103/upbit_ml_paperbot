# Upbit ML Paper Trading Bot (5m, WebSocket)

This project implements:
- 5m candle building from Upbit WebSocket trades (asyncio)
- Feature engineering (trend/pullback/breakout/volume/momentum)
- TP/SL-first labeling (max holding = 60 candles)
- LightGBM time-series training (walk-forward CV)
- Paper trading engine + risk engine + position sizing + trade guard
- Optional performance dashboard (matplotlib)

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 1) Collect 1y of 5m data (per market)
```bash
python upbit_ohlcv_collector.py --market KRW-BTC
```

## 2) Generate features
```bash
python feature_engineering.py --in data_KRW_BTC_5m.csv --out features_KRW_BTC_5m.csv
```

## 3) Label TP/SL-first (example TP=1.5%, SL=0.9%, max_holding=60)
```bash
python tp_sl_labeling.py --in features_KRW_BTC_5m.csv --out labeled_KRW_BTC_5m.csv --tp 0.015 --sl 0.009 --max_holding 60
```

## 4) Train LightGBM (saves model_champion.pkl)
```bash
python train_lightgbm.py --in labeled_KRW_BTC_5m.csv
```

## 5) Run real-time paper bot (requires model_champion.pkl)
```bash
python realtime_trader.py
```

## Notes
- This is **paper trading** only. No live orders are placed.
- Start with KRW-BTC only. Multi-coin expansion can be added next.
