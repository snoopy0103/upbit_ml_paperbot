import argparse
import json
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

MODEL_PATH = "model_champion.pkl"
META_PATH = "model_meta.json"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def split_features_labels(df: pd.DataFrame):
    X = df.drop(columns=["datetime", "label"], errors="ignore")
    y = df["label"]
    return X, y

def train_lightgbm(X, y):
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    best_model = None
    best_auc = float("-inf")
    best_fold = None

    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va)

        model = lgb.train(
            params,
            dtr,
            num_boost_round=3000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        p = model.predict(X_va)
        auc = roc_auc_score(y_va, p)
        aucs.append(auc)
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_fold = fold

        print(f"Fold {fold} AUC={auc:.4f} best_iter={model.best_iteration}")

    mean_auc = sum(aucs) / len(aucs)
    return best_model, mean_auc, best_auc, best_fold

def save_model(model, mean_auc: float, best_auc: float, best_fold: int | None):
    joblib.dump(model, MODEL_PATH)
    meta = {
        "mean_cv_auc": float(mean_auc),
        "best_fold_auc": float(best_auc),
        "best_fold": int(best_fold) if best_fold is not None else None,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()

    df = load_data(args.inp)
    X, y = split_features_labels(df)
    model, mean_auc, best_auc, best_fold = train_lightgbm(X, y)

    print(f"Mean CV AUC: {mean_auc:.4f} | Best fold AUC: {best_auc:.4f} (fold={best_fold})")
    save_model(model, mean_auc, best_auc, best_fold)
    print(f"Saved model: {MODEL_PATH} and meta: {META_PATH}")

if __name__ == "__main__":
    main()
