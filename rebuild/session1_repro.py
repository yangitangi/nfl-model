"""
rebuild/session1_repro.py
-------------------------
Cloud session 1: reproduce the production XGBoost baseline on the DEV window
with the rebuild protocol. Walk-forward over DEV_TEST_SEASONS: for each test
season T, train on every season < T in data/processed/game_level_features.parquet
using the production functions in models/train_model.py (unchanged), predict T,
then score all DEV predictions with rebuild.protocol.evaluate.

Holdout seasons (2024-2025) are never used as test seasons here, and since
training uses only seasons < T <= 2023 they are never used for training either.

    python rebuild/session1_repro.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.train_model import get_feature_columns, reconstruct_margin, train_margin_model  # noqa: E402
from rebuild.protocol import DEV_TEST_SEASONS, evaluate  # noqa: E402


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data/processed/game_level_features.parquet")
    feature_cols = get_feature_columns(df)
    # same row filter as prepare_data(): drop rows missing features or targets
    required = [c for c in feature_cols + ["actual_margin", "home_win"] if c in df.columns]
    clean = df.dropna(subset=required).copy()

    parts = []
    for T in DEV_TEST_SEASONS:
        train = clean[clean["season"] < T]
        test = clean[clean["season"] == T]
        model = train_margin_model(train, feature_cols)
        raw = model.predict(test[feature_cols])
        out = test[["game_id", "season", "week", "spread_line", "actual_margin"]].copy()
        out["pred_margin"] = reconstruct_margin(test, raw)
        parts.append(out)
        print(f"T={T}: train {len(train):,}  test {len(test):,}")
    preds = pd.concat(parts, ignore_index=True)

    res = evaluate(preds, "xgb_repro", seasons=DEV_TEST_SEASONS)
    res["n_features"] = len(feature_cols)
    res["seconds"] = round(time.time() - t0, 1)
    print(f"\n{len(feature_cols)} features, {res['seconds']}s")
    out = ROOT / "rebuild/session1_repro.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
