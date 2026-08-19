"""
models/tune_blend_weight.py
----------------------------
Finds config.MARKET_BLEND_WEIGHT for MARGIN_TARGET_MODE="blend" via
walk-forward validation across several past seasons -- deliberately NOT the
true 2025 holdout, which train_model.py's evaluate() reports on. Tuning a
hyperparameter on the same data used for the final honest evaluation would
leak; this is why nfelo describes fitting their own market-weight on
validation data, not on the season they report performance for.

For each validation season Y (2021-2024), trains a raw-margin model on
every season before Y (a fresh walk-forward fold, same discipline as the
production train/test split), then tests every candidate blend weight
against Y's actual results. The weight is picked by average ATS hit rate
across all four folds -- that's literally the thing we're trying to fix,
not just a proxy like MAE.

Run with:
    python models/tune_blend_weight.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import get_feature_columns, train_margin_model

VAL_SEASONS = [2021, 2022, 2023, 2024]
WEIGHT_GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)


def prepare_fold(df: pd.DataFrame, feature_cols: list[str], cutoff_season: int):
    required = feature_cols + ["actual_margin", "home_win", "spread_line"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    train = clean[clean["season"] < cutoff_season]
    val = clean[clean["season"] == cutoff_season]
    return train, val


def ats_hit_rate(pred_margin: np.ndarray, spread_line: np.ndarray, actual_margin: np.ndarray) -> float:
    covered_home = (actual_margin - spread_line) > 0
    leans_home = (pred_margin - spread_line) > 0
    hit = np.where(leans_home, covered_home, ~covered_home)
    return hit.mean()


def main():
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")

    # Train every fold's model in "raw" mode -- an independent from-scratch
    # prediction is what gets blended with the market, not a residual.
    config.MARGIN_TARGET_MODE = "raw"
    feature_cols = get_feature_columns(df)

    fold_data = []
    for y in VAL_SEASONS:
        train, val = prepare_fold(df, feature_cols, y)
        model = train_margin_model(train, feature_cols)
        raw_pred = model.predict(val[feature_cols])
        fold_data.append({
            "season": y,
            "raw_pred": raw_pred,
            "spread_line": val["spread_line"].values,
            "actual_margin": val["actual_margin"].values,
        })
        print(f"Fold {y}: trained on {len(train):,} games, validating on {len(val):,} games")

    print(f"\n{'Weight':<8}{'Avg ATS':<10}{'Avg MAE':<10}  Per-fold ATS ({'/'.join(str(y) for y in VAL_SEASONS)})")
    print("-" * 70)

    best_key = None
    best_w = None
    for w in WEIGHT_GRID:
        ats_list, mae_list = [], []
        for fold in fold_data:
            blended = w * fold["spread_line"] + (1 - w) * fold["raw_pred"]
            ats_list.append(ats_hit_rate(blended, fold["spread_line"], fold["actual_margin"]))
            mae_list.append(np.mean(np.abs(blended - fold["actual_margin"])))
        avg_ats = float(np.mean(ats_list))
        avg_mae = float(np.mean(mae_list))
        per_fold = " / ".join(f"{a:.1%}" for a in ats_list)
        print(f"{w:<8.2f}{avg_ats:<10.1%}{avg_mae:<10.2f}  {per_fold}")

        key = (round(avg_ats, 4), -round(avg_mae, 4))
        if best_key is None or key > best_key:
            best_key = key
            best_w = w

    print(f"\nBest weight by avg ATS hit rate across {len(VAL_SEASONS)} folds: "
          f"{best_w:.2f} (ATS {best_key[0]:.1%}, MAE {-best_key[1]:.2f})")
    print(f"\nUpdate config.py: MARGIN_TARGET_MODE = \"blend\", MARKET_BLEND_WEIGHT = {best_w}")


if __name__ == "__main__":
    main()
