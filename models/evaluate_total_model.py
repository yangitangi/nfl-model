"""
models/evaluate_total_model.py
--------------------------------
Walk-forward validation for the total-points model (see train_total_model
in train_model.py) across the last N seasons, same discipline as
feature_ablation.py -- a single holdout season (what train_model.py's main
pipeline reports) can be a fluke; walking forward checks it's not.

Usage:
    python models/evaluate_total_model.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import get_feature_columns, reconstruct_total, train_total_model

N_WALKFORWARD_SEASONS = 5


def run():
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")
    feature_cols = get_feature_columns(df)
    required = feature_cols + ["actual_total", "total_line"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    seasons = sorted(clean["season"].unique())
    test_seasons = seasons[-N_WALKFORWARD_SEASONS:]

    print(f"{'Season':<8}{'N':>6}{'Model MAE':>12}{'Vegas MAE':>12}{'Delta':>10}")
    model_maes, vegas_maes, ns = [], [], []
    for test_season in test_seasons:
        train = clean[clean["season"] < test_season]
        test = clean[clean["season"] == test_season]
        if len(train) < 200 or len(test) == 0:
            continue

        model = train_total_model(train, feature_cols)
        pred_total = reconstruct_total(test, model.predict(test[feature_cols]))
        actual_total = test["actual_total"].values
        vegas_total = test["total_line"].values

        model_mae = mean_absolute_error(actual_total, pred_total)
        vegas_mae = mean_absolute_error(actual_total, vegas_total)
        delta = model_mae - vegas_mae
        print(f"{test_season:<8}{len(test):>6}{model_mae:>12.2f}{vegas_mae:>12.2f}{delta:>+10.2f}")

        model_maes.append(model_mae * len(test))
        vegas_maes.append(vegas_mae * len(test))
        ns.append(len(test))

    n_total = sum(ns)
    overall_model_mae = sum(model_maes) / n_total
    overall_vegas_mae = sum(vegas_maes) / n_total
    print(f"\n{'OVERALL':<8}{n_total:>6}{overall_model_mae:>12.2f}{overall_vegas_mae:>12.2f}"
          f"{overall_model_mae - overall_vegas_mae:>+10.2f}")


if __name__ == "__main__":
    run()
