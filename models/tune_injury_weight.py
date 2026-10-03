"""
models/tune_injury_weight.py
------------------------------
Tests whether PARTIALLY weighting injury_burden's contribution to the
final prediction beats using it at full strength or not at all.

Two transform-based fixes (bucketing, week-dependent smoothing -- see
config.FEATURES["injuries_bucketed"]/["injuries_smoothed_early"]) both
landed at 2-of-3 metrics improved, not a clean pass. This tests a
different lever: not changing the FEATURE's values, but dampening how
much the MODEL'S FINAL PREDICTION depends on it.

Simply scaling injury_burden's raw values doesn't do this -- a tree
splits on thresholds, so rescaling a feature just rescales the threshold
numbers, not which rows land on which side of each split (no functional
change to the model at all). The only way to actually dampen a feature's
NET effect on the final prediction is at the ensemble level: train one
model WITH injury_burden and one WITHOUT, then blend their predictions
with a tunable weight alpha:

    final = alpha * pred_with_injury_burden + (1 - alpha) * pred_without

alpha=1.0 is today's production behavior (full weight). alpha=0.0 is
dropping the feature entirely. Anything in between is a genuine "turn
the dial down" -- this sweeps the full grid via the same 5-season
walk-forward discipline as every other test in this project, to see if
some middle value beats BOTH endpoints on MAE, win accuracy, AND ATS.

Run with:
    python models/tune_injury_weight.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import get_feature_columns, train_margin_model, reconstruct_margin
from models.feature_ablation import _ats_hit_rate, N_WALKFORWARD_SEASONS

ALPHA_GRID = np.round(np.arange(0.0, 1.001, 0.1), 2)


def run():
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")

    config.FEATURES["injuries"] = True
    feature_cols_with = get_feature_columns(df)
    config.FEATURES["injuries"] = False
    feature_cols_without = get_feature_columns(df)
    config.FEATURES["injuries"] = True  # restore default

    required = feature_cols_with + ["actual_margin", "home_win", "spread_line"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    seasons = sorted(clean["season"].unique())
    test_seasons = seasons[-N_WALKFORWARD_SEASONS:]

    fold_data = []
    for y in test_seasons:
        train = clean[clean["season"] < y]
        test = clean[clean["season"] == y]
        if len(train) < 200 or len(test) == 0:
            continue

        model_with = train_margin_model(train, feature_cols_with)
        model_without = train_margin_model(train, feature_cols_without)
        pred_with = reconstruct_margin(test, model_with.predict(test[feature_cols_with]))
        pred_without = reconstruct_margin(test, model_without.predict(test[feature_cols_without]))

        fold_data.append({
            "season": y, "pred_with": pred_with, "pred_without": pred_without,
            "actual_margin": test["actual_margin"].values,
            "spread_line": test["spread_line"].values,
            "home_win": test["home_win"].values,
        })
        print(f"Fold {y}: trained on {len(train):,} games, validating on {len(test):,} games")

    print(f"\n{'Alpha':<8}{'MAE':<10}{'WinAcc':<10}{'ATS':<10}")
    print("-" * 38)

    results = []
    for alpha in ALPHA_GRID:
        all_pred, all_actual, all_spread, all_win = [], [], [], []
        for fold in fold_data:
            blended = alpha * fold["pred_with"] + (1 - alpha) * fold["pred_without"]
            all_pred.append(blended)
            all_actual.append(fold["actual_margin"])
            all_spread.append(fold["spread_line"])
            all_win.append(fold["home_win"])
        pred = np.concatenate(all_pred)
        actual = np.concatenate(all_actual)
        spread = np.concatenate(all_spread)
        win = np.concatenate(all_win)

        mae = mean_absolute_error(actual, pred)
        win_acc = accuracy_score(win, (pred > 0).astype(int))
        ats = _ats_hit_rate(pred, spread, actual)
        results.append({"alpha": alpha, "mae": mae, "win_acc": win_acc, "ats": ats})
        print(f"{alpha:<8.1f}{mae:<10.3f}{win_acc:<10.1%}{ats:<10.1%}")

    full = results[-1]   # alpha=1.0, today's production behavior
    none = results[0]    # alpha=0.0, feature dropped entirely
    print(f"\nFull weight (alpha=1.0, today's behavior): MAE {full['mae']:.3f}  WinAcc {full['win_acc']:.1%}  ATS {full['ats']:.1%}")
    print(f"No weight   (alpha=0.0, feature dropped):   MAE {none['mae']:.3f}  WinAcc {none['win_acc']:.1%}  ATS {none['ats']:.1%}")

    winners = [r for r in results[1:-1]
               if r["mae"] < full["mae"] and r["mae"] < none["mae"]
               and r["win_acc"] > full["win_acc"] and r["win_acc"] > none["win_acc"]
               and r["ats"] > full["ats"] and r["ats"] > none["ats"]]
    if winners:
        print(f"\nFound {len(winners)} alpha(s) that beat BOTH endpoints on all three metrics: "
              f"{[r['alpha'] for r in winners]}")
    else:
        print("\nNo alpha beats both endpoints on all three metrics together.")


if __name__ == "__main__":
    run()
