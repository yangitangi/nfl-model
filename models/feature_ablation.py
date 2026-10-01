"""
models/feature_ablation.py
---------------------------
Walk-forward ablation test for a candidate feature (or set of features):
does adding it actually improve the model, or is it just noise?

Unlike train_model.py's single train/test split (train on everything
before the final season, test on just that one), this walks forward across
the last N seasons one at a time -- train on everything before season T,
test on season T, for each T in the window -- and aggregates results. A
single-season test is noisy enough that a feature can look good or bad by
chance; walking forward across several seasons is the bar every feature in
this project has been held to (see the validation comments on "injuries"
and "qb_specific_form" in config.FEATURES).

A feature is only adopted if it improves MAE, win accuracy, AND ATS hit
rate TOGETHER across the walk-forward window -- not just one or two of the
three. Features that fail this bar stay off by default in config.FEATURES.

Usage:
    python models/feature_ablation.py --flags turnover_epa penalty_epa
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import (
    get_feature_columns, get_margin_target, reconstruct_margin, train_margin_model,
)

N_WALKFORWARD_SEASONS = 5


def _ats_hit_rate(pred_margin: np.ndarray, spread_line: np.ndarray, actual_margin: np.ndarray) -> float:
    """Fraction of games where the model's lean relative to the market
    (pred_margin vs spread_line) landed on the side that actually covered.
    Pushes (actual_margin == spread_line) are excluded, same as a real bet."""
    model_lean = np.sign(pred_margin - spread_line)
    actual_cover = np.sign(actual_margin - spread_line)
    no_push = actual_cover != 0
    model_no_lean = model_lean != 0
    valid = no_push & model_no_lean
    if valid.sum() == 0:
        return float("nan")
    return float((model_lean[valid] == actual_cover[valid]).mean())


def walk_forward_eval(df: pd.DataFrame, feature_flag_overrides: dict) -> dict:
    """Runs the walk-forward loop with the given config.FEATURES overrides
    applied (restored afterward), returns aggregated MAE/win-acc/ATS."""
    original = {k: config.FEATURES[k] for k in feature_flag_overrides}
    config.FEATURES.update(feature_flag_overrides)
    try:
        feature_cols = get_feature_columns(df)
        required = feature_cols + ["actual_margin", "home_win", "spread_line"]
        clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()

        seasons = sorted(clean["season"].unique())
        test_seasons = seasons[-N_WALKFORWARD_SEASONS:]

        all_pred_margin, all_actual_margin, all_spread, all_pred_win, all_actual_win = [], [], [], [], []
        per_season = []
        for test_season in test_seasons:
            train = clean[clean["season"] < test_season]
            test = clean[clean["season"] == test_season]
            if len(train) < 200 or len(test) == 0:
                continue

            model = train_margin_model(train, feature_cols)
            pred_margin = reconstruct_margin(test, model.predict(test[feature_cols]))
            actual_margin = test["actual_margin"].values
            spread = test["spread_line"].values

            mae = mean_absolute_error(actual_margin, pred_margin)
            win_acc = accuracy_score(test["home_win"].values, (pred_margin > 0).astype(int))
            ats = _ats_hit_rate(pred_margin, spread, actual_margin)
            per_season.append({"season": test_season, "n": len(test), "mae": mae, "win_acc": win_acc, "ats": ats})

            all_pred_margin.append(pred_margin)
            all_actual_margin.append(actual_margin)
            all_spread.append(spread)
            all_pred_win.append((pred_margin > 0).astype(int))
            all_actual_win.append(test["home_win"].values)

        pred_margin = np.concatenate(all_pred_margin)
        actual_margin = np.concatenate(all_actual_margin)
        spread = np.concatenate(all_spread)
        pred_win = np.concatenate(all_pred_win)
        actual_win = np.concatenate(all_actual_win)

        return {
            "n_features": len(feature_cols),
            "n_games": len(pred_margin),
            "mae": mean_absolute_error(actual_margin, pred_margin),
            "win_acc": accuracy_score(actual_win, pred_win),
            "ats": _ats_hit_rate(pred_margin, spread, actual_margin),
            "per_season": per_season,
        }
    finally:
        config.FEATURES.update(original)


def run_ablation(flags: list[str]):
    feature_path = config.PROCESSED_DATA_DIR / "game_level_features.parquet"
    df = pd.read_parquet(feature_path)

    baseline_overrides = {f: False for f in flags}
    treatment_overrides = {f: True for f in flags}

    print(f"Testing flags {flags} over the last {N_WALKFORWARD_SEASONS} seasons (walk-forward)...\n")

    baseline = walk_forward_eval(df, baseline_overrides)
    treatment = walk_forward_eval(df, treatment_overrides)

    print(f"{'':20s} {'BASELINE (off)':>16s} {'TREATMENT (on)':>16s} {'DELTA':>10s}")
    print(f"{'Features used':20s} {baseline['n_features']:>16d} {treatment['n_features']:>16d}")
    print(f"{'Games evaluated':20s} {baseline['n_games']:>16d} {treatment['n_games']:>16d}")
    mae_delta = treatment["mae"] - baseline["mae"]
    acc_delta = treatment["win_acc"] - baseline["win_acc"]
    ats_delta = treatment["ats"] - baseline["ats"]
    print(f"{'MAE (lower=better)':20s} {baseline['mae']:>16.3f} {treatment['mae']:>16.3f} {mae_delta:>+10.3f}")
    print(f"{'Win accuracy':20s} {baseline['win_acc']:>15.1%} {treatment['win_acc']:>15.1%} {acc_delta:>+9.1%}")
    print(f"{'ATS hit rate':20s} {baseline['ats']:>15.1%} {treatment['ats']:>15.1%} {ats_delta:>+9.1%}")

    print("\nPer-season breakdown (treatment):")
    for row in treatment["per_season"]:
        print(f"  {row['season']}: n={row['n']:>3d}  MAE={row['mae']:.2f}  WinAcc={row['win_acc']:.1%}  ATS={row['ats']:.1%}")

    improved_all = mae_delta < 0 and acc_delta > 0 and ats_delta > 0
    print(f"\n{'ADOPT' if improved_all else 'REJECT'}: "
          f"{'improves all three metrics together' if improved_all else 'does not improve all three metrics together'}")
    return baseline, treatment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward ablation test for candidate config.FEATURES flags")
    parser.add_argument("--flags", nargs="+", required=True, help="config.FEATURES keys to toggle on for the treatment run")
    args = parser.parse_args()
    run_ablation(args.flags)
