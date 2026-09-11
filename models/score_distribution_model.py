"""
models/score_distribution_model.py
------------------------------------
An alternative to predicting a single margin: two independent Poisson
regressions (XGBoost's count:poisson objective) predicting home_score and
away_score directly, the same idea as the CS scientist's model in spirit --
though hers was presumably a richer joint model; this is a standard,
well-established first cut at "predict goals/points instead of margin"
(the Poisson approach to scorelines is standard in soccer analytics, e.g.
the basis for Dixon-Coles models).

From the two predicted means (lambda_home, lambda_away):
  - predicted_margin = lambda_home - lambda_away
  - P(home win) via the Skellam distribution (the exact distribution of
    the difference of two independent Poisson variables) -- not a
    simulation or approximation.
  - A full scoreline probability grid (P(home=h, away=a) for a range of
    h, a), the same kind of "most likely score: 27-24, 1.1%" output the
    CS scientist's model gave.

Independence between home and away scores is a simplifying assumption --
real models (Dixon-Coles) add a low-score correlation adjustment. Not
attempted here; flagged as a known limitation, not hidden.

Graded with the identical walk-forward panel and ATS/win-accuracy math
used throughout this project, so it's directly comparable to the
production single-margin XGBoost model.

Run with:
    python models/score_distribution_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.fetch_data import fetch_schedules, standardize_team_abbrs
from models.train_model import get_feature_columns, train_margin_model, reconstruct_margin

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

EVAL_SEASONS = [2021, 2022, 2023, 2024, 2025]
MAX_SCORE_GRID = 60  # scoreline probability grid spans 0..MAX_SCORE_GRID for each team


def load_data_with_scores() -> pd.DataFrame:
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")
    schedules = fetch_schedules(max_season=config.END_SEASON)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    df = df.merge(schedules[["game_id", "home_score", "away_score"]], on="game_id", how="left")
    return df


def train_poisson_model(train: pd.DataFrame, feature_cols: list[str], target: str):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("xgboost required for the Poisson score model")
    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_SEED,
    )
    model.fit(train[feature_cols], train[target])
    return model


def most_likely_scorelines(lam_home: float, lam_away: float, top_n: int = 3):
    """Top N (home_score, away_score, probability) combinations under the
    independent-Poisson assumption."""
    hs = np.arange(0, MAX_SCORE_GRID)
    as_ = np.arange(0, MAX_SCORE_GRID)
    p_home = poisson.pmf(hs, lam_home)
    p_away = poisson.pmf(as_, lam_away)
    grid = np.outer(p_home, p_away)  # grid[h, a] = P(home=h, away=a)
    flat_idx = np.argsort(grid.ravel())[::-1][:top_n]
    results = []
    for idx in flat_idx:
        h, a = np.unravel_index(idx, grid.shape)
        results.append((int(h), int(a), grid[h, a]))
    return results


def grade(pred_margin, spread, actual, home_win):
    mae = np.mean(np.abs(pred_margin - actual))
    win_acc = np.mean((pred_margin > 0).astype(int) == home_win)
    covered = (actual - spread) > 0
    leans_home = (pred_margin - spread) > 0
    ats = np.where(leans_home, covered, ~covered).mean()
    return mae, win_acc, ats


def main():
    config.MARGIN_TARGET_MODE = "residual"  # only used to build the comparable feature_cols list
    df = load_data_with_scores()
    feature_cols = get_feature_columns(df)
    required = feature_cols + ["actual_margin", "home_win", "spread_line", "home_score", "away_score"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    poisson_results = {"mae": [], "acc": [], "ats": []}
    margin_results = {"mae": [], "acc": [], "ats": []}

    example_printed = False
    for y in EVAL_SEASONS:
        train = clean[clean["season"] < y]
        val = clean[clean["season"] == y]

        # --- Poisson score model ---
        home_model = train_poisson_model(train, feature_cols, "home_score")
        away_model = train_poisson_model(train, feature_cols, "away_score")
        lam_home = home_model.predict(val[feature_cols])
        lam_away = away_model.predict(val[feature_cols])
        lam_home = np.clip(lam_home, 0.1, None)  # Poisson mean must be positive
        lam_away = np.clip(lam_away, 0.1, None)
        pred_margin_poisson = lam_home - lam_away

        mae, acc, ats = grade(pred_margin_poisson, val["spread_line"].values,
                               val["actual_margin"].values, val["home_win"].values)
        poisson_results["mae"].append(mae); poisson_results["acc"].append(acc); poisson_results["ats"].append(ats)

        # --- Current production: single margin regression (residual mode) ---
        margin_model = train_margin_model(train, feature_cols)
        pred_margin_reg = reconstruct_margin(val, margin_model.predict(val[feature_cols]))
        mae2, acc2, ats2 = grade(pred_margin_reg, val["spread_line"].values,
                                  val["actual_margin"].values, val["home_win"].values)
        margin_results["mae"].append(mae2); margin_results["acc"].append(acc2); margin_results["ats"].append(ats2)

        print(f"{y}: Poisson  MAE={mae:.2f} WinAcc={acc:.1%} ATS={ats:.1%}   |   "
              f"Margin-reg  MAE={mae2:.2f} WinAcc={acc2:.1%} ATS={ats2:.1%}")

        if not example_printed:
            # Show one worked example: predicted scoreline distribution for
            # the first game in this fold, same style as "most likely score."
            row = val.iloc[0]
            top = most_likely_scorelines(lam_home[0], lam_away[0])
            print(f"\n  Example -- {row['away_team']} @ {row['home_team']} ({y}):")
            print(f"    Predicted mean score: {row['home_team']} {lam_home[0]:.1f} - "
                  f"{row['away_team']} {lam_away[0]:.1f}")
            print(f"    P(home win) via Skellam: {skellam.sf(0, lam_home[0], lam_away[0]):.1%}")
            for h, a, p in top:
                print(f"    Most likely: {row['home_team']} {h}-{a} {row['away_team']}  ({p:.1%})")
            print()
            example_printed = True

    print("\n=== SUMMARY (avg over 2021-2025) ===")
    print(f"{'Model':<20}{'MAE':<8}{'WinAcc':<10}{'ATS':<8}")
    print(f"{'Poisson scores':<20}{np.mean(poisson_results['mae']):<8.2f}"
          f"{np.mean(poisson_results['acc']):<10.1%}{np.mean(poisson_results['ats']):<8.1%}")
    print(f"{'Margin regression':<20}{np.mean(margin_results['mae']):<8.2f}"
          f"{np.mean(margin_results['acc']):<10.1%}{np.mean(margin_results['ats']):<8.1%}")


if __name__ == "__main__":
    main()
