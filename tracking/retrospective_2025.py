"""
tracking/retrospective_2025.py
-------------------------------
Backtest: apply the currently deployed model (trained on 2010-2024) to
EVERY 2025 game -- regular season AND playoffs -- and grade every single
one against the real result. Unlike the live weekly tracker, this uses
the actual final market lines (not an open/close snapshot) and doesn't
need results graded incrementally, since 2025 is already fully known.

Produces:
  - outputs/retrospective_2025.csv -- every game, full detail
  - A summary printed to console: overall + week-by-week + edge-bucket
    breakdowns, same grading methodology as tracking/prediction_tracker.py

Run with:
    python tracking/retrospective_2025.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.build_features import build_playoff_features
from data.fetch_data import fetch_schedules, standardize_team_abbrs
from models.predict import load_model_artifacts, predict, add_market_edges, format_spread, format_ml

SEASON = 2025


def build_full_season(season: int = SEASON) -> tuple[pd.DataFrame, list[str]]:
    """
    Combines the regular season (from the processed training table -- its
    rolling features are already the correct week-by-week walk-forward
    values) with playoffs (freshly built, frozen end-of-regular-season
    snapshot -- see build_playoff_features) into one game-level dataframe.
    Real moneyline/odds/scores are merged in for the regular season rows
    too, since the training table doesn't carry them (never used as a
    model feature, only spread_line/total_line are).
    """
    _, _, feature_cols = load_model_artifacts()

    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")
    reg = df[df["season"] == season].copy()

    schedules = fetch_schedules(max_season=season)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    odds_cols = ["game_id", "home_spread_odds", "away_spread_odds",
                 "home_moneyline", "away_moneyline", "over_odds", "under_odds",
                 "home_score", "away_score"]
    reg = reg.merge(schedules[odds_cols], on="game_id", how="left")

    print(f"Regular season: {len(reg)} games")
    playoffs = build_playoff_features(season)
    print(f"Playoffs: {len(playoffs)} games")

    shared_cols = [c for c in reg.columns if c in playoffs.columns]
    combined = pd.concat([reg[shared_cols], playoffs[shared_cols]], ignore_index=True)
    combined = combined.sort_values(["week", "gameday"]).reset_index(drop=True)
    return combined, feature_cols


def grade(df: pd.DataFrame) -> pd.DataFrame:
    """Adds home_win, spread_hit (did the side the model leaned toward vs.
    the actual spread_line cover?), and ml_hit (did the side the model
    favored vs. the fair/vig-free moneyline actually win?) -- same
    methodology as tracking/prediction_tracker.py's _add_grading_columns,
    just against the real, final market line instead of an open snapshot."""
    df = df.copy()
    df["home_win"] = (df["actual_winner"] == "home").astype(int)

    covered_home = (df["actual_margin"] - df["spread_line"]) > 0
    model_leans_home = df["spread_edge"] > 0
    df["spread_hit"] = np.where(model_leans_home, covered_home, ~covered_home)

    has_ml = df["ml_edge"].notna()
    ml_leans_home = df["ml_edge"] > 0
    ml_hit = np.where(ml_leans_home, df["home_win"] == 1, df["home_win"] == 0)
    df["ml_hit"] = np.where(has_ml, ml_hit, np.nan)

    return df


def print_summary(df: pd.DataFrame):
    n = len(df)
    mae = (df["pred_margin"] - df["actual_margin"]).abs().mean()
    vegas_mae = (df["spread_line"] - df["actual_margin"]).abs().mean()
    model_win_acc = ((df["pred_margin"] > 0).astype(int) == df["home_win"]).mean()
    vegas_win_acc = ((df["spread_line"] > 0).astype(int) == df["home_win"]).mean()

    print(f"\n=== {SEASON} SEASON RETROSPECTIVE: {n} games (272 regular season + 13 playoff) ===\n")
    print(f"Model MAE:            {mae:.2f} pts")
    print(f"Vegas MAE:             {vegas_mae:.2f} pts")
    print(f"Model win accuracy:    {model_win_acc:.1%}")
    print(f"Vegas win accuracy:    {vegas_win_acc:.1%}")

    print(f"\n--- SPREAD performance (ATS) ---")
    print(f"Following the model's lean vs. the actual spread: {df['spread_hit'].mean():.1%} ({n} games)")
    df["_edge_bucket"] = pd.cut(
        df["spread_edge"].abs(), bins=[0, 1, 2, 4, np.inf],
        labels=["0-1 pt", "1-2 pt", "2-4 pt", "4+ pt"],
    )
    bucket_stats = df.groupby("_edge_bucket", observed=True)["spread_hit"].agg(["mean", "count"])
    for label, row in bucket_stats.iterrows():
        print(f"  {label:8s}  hit rate {row['mean']:.1%}  (n={int(row['count'])})")

    has_ml = df["ml_hit"].notna()
    print(f"\n--- MONEYLINE performance ---")
    print(f"Model's favored side (vs. fair market) won {df['ml_hit'].mean():.1%} of the time "
          f"({int(has_ml.sum())} games)")

    print("\n--- Week-by-week ---")
    header = f"{'Week':<6}{'Type':<6}{'N':<5}{'Model MAE':<12}{'Spread ATS':<13}{'Moneyline':<14}"
    print(header)
    print("-" * len(header))
    for wk, g in sorted(df.groupby("week")):
        mae_wk = (g["pred_margin"] - g["actual_margin"]).abs().mean()
        gtype = g["game_type"].iloc[0]
        ml_n = int(g["ml_hit"].notna().sum())
        ml_str = f"{g['ml_hit'].mean():.0%} (n={ml_n})" if ml_n else "n/a"
        print(f"{wk:<6}{gtype:<6}{len(g):<5}{mae_wk:<12.2f}{g['spread_hit'].mean():<13.1%}{ml_str:<14}")

    print("\n--- Other cuts worth a look ---")
    for label, mask in [
        ("Home favorite (spread_line > 0)", df["spread_line"] > 0),
        ("Away favorite (spread_line < 0)", df["spread_line"] < 0),
        ("Division games", df["div_game"] == 1),
        ("Non-division games", df["div_game"] == 0),
        ("Dome/indoor", df["is_outdoor"] == 0),
        ("Outdoor", df["is_outdoor"] == 1),
        ("Playoffs", df["game_type"] != "REG"),
        ("Regular season", df["game_type"] == "REG"),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        print(f"  {label:32s}  ATS {sub['spread_hit'].mean():.1%}  "
              f"ML {sub['ml_hit'].mean():.1%}  MAE {(sub['pred_margin']-sub['actual_margin']).abs().mean():.2f}  (n={len(sub)})")


def main():
    combined, feature_cols = build_full_season()
    predicted = predict(combined, feature_cols)
    predicted = add_market_edges(predicted)
    graded = grade(predicted)

    print_summary(graded)

    out = graded.copy()
    out["model_spread_display"] = out.apply(
        lambda r: format_spread(r["home_team"], r["away_team"], r["pred_margin"]), axis=1)
    out["vegas_spread_display"] = out.apply(
        lambda r: format_spread(r["home_team"], r["away_team"], r["spread_line"]), axis=1)

    export_cols = [
        "game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "div_game",
        "vegas_spread_display", "model_spread_display", "spread_line", "pred_margin", "spread_edge", "spread_hit",
        "home_moneyline", "away_moneyline", "fair_home_ml_prob", "pred_home_win_prob", "ml_edge", "ml_hit",
        "total_line", "home_score", "away_score", "actual_margin", "actual_winner",
        "is_outdoor", "temp", "wind", "home_rest_days", "away_rest_days",
    ]
    out_path = config.OUTPUTS_DIR / "retrospective_2025.csv"
    out[export_cols].to_csv(out_path, index=False)
    print(f"\n[saved] {len(out)} games -> {out_path}")


if __name__ == "__main__":
    main()
