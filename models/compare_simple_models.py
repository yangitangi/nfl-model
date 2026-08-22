"""
models/compare_simple_models.py
---------------------------------
Tests whether a much simpler model beats our 30-feature XGBoost, the same
lesson a La Liga soccer model builder found: a model using only goals
scored/conceded outperformed a fancier one using team ratings, form, and
home/away splits. Three candidates, all graded with the identical
walk-forward methodology (2021-2025, five independent train/test folds,
never touching future data) and the identical ATS/win-accuracy math used
everywhere else in this project:

  1. XGBoost (current production, residual-vs-spread mode) -- the bar to beat.
  2. Weighted offense/defense form -- exponentially time-decayed average of
     each team's own points scored and points allowed (all history, not
     just a 5-game window), analogous to the soccer model's "vatt"/"vdeff".
     Predicted home score = avg(home's scoring form, away's allowed form);
     mirror for away. No ML, no tree model -- just two weighted averages
     and a fitted home-field constant.
  3. Elo rating -- FiveThirtyEight's public NFL Elo formula (source-
     verified constants: HFA=65 Elo pts, K=20, 1/3 reversion to the mean
     at each season boundary, MOV dampening multiplier), converted to a
     point margin via the standard elo_diff/25 rule of thumb. Sequential
     by construction, so every prediction only ever uses games that
     already happened -- no separate walk-forward machinery needed.

Run with:
    python models/compare_simple_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.fetch_data import fetch_schedules, standardize_team_abbrs
from models.train_model import get_feature_columns, train_margin_model, reconstruct_margin

EVAL_SEASONS = [2021, 2022, 2023, 2024, 2025]

# ---------------------------------------------------------------------------
# Shared grading (identical to tracking/prediction_tracker.py's methodology)
# ---------------------------------------------------------------------------
def grade(pred_margin: np.ndarray, spread_line: np.ndarray, actual_margin: np.ndarray, home_win: np.ndarray) -> dict:
    valid = ~np.isnan(pred_margin) & ~np.isnan(spread_line) & ~np.isnan(actual_margin)
    pred_margin, spread_line, actual_margin, home_win = (
        pred_margin[valid], spread_line[valid], actual_margin[valid], home_win[valid])

    mae = np.mean(np.abs(pred_margin - actual_margin))
    win_acc = np.mean((pred_margin > 0).astype(int) == home_win)

    covered_home = (actual_margin - spread_line) > 0
    leans_home = (pred_margin - spread_line) > 0
    ats_hit = np.where(leans_home, covered_home, ~covered_home).mean()

    return {"n": int(valid.sum()), "mae": mae, "win_acc": win_acc, "ats": ats_hit}


# ---------------------------------------------------------------------------
# Model 1: XGBoost production (reuses existing training code exactly)
# ---------------------------------------------------------------------------
def xgboost_predictions() -> pd.DataFrame:
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")
    config.MARGIN_TARGET_MODE = "residual"
    feature_cols = get_feature_columns(df)
    required = feature_cols + ["actual_margin", "home_win", "spread_line"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    rows = []
    for y in EVAL_SEASONS:
        train = clean[clean["season"] < y]
        val = clean[clean["season"] == y]
        model = train_margin_model(train, feature_cols)
        pred = reconstruct_margin(val, model.predict(val[feature_cols]))
        rows.append(pd.DataFrame({
            "season": y, "pred_margin": pred,
            "spread_line": val["spread_line"].values,
            "actual_margin": val["actual_margin"].values,
            "home_win": val["home_win"].values,
        }))
        print(f"  [xgboost] fold {y}: trained on {len(train):,}, predicted {len(val):,}")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Model 2: weighted offense/defense form (no ML)
# ---------------------------------------------------------------------------
def build_long_schedule() -> pd.DataFrame:
    schedules = fetch_schedules(max_season=config.END_SEASON)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    schedules = schedules[(schedules["game_type"] == "REG") & schedules["home_score"].notna()].copy()
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    cols = ["game_id", "season", "week", "gameday", "spread_line"]
    home = schedules[cols + ["home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "away_team": "opp", "home_score": "team_score", "away_score": "opp_score"})
    away = schedules[cols + ["home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"away_team": "team", "home_team": "opp", "away_score": "team_score", "home_score": "opp_score"})
    long_df = pd.concat([home, away], ignore_index=True)
    return schedules, long_df


def weighted_form_predictions(half_life_days: int = 365) -> pd.DataFrame:
    schedules, long_df = build_long_schedule()

    def per_team_form(g):
        team_name = g.name
        g = g.sort_values("gameday").set_index("gameday")
        scored = g["team_score"].shift(1)
        allowed = g["opp_score"].shift(1)
        g["form_scored"] = scored.ewm(halflife=pd.Timedelta(days=half_life_days), times=g.index, min_periods=3).mean()
        g["form_allowed"] = allowed.ewm(halflife=pd.Timedelta(days=half_life_days), times=g.index, min_periods=3).mean()
        g = g.reset_index()
        g["team"] = team_name
        return g

    long_df = long_df.groupby("team", group_keys=False).apply(per_team_form)

    home_form = long_df.merge(
        schedules[["game_id", "home_team"]], left_on=["game_id", "team"], right_on=["game_id", "home_team"]
    )[["game_id", "form_scored", "form_allowed"]].rename(
        columns={"form_scored": "home_form_scored", "form_allowed": "home_form_allowed"})
    away_form = long_df.merge(
        schedules[["game_id", "away_team"]], left_on=["game_id", "team"], right_on=["game_id", "away_team"]
    )[["game_id", "form_scored", "form_allowed"]].rename(
        columns={"form_scored": "away_form_scored", "form_allowed": "away_form_allowed"})

    game_level = schedules.merge(home_form, on="game_id").merge(away_form, on="game_id")
    game_level["actual_margin"] = game_level["home_score"] - game_level["away_score"]
    game_level["home_win"] = (game_level["actual_margin"] > 0).astype(int)
    game_level["pred_home_score"] = (game_level["home_form_scored"] + game_level["away_form_allowed"]) / 2
    game_level["pred_away_score"] = (game_level["away_form_scored"] + game_level["home_form_allowed"]) / 2
    game_level["pred_margin_raw"] = game_level["pred_home_score"] - game_level["pred_away_score"]

    rows = []
    for y in EVAL_SEASONS:
        train = game_level[game_level["season"] < y].dropna(subset=["pred_margin_raw", "actual_margin"])
        val = game_level[game_level["season"] == y].dropna(subset=["pred_margin_raw"])
        hfa = (train["actual_margin"] - train["pred_margin_raw"]).mean()  # fit HFA on training folds only
        pred = val["pred_margin_raw"].values + hfa
        rows.append(pd.DataFrame({
            "season": y, "pred_margin": pred,
            "spread_line": val["spread_line"].values,
            "actual_margin": val["actual_margin"].values,
            "home_win": val["home_win"].values,
        }))
        print(f"  [weighted-form] fold {y}: HFA fit at {hfa:+.2f} pts, predicted {len(val):,}")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Model 3: Elo (FiveThirtyEight's public NFL formula, source-verified constants)
# ---------------------------------------------------------------------------
def elo_predictions() -> pd.DataFrame:
    schedules, _ = build_long_schedule()
    schedules = schedules.sort_values("gameday").reset_index(drop=True)

    INITIAL = 1500.0
    HFA_ELO = 65.0
    K = 20.0
    REVERSION = 1.0 / 3.0
    MARGIN_SCALE = 25.0

    ratings = {}

    def get(team):
        return ratings.setdefault(team, INITIAL)

    rows = []
    last_season = None
    for _, g in schedules.iterrows():
        if last_season is not None and g["season"] != last_season:
            for t in ratings:
                ratings[t] = ratings[t] * (1 - REVERSION) + INITIAL * REVERSION
        last_season = g["season"]

        home, away = g["home_team"], g["away_team"]
        elo_diff = get(home) + HFA_ELO - get(away)
        pred_margin = elo_diff / MARGIN_SCALE  # this game's prediction, using PRE-game ratings only

        actual_margin = g["home_score"] - g["away_score"]
        result_home = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)
        expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        pd_ = max(abs(actual_margin), 1)
        mov_mult = np.log(pd_ + 1.0) * (2.2 / (1.0 if result_home == 0.5 else
                     ((elo_diff if result_home == 1.0 else -elo_diff) * 0.001 + 2.2)))
        shift = K * mov_mult * (result_home - expected_home)
        ratings[home] = get(home) + shift
        ratings[away] = get(away) - shift

        rows.append({
            "season": g["season"], "pred_margin": pred_margin,
            "spread_line": g["spread_line"], "actual_margin": actual_margin,
            "home_win": int(actual_margin > 0),
        })

    all_preds = pd.DataFrame(rows)
    return all_preds[all_preds["season"].isin(EVAL_SEASONS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_comparison(name: str, preds: pd.DataFrame):
    print(f"\n=== {name} ===")
    per_season = {}
    for y in EVAL_SEASONS:
        g = preds[preds["season"] == y]
        r = grade(g["pred_margin"].values, g["spread_line"].values, g["actual_margin"].values, g["home_win"].values)
        per_season[y] = r
        print(f"  {y}: n={r['n']:<4} MAE={r['mae']:.2f}  WinAcc={r['win_acc']:.1%}  ATS={r['ats']:.1%}")
    overall = grade(preds["pred_margin"].values, preds["spread_line"].values,
                     preds["actual_margin"].values, preds["home_win"].values)
    print(f"  ALL {EVAL_SEASONS[0]}-{EVAL_SEASONS[-1]}: n={overall['n']:<4} MAE={overall['mae']:.2f}  "
          f"WinAcc={overall['win_acc']:.1%}  ATS={overall['ats']:.1%}")
    return overall


def main():
    print("Building predictions for each model across folds", EVAL_SEASONS, "...")

    print("\n--- XGBoost (production, residual mode) ---")
    xgb_preds = xgboost_predictions()

    print("\n--- Weighted offense/defense form (no ML) ---")
    form_preds = weighted_form_predictions()

    print("\n--- Elo (538 formula) ---")
    elo_preds = elo_predictions()

    print("\n\n" + "=" * 70)
    print("COMPARISON: identical folds, identical grading")
    print("=" * 70)
    results = {}
    results["XGBoost"] = print_comparison("XGBoost (production, residual)", xgb_preds)
    results["Weighted form"] = print_comparison("Weighted offense/defense form", form_preds)
    results["Elo"] = print_comparison("Elo (538 formula)", elo_preds)

    # Vegas benchmark for reference
    veg = xgb_preds.copy()
    veg_result = grade(veg["spread_line"].values, veg["spread_line"].values,
                        veg["actual_margin"].values, veg["home_win"].values)
    print(f"\n  (Vegas closing line win accuracy for reference: {veg_result['win_acc']:.1%} "
          f"-- ATS is undefined for the market grading against itself)")

    print(f"\n{'Model':<16}{'MAE':<8}{'WinAcc':<10}{'ATS':<8}")
    for name, r in results.items():
        print(f"{name:<16}{r['mae']:<8.2f}{r['win_acc']:<10.1%}{r['ats']:<8.1%}")


if __name__ == "__main__":
    main()
