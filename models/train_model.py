"""
train_model.py
---------------
Trains a model on the game-level feature table and evaluates it with
walk-forward validation (train on past seasons, test on the most recent
complete season) — never a random split, which would leak future
information into training.

Critically, this also benchmarks the model against the Vegas closing line.
A model that can't match or beat that benchmark isn't adding predictive
value yet — see the docstring on evaluate() for why this matters.

Run directly to train + evaluate + save the model:
    python models/train_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Picks model input columns from the processed table, gated by
    config.FEATURES so flags actually control what the model sees (rather
    than just documenting intent) — kept separate so you can turn a feature
    family on/off and see how much it's actually contributing.

    Rolling stat columns (home_/away_ + _rollN) are matched to a flag via
    config.FEATURE_COLUMN_GROUPS; any rolling column with no entry there
    (e.g. point_margin_roll5) is always included as a core signal.
    """
    roll_suffix = f"_roll{config.ROLLING_WINDOW_GAMES}"
    all_roll_cols = [c for c in df.columns if c.endswith(roll_suffix)]

    def base_stat_name(col: str) -> str:
        name = col[len("home_"):] if col.startswith("home_") else col[len("away_"):]
        return name[: -len(roll_suffix)]

    feature_cols = []
    for c in all_roll_cols:
        group = config.FEATURE_COLUMN_GROUPS.get(base_stat_name(c))
        if group is None or config.FEATURES.get(group, True):
            feature_cols.append(c)

    if config.FEATURES.get("rest_travel", False):
        feature_cols += ["home_rest_days", "away_rest_days", "div_game",
                          "home_games_played_this_season", "away_games_played_this_season"]

    if config.FEATURES.get("market", False):
        feature_cols += ["spread_line", "total_line"]

    if config.FEATURES.get("weather", False):
        feature_cols += ["is_outdoor", "temp", "wind"]

    return feature_cols


def prepare_data(df: pd.DataFrame):
    """
    Drops games without enough rolling history (early-season games each
    team's first few appearances) and splits into train (all seasons before
    the final one) / test (the final season) — a walk-forward split, not
    a random one, so no future information leaks into training.
    """
    feature_cols = get_feature_columns(df)
    required = feature_cols + ["actual_margin", "home_win"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    test_season = clean["season"].max()
    train = clean[clean["season"] < test_season]
    test = clean[clean["season"] == test_season]

    print(f"Train: {len(train):,} games (seasons {train['season'].min()}-{train['season'].max()})")
    print(f"Test:  {len(test):,} games (season {test_season})")

    return train, test, feature_cols


# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------
def train_margin_model(train: pd.DataFrame, feature_cols: list[str]):
    """
    Trains a regression model predicting point margin (home_score - away_score).
    Point margin is used as the primary target (rather than plain win/loss)
    because it's more informative — win probability and spread predictions
    can both be derived from it, but not vice versa.
    """
    X_train = train[feature_cols]
    y_train = train["actual_margin"]

    if XGBOOST_AVAILABLE:
        model = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_SEED,
        )
    else:
        print("[note] xgboost not installed, using sklearn HistGradientBoostingRegressor instead")
        model = HistGradientBoostingRegressor(max_depth=3, random_state=config.RANDOM_SEED)

    model.fit(X_train, y_train)
    return model


def train_calibration_model(train_margins: np.ndarray, train_wins: np.ndarray):
    """
    Converts predicted point margin into a calibrated win probability.
    Raw margin predictions aren't automatically well-calibrated probabilities,
    so this fits a small logistic regression on top: margin -> P(win).
    """
    calibrator = LogisticRegression()
    calibrator.fit(train_margins.reshape(-1, 1), train_wins)
    return calibrator


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def evaluate(model, calibrator, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Evaluates the model on held-out (future) games and — importantly —
    compares it against the Vegas closing spread on the SAME games.

    Why compare to Vegas: the betting market is one of the most efficient
    predictors that exists for NFL games, since it aggregates enormous
    amounts of public and sharp information. A model that can't match its
    accuracy isn't necessarily "bad," but it means the model isn't yet
    adding predictive value beyond what's already publicly priced in.
    Matching it is a solid result; beating it consistently is hard and rare.
    """
    X_test = test[feature_cols]
    y_test_margin = test["actual_margin"].values
    y_test_win = test["home_win"].values

    pred_margin = model.predict(X_test)
    pred_win_prob = calibrator.predict_proba(pred_margin.reshape(-1, 1))[:, 1]
    pred_win = (pred_win_prob > 0.5).astype(int)

    results = {
        "model_mae_margin": mean_absolute_error(y_test_margin, pred_margin),
        "model_win_accuracy": accuracy_score(y_test_win, pred_win),
        "model_brier_score": brier_score_loss(y_test_win, pred_win_prob),
        "n_test_games": len(test),
    }

    # Vegas benchmark — only on games where a spread is available
    has_spread = test["spread_line"].notna()
    if has_spread.sum() > 0:
        vegas_pred_margin = test.loc[has_spread, "spread_line"].values
        vegas_actual_margin = test.loc[has_spread, "actual_margin"].values
        vegas_win_actual = test.loc[has_spread, "home_win"].values
        # Vegas "predicted winner" = whichever team it favored
        vegas_pred_win = (vegas_pred_margin > 0).astype(int)

        results["vegas_mae_margin"] = mean_absolute_error(vegas_actual_margin, vegas_pred_margin)
        results["vegas_win_accuracy"] = accuracy_score(vegas_win_actual, vegas_pred_win)
        results["n_games_with_spread"] = int(has_spread.sum())

    return results


def print_results(results: dict):
    print("\n=== MODEL PERFORMANCE (held-out test season) ===")
    print(f"Games evaluated:        {results['n_test_games']:,}")
    print(f"Model MAE (margin):     {results['model_mae_margin']:.2f} points")
    print(f"Model win accuracy:     {results['model_win_accuracy']:.1%}")
    print(f"Model Brier score:      {results['model_brier_score']:.4f}  (lower is better, 0=perfect)")

    if "vegas_mae_margin" in results:
        print(f"\n=== VEGAS BENCHMARK (same games, {results['n_games_with_spread']:,} with spread data) ===")
        print(f"Vegas MAE (margin):     {results['vegas_mae_margin']:.2f} points")
        print(f"Vegas win accuracy:     {results['vegas_win_accuracy']:.1%}")

        mae_gap = results["model_mae_margin"] - results["vegas_mae_margin"]
        acc_gap = results["model_win_accuracy"] - results["vegas_win_accuracy"]
        print(f"\nModel vs Vegas:  MAE {'+' if mae_gap > 0 else ''}{mae_gap:.2f} pts "
              f"({'worse' if mae_gap > 0 else 'better'}),  "
              f"Accuracy {'+' if acc_gap > 0 else ''}{acc_gap:.1%} "
              f"({'worse' if acc_gap < 0 else 'better'})")
        print("\nNote: matching Vegas is already a solid result. Consistently beating")
        print("it is difficult even for professional modelers — treat any edge with caution")
        print("and validate over many more seasons/games before trusting it.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_training_pipeline():
    feature_path = config.PROCESSED_DATA_DIR / "game_level_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"{feature_path} not found. Run data/build_features.py first."
        )

    df = pd.read_parquet(feature_path)
    train, test, feature_cols = prepare_data(df)

    print(f"\nFeature columns used ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")

    print("\nTraining margin model...")
    model = train_margin_model(train, feature_cols)

    print("Training win-probability calibrator...")
    train_pred_margin = model.predict(train[feature_cols])
    calibrator = train_calibration_model(train_pred_margin, train["home_win"].values)

    results = evaluate(model, calibrator, test, feature_cols)
    print_results(results)

    # Save model artifacts
    import joblib
    model_path = config.MODELS_DIR / "margin_model.joblib"
    calibrator_path = config.MODELS_DIR / "win_calibrator.joblib"
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)
    joblib.dump(feature_cols, config.MODELS_DIR / "feature_columns.joblib")
    print(f"\n[saved] {model_path}")
    print(f"[saved] {calibrator_path}")

    return model, calibrator, results


if __name__ == "__main__":
    print(f"=== Training NFL Prediction Model ===\n")
    run_training_pipeline()
