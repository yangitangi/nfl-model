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

    if config.FEATURES.get("injuries", False):
        feature_cols += ["home_injury_burden", "away_injury_burden"]

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
# MARGIN TARGET TRANSFORM (config.MARGIN_TARGET_MODE)
# ---------------------------------------------------------------------------
def get_margin_target(df: pd.DataFrame) -> np.ndarray:
    """
    The regression target for the margin model. See config.MARGIN_TARGET_MODE
    for why "residual" mode (predicting the gap to the market line, rather
    than the margin itself) is worth trying.
    """
    if config.MARGIN_TARGET_MODE == "residual":
        return (df["actual_margin"] - df["spread_line"]).values
    return df["actual_margin"].values


def get_total_target(df: pd.DataFrame) -> np.ndarray:
    """The regression target for the total-points model. See
    config.TOTAL_TARGET_MODE -- same idea as get_margin_target, just
    anchored on total_line (the market's combined-score number) instead of
    spread_line."""
    if config.TOTAL_TARGET_MODE == "residual":
        return (df["actual_total"] - df["total_line"]).values
    return df["actual_total"].values


def reconstruct_total(df: pd.DataFrame, raw_pred: np.ndarray) -> np.ndarray:
    """Converts the total model's raw output back to an actual-total-scale
    prediction. "residual": adds total_line back; a missing total_line
    falls back to the raw prediction's own scale being undefined, so this
    uses a 45.0-point league-average prior instead (roughly the long-run
    mean total), same "don't leave it undefined" principle as
    reconstruct_margin."""
    if config.TOTAL_TARGET_MODE == "residual":
        offset = df["total_line"].fillna(45.0).values
        return raw_pred + offset
    return raw_pred


def reconstruct_margin(df: pd.DataFrame, raw_pred: np.ndarray) -> np.ndarray:
    """
    Converts the model's raw output back to an actual-margin-scale
    prediction.

    "residual": adds the market line back; a missing spread_line (a future
    game with no line posted yet) falls back to a 0-point offset rather
    than leaving the prediction undefined.

    "blend": a weighted average of the raw prediction and the market line
    (config.MARKET_BLEND_WEIGHT on the market side). A missing spread_line
    falls back to the raw prediction alone (market weight 0 for that game),
    same "don't leave it undefined" principle as residual mode.
    """
    if config.MARGIN_TARGET_MODE == "residual":
        offset = df["spread_line"].fillna(0).values
        return raw_pred + offset
    if config.MARGIN_TARGET_MODE == "blend":
        spread = df["spread_line"].values.astype(float)
        w = config.MARKET_BLEND_WEIGHT
        has_spread = ~np.isnan(spread)
        blended = raw_pred.copy()
        blended[has_spread] = w * spread[has_spread] + (1 - w) * raw_pred[has_spread]
        return blended
    return raw_pred


# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------
def _build_regressor():
    """Same model type/hyperparameters for every regression target in this
    project (margin, total) -- kept in one place so they can't drift apart."""
    if XGBOOST_AVAILABLE:
        return XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_SEED,
        )
    print("[note] xgboost not installed, using sklearn HistGradientBoostingRegressor instead")
    return HistGradientBoostingRegressor(max_depth=3, random_state=config.RANDOM_SEED)


def train_margin_model(train: pd.DataFrame, feature_cols: list[str]):
    """
    Trains a regression model predicting point margin (home_score - away_score)
    -- or, in "residual" mode, the gap between that margin and the market
    line (see get_margin_target). Point margin (rather than plain win/loss)
    is the underlying target either way, since win probability and spread
    predictions can both be derived from it, but not vice versa.
    """
    model = _build_regressor()
    model.fit(train[feature_cols], get_margin_target(train))
    return model


def train_total_model(train: pd.DataFrame, feature_cols: list[str]):
    """
    Trains a regression model predicting the game's combined score
    (home_score + away_score) -- or, in "residual" mode, the gap between
    that total and the market's total_line (see get_total_target).

    Paired with the margin model, this lets implied per-team scores be
    backed out: home = (total + margin) / 2, away = (total - margin) / 2 --
    the same "projected final score" display public models like David
    Sasser's (davidsasser.com/nfl) show, built from two independently
    validated numbers rather than guessed directly.
    """
    model = _build_regressor()
    model.fit(train[feature_cols], get_total_target(train))
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

    pred_margin = reconstruct_margin(test, model.predict(X_test))
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


def evaluate_total(model, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Same idea as evaluate(), for the total-points model: MAE against the
    actual combined score, benchmarked against Vegas's total_line on the
    same games. There's no win/loss analog for a total, so this is simpler
    than evaluate() -- MAE is the whole story."""
    y_test_total = test["actual_total"].values
    pred_total = reconstruct_total(test, model.predict(test[feature_cols]))

    results = {
        "model_mae_total": mean_absolute_error(y_test_total, pred_total),
        "n_test_games": len(test),
    }

    has_total = test["total_line"].notna()
    if has_total.sum() > 0:
        vegas_pred_total = test.loc[has_total, "total_line"].values
        vegas_actual_total = y_test_total[has_total.values]
        results["vegas_mae_total"] = mean_absolute_error(vegas_actual_total, vegas_pred_total)
        results["n_games_with_total"] = int(has_total.sum())

    return results


def print_total_results(results: dict):
    print("\n=== TOTAL-POINTS MODEL PERFORMANCE (held-out test season) ===")
    print(f"Games evaluated:        {results['n_test_games']:,}")
    print(f"Model MAE (total):      {results['model_mae_total']:.2f} points")

    if "vegas_mae_total" in results:
        print(f"\n=== VEGAS BENCHMARK (same games, {results['n_games_with_total']:,} with a total line) ===")
        print(f"Vegas MAE (total):      {results['vegas_mae_total']:.2f} points")
        mae_gap = results["model_mae_total"] - results["vegas_mae_total"]
        print(f"\nModel vs Vegas:  MAE {'+' if mae_gap > 0 else ''}{mae_gap:.2f} pts "
              f"({'worse' if mae_gap > 0 else 'better'})")


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
    train_pred_margin = reconstruct_margin(train, model.predict(train[feature_cols]))
    calibrator = train_calibration_model(train_pred_margin, train["home_win"].values)

    results = evaluate(model, calibrator, test, feature_cols)
    print_results(results)

    print("\nTraining total-points model...")
    total_model = train_total_model(train, feature_cols)
    total_results = evaluate_total(total_model, test, feature_cols)
    print_total_results(total_results)

    # Save model artifacts
    import joblib
    model_path = config.MODELS_DIR / "margin_model.joblib"
    calibrator_path = config.MODELS_DIR / "win_calibrator.joblib"
    total_model_path = config.MODELS_DIR / "total_model.joblib"
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)
    joblib.dump(total_model, total_model_path)
    joblib.dump(feature_cols, config.MODELS_DIR / "feature_columns.joblib")
    # Pinned alongside the models so inference code (the dashboard) always
    # reconstructs margins/totals the same way these models were trained,
    # even if config.MARGIN_TARGET_MODE/TOTAL_TARGET_MODE change later
    # without a retrain.
    joblib.dump(config.MARGIN_TARGET_MODE, config.MODELS_DIR / "margin_target_mode.joblib")
    joblib.dump(config.TOTAL_TARGET_MODE, config.MODELS_DIR / "total_target_mode.joblib")
    print(f"\n[saved] {model_path}")
    print(f"[saved] {calibrator_path}")
    print(f"[saved] {total_model_path}")

    return model, calibrator, total_model, results, total_results


if __name__ == "__main__":
    print(f"=== Training NFL Prediction Model ===\n")
    run_training_pipeline()
