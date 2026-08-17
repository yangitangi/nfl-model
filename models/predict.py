"""
models/predict.py
------------------
Shared prediction + market-odds math used by both the dashboard
(dashboard/app.py) and the weekly prediction tracker
(tracking/prediction_tracker.py), so the two can't drift out of sync.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import reconstruct_margin


def load_model_artifacts():
    """
    Loads the saved model, calibrator, and feature columns, and pins
    config.MARGIN_TARGET_MODE to whatever this model was actually trained
    with (see train_model.py) so callers reconstruct margins correctly even
    if config.py has since changed without a retrain.
    """
    model = joblib.load(config.MODELS_DIR / "margin_model.joblib")
    calibrator = joblib.load(config.MODELS_DIR / "win_calibrator.joblib")
    feature_cols = joblib.load(config.MODELS_DIR / "feature_columns.joblib")
    target_mode_path = config.MODELS_DIR / "margin_target_mode.joblib"
    if target_mode_path.exists():
        config.MARGIN_TARGET_MODE = joblib.load(target_mode_path)
    return model, calibrator, feature_cols


def predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Adds pred_margin (home team perspective, positive = home favored) and
    pred_home_win_prob columns."""
    model, calibrator, _ = load_model_artifacts()
    pred_margin = reconstruct_margin(df, model.predict(df[feature_cols]))
    pred_win_prob = calibrator.predict_proba(pred_margin.reshape(-1, 1))[:, 1]
    out = df.copy()
    out["pred_margin"] = pred_margin
    out["pred_home_win_prob"] = pred_win_prob
    return out


def american_to_implied_prob(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def fair_home_win_prob(home_ml: float, away_ml: float) -> float:
    """Vig-free home win probability: normalizes both sides' implied
    probabilities (which sum to >100% due to the sportsbook's cut) to sum
    to 1, so the model's probability is compared against a fair number
    rather than one inflated by the vig."""
    home_implied = american_to_implied_prob(home_ml)
    away_implied = american_to_implied_prob(away_ml)
    return home_implied / (home_implied + away_implied)


def add_market_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fair_home_ml_prob, spread_edge, and ml_edge to an already-predicted
    dataframe (must have pred_margin, pred_home_win_prob, spread_line,
    home_moneyline, away_moneyline columns). Positive edge = model leans
    more toward the home team than the market does; negative = leans away.
    """
    out = df.copy()
    out["fair_home_ml_prob"] = out.apply(
        lambda r: fair_home_win_prob(r["home_moneyline"], r["away_moneyline"])
        if pd.notna(r["home_moneyline"]) and pd.notna(r["away_moneyline"]) else pd.NA,
        axis=1,
    )
    out["spread_edge"] = out["pred_margin"] - out["spread_line"]
    out["ml_edge"] = out["pred_home_win_prob"] - out["fair_home_ml_prob"].astype(float)
    return out


def format_spread(home_team: str, away_team: str, home_margin: float) -> str:
    """home_margin: positive = home favored by that many points (this
    project's sign convention throughout). Converted to standard betting
    notation, favorite shown negative, e.g. 'SEA -3.5'."""
    if pd.isna(home_margin):
        return "TBD"
    if home_margin > 0:
        return f"{home_team} -{home_margin:.1f}"
    if home_margin < 0:
        return f"{away_team} -{-home_margin:.1f}"
    return "Pick'em"


def format_ml(odds: float) -> str:
    return "TBD" if pd.isna(odds) else f"{odds:+.0f}"
