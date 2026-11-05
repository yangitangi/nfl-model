"""
rebuild/models/common.py
------------------------
Shared data access and conventions for the session-3 candidates.

* Only DEV-era rows are ever loaded: seasons <= 2023 (the 2024-2025 holdout is filtered out at load time
  and load_dev() asserts it).
* "Chosen once on 2010-2015": hyperparameters are picked by walk-forward inside 2010-2015 only
  (validate on 2013, 2014, 2015, each fitted on the seasons before it) and then reused for every
  DEV fold 2016-2023. 2010-2015 is inside every DEV fold's training window, so this never sees a test season.
* Each candidate returns {T: (test_frame, train_frame)} with a `pred_margin` column on both: test = the
  walk-forward prediction for season T, train = the same fitted model's predictions on seasons < T
  (used only for the win-probability calibration fitted inside the training window).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rebuild.build_asof_store import feature_columns  # noqa: E402
from rebuild.protocol import DEV_TEST_SEASONS  # noqa: E402

SELECT_VAL_SEASONS = [2013, 2014, 2015]
SELECT_LAST = 2015
KEEP = ["game_id", "season", "week", "gw", "spread_line", "total_line", "actual_margin", "actual_total",
        "home_win", "home_score", "away_score", "home_team", "away_team"]


def load_dev() -> pd.DataFrame:
    store = pd.read_parquet(ROOT / "data/processed/asof_features.parquet")
    df = store[(store["season"] <= max(DEV_TEST_SEASONS)) & store["spread_line"].notna()
               & store["actual_margin"].notna()].copy()
    assert df["season"].max() <= 2023, "holdout rows must never be loaded"
    return df.sort_values(["gw", "kickoff", "game_id"]).reset_index(drop=True)


def base_features(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(df) if pd.api.types.is_numeric_dtype(df[c])]


INTERACTIONS = ["ix_spread_x_rest_diff", "ix_spread_x_qb_diff", "ix_total_x_wind", "ix_total_x_temp"]


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ix_spread_x_rest_diff"] = d["spread_line"] * d["rest_diff"]
    d["ix_spread_x_qb_diff"] = d["spread_line"] * (d["home_qb_epa_db_career"] - d["away_qb_epa_db_career"])
    d["ix_total_x_wind"] = d["total_line"] * d["wind"]
    d["ix_total_x_temp"] = d["total_line"] * d["temp"]
    return d


def residual(df: pd.DataFrame) -> np.ndarray:
    return (df["actual_margin"] - df["spread_line"]).values


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(np.asarray(pred) - np.asarray(actual)).mean())
