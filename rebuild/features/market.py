"""
rebuild/features/market.py
--------------------------
Closing market for the game itself (known at kickoff). Never uses another game's line.
  spread_line  home perspective, positive = home favoured (nflverse convention)
  total_line
  ml_home_prob vig-free home win probability from the two moneylines: p_h / (p_h + p_a), where
               p = 100/(ml+100) for ml > 0 and -ml/(-ml+100) for ml < 0.
  ml_vig       overround (p_h + p_a - 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied(ml: pd.Series) -> pd.Series:
    ml = ml.astype(float)
    return pd.Series(np.where(ml > 0, 100.0 / (ml + 100.0), -ml / (-ml + 100.0)), index=ml.index)


def build(sched: pd.DataFrame) -> pd.DataFrame:
    ph, pa = implied(sched["home_moneyline"]), implied(sched["away_moneyline"])
    return pd.DataFrame({
        "game_id": sched["game_id"].values,
        "spread_line": sched["spread_line"].values,
        "total_line": sched["total_line"].values,
        "ml_home_prob": (ph / (ph + pa)).values,
        "ml_vig": (ph + pa - 1.0).values,
    })
