"""
rebuild/features/context.py
---------------------------
Game context from the pre-game schedule row.
  home_rest, away_rest, rest_diff   days since the team's previous game (nflverse; known in advance)
  home/away_short_week              rest <= 5 days
  home/away_bye_before              rest >= 13 days (team had a bye; week 1 rest is not a bye)
  tz_diff                           home stadium time zone minus away team's home time zone, hours
                                    (positive = away team travelled east). 0 for neutral sites.
  neutral_site                      nflverse location == "Neutral" (international / relocated games)
  is_dome                           roof in {dome, closed}
  temp, wind                        game-time weather from nflverse (a near-as-of proxy: a forecast is known
                                    before kickoff). Domes: temp 70, wind 0. Outdoor games with missing
                                    weather keep NaN plus weather_missing = 1 (structural NaN, about 25-65%
                                    of outdoor games per season; impute inside the model's training fold).
  div_game
Time zones use the HISTORIC team code (STL was Central time before moving to LA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ = {**{t: 0 for t in ["ATL", "BAL", "BUF", "CAR", "CIN", "CLE", "DET", "IND", "JAX", "MIA", "NE", "NYG",
                        "NYJ", "PHI", "PIT", "TB", "WAS"]},
      **{t: -1 for t in ["CHI", "DAL", "GB", "HOU", "KC", "MIN", "NO", "TEN", "STL"]},
      **{t: -2 for t in ["DEN", "ARI"]},
      **{t: -3 for t in ["LA", "LAC", "LV", "OAK", "SD", "SF", "SEA"]}}


def build(sched: pd.DataFrame) -> pd.DataFrame:
    s = sched
    neutral = (s["location"] == "Neutral").astype(int)
    dome = s["roof"].isin(["dome", "closed"]).astype(int)
    tz = (s["home_team_orig"].map(TZ) - s["away_team_orig"].map(TZ)).where(neutral == 0, 0)
    temp = np.where(dome == 1, 70.0, s["temp"].astype(float))
    wind = np.where(dome == 1, 0.0, s["wind"].astype(float))
    hr, ar = s["home_rest"].astype(float), s["away_rest"].astype(float)
    week1 = s["week"] == 1
    return pd.DataFrame({
        "game_id": s["game_id"].values,
        "home_rest": hr.values, "away_rest": ar.values, "rest_diff": (hr - ar).values,
        "home_short_week": (hr <= 5).astype(int).values, "away_short_week": (ar <= 5).astype(int).values,
        "home_bye_before": ((hr >= 13) & ~week1).astype(int).values,
        "away_bye_before": ((ar >= 13) & ~week1).astype(int).values,
        "tz_diff": tz.astype(float).values, "neutral_site": neutral.values, "is_dome": dome.values,
        "temp": temp, "wind": wind, "weather_missing": (np.isnan(temp) | np.isnan(wind)).astype(int),
        "div_game": s["div_game"].astype(int).values,
    })
