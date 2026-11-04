"""
rebuild/features/common.py
--------------------------
Raw-data loading and the shared as-of conventions for the feature store.

AS-OF CONVENTION (week-granular information set)
  Each regular-season game gets a global week index `gw` (ordinal of (season, week)).
  The features of a game in week gw may use only:
    * results / play-by-play of games in weeks < gw (this and earlier seasons),
    * the game's own PRE-GAME schedule row (teams, date, closing lines, roof, rest, announced
      starting QB id, weather; see notes in market.py / context.py / qb.py),
    * injury reports for weeks <= gw (the week-gw report is the final pre-game report;
      all of them are timestamped before kickoff in nflverse).
  This is slightly conservative: a Sunday game does not see that week's Thursday result.
  It makes "delete every row dated on or after the week's first kickoff" an exact test.

Team codes are normalised to franchise codes (OAK->LV, SD->LAC, STL->LA) because nflverse pbp
uses current codes for every season while schedules/injuries use the historic ones.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
FIRST_SEASON, LAST_SEASON = 2010, 2025
HALF_LIFE_GAMES = 6.0
DECAY = 0.5 ** (1.0 / HALF_LIFE_GAMES)

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "pass", "rush", "epa", "success",
    "yards_gained", "qb_dropback", "passer_player_id", "rusher_player_id", "receiver_player_id", "cpoe",
    "sack", "two_point_attempt", "solo_tackle_1_player_id", "assist_tackle_1_player_id", "sack_player_id",
    "half_sack_1_player_id", "pass_defense_1_player_id", "interception_player_id", "qb_hit_1_player_id",
    "tackle_for_loss_1_player_id",
]
RESULT_COLS = ["home_score", "away_score", "result", "total", "overtime"]


def norm_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_MAP)


def load_raw(first: int = FIRST_SEASON, last: int = LAST_SEASON, raw_dir: Path = RAW) -> dict:
    """Load regular-season schedules, pbp and injury reports into memory (about 10 s)."""
    s = pd.read_parquet(raw_dir / "schedules.parquet")
    s = s[(s["game_type"] == "REG") & s["season"].between(first, last)].copy()
    s["home_team_orig"], s["away_team_orig"] = s["home_team"], s["away_team"]
    s["home_team"], s["away_team"] = norm_team(s["home_team"]), norm_team(s["away_team"])
    s["kickoff"] = pd.to_datetime(s["gameday"] + " " + s["gametime"])  # local ET, naive

    pbp = []
    for season in range(first, last + 1):
        f = raw_dir / f"pbp_{season}.parquet"
        p = pd.read_parquet(f, columns=PBP_COLS)
        pbp.append(p[p["season_type"] == "REG"])
    pbp = pd.concat(pbp, ignore_index=True)
    for c in ("posteam", "defteam"):
        pbp[c] = norm_team(pbp[c])

    inj = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(raw_dir / "injuries_*.parquet")))],
                    ignore_index=True)
    inj = inj[(inj["game_type"] == "REG") & inj["season"].between(first, last)].copy()
    inj["season"], inj["week"] = inj["season"].astype(int), inj["week"].astype(int)
    inj["team"] = norm_team(inj["team"])
    return {"sched": s.reset_index(drop=True), "pbp": pbp, "inj": inj.reset_index(drop=True)}


def truncate_raw(raw: dict, season: int, week: int) -> dict:
    """Simulate the information state at the first kickoff of (season, week):
    every game with kickoff on/after that moment is deleted from schedules and pbp, EXCEPT the
    target week's own games, which keep only their pre-game schedule row (results blanked).
    Injury reports for later weeks are deleted, and any report row timestamped on/after its
    game's kickoff is deleted."""
    s = raw["sched"]
    target = (s["season"] == season) & (s["week"] == week)
    cutoff = s.loc[target, "kickoff"].min()
    keep_past = s["kickoff"] < cutoff
    sched = pd.concat([s[keep_past & ~target], s[target].assign(**{c: np.nan for c in RESULT_COLS})])
    sched = sched.sort_values(["season", "week", "kickoff", "game_id"]).reset_index(drop=True)

    past_ids = set(s.loc[keep_past & ~target, "game_id"])
    pbp = raw["pbp"][raw["pbp"]["game_id"].isin(past_ids)]

    inj = raw["inj"]
    later = (inj["season"] > season) | ((inj["season"] == season) & (inj["week"] > week))
    inj = inj[~later]
    ko = pd.concat([sched[["season", "week", "home_team", "kickoff"]].rename(columns={"home_team": "team"}),
                    sched[["season", "week", "away_team", "kickoff"]].rename(columns={"away_team": "team"})])
    m = inj.merge(ko, on=["season", "week", "team"], how="left")
    ko_utc = m["kickoff"].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    dated_after = (m["date_modified"] >= ko_utc).fillna(False).values
    inj = inj[~dated_after]
    return {"sched": sched, "pbp": pbp, "inj": inj.reset_index(drop=True)}


def add_gw(sched: pd.DataFrame) -> pd.DataFrame:
    """Global week index. Uses the full calendar of (season, week) pairs 2010-2025 so that it does
    not depend on which games are present (stable under truncation)."""
    weeks = [(y, w) for y in range(FIRST_SEASON, LAST_SEASON + 1) for w in range(1, 23)]
    idx = {k: i for i, k in enumerate(weeks)}
    sched = sched.copy()
    sched["gw"] = [idx[(int(y), int(w))] for y, w in zip(sched["season"], sched["week"])]
    return sched


def team_games(sched: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (game, team) with opponent and home flag, sorted by team, gw."""
    cols = ["game_id", "season", "week", "gw", "kickoff"]
    h = sched[cols + ["home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opp"})
    a = sched[cols + ["away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opp"})
    h["is_home"], a["is_home"] = 1, 0
    tg = pd.concat([h, a], ignore_index=True).sort_values(["team", "gw"]).reset_index(drop=True)
    return tg


def window_mean_by_gw(gw_vals: np.ndarray, sums: np.ndarray, counts: np.ndarray, n_gw: int,
                      window: int) -> np.ndarray:
    """For each global week g: sum(sums)/sum(counts) over entries with gw in [g-window, g-1].
    sums: (n, k), counts: (n, k) or (n,). Returns (n_gw, k); NaN where the window is empty."""
    sums = sums.reshape(len(sums), -1).astype(float)
    counts = np.broadcast_to(counts.reshape(len(counts), -1), sums.shape).astype(float)
    S =np.zeros((n_gw + 1, sums.shape[1]))
    N = np.zeros((n_gw + 1, sums.shape[1]))
    np.add.at(S, gw_vals + 1, np.nan_to_num(sums))
    np.add.at(N, gw_vals + 1, np.where(np.isnan(sums), 0, counts))
    cS, cN = np.cumsum(S, axis=0), np.cumsum(N, axis=0)       # cS[g] = sum over gw < g
    g = np.arange(n_gw)
    lo = np.clip(g - window, 0, None)
    num, den = cS[g] - cS[lo], cN[g] - cN[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
