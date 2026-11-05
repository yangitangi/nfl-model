"""
rebuild/features/team_strength.py
---------------------------------
Opponent-adjusted, exponentially decayed team ratings (as-of the start of the game's week).

Per team-game (from pbp, REG, pass or rush plays with non-missing EPA, no 2-pt tries):
  pass = dropbacks (incl. sacks/scrambles), rush = designed runs.
  epa = EPA/play, sr = success rate, expl = explosive rate (pass >= 20 yds, rush >= 10 yds).
  off_* = the team on offense; def_* = what the team allowed (higher = worse defense).

Rating before game i (per team, per metric):
  cur  = play-weighted EWM over this season's prior games, decay 0.5^(1/6) per game (half-life 6 games).
  prior = L + RETAIN * (last_season_final - L), RETAIN = 0.5 (i.e. regressed halfway to the league mean L),
          last_season_final = the team's end-of-previous-season EWM.
  team games played this season g: g=0 -> prior; g=1..3 -> w*prior + (1-w)*cur with w=(4-g)/4; g>=4 -> cur.
  (no prior available, i.e. 2010 -> cur only; NaN only before a team's first 2010 game.)
  L = league mean over the previous 17 global weeks (as-of).

Opponent adjustment (one pass, as-of only): each game's per-play value is shifted by the opponent's
UNADJUSTED as-of rating minus L (offense: minus the opponent defense's deviation; defense: minus the
opponent offense's deviation); the adjusted per-game values are then rated with the same EWM/carry-over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DECAY, team_games, window_mean_by_gw

RETAIN = 0.5
CARRY_GAMES = 4
LEAGUE_WINDOW = 17
SPLITS = ("pass", "rush")
STATS = ("epa", "sr", "expl")
METRICS = [f"{sp}_{st}" for sp in SPLITS for st in STATS]


def _game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1)) & pbp["epa"].notna()
            & (pbp["two_point_attempt"].fillna(0) == 0)].copy()
    p["split"] = np.where(p["pass"] == 1, "pass", "rush")
    p["expl"] = np.where(p["split"] == "pass", p["yards_gained"] >= 20, p["yards_gained"] >= 10).astype(float)
    p["one"] = 1.0
    g = p.groupby(["game_id", "posteam", "defteam", "split"])[["epa", "success", "expl", "one"]].sum()
    g = g.rename(columns={"success": "sr", "one": "n"}).unstack("split")
    g.columns = [f"{sp}_{st}" for st, sp in g.columns]
    return g.reset_index()


def _asof(season: np.ndarray, team_idx: list, sums: np.ndarray, counts: np.ndarray,
          L: np.ndarray) -> np.ndarray:
    """Rating BEFORE each row (rows of one team must be in gw order). sums/counts: (n, k)."""
    out = np.full(sums.shape, np.nan)
    for idx in team_idx:
        S = np.zeros(sums.shape[1]); N = np.zeros(sums.shape[1])
        cur_season, g, last_final = None, 0, np.full(sums.shape[1], np.nan)
        for i in idx:
            if season[i] != cur_season:
                if cur_season is not None:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        last_final = np.where(N > 0, S / np.where(N > 0, N, 1), np.nan)
                S[:] = 0; N[:] = 0; g = 0; cur_season = season[i]
            with np.errstate(invalid="ignore", divide="ignore"):
                cur = np.where(N > 0, S / np.where(N > 0, N, 1), np.nan)
            prior = L[i] + RETAIN * (last_final - L[i])
            if g >= CARRY_GAMES:
                r = cur
            elif g == 0:
                r = prior
            else:
                w = (CARRY_GAMES - g) / CARRY_GAMES
                r = np.where(np.isnan(prior), cur, w * prior + (1 - w) * cur)
            out[i] = r
            ok = ~np.isnan(sums[i]) & (counts[i] > 0)
            if ok.any():
                S[ok] = DECAY * S[ok] + sums[i][ok]
                N[ok] = DECAY * N[ok] + counts[i][ok]
                g += 1
    return out


def build(sched: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (game_id, team) with off_/def_ adjusted ratings."""
    tg = team_games(sched)
    gs = _game_stats(pbp)
    off = gs.drop(columns="defteam").rename(columns={"posteam": "team"})
    dfn = gs.drop(columns="posteam").rename(columns={"defteam": "team"})
    tg = tg.merge(off, on=["game_id", "team"], how="left").merge(
        dfn, on=["game_id", "team"], how="left", suffixes=("", "_d"))
    n_gw = int(tg["gw"].max()) + 2

    def arrays(suffix):
        s = np.column_stack([tg[f"{m}{suffix}"] for m in METRICS])
        n = np.column_stack([tg[f"{m.split('_')[0]}_n{suffix}"] for m in METRICS])
        return s.astype(float), np.nan_to_num(n.astype(float))

    s_off, n_off = arrays("")
    s_def, n_def = arrays("_d")
    Lg = window_mean_by_gw(tg["gw"].values, s_off, n_off, n_gw, LEAGUE_WINDOW)   # league mean per metric
    L = Lg[tg["gw"].values]
    Lfill = np.where(np.isnan(L), 0.0, L)
    team_idx = [g.index.values for _, g in tg.groupby("team", sort=False)]
    season = tg["season"].values

    r_off = _asof(season, team_idx, s_off, n_off, Lfill)
    r_def = _asof(season, team_idx, s_def, n_def, Lfill)

    # opponent's unadjusted as-of ratings in the same game
    key = tg[["game_id", "team"]].copy()
    pos = pd.Series(np.arange(len(tg)), index=pd.MultiIndex.from_frame(key))
    opp_pos = pos.reindex(pd.MultiIndex.from_frame(tg[["game_id", "opp"]].rename(columns={"opp": "team"}))).values
    opp_def_dev = np.nan_to_num(r_def[opp_pos] - L)
    opp_off_dev = np.nan_to_num(r_off[opp_pos] - L)
    a_off = _asof(season, team_idx, s_off - n_off * opp_def_dev, n_off, Lfill)
    a_def = _asof(season, team_idx, s_def - n_def * opp_off_dev, n_def, Lfill)

    out = tg[["game_id", "team"]].copy()
    for j, m in enumerate(METRICS):
        out[f"off_{m}"] = a_off[:, j]
        out[f"def_{m}"] = a_def[:, j]
    return out
