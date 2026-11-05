"""
rebuild/features/qb.py
----------------------
Starting-QB features with empirical-Bayes-style shrinkage to a replacement-level prior. Never NaN.

Starter identity (as-of-announcement proxy): the schedule's home_qb_id / away_qb_id. nflverse derives
these from the game's own first offensive snaps, i.e. the QB who actually started. For historical rows
this is the best available proxy for the starter the market knew at kickoff (starters are normally
announced or known before the game). It is the ONLY field taken from the game itself, and it carries
no performance information. Backtests therefore assume perfect knowledge of late QB switches.

QB history (pbp, REG, qb_dropback == 1, QB = passer id, else scrambler id), per QB-game:
  dropbacks, EPA sum, sacks, CPOE sum/count (pass attempts with CPOE).
Before the game's week gw:
  career = all prior games; recent = EWM over the QB's prior games, half-life 6 games.
  shrunk = (sum + K * prior) / (n + K), K_EPA = 300 dropbacks, K_SACK = 200 dropbacks, K_CPOE = 300 attempts
  (fixed pseudo-counts in the range of published stabilisation points; not tuned).
  prior = replacement level: as-of mean over dropbacks by QBs with < 300 career dropbacks at the time,
  previous 51 global weeks (about 3 seasons); fallback constants when that window has < 500 dropbacks
  (only the first weeks of 2010): EPA -0.10, CPOE -3.0, sack rate 0.08.
Flags: qb_dropbacks_prior (career dropbacks before the game), qb_rookie_or_unknown (< 100 career dropbacks
or missing id), qb_changed (starter differs from the team's previous game starter; 0 for a team's first game).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DECAY, team_games, window_mean_by_gw

K_EPA, K_SACK, K_CPOE = 300.0, 200.0, 300.0
INEXPERIENCED_DB = 300
PRIOR_WINDOW = 51
PRIOR_MIN_DB = 500
FALLBACK = {"epa": -0.10, "cpoe": -3.0, "sack": 0.08}
ROOKIE_DB = 100


def _qb_games(pbp: pd.DataFrame, gw_of_game: pd.Series) -> pd.DataFrame:
    p = pbp[(pbp["qb_dropback"] == 1) & pbp["epa"].notna()].copy()
    p["qb"] = p["passer_player_id"].fillna(p["rusher_player_id"])
    p = p[p["qb"].notna()]
    p["has_cpoe"] = p["cpoe"].notna().astype(float)
    p["one"] = 1.0
    g = p.groupby(["game_id", "qb"]).agg(db=("one", "sum"), epa=("epa", "sum"), sack=("sack", "sum"),
                                         cpoe=("cpoe", "sum"), cpoe_n=("has_cpoe", "sum")).reset_index()
    g["gw"] = g["game_id"].map(gw_of_game)
    g = g[g["gw"].notna()].astype({"gw": int}).sort_values(["qb", "gw"]).reset_index(drop=True)
    return g


def build(sched: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    gw_of_game = sched.set_index("game_id")["gw"]
    qg = _qb_games(pbp, gw_of_game)
    n_gw = int(sched["gw"].max()) + 2

    # career / recent state AFTER each QB-game
    vals = ["db", "epa", "sack", "cpoe", "cpoe_n"]
    car = qg.groupby("qb")[vals].cumsum()
    rec = np.zeros((len(qg), len(vals)))
    arr = qg[vals].values
    for _, idx in qg.groupby("qb").indices.items():
        state = np.zeros(len(vals))
        for i in idx:
            state = DECAY * state + arr[i]
            rec[i] = state
    state_tbl = qg[["qb", "gw"]].copy()
    for j, v in enumerate(vals):
        state_tbl[f"car_{v}"] = car[v].values
        state_tbl[f"rec_{v}"] = rec[:, j]

    # replacement prior per gw: dropbacks by QBs with < 300 career dropbacks before that game
    prev_db = car["db"].values - qg["db"].values
    inexp = prev_db < INEXPERIENCED_DB
    q = qg[inexp]
    pri_epa = window_mean_by_gw(q["gw"].values, q[["epa"]].values, q["db"].values, n_gw, PRIOR_WINDOW)[:, 0]
    pri_sack = window_mean_by_gw(q["gw"].values, q[["sack"]].values, q["db"].values, n_gw, PRIOR_WINDOW)[:, 0]
    pri_cpoe = window_mean_by_gw(q["gw"].values, q[["cpoe"]].values, q["cpoe_n"].values, n_gw, PRIOR_WINDOW)[:, 0]
    # total inexperienced-QB dropbacks in the window (to decide whether to use the fallback)
    S =np.zeros(n_gw + 1); np.add.at(S, q["gw"].values + 1, q["db"].values); cS = np.cumsum(S)
    g = np.arange(n_gw); tot = cS[g] - cS[np.clip(g - PRIOR_WINDOW, 0, None)]
    enough = tot >= PRIOR_MIN_DB
    pri_epa = np.where(enough, pri_epa, FALLBACK["epa"])
    pri_sack = np.where(enough, pri_sack, FALLBACK["sack"])
    pri_cpoe = np.where(enough, pri_cpoe, FALLBACK["cpoe"])

    # starters
    tg = team_games(sched)
    qbid = pd.concat([sched[["game_id", "home_team", "home_qb_id"]].rename(columns={"home_team": "team", "home_qb_id": "qb"}),
                      sched[["game_id", "away_team", "away_qb_id"]].rename(columns={"away_team": "team", "away_qb_id": "qb"})])
    tg = tg.merge(qbid, on=["game_id", "team"], how="left")
    tg["qb_changed"] = (tg["qb"] != tg.groupby("team")["qb"].shift(1)) & tg.groupby("team").cumcount().gt(0)

    # as-of lookup: latest QB state with state gw < game gw
    look = tg[["game_id", "team", "qb", "gw"]].copy()
    look["qb_key"] = look["qb"].fillna("__unknown__")
    st = state_tbl.rename(columns={"qb": "qb_key", "gw": "state_gw"}).sort_values("state_gw")
    look = look.sort_values("gw")
    look["gw_m1"] = look["gw"] - 1
    m = pd.merge_asof(look, st, left_on="gw_m1", right_on="state_gw", by="qb_key", direction="backward")
    m = m.set_index(["game_id", "team"]).reindex(pd.MultiIndex.from_frame(tg[["game_id", "team"]])).reset_index()
    # decay recent state for weeks without a game is NOT applied (decay is per game played)
    for c in [c for c in m.columns if c.startswith(("car_", "rec_"))]:
        m[c] = m[c].fillna(0.0)
    gw = tg["gw"].values
    pe, ps, pc = pri_epa[gw], pri_sack[gw], pri_cpoe[gw]

    out = tg[["game_id", "team"]].copy()
    out["qb_epa_db_career"] = (m["car_epa"] + K_EPA * pe) / (m["car_db"] + K_EPA)
    out["qb_epa_db_recent"] = (m["rec_epa"] + K_EPA * pe) / (m["rec_db"] + K_EPA)
    out["qb_sack_rate"] = (m["car_sack"] + K_SACK * ps) / (m["car_db"] + K_SACK)
    out["qb_cpoe"] = (m["car_cpoe"] + K_CPOE * pc) / (m["car_cpoe_n"] + K_CPOE)
    out["qb_dropbacks_prior"] = m["car_db"].values
    out["qb_rookie_or_unknown"] = ((m["car_db"] < ROOKIE_DB) | tg["qb"].isna()).astype(int).values
    out["qb_changed"] = tg["qb_changed"].astype(int).values
    out["qb_prior_epa"] = pe
    return out
