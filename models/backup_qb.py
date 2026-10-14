"""
models/backup_qb.py
-------------------
Prospective "backup QB" flag: a team is starting a backup-level QB when its
first-dropback passer is NOT the team's established QB and has little recent
playing time of his own. Everything here is computed from play-by-play
strictly BEFORE the game being flagged, so it can be applied to upcoming
games and walk-forward back-tests alike.

Validated in models/backup_qb_test.py: teams in this situation covered the
closing line only ~44% of the time (2014-2025), and the model was too
bullish on them by ~1.6 pts on average (leave-one-season-out).
"""

import numpy as np
import pandas as pd

import config

WINDOW_GAMES = 17          # trailing games used for "established" / recent usage
ESTABLISHED_MIN_DB = 250   # leader needs this many dropbacks in the window
STARTER_MAX_DB = 250       # starter's own trailing dropbacks below this => backup-level


def _passer_games() -> pd.DataFrame:
    frames = []
    for path in sorted(config.RAW_DATA_DIR.glob("pbp_*.parquet")):
        cols = ["game_id", "season", "week", "play_id", "posteam", "passer_player_id",
                "qb_dropback", "season_type"]
        p = pd.read_parquet(path, columns=cols)
        frames.append(p[(p["season_type"] == "REG") & (p["qb_dropback"] == 1)
                        & p["passer_player_id"].notna() & p["posteam"].notna()])
    p = pd.concat(frames, ignore_index=True)
    first = (p.sort_values("play_id").groupby(["game_id", "posteam"]).head(1)
             [["game_id", "posteam", "passer_player_id"]]
             .rename(columns={"passer_player_id": "starter_id"}))
    db = (p.groupby(["season", "week", "game_id", "posteam", "passer_player_id"]).size()
          .reset_index(name="db"))
    return first, db


def historical_backup_flags() -> pd.DataFrame:
    """One row per (game_id, team): starter_id, established_id, starter_recent_db,
    established_db, backup_flag -- all from games strictly before this one."""
    first, db = _passer_games()
    order = (db[["season", "week", "game_id"]].drop_duplicates()
             .sort_values(["season", "week", "game_id"]).reset_index(drop=True))
    order["t"] = np.arange(len(order))
    db = db.merge(order[["game_id", "t"]], on="game_id")
    first = first.merge(order[["game_id", "season", "week", "t"]], on="game_id")

    # per-passer chronological dropback history
    by_passer = {pid: g.sort_values("t")[["t", "db"]].to_numpy() for pid, g in db.groupby("passer_player_id")}
    # per-team chronological games (to define "last 17 team games")
    team_games = (first.sort_values("t").groupby("posteam")["t"].apply(np.array).to_dict())
    team_db = {tm: g for tm, g in db.groupby("posteam")}

    rows = []
    for r in first.itertuples(index=False):
        tg = team_games[r.posteam]
        prior_t = tg[tg < r.t][-WINDOW_GAMES:]
        if len(prior_t) == 0:
            rows.append((r.game_id, r.posteam, r.starter_id, None, 0, 0, False))
            continue
        td = team_db[r.posteam]
        w = td[td["t"].isin(prior_t)].groupby("passer_player_id")["db"].sum()
        est_id, est_db = (w.idxmax(), int(w.max())) if len(w) else (None, 0)
        hist = by_passer.get(r.starter_id)
        if hist is None:
            s_db = 0
        else:
            prior = hist[hist[:, 0] < r.t][-WINDOW_GAMES:]
            s_db = int(prior[:, 1].sum()) if len(prior) else 0
        flag = bool(est_id is not None and r.starter_id != est_id
                    and est_db >= ESTABLISHED_MIN_DB and s_db < STARTER_MAX_DB)
        rows.append((r.game_id, r.posteam, r.starter_id, est_id, est_db, s_db, flag))
    return pd.DataFrame(rows, columns=["game_id", "team", "starter_id", "established_id",
                                       "established_db", "starter_recent_db", "backup_flag"])
