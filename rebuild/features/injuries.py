"""
rebuild/features/injuries.py
----------------------------
Player-level injury value from the final pre-game injury report (no counts, no zero-reading artifact).

For each player on the team's report for the game's week:
  status weight: Out 1.0, Doubtful 1.0, Questionable 0.25, Probable/other/blank 0.
  usage (PRIOR games only): the player's share of the team's usage over the team's previous 6 games
    (games he missed count as zero, so a long-term absence the market already prices has low usage):
      QB          dropback share, ref 0.5
      WR/TE/RB/FB share of targets + carries, refs WR 0.18, TE 0.12, RB 0.25, FB 0.05
      defenders   share of defensive involvements (solo/assist tackles, sacks, half sacks, TFL,
                  QB hits, passes defensed, INTs), ref 0.08
      OL (T/G/C)  no usage signal in pbp and no snap counts in data/raw -> constant 0.6 (limitation)
      K/P/LS      0
    usage = min(1, share / ref).
  value = status weight x position value x usage. Position values (fixed, not fitted):
      off: WR 1.0, T 0.8, TE 0.7, RB 0.6, C 0.6, G 0.5, FB 0.2;  def: DE 0.9, CB 0.9, OLB 0.8,
      DT 0.6, LB 0.6, S/FS/SS/DB 0.6, NT 0.5, ILB/MLB 0.5.
Team outputs: inj_off_value (non-QB offense), inj_def_value, inj_qb_out (max over the team's QBs of
status weight x QB usage, so 1.0 = the usual starter is Out/Doubtful).
Missing report (no rows for the team-week): inj_report_missing = 1 and values carried forward from
the team's previous game (0 if none). Never imputed as an explicit zero reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import team_games

STATUS_W = {"Out": 1.0, "Doubtful": 1.0, "Questionable": 0.25}
OFF_VALUE = {"WR": 1.0, "T": 0.8, "OT": 0.8, "TE": 0.7, "RB": 0.6, "C": 0.6, "G": 0.5, "OG": 0.5, "FB": 0.2}
DEF_VALUE = {"DE": 0.9, "CB": 0.9, "OLB": 0.8, "DT": 0.6, "LB": 0.6, "S": 0.6, "FS": 0.6, "SS": 0.6,
             "DB": 0.6, "NT": 0.5, "ILB": 0.5, "MLB": 0.5}
REF = {"QB": 0.5, "WR": 0.18, "TE": 0.12, "RB": 0.25, "FB": 0.05}
DEF_REF = 0.08
OL_USAGE = 0.6
OL = {"T", "OT", "G", "OG", "C"}
WINDOW = 6
DEF_COLS = ["solo_tackle_1_player_id", "assist_tackle_1_player_id", "sack_player_id", "half_sack_1_player_id",
            "tackle_for_loss_1_player_id", "qb_hit_1_player_id", "pass_defense_1_player_id",
            "interception_player_id"]


def _usage_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """Long table (game_id, team, player, cat) with one row per usage event."""
    ev = []
    db = pbp[pbp["qb_dropback"] == 1]
    ev.append(pd.DataFrame({"game_id": db["game_id"], "team": db["posteam"],
                            "player": db["passer_player_id"].fillna(db["rusher_player_id"]), "cat": "db"}))
    tgt = pbp[(pbp["pass"] == 1) & pbp["receiver_player_id"].notna()]
    ev.append(pd.DataFrame({"game_id": tgt["game_id"], "team": tgt["posteam"], "player": tgt["receiver_player_id"],
                            "cat": "touch"}))
    car = pbp[(pbp["rush"] == 1) & (pbp["qb_dropback"] != 1) & pbp["rusher_player_id"].notna()]
    ev.append(pd.DataFrame({"game_id": car["game_id"], "team": car["posteam"], "player": car["rusher_player_id"],
                            "cat": "touch"}))
    for c in DEF_COLS:
        d = pbp[pbp[c].notna()]
        ev.append(pd.DataFrame({"game_id": d["game_id"], "team": d["defteam"], "player": d[c], "cat": "def"}))
    ev = pd.concat(ev, ignore_index=True)
    return ev[ev["player"].notna()]


def build(sched: pd.DataFrame, pbp: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    tg = team_games(sched)
    tg["seq"] = tg.groupby("team").cumcount()
    seq = tg.set_index(["game_id", "team"])["seq"]

    ev = _usage_events(pbp)
    pg = ev.groupby(["game_id", "team", "player", "cat"]).size().rename("cnt").reset_index()
    tt = ev.groupby(["game_id", "team", "cat"]).size().rename("tot").reset_index()
    pg["seq"] = seq.reindex(pd.MultiIndex.from_frame(pg[["game_id", "team"]])).values
    tt["seq"] = seq.reindex(pd.MultiIndex.from_frame(tt[["game_id", "team"]])).values
    pg, tt = pg[pg["seq"].notna()], tt[tt["seq"].notna()]

    # report rows -> team-game
    r = inj.merge(tg[["season", "week", "team", "game_id", "seq"]], on=["season", "week", "team"], how="inner")
    has_report = r.groupby(["game_id", "team"]).size()
    r["w"] = r["report_status"].map(STATUS_W).fillna(0.0)
    r["position"] = r["position"].astype(str).str.strip()
    r = r[(r["w"] > 0) & r["gsis_id"].notna()].drop_duplicates(["game_id", "team", "gsis_id"])
    r["cat"] = np.select([r["position"] == "QB", r["position"].isin(list(REF)), r["position"].isin(list(DEF_VALUE))],
                         ["db", "touch", "def"], "none")

    # player share over the team's previous WINDOW games
    x = r[["game_id", "team", "gsis_id", "cat", "seq"]].merge(
        pg.rename(columns={"player": "gsis_id", "seq": "pseq", "game_id": "pgame"}), on=["team", "gsis_id", "cat"],
        how="left")
    x = x[(x["pseq"] < x["seq"]) & (x["pseq"] >= x["seq"] - WINDOW)]
    num = x.groupby(["game_id", "team", "gsis_id"])["cnt"].sum()
    # team totals over the same window
    t = r[["game_id", "team", "cat", "seq"]].drop_duplicates(["game_id", "team", "cat"]).merge(
        tt.rename(columns={"seq": "tseq", "game_id": "tgame"}), on=["team", "cat"], how="left")
    t = t[(t["tseq"] < t["seq"]) & (t["tseq"] >= t["seq"] - WINDOW)]
    den = t.groupby(["game_id", "team", "cat"])["tot"].sum()

    r["num"] = num.reindex(pd.MultiIndex.from_frame(r[["game_id", "team", "gsis_id"]])).fillna(0).values
    r["den"] = den.reindex(pd.MultiIndex.from_frame(r[["game_id", "team", "cat"]])).values
    share = np.where(r["den"] > 0, r["num"] / r["den"].where(r["den"] > 0, 1), 0.0)
    ref = r["position"].map(REF).fillna(DEF_REF).values
    usage = np.minimum(1.0, share / ref)
    usage = np.where(r["position"].isin(OL), OL_USAGE, usage)
    usage = np.where(r["cat"].eq("none") & ~r["position"].isin(OL), 0.0, usage)
    r["usage"] = usage
    pos_val = r["position"].map({**OFF_VALUE, **DEF_VALUE}).fillna(0.0)
    r["value"] = r["w"] * pos_val * r["usage"]
    is_def = r["position"].isin(list(DEF_VALUE))

    out = tg[["game_id", "team", "seq"]].copy()
    idx = pd.MultiIndex.from_frame(out[["game_id", "team"]])
    out["inj_off_value"] = r[~is_def & (r["position"] != "QB")].groupby(["game_id", "team"])["value"].sum().reindex(idx).fillna(0).values
    out["inj_def_value"] = r[is_def].groupby(["game_id", "team"])["value"].sum().reindex(idx).fillna(0).values
    q = r[r["position"] == "QB"]
    out["inj_qb_out"] = (q["w"] * q["usage"]).groupby([q["game_id"], q["team"]]).max().reindex(idx).fillna(0).values
    out["inj_report_missing"] = (~idx.isin(has_report.index)).astype(int)
    vals = ["inj_off_value", "inj_def_value", "inj_qb_out"]
    out.loc[out["inj_report_missing"] == 1, vals] = np.nan
    out[vals] = out.groupby("team")[vals].ffill().fillna(0.0)
    return out.drop(columns="seq")
