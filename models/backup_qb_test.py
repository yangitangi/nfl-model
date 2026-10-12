"""
models/backup_qb_test.py
------------------------
Does the model (and the market) systematically misprice teams that start a
backup QB? Walk-forward predictions for every season from 2014 on are
compared with the closing line for team-games where the primary passer was
NOT that team's season-leading QB.

Usage:
    python models/backup_qb_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.train_model import get_feature_columns, reconstruct_margin, train_margin_model

FIRST_TEST_SEASON = 2014
MIN_LEADER_GAMES = 6


def primary_passers() -> pd.DataFrame:
    frames = []
    for path in sorted(config.RAW_DATA_DIR.glob("pbp_*.parquet")):
        cols = ["game_id", "season", "week", "play_id", "posteam", "passer_player_id",
                "passer_player_name", "qb_dropback", "season_type"]
        p = pd.read_parquet(path, columns=cols)
        p = p[(p["season_type"] == "REG") & (p["qb_dropback"] == 1)
              & p["passer_player_id"].notna() & p["posteam"].notna()]
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    g = (p.groupby(["season", "game_id", "posteam", "passer_player_id", "passer_player_name"])
         .size().reset_index(name="db"))
    # The QB who took the team's first dropback is who the line was priced
    # for -- an in-game injury replacement isn't a "backup start".
    first = (p.sort_values("play_id").groupby(["game_id", "posteam"]).head(1)
             [["season", "game_id", "posteam", "passer_player_id", "passer_player_name"]].copy())
    first["db"] = 1
    prim = first
    leader = (g.groupby(["season", "posteam", "passer_player_id"])["db"].sum().reset_index()
              .sort_values("db", ascending=False).groupby(["season", "posteam"]).head(1)
              .rename(columns={"passer_player_id": "leader_id"})[["season", "posteam", "leader_id"]])
    prim = prim.merge(leader, on=["season", "posteam"])
    leader_games = (prim[prim["passer_player_id"] == prim["leader_id"]]
                    .groupby(["season", "posteam"]).size().rename("leader_games").reset_index())
    prim = prim.merge(leader_games, on=["season", "posteam"], how="left")
    prim["backup_start"] = (prim["passer_player_id"] != prim["leader_id"]) & (prim["leader_games"] >= MIN_LEADER_GAMES)
    return prim[["game_id", "season", "posteam", "passer_player_name", "backup_start"]]


def walk_forward_predictions(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    required = feature_cols + ["actual_margin", "home_win", "spread_line"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    out = []
    for season in sorted(s for s in clean["season"].unique() if s >= FIRST_TEST_SEASON):
        train = clean[clean["season"] < season]
        test = clean[clean["season"] == season]
        if len(train) < 200 or test.empty:
            continue
        model = train_margin_model(train, feature_cols)
        pred = reconstruct_margin(test, model.predict(test[feature_cols]))
        o = test[["game_id", "season", "week", "home_team", "away_team", "spread_line", "actual_margin"]].copy()
        o["pred_margin"] = pred
        out.append(o)
        print(f"  walk-forward {season}: {len(test)} games")
    return pd.concat(out, ignore_index=True)


def main():
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")
    preds = walk_forward_predictions(df)
    prim = primary_passers()

    rows = []
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        m = preds.merge(prim, left_on=["game_id", team_col], right_on=["game_id", "posteam"], how="inner",
                        suffixes=("", "_q"))
        m = m[m["backup_start"]].copy()
        sign = 1 if side == "home" else -1
        m["team"] = m[team_col]
        m["team_actual"] = sign * m["actual_margin"]
        m["team_mkt"] = sign * m["spread_line"]
        m["team_model"] = sign * m["pred_margin"]
        rows.append(m)
    b = pd.concat(rows, ignore_index=True)
    b = b[b["team_actual"] != b["team_mkt"]].copy()

    b["edge"] = b["team_model"] - b["team_mkt"]
    b["cover"] = b["team_actual"] > b["team_mkt"]
    b["underdog"] = b["team_mkt"] < 0
    b["model_likes_team"] = b["edge"] > 0
    b["model_right"] = b["model_likes_team"] == b["cover"]

    def summarize(x: pd.DataFrame, label: str):
        n = len(x)
        if n == 0:
            print(f"{label:<44} n=0")
            return
        print(f"{label:<44} n={n:<4} cover={x['cover'].mean():.1%}  "
              f"mkt_err={ (x['team_actual'] - x['team_mkt']).mean():+.2f}  "
              f"model_err={(x['team_actual'] - x['team_model']).mean():+.2f}  "
              f"model_ATS={x['model_right'].mean():.1%}")

    print(f"\nBackup-QB team-games (leader had >= {MIN_LEADER_GAMES} starts), "
          f"{FIRST_TEST_SEASON}-{int(b['season'].max())}, vs closing line\n")
    summarize(b, "ALL backup-QB teams")
    summarize(b[b["underdog"]], "  backup team is underdog")
    summarize(b[~b["underdog"]], "  backup team is favorite")
    summarize(b[b["model_likes_team"]], "  model LIKES backup team vs market")
    summarize(b[~b["model_likes_team"]], "  model FADES backup team vs market")
    summarize(b[b["model_likes_team"] & b["underdog"]], "  model likes backup team AND underdog")
    summarize(b[(~b["model_likes_team"]) & b["underdog"]], "  model fades backup team, underdog")
    big = b[b["edge"].abs() >= 2]
    summarize(big, "  |model - market| >= 2 pts")
    summarize(big[big["model_likes_team"]], "    ...likes backup team")
    summarize(big[~big["model_likes_team"]], "    ...fades backup team")
    print("\nBy season (cover rate of backup-QB teams):")
    print(b.groupby("season").agg(n=("cover", "size"), cover=("cover", "mean")).round(3).T.to_string())
    print("\nAll games for comparison (model ATS across the same seasons):")
    allg = preds[preds["actual_margin"] != preds["spread_line"]]
    allg_edge = allg["pred_margin"] - allg["spread_line"]
    print(f"  n={len(allg)}  model_ATS={((allg_edge > 0) == (allg['actual_margin'] > allg['spread_line'])).mean():.1%}")

    b.to_pickle(config.OUTPUTS_DIR / "backup_qb_test_games.pkl")


if __name__ == "__main__":
    main()
