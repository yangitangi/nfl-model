"""
rebuild/build_asof_store.py
---------------------------
Builds data/processed/asof_features.parquet: one row per regular-season game 2010-2025 with
as-of-kickoff features (see rebuild/features/*.py and rebuild/session2_report.md).

    python rebuild/bootstrap_data.py      # raw data first
    python rebuild/build_asof_store.py

build_store(raw) is a pure function of the raw tables, so the leak tests can rebuild it from
truncated raw data (rebuild/features/common.truncate_raw) and compare.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rebuild.features import context, injuries, market, qb, team_strength  # noqa: E402
from rebuild.features.common import add_gw, load_raw  # noqa: E402

OUT = ROOT / "data" / "processed" / "asof_features.parquet"
ID_COLS = ["game_id", "season", "week", "gw", "kickoff", "home_team", "away_team", "home_qb_id", "away_qb_id"]
LABEL_COLS = ["actual_margin", "actual_total", "home_win", "home_score", "away_score"]


def _side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    f = df.drop(columns="game_id")
    f = f.rename(columns={c: f"{side}_{c}" for c in f.columns if c != "team"})
    return f.rename(columns={"team": f"{side}_team"}).assign(game_id=df["game_id"].values)


def build_store(raw: dict, inject_leak: bool = False) -> pd.DataFrame:
    sched = add_gw(raw["sched"]).sort_values(["gw", "kickoff", "game_id"]).reset_index(drop=True)
    pbp, inj = raw["pbp"], raw["inj"]

    base = sched[ID_COLS].copy()
    base["actual_margin"] = sched["result"].values
    base["actual_total"] = sched["total"].values
    base["home_win"] = np.where(sched["result"].isna(), np.nan, (sched["result"] > 0).astype(float))
    base["home_score"], base["away_score"] = sched["home_score"].values, sched["away_score"].values

    per_team = [team_strength.build(sched, pbp), qb.build(sched, pbp), injuries.build(sched, pbp, inj)]
    t = per_team[0]
    for f in per_team[1:]:
        t = t.merge(f, on=["game_id", "team"], how="left")

    if inject_leak:  # deliberately leaky feature for the sanity test: the team's NEXT game EPA rating
        t = t.merge(base[["game_id", "gw"]], on="game_id").sort_values(["team", "gw"])
        t["leak_next_off_pass_epa"] = t.groupby("team")["off_pass_epa"].shift(-1)
        t = t.drop(columns="gw")

    store = base.merge(_side(t, "home"), on=["game_id", "home_team"], how="left")
    store = store.merge(_side(t, "away"), on=["game_id", "away_team"], how="left")
    store = store.merge(market.build(sched), on="game_id").merge(context.build(sched), on="game_id")
    return store.sort_values(["gw", "kickoff", "game_id"]).reset_index(drop=True)


def feature_columns(store: pd.DataFrame) -> list[str]:
    return [c for c in store.columns if c not in ID_COLS + LABEL_COLS]


def nan_report(store: pd.DataFrame) -> dict:
    return {c: int(store[c].isna().sum()) for c in feature_columns(store) if store[c].isna().any()}


def main():
    t0 = time.time()
    raw = load_raw()
    t1 = time.time()
    store = build_store(raw)
    t2 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(OUT, index=False)
    rep = {"rows": len(store), "n_features": len(feature_columns(store)), "load_s": round(t1 - t0, 1),
           "build_s": round(t2 - t1, 1), "file_mb": round(OUT.stat().st_size / 1e6, 2),
           "rows_per_season": store.groupby("season").size().to_dict(), "nan_counts": nan_report(store)}
    (ROOT / "rebuild" / "asof_store_report.json").write_text(json.dumps(rep, indent=2, default=int))
    print(json.dumps(rep, indent=2, default=int))


if __name__ == "__main__":
    main()
