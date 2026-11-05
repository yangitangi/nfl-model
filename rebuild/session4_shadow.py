"""
rebuild/session4_shadow.py
--------------------------
Session 4, step 2: 2026 shadow backfill with the FROZEN F1 (one fit on 2010-2025, no refit in 2026) and the
frozen A-lite z (L1), compared with the production model's 2026 log (read-only copy from origin/master).

Data: data/raw/shadow2026/ (gitignored) holds symlinks to the 2010-2025 raw files plus fresh nflverse downloads
in dl/ (play_by_play_2026, injuries_2026, games.parquet). The session-2 feature code is reused unchanged; the
season calendar is extended to 2026 at runtime (common.LAST_SEASON), and the script asserts that every
2010-2025 row is identical to the committed store.

    python rebuild/session4_shadow.py <prod_log.parquet> <weekly_results.csv>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rebuild.features.common as common  # noqa: E402
from rebuild.build_asof_store import build_store, feature_columns  # noqa: E402
from rebuild.models.a_kalman import kalman  # noqa: E402

RAW = ROOT / "data/raw/shadow2026"
FROZEN = json.loads((ROOT / "rebuild/session4_frozen.json").read_text())
SEASON = 2026


def prepare_raw_dir():
    dl = RAW / "dl"
    shutil.copyfile(dl / "games.parquet", RAW / "schedules.parquet")
    shutil.copyfile(dl / "play_by_play_2026.parquet", RAW / "pbp_2026.parquet")
    shutil.copyfile(dl / "injuries_2026.parquet", RAW / "injuries_2026.parquet")


def sign_cover(pred, line, act):
    lean, cov = np.sign(pred - line), np.sign(act - line)
    return np.where((lean == 0) | (cov == 0), np.nan, (lean == cov).astype(float))


def main(prod_log: str, weekly: str):
    prepare_raw_dir()
    common.LAST_SEASON = SEASON                      # runtime calendar extension; no source edits
    raw = common.load_raw(2010, SEASON, RAW)
    store = build_store(raw)

    old = pd.read_parquet(ROOT / "data/processed/asof_features.parquet").set_index("game_id").sort_index()
    new = store[store.season <= 2025].set_index("game_id").sort_index()
    cols = feature_columns(old)
    assert old.index.equals(new.index)
    pd.testing.assert_frame_equal(old[cols], new[cols], check_exact=False, rtol=0, atol=1e-9)

    done = store[store.spread_line.notna() & store.actual_margin.notna()].sort_values(["gw", "kickoff", "game_id"])
    done = done.reset_index(drop=True)
    tr, te = done[done.season <= 2025], done[done.season == SEASON].copy()
    feats, k = FROZEN["F1"]["features"], FROZEN["F1"]["k"]
    m = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=FROZEN["F1"]["alpha"]))
    m.fit(tr[feats], tr.actual_margin - tr.spread_line)
    te["f1_spread"] = te.spread_line + k * m.predict(te[feats])

    L = FROZEN["L1"]
    _, pr, pv, _ = kalman(done, L["rho"], L["q"], L["p0"], L["r"])
    done["alite_z"] = pr / np.sqrt(pv)
    te["alite_z"] = done.loc[done.season == SEASON, "alite_z"].values
    te["l1_top_quintile"] = (te.alite_z.abs() >= L["top_quintile_abs_z_cut"]).astype(int)

    w = pd.read_csv(weekly)[["game_id", "pred_margin", "vegas_spread_line", "spread_correct"]]
    w = w.rename(columns={"pred_margin": "prod_pred_margin", "vegas_spread_line": "prod_logged_line"})
    lg = pd.read_parquet(prod_log)[["game_id", "close_spread_line"]].rename(columns={"close_spread_line": "prod_log_close_line"})
    te = te.merge(w, on="game_id", how="left").merge(lg, on="game_id", how="left")

    out = pd.DataFrame({
        "game_id": te.game_id, "week": te.week, "home_team": te.home_team, "away_team": te.away_team,
        "market_spread": te.spread_line, "f1_spread": te.f1_spread.round(3),
        "f1_edge": (te.f1_spread - te.spread_line).round(3), "alite_z": te.alite_z.round(4),
        "l1_top_quintile": te.l1_top_quintile, "actual_margin": te.actual_margin,
        "home_covered": np.where(te.actual_margin == te.spread_line, np.nan,
                                 (te.actual_margin > te.spread_line).astype(float)),
        "f1_pick_covered": sign_cover(te.f1_spread, te.spread_line, te.actual_margin),
        "l1_fade_covered": np.where(te.l1_top_quintile == 1,
                                    sign_cover(te.spread_line - te.alite_z, te.spread_line, te.actual_margin), np.nan),
        "prod_pred_margin": te.prod_pred_margin.round(3), "prod_logged_line": te.prod_logged_line,
        "prod_pick_covered_vs_logged_line": sign_cover(te.prod_pred_margin, te.prod_logged_line, te.actual_margin),
        "prod_pick_covered_vs_close": sign_cover(te.prod_pred_margin, te.spread_line, te.actual_margin),
    })
    out.to_csv(ROOT / "rebuild/shadow_2026.csv", index=False)

    def ats(col, mask=None):
        s = out[col] if mask is None else out.loc[mask, col]
        s = s.dropna()
        return {"ats": float(s.mean()) if len(s) else None, "n": int(len(s)),
                "se": float(np.sqrt(s.mean() * (1 - s.mean()) / len(s))) if len(s) else None}

    err = lambda p: (p - out.actual_margin).abs()  # noqa: E731
    both = out.prod_pred_margin.notna()
    dif = err(out.f1_spread) - err(out.market_spread)
    difp = (err(out.prod_pred_margin) - err(out.market_spread))[both]
    summ = {
        "weeks": sorted(int(x) for x in out.week.unique()), "n_games": int(len(out)),
        "n_games_with_prod_log": int(both.sum()),
        "mae": {"market": float(err(out.market_spread).mean()), "f1": float(err(out.f1_spread).mean()),
                "prod_same_games": float(err(out.prod_pred_margin)[both].mean()),
                "market_same_games_as_prod": float(err(out.market_spread)[both].mean())},
        "f1_minus_market": {"diff": float(dif.mean()), "se": float(dif.std(ddof=1) / np.sqrt(len(dif)))},
        "prod_minus_market": {"diff": float(difp.mean()), "se": float(difp.std(ddof=1) / np.sqrt(len(difp)))},
        "f1_ats": ats("f1_pick_covered"), "f1_ats_same_games_as_prod": ats("f1_pick_covered", both),
        "prod_ats_vs_logged_line": ats("prod_pick_covered_vs_logged_line"),
        "prod_ats_vs_nflverse_close": ats("prod_pick_covered_vs_close"),
        "prod_spread_correct_column": {"rate": float(te.spread_correct.dropna().astype(float).mean()),
                                       "n": int(te.spread_correct.notna().sum())},
        "f1_max_abs_edge": float(out.f1_edge.abs().max()),
        "l1_fade": ats("l1_fade_covered"), "l1_share_top_quintile": float(out.l1_top_quintile.mean()),
        "logged_line_differs_from_nflverse_close": int((out.prod_logged_line != out.market_spread)[both].sum()),
    }
    (ROOT / "rebuild/session4_shadow.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
