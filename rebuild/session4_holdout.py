"""
rebuild/session4_holdout.py
---------------------------
Session 4, step 1: score the pre-registered F1 and L1 (rebuild/PREREGISTRATION.md, parameters from
rebuild/session4_frozen.json) on the 2024-2025 holdout. RUN ONCE. Writes rebuild/session4_holdout.json
before printing anything.

    python rebuild/session4_holdout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rebuild.models.a_kalman import kalman  # noqa: E402
from rebuild.protocol import FINAL_HOLDOUT_SEASONS, evaluate, paired_mae_diff  # noqa: E402

FROZEN = json.loads((ROOT / "rebuild/session4_frozen.json").read_text())
OUT = ROOT / "rebuild/session4_holdout.json"
BREAK_EVEN = 0.524


def f1_model():
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=FROZEN["F1"]["alpha"]))


def calib(train_pred, train_margin, test_pred):
    lr = LogisticRegression(C=1e6).fit(np.asarray(train_pred).reshape(-1, 1), (np.asarray(train_margin) > 0).astype(int))
    return lr.predict_proba(np.asarray(test_pred).reshape(-1, 1))[:, 1]


def ats_mask(pred, line, act, mask):
    lean, cov = np.sign(pred - line)[mask], np.sign(act - line)[mask]
    ok = (lean != 0) & (cov != 0)
    return float((lean[ok] == cov[ok]).mean()) if ok.sum() else None, int(ok.sum())


def load_all() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/processed/asof_features.parquet")
    df = df[(df.season <= max(FINAL_HOLDOUT_SEASONS)) & df.spread_line.notna() & df.actual_margin.notna()]
    return df.sort_values(["gw", "kickoff", "game_id"]).reset_index(drop=True)


def main():
    df = load_all()
    feats, k = FROZEN["F1"]["features"], FROZEN["F1"]["k"]
    res = {"holdout_seasons": FINAL_HOLDOUT_SEASONS}

    # ---- F1 and market, walk-forward over the holdout seasons
    f1_parts, mk_parts = [], []
    for T in FINAL_HOLDOUT_SEASONS:
        tr, te = df[df.season < T], df[df.season == T].copy()
        m = f1_model().fit(tr[feats], tr.actual_margin - tr.spread_line)
        tr_pred = tr.spread_line + k * m.predict(tr[feats])
        te["pred_margin"] = te.spread_line + k * m.predict(te[feats])
        te["pred_home_win_prob"] = calib(tr_pred, tr.actual_margin, te.pred_margin)
        f1_parts.append(te)
        mk = te.copy()
        mk["pred_margin"] = mk.spread_line
        mk["pred_home_win_prob"] = calib(tr.spread_line, tr.actual_margin, mk.spread_line)
        mk_parts.append(mk)
    f1, mk = pd.concat(f1_parts), pd.concat(mk_parts)

    res["F1"] = evaluate(f1, "F1 (frozen B2 ridge + k)", seasons=FINAL_HOLDOUT_SEASONS, verbose=False)
    res["market"] = evaluate(mk, "market", seasons=FINAL_HOLDOUT_SEASONS, verbose=False)
    d, se = paired_mae_diff(f1, mk)
    res["F1"]["paired_vs_market"] = {"diff": d, "se": se}
    win = (f1.actual_margin > 0).astype(float).values
    bd = (f1.pred_home_win_prob.values - win) ** 2 - (mk.pred_home_win_prob.values - win) ** 2
    res["F1"]["paired_brier_vs_market"] = {"diff": float(bd.mean()), "se": float(bd.std(ddof=1) / np.sqrt(len(bd)))}
    edge = (f1.pred_margin - f1.spread_line).abs().values
    a2, n2 = ats_mask(f1.pred_margin.values, f1.spread_line.values, f1.actual_margin.values, edge >= 2)
    res["F1"]["ats_edge_ge2"] = {"ats": a2, "n": n2}
    res["F1"]["max_abs_edge"] = float(edge.max())
    res["F1"]["by_season_mae_minus_market"] = {
        int(s): float((g.pred_margin - g.actual_margin).abs().mean() - (g.spread_line - g.actual_margin).abs().mean())
        for s, g in f1.groupby("season")}
    res["F1"]["passes"] = bool(d <= 0 or (a2 is not None and a2 >= BREAK_EVEN and n2 >= 100))

    # ---- production XGBoost (BASELINES.md source), holdout seasons
    xgb = pd.read_pickle(ROOT / "outputs/walk_forward_preds_2014_2025.pkl")
    xgb = xgb[xgb.season.isin(FINAL_HOLDOUT_SEASONS)][["game_id", "season", "week", "spread_line", "actual_margin",
                                                        "pred_margin"]]
    res["xgb"] = evaluate(xgb, "production XGBoost", seasons=FINAL_HOLDOUT_SEASONS, verbose=False)
    dx, sex = paired_mae_diff(xgb, xgb.assign(pred_margin=xgb.spread_line))
    res["xgb"]["paired_vs_market"] = {"diff": dx, "se": sex}
    sh = f1[f1.game_id.isin(xgb.game_id)]
    d2, se2 = paired_mae_diff(sh, xgb)
    res["F1"]["paired_vs_xgb_shared"] = {"diff": d2, "se": se2, "n": int(len(sh))}
    res["xgb"]["by_season_mae_minus_market"] = {
        int(s): float((g.pred_margin - g.actual_margin).abs().mean() - (g.spread_line - g.actual_margin).abs().mean())
        for s, g in xgb.groupby("season")}

    # ---- L1: fade A-lite's top-|z| quintile (frozen params and cut)
    L = FROZEN["L1"]
    _, pr, pv, _ = kalman(df, L["rho"], L["q"], L["p0"], L["r"])
    df["z"] = pr / np.sqrt(pv)
    h = df[df.season.isin(FINAL_HOLDOUT_SEASONS)]
    top = h[h.z.abs() >= L["top_quintile_abs_z_cut"]]
    fade = -np.sign(top.z.values)
    cover = np.sign((top.actual_margin - top.spread_line).values)
    ok = cover != 0
    wins = (fade[ok] == cover[ok])
    n = int(ok.sum())
    p = float(wins.mean()) if n else float("nan")
    half = 1.96 * np.sqrt(p * (1 - p) / n) if n else float("nan")
    res["L1"] = {"n_holdout_games": int(len(h)), "n_top_quintile": int(len(top)),
                 "share_top_quintile": float(len(top) / len(h)), "n_bets_excl_push": n,
                 "fade_ats": p, "ci95": [p - half, p + half],
                 "by_season": {int(s): {"n": int(((g.actual_margin - g.spread_line) != 0).sum()),
                                        "fade_ats": float((-np.sign(g.z) == np.sign(g.actual_margin - g.spread_line))
                                                          [(g.actual_margin - g.spread_line) != 0].mean())}
                               for s, g in top.groupby("season")}}
    res["L1"]["passes"] = bool(n >= 150 and p >= BREAK_EVEN and p - half > 0.5)

    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
