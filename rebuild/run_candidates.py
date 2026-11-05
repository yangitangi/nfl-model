"""
rebuild/run_candidates.py
-------------------------
Session 3: candidates B2 (fixed-shrinkage linear), C (monotone LightGBM) and A-lite (Kalman ratings with
the market as prior) on the DEV window (walk-forward test seasons 2016-2023). Never loads 2024-2025.

    python rebuild/run_candidates.py      -> rebuild/session3_results.json (+ printed tables)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rebuild.models import a_kalman, b2_linear, c_lgbm  # noqa: E402
from rebuild.models.common import (DEV_TEST_SEASONS, SELECT_VAL_SEASONS, add_interactions, base_features,  # noqa: E402
                                   load_dev, mae)
from rebuild.protocol import evaluate, paired_mae_diff  # noqa: E402

BREAK_EVEN = 0.524


def calibrate(folds: dict) -> pd.DataFrame:
    """Logistic calibration of pred_margin -> home win, fitted on each fold's TRAINING-window predictions."""
    parts = []
    for T, (te, tr) in folds.items():
        lr = LogisticRegression(C=1e6).fit(tr[["pred_margin"]].values, (tr["actual_margin"] > 0).astype(int))
        te = te.copy()
        te["pred_home_win_prob"] = lr.predict_proba(te[["pred_margin"]].values)[:, 1]
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def ats_edge2_by_season(p: pd.DataFrame) -> dict:
    out = {}
    for s, g in p.groupby("season"):
        g = g[(g.pred_margin - g.spread_line).abs() >= 2]
        lean, cov = np.sign(g.pred_margin - g.spread_line), np.sign(g.actual_margin - g.spread_line)
        ok = (lean != 0) & (cov != 0)
        out[int(s)] = {"ats": float((lean[ok] == cov[ok]).mean()) if ok.sum() else None, "n": int(ok.sum())}
    return out


def score(p: pd.DataFrame, name: str, market: pd.DataFrame, xgb: pd.DataFrame | None) -> dict:
    r = evaluate(p, name, seasons=DEV_TEST_SEASONS)
    d, se = paired_mae_diff(p, market)
    if "pred_home_win_prob" in p.columns and "pred_home_win_prob" in market.columns:
        m = p[["game_id", "pred_home_win_prob", "actual_margin"]].merge(
            market[["game_id", "pred_home_win_prob"]], on="game_id", suffixes=("", "_mkt"))
        win = (m.actual_margin > 0).astype(float)
        bd = (m.pred_home_win_prob - win) ** 2 - (m.pred_home_win_prob_mkt - win) ** 2
        r["paired_brier_vs_market"] = {"diff": float(bd.mean()), "se": float(bd.std(ddof=1) / np.sqrt(len(bd)))}
    r["paired_vs_market"] = {"diff": d, "se": se, "z": d / se if se > 0 else None}
    per = {int(s): mae(g.pred_margin, g.actual_margin) - mae(g.spread_line, g.actual_margin)
           for s, g in p.groupby("season")}
    r["by_season_mae_minus_market"] = per
    r["seasons_better_than_market"] = int(sum(v < 0 for v in per.values()))
    e2 = ats_edge2_by_season(p)
    r["ats_edge2_by_season"] = e2
    r["folds_edge2_above_breakeven"] = int(sum((v["ats"] or 0) > BREAK_EVEN for v in e2.values()))
    if xgb is not None:
        sh = p[p.game_id.isin(xgb.game_id)]
        d2, se2 = paired_mae_diff(sh, xgb)
        r["paired_vs_xgb_shared"] = {"diff": d2, "se": se2, "n": len(sh)}
    print(f"   paired vs market {d:+.4f} (SE {se:.4f}); seasons better {r['seasons_better_than_market']}/8; "
          f"edge>=2 folds above 52.4%: {r['folds_edge2_above_breakeven']}/8")
    return r


def z_buckets(p: pd.DataFrame) -> dict:
    """A-lite: ATS by |pred_resid / post_sd| quintile, vs. ATS by |edge| quintile, plus sd informativeness."""
    d = p.copy()
    d["edge"] = d.pred_margin - d.spread_line
    d["z"] = d.edge / d.post_sd
    lean, cov = np.sign(d.edge), np.sign(d.actual_margin - d.spread_line)
    d["ok"] = (lean != 0) & (cov != 0)
    d["hit"] = (lean == cov)
    res = {}
    for key in ("z", "edge"):
        q = pd.qcut(d[key].abs(), 5, labels=False)
        rows = []
        for b in range(5):
            g = d[(q == b) & d.ok]
            rows.append({"bucket": f"Q{b + 1}", "abs_lo": float(d[key].abs()[q == b].min()),
                         "abs_hi": float(d[key].abs()[q == b].max()), "ats": float(g.hit.mean()), "n": int(len(g))})
        res[f"ats_by_abs_{key}_quintile"] = rows
    err = (d.actual_margin - d.pred_margin).abs()
    res["corr_post_sd_vs_abs_error"] = float(np.corrcoef(d.post_sd, err)[0, 1])
    res["post_sd_mean"], res["post_sd_min"], res["post_sd_max"] = (float(d.post_sd.mean()), float(d.post_sd.min()),
                                                                   float(d.post_sd.max()))
    res["by_season_post_sd"] = d.groupby("season").post_sd.mean().round(3).to_dict()
    # exploratory (thresholds come from DEV itself): per-season ATS in the top |z| quintile
    top = d[(pd.qcut(d["z"].abs(), 5, labels=False) == 4) & d.ok]
    res["top_z_quintile_ats_by_season"] = {int(s): {"ats": float(g.hit.mean()), "n": int(len(g))}
                                           for s, g in top.groupby("season")}
    return res


def totals(df: pd.DataFrame, cols: list[str], kalman_full: pd.DataFrame) -> dict:
    """Ridge on (actual_total - total_line), alpha chosen once on 2010-2015; plus A-lite's total."""
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def mk(a):
        return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=a))
    d = df[df.total_line.notna() & df.actual_total.notna()]
    y = lambda x: (x.actual_total - x.total_line).values  # noqa: E731
    sc = {}
    for a in b2_linear.ALPHAS:
        sc[a] = float(np.mean([mae(d[d.season == v].total_line + mk(a).fit(d[d.season < v][cols], y(d[d.season < v]))
                                   .predict(d[d.season == v][cols]), d[d.season == v].actual_total)
                               for v in SELECT_VAL_SEASONS]))
    alpha = min(sc, key=sc.get)
    parts = []
    for T in DEV_TEST_SEASONS:
        tr, te = d[d.season < T], d[d.season == T].copy()
        te["pred_total"] = te.total_line + mk(alpha).fit(tr[cols], y(tr)).predict(te[cols])
        parts.append(te)
    out = {"alpha": alpha, "alpha_val_mae": sc}
    for name, p in (("ridge_total", pd.concat(parts)),
                    ("a_lite_total", kalman_full[kalman_full.season.isin(DEV_TEST_SEASONS)])):
        e_m, e_l = (p.pred_total - p.actual_total).abs(), (p.total_line - p.actual_total).abs()
        diff = e_m - e_l
        lean, res = np.sign(p.pred_total - p.total_line), np.sign(p.actual_total - p.total_line)
        ok = (lean != 0) & (res != 0)
        out[name] = {"n": int(len(p)), "mae": float(e_m.mean()), "line_mae": float(e_l.mean()),
                     "paired_diff": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                     "ou_acc": float((lean[ok] == res[ok]).mean()), "ou_n": int(ok.sum()),
                     "by_season_mae_minus_line": {int(s): float(g.mean()) for s, g in diff.groupby(p.season)}}
        o = out[name]
        print(f"== {name}: MAE {o['mae']:.3f} vs line {o['line_mae']:.3f}; paired {o['paired_diff']:+.4f} "
              f"(SE {o['se']:.4f}); O/U {o['ou_acc']:.1%} (n={o['ou_n']})")
    return out


def main():
    t0 = time.time()
    df = load_dev()
    cols = base_features(df)
    dfi = add_interactions(df)
    cols_ix = cols + [c for c in dfi.columns if c.startswith("ix_")]
    timings, meta, folds = {}, {}, {}

    folds["market"] = {T: (df[df.season == T].assign(pred_margin=lambda x: x.spread_line),
                           df[df.season < T].assign(pred_margin=lambda x: x.spread_line)) for T in DEV_TEST_SEASONS}
    specs = [("B2_ridge_fixed_alpha", lambda: b2_linear.run(df, cols, "ridge", tune_k=False)),
             ("B2_ridge_shrink_k", lambda: b2_linear.run(df, cols, "ridge", tune_k=True)),
             ("B2_huber_shrink_k", lambda: b2_linear.run(df, cols, "huber", tune_k=True)),
             ("B2_ridge_interactions_k", lambda: b2_linear.run(dfi, cols_ix, "ridge", tune_k=True)),
             ("C_lgbm_monotone_5seed", lambda: c_lgbm.run(df, cols)),
             ("A_lite_kalman", lambda: a_kalman.run(df))]
    for name, fn in specs:
        t = time.time()
        folds[name], meta[name] = fn()
        timings[name] = round(time.time() - t, 1)
        print(f"[{name}] {timings[name]}s")

    xgb = pd.read_pickle(ROOT / "outputs/walk_forward_preds_2014_2025.pkl")
    xgb = xgb[xgb.season.isin(DEV_TEST_SEASONS)][["game_id", "season", "week", "spread_line", "actual_margin",
                                                    "pred_margin"]]
    market_preds = calibrate(folds["market"])

    results = {"xgb_baseline": score(xgb, "XGBoost (BASELINES.md)", xgb.assign(pred_margin=xgb.spread_line), None)}
    preds = {}
    for name in folds:
        preds[name] = calibrate(folds[name])
        results[name] = score(preds[name], name, market_preds, xgb)

    c_info = meta["C_lgbm_monotone_5seed"]
    seed_preds = c_info.pop("seed_preds")
    c_info["seed_paired_vs_market"] = {}
    for s, sp in seed_preds.items():
        p = df.loc[sp.index, ["game_id", "actual_margin"]].assign(pred_margin=sp.values)
        c_info["seed_paired_vs_market"][s] = paired_mae_diff(p, market_preds)[0]
    c_info["seed_paired_sd"] = float(np.std(list(c_info["seed_paired_vs_market"].values()), ddof=1))

    results["A_lite_z_analysis"] = z_buckets(preds["A_lite_kalman"])
    kal_full = pd.concat([folds["A_lite_kalman"][T][0] for T in DEV_TEST_SEASONS])
    results["totals"] = totals(df, cols, kal_full)
    results["meta"] = meta
    results["timings_s"] = {**timings, "total": round(time.time() - t0, 1)}
    (ROOT / "rebuild/session3_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({"meta": meta, "z": results["A_lite_z_analysis"], "timings": results["timings_s"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
