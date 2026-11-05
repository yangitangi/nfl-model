"""
rebuild/session2_smoke.py
-------------------------
Session 2 smoke test on the DEV window (walk-forward, test seasons 2016-2023, train on 2010..T-1).
Uses ONLY data/processed/asof_features.parquet; never touches 2024-2025.

  (a) market      pred_margin = spread_line (residual 0)
  (b) ridge_all   ridge on the residual (actual_margin - spread_line), all store features
  (c) ridge_noqi  same, with the QB and injury groups dropped

Ridge pipeline: mean imputation (fitted on the training fold) -> StandardScaler -> Ridge.
Alpha grid {0.1, 1, 10, 100, 1e3, 1e4}, chosen inside the training window only: 3 season folds,
validating on each of the last 3 training seasons (T-3, T-2, T-1) with a model fitted on the
seasons before it; the alpha with the lowest mean validation MAE is then refitted on 2010..T-1.

    python rebuild/session2_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rebuild.build_asof_store import feature_columns  # noqa: E402
from rebuild.protocol import DEV_TEST_SEASONS, evaluate, paired_mae_diff  # noqa: E402

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
OUT_COLS = ["game_id", "season", "week", "spread_line", "actual_margin"]


def model(alpha):
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=alpha))


def pick_alpha(train: pd.DataFrame, cols: list[str]) -> tuple[float, dict]:
    last = sorted(train["season"].unique())[-3:]
    scores = {}
    for a in ALPHAS:
        maes = []
        for v in last:
            tr, va = train[train.season < v], train[train.season == v]
            m = model(a).fit(tr[cols], tr["actual_margin"] - tr["spread_line"])
            maes.append(np.abs(va["spread_line"] + m.predict(va[cols]) - va["actual_margin"]).mean())
        scores[a] = float(np.mean(maes))
    return min(scores, key=scores.get), scores


def walk_forward(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    parts, alphas, coefs = [], {}, {}
    for T in DEV_TEST_SEASONS:
        train, test = df[df.season < T], df[df.season == T]
        a, _ = pick_alpha(train, cols)
        m = model(a).fit(train[cols], train["actual_margin"] - train["spread_line"])
        o = test[OUT_COLS].copy()
        o["pred_margin"] = test["spread_line"].values + m.predict(test[cols])
        parts.append(o)
        alphas[T] = a
        coefs[T] = dict(zip(cols, m[-1].coef_))
    return pd.concat(parts, ignore_index=True), alphas, coefs


def main():
    t0 = time.time()
    store = pd.read_parquet(ROOT / "data/processed/asof_features.parquet")
    df = store[(store.season <= max(DEV_TEST_SEASONS)) & store.spread_line.notna() & store.actual_margin.notna()]
    all_cols = [c for c in feature_columns(store) if pd.api.types.is_numeric_dtype(store[c])]
    noqi = [c for c in all_cols if "_qb_" not in c and "_inj_" not in c]

    market = df[df.season.isin(DEV_TEST_SEASONS)][OUT_COLS].copy()
    market["pred_margin"] = market["spread_line"]
    preds = {"market": market}
    meta = {}
    for name, cols in (("ridge_all", all_cols), ("ridge_noqi", noqi)):
        p, alphas, coefs = walk_forward(df, cols)
        preds[name] = p
        last = pd.Series(coefs[max(DEV_TEST_SEASONS)])
        meta[name] = {"n_features": len(cols), "alpha_by_season": alphas,
                      "top_coefs_2023": last.reindex(last.abs().sort_values(ascending=False).index[:10]).round(3).to_dict()}

    # production XGBoost walk-forward predictions (read-only), DEV seasons only
    xgb = pd.read_pickle(ROOT / "outputs/walk_forward_preds_2014_2025.pkl")
    xgb = xgb[xgb.season.isin(DEV_TEST_SEASONS)][["game_id", "season", "week", "spread_line", "actual_margin", "pred_margin"]]

    results = {}
    for name, p in preds.items():
        r = evaluate(p, name, seasons=DEV_TEST_SEASONS)
        d, se = paired_mae_diff(p, preds["market"])
        r["paired_vs_market"] = {"diff": d, "se": se}
        shared = p[p.game_id.isin(xgb.game_id)]
        d2, se2 = paired_mae_diff(shared, xgb)
        r["paired_vs_xgb_shared"] = {"diff": d2, "se": se2, "n": len(shared)}
        r["by_season_mae_minus_market"] = {
            int(s): float((g.pred_margin - g.actual_margin).abs().mean() - (g.spread_line - g.actual_margin).abs().mean())
            for s, g in p.groupby("season")}
        results[name] = r
        print(f"  paired vs market: {d:+.4f} (SE {se:.4f}) | vs XGB on {len(shared)} shared games: {d2:+.4f} (SE {se2:.4f})")
    # XGBoost on the same shared-game subset, for reference
    results["xgb_baseline_shared"] = evaluate(xgb, "xgb (BASELINES.md, outputs pkl)", seasons=DEV_TEST_SEASONS)
    results["meta"] = meta
    results["seconds"] = round(time.time() - t0, 1)
    (ROOT / "rebuild/session2_smoke.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(meta, indent=2, default=float))
    print(f"{results['seconds']}s")


if __name__ == "__main__":
    main()
