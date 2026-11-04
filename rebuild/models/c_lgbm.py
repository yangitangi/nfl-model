"""
rebuild/models/c_lgbm.py
------------------------
Candidate C: LightGBM on the residual (margin - closing spread) with monotonic constraints.

Fixed settings (not tuned): max_depth 3, num_leaves 7, learning_rate 0.05, min_child_samples 150,
subsample 0.8 (freq 1), colsample_bytree 0.8, reg_lambda 1, L2 objective.
Number of trees (max 300): early stopping (30 rounds, L1 metric) on an inner season fold — fit on
seasons < T-1, validate on T-1 — then refit on all seasons < T with that many trees.
Seeds 0-4; the reported prediction is the 5-seed average, and the seed SD of MAE is reported.

Monotone signs (home margin residual):
  +1: home offense ratings, away defense-allowed ratings, home QB EPA/CPOE, away QB sack rate,
      away injury values / away QB out
  -1: the mirror images.
  0 : everything else (market, context, flags, sample sizes). Injury inputs are player-value sums, not counts.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import DEV_TEST_SEASONS, residual

SEEDS = [0, 1, 2, 3, 4]
MAX_TREES = 300
PARAMS = dict(objective="regression", max_depth=3, num_leaves=7, learning_rate=0.05, min_child_samples=150,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4)


def monotone_sign(col: str) -> int:
    side = 1 if col.startswith("home_") else -1 if col.startswith("away_") else 0
    if side == 0:
        return 0
    c = col.split("_", 1)[1]
    if c.startswith("off_") or c in ("qb_epa_db_career", "qb_epa_db_recent", "qb_cpoe"):
        return side
    if c.startswith("def_") or c in ("qb_sack_rate", "inj_off_value", "inj_def_value", "inj_qb_out"):
        return -side
    return 0


def _fit(X, y, seed, n_trees, mono, Xv=None, yv=None):
    m = lgb.LGBMRegressor(n_estimators=n_trees, random_state=seed, monotone_constraints=mono, **PARAMS)
    if Xv is None:
        return m.fit(X, y)
    return m.fit(X, y, eval_X=Xv, eval_y=yv, eval_metric="l1",
                 callbacks=[lgb.early_stopping(30, verbose=False)])


def run(df: pd.DataFrame, cols: list[str]) -> tuple[dict, dict]:
    mono = [monotone_sign(c) for c in cols]
    folds, info = {}, {"trees_by_fold": {}, "seed_mae": {}, "n_monotone": int(np.count_nonzero(mono))}
    seed_preds = {s: [] for s in SEEDS}
    for T in DEV_TEST_SEASONS:
        tr, te = df[df.season < T].copy(), df[df.season == T].copy()
        inner_tr, inner_va = tr[tr.season < T - 1], tr[tr.season == T - 1]
        p_te, p_tr, trees = [], [], []
        for s in SEEDS:
            es = _fit(inner_tr[cols], residual(inner_tr), s, MAX_TREES, mono, inner_va[cols], residual(inner_va))
            n = max(1, int(es.best_iteration_ or MAX_TREES))
            m = _fit(tr[cols], residual(tr), s, n, mono)
            p_te.append(m.predict(te[cols])); p_tr.append(m.predict(tr[cols])); trees.append(n)
            seed_preds[s].append(pd.Series(te["spread_line"].values + p_te[-1], index=te.index))
        te["pred_margin"] = te["spread_line"] + np.mean(p_te, axis=0)
        tr["pred_margin"] = tr["spread_line"] + np.mean(p_tr, axis=0)
        folds[T] = (te, tr)
        info["trees_by_fold"][T] = trees
    for s in SEEDS:
        p = pd.concat(seed_preds[s])
        info["seed_mae"][s] = float(np.abs(p - df.loc[p.index, "actual_margin"]).mean())
    info["seed_mae_sd"] = float(np.std(list(info["seed_mae"].values()), ddof=1))
    info["seed_preds"] = {s: pd.concat(v) for s, v in seed_preds.items()}
    return folds, info
