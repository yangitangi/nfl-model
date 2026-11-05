"""
rebuild/models/b2_linear.py
---------------------------
Candidate B2: regularized linear model of the residual (margin - closing spread) with FIXED shrinkage.

Fixes the session-2 failure mode (alpha re-picked per fold, one fold jumping to a weak penalty):
  * one alpha from {0.1, 1, 10, 100, 1e3, 1e4}, chosen ONCE on 2010-2015 (inner validation 2013/14/15),
    reused for every DEV fold;
  * an optional prediction shrink factor k in {0.25, 0.5, 1.0}: pred = spread + k * residual_hat,
    k chosen ONCE on 2010-2015 given that alpha;
  * a Huber-loss variant (sklearn HuberRegressor, epsilon 1.35, same alpha grid), to test whether
    outlier games drive the instability;
  * an interaction variant: spread x rest_diff, spread x QB shrunk EPA diff, total x wind, total x temp.
Pipeline: mean imputation -> StandardScaler -> model (all fitted on the training fold only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import DEV_TEST_SEASONS, SELECT_VAL_SEASONS, mae, residual

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
KS = [0.25, 0.5, 1.0]


def make(kind: str, alpha: float):
    reg = Ridge(alpha=alpha) if kind == "ridge" else HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=1000)
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), reg)


def _val_resid_preds(df, cols, kind, alpha):
    out = []
    for v in SELECT_VAL_SEASONS:
        tr, va = df[df.season < v], df[df.season == v]
        m = make(kind, alpha).fit(tr[cols], residual(tr))
        out.append((va, m.predict(va[cols])))
    return out


def select(df: pd.DataFrame, cols: list[str], kind: str, tune_k: bool) -> dict:
    alpha_scores = {}
    cache = {}
    for a in ALPHAS:
        vp = _val_resid_preds(df, cols, kind, a)
        cache[a] = vp
        alpha_scores[a] = float(np.mean([mae(va.spread_line + r, va.actual_margin) for va, r in vp]))
    alpha = min(alpha_scores, key=alpha_scores.get)
    k_scores = {k: float(np.mean([mae(va.spread_line + k * r, va.actual_margin) for va, r in cache[alpha]]))
                for k in (KS if tune_k else [1.0])}
    k = min(k_scores, key=k_scores.get)
    return {"alpha": alpha, "k": k, "alpha_val_mae": alpha_scores, "k_val_mae": k_scores}


def run(df: pd.DataFrame, cols: list[str], kind: str = "ridge", tune_k: bool = True) -> tuple[dict, dict]:
    sel = select(df, cols, kind, tune_k)
    folds = {}
    coefs = None
    for T in DEV_TEST_SEASONS:
        tr, te = df[df.season < T].copy(), df[df.season == T].copy()
        m = make(kind, sel["alpha"]).fit(tr[cols], residual(tr))
        te["pred_margin"] = te["spread_line"] + sel["k"] * m.predict(te[cols])
        tr["pred_margin"] = tr["spread_line"] + sel["k"] * m.predict(tr[cols])
        folds[T] = (te, tr)
        coefs = pd.Series(m[-1].coef_, index=cols)
    sel["top_coefs_last_fold"] = coefs.reindex(coefs.abs().sort_values(ascending=False).index[:8]).round(3).to_dict()
    return folds, sel
