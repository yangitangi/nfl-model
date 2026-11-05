"""
rebuild/models/a_kalman.py
--------------------------
Candidate A-lite: state-space team ratings with the MARKET AS PRIOR (closed-form Kalman filter, numpy).

State: for each of 32 franchises an offense rating o and a defense rating d (points vs. market expectation;
d > 0 = good defense). 64-dim Gaussian state x ~ N(m, P). The ratings describe only what the line misses.

Observations per game (two, in points):
  y_home = home_score - (total_line + spread_line)/2  =  o_home - d_away + e
  y_away = away_score - (total_line - spread_line)/2  =  o_away - d_home + e,   e ~ N(0, r)
i.e. the market-implied team scores are the baseline.
Dynamics:
  each new global week:  P += q I                       (ratings drift)
  each new season:       m <- rho m ; P <- rho^2 P + (1 - rho^2) p0 I
                         (last season's posterior regressed toward 0 with shrink weight rho)
  start (2010):          m = 0, P = p0 I
Prediction for a game (state at the START of its week, so same-week results are never used):
  margin residual  = (o_h - d_a) - (o_a - d_h) = h . m,   posterior variance h P h'
  pred_margin = spread_line + h . m,  pred_total = total_line + (o_h - d_a + o_a - d_h)
Parameters (rho, q, p0, r) are fitted ONCE by maximising the one-step-ahead Gaussian predictive
log-likelihood of the observations on 2010-2015 (scipy Nelder-Mead), then frozen for every DEV fold.
Within a test season the filter updates week by week from completed games (that is state, not fitting),
exactly like the as-of rolling features of the other candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .common import SELECT_LAST

N_TEAMS = 32


def _setup(df: pd.DataFrame):
    teams = sorted(set(df.home_team) | set(df.away_team))
    assert len(teams) == N_TEAMS
    ti = {t: i for i, t in enumerate(teams)}
    h = df.home_team.map(ti).values
    a = df.away_team.map(ti).values
    yh = (df.home_score - (df.total_line + df.spread_line) / 2).values
    ya = (df.away_score - (df.total_line - df.spread_line) / 2).values
    return h, a, yh, ya


def kalman(df: pd.DataFrame, rho: float, q: float, p0: float, r: float, collect: bool = True):
    """df sorted by gw. Returns (loglik, pred_margin_resid, post_var_margin, pred_total_resid)."""
    h, a, yh, ya = _setup(df)
    gw, season = df.gw.values, df.season.values
    n, k = len(df), 2 * N_TEAMS
    m, P = np.zeros(k), np.eye(k) * p0
    I = np.eye(k)
    out_r, out_v, out_t = np.zeros(n), np.zeros(n), np.zeros(n)
    ll = 0.0
    cur_gw, cur_season = None, season[0]
    starts = np.flatnonzero(np.r_[True, gw[1:] != gw[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        if season[s] != cur_season:
            m = rho * m
            P = rho ** 2 * P + (1 - rho ** 2) * p0 * I
            cur_season = season[s]
        elif cur_gw is not None:
            P = P + q * I
        cur_gw = gw[s]
        # predictions for the whole week from the start-of-week state
        for i in range(s, e):
            H = np.zeros((2, k))
            H[0, h[i]], H[0, N_TEAMS + a[i]] = 1, -1
            H[1, a[i]], H[1, N_TEAMS + h[i]] = 1, -1
            hm = H[0] - H[1]
            out_r[i] = hm @ m
            out_v[i] = hm @ P @ hm
            out_t[i] = (H[0] + H[1]) @ m
        # sequential updates with the week's results
        for i in range(s, e):
            idx = [h[i], a[i], N_TEAMS + h[i], N_TEAMS + a[i]]
            H = np.zeros((2, 4))
            H[0, 0], H[0, 3] = 1, -1   # o_h - d_a
            H[1, 1], H[1, 2] = 1, -1   # o_a - d_h
            Pi = P[:, idx]                        # k x 4
            S = H @ P[np.ix_(idx, idx)] @ H.T + r * np.eye(2)
            v = np.array([yh[i], ya[i]]) - H @ m[idx]
            Sinv = np.linalg.inv(S)
            K = Pi @ H.T @ Sinv                   # k x 2
            m = m + K @ v
            P = P - K @ (H @ Pi.T)
            ll += -0.5 * (np.log(np.linalg.det(S)) + v @ Sinv @ v + 2 * np.log(2 * np.pi))
    return ll, out_r, out_v, out_t


def fit_params(df: pd.DataFrame) -> dict:
    d = df[df.season <= SELECT_LAST]

    def nll(theta):
        rho = 1 / (1 + np.exp(-theta[0]))
        q, p0, r = np.exp(theta[1:])
        return -kalman(d, rho, q, p0, r)[0]

    x0 = np.array([0.0, np.log(0.05), np.log(4.0), np.log(90.0)])
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-2})
    rho = float(1 / (1 + np.exp(-res.x[0])))
    q, p0, r = (float(v) for v in np.exp(res.x[1:]))
    return {"rho": rho, "q": q, "p0": p0, "r": r, "nll": float(res.fun), "nfev": int(res.nfev),
            "converged": bool(res.success)}


def run(df: pd.DataFrame) -> tuple[dict, dict]:
    from .common import DEV_TEST_SEASONS
    par = fit_params(df)
    _, pr, pv, pt = kalman(df, par["rho"], par["q"], par["p0"], par["r"])
    full = df.copy()
    full["pred_margin"] = full["spread_line"] + pr
    full["pred_resid"], full["post_sd"] = pr, np.sqrt(pv)
    full["pred_total"] = full["total_line"] + pt
    folds = {T: (full[full.season == T].copy(), full[full.season < T].copy()) for T in DEV_TEST_SEASONS}
    return folds, par
