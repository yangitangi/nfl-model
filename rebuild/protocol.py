"""
rebuild/protocol.py
-------------------
The single evaluation protocol for the 2027 rebuild. EVERY candidate model
must be scored with evaluate() on the same folds so results are comparable.

Rules (see docs/REBUILD_PLAN.md):
  * Development window: walk-forward test seasons 2016-2023 (train on all
    seasons before each test season, no peeking at later seasons).
  * Final holdout: 2024-2025. Do not look at these until a model is frozen;
    report them once, separately.
  * Every feature must be computable from information available BEFORE
    kickoff of the game it is used for ("as-of" features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEV_TEST_SEASONS = list(range(2016, 2024))     # 2016..2023
FINAL_HOLDOUT_SEASONS = [2024, 2025]
EDGE_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]
REQUIRED = ["game_id", "season", "week", "pred_margin", "spread_line", "actual_margin"]


def train_seasons_for(test_season: int, first_season: int = 2010) -> list[int]:
    """Walk-forward: all seasons strictly before the test season."""
    return list(range(first_season, test_season))


def _ats(pred: np.ndarray, line: np.ndarray, actual: np.ndarray) -> tuple[float, int]:
    lean = np.sign(pred - line)
    cover = np.sign(actual - line)
    ok = (lean != 0) & (cover != 0)
    if ok.sum() == 0:
        return float("nan"), 0
    return float((lean[ok] == cover[ok]).mean()), int(ok.sum())


def evaluate(preds: pd.DataFrame, name: str, seasons: list[int] | None = None, verbose: bool = True) -> dict:
    """preds columns: game_id, season, week, pred_margin, spread_line, actual_margin
    (home perspective, positive = home favored). Optional: pred_home_win_prob,
    pred_total, total_line, actual_total. Returns a dict of metrics."""
    missing = [c for c in REQUIRED if c not in preds.columns]
    if missing:
        raise ValueError(f"preds is missing columns: {missing}")
    d = preds.copy()
    if seasons is not None:
        d = d[d["season"].isin(seasons)]
    d = d.dropna(subset=REQUIRED)
    p, line, act = d["pred_margin"].values, d["spread_line"].values, d["actual_margin"].values

    out = {
        "model": name,
        "n_games": len(d),
        "mae": float(np.abs(p - act).mean()),
        "market_mae": float(np.abs(line - act).mean()),
        "win_acc": float(((p > 0) == (act > 0))[act != 0].mean()),
        "market_win_acc": float(((line > 0) == (act > 0))[(act != 0) & (line != 0)].mean()),
    }
    out["ats"], out["ats_n"] = _ats(p, line, act)

    edge = np.abs(p - line)
    buckets = {}
    for lo, hi in EDGE_BUCKETS:
        m = (edge >= lo) & (edge < hi)
        a, n = _ats(p[m], line[m], act[m])
        buckets[f"{lo}-{hi}"] = {"ats": a, "n": n}
    out["ats_by_edge"] = buckets

    if "pred_home_win_prob" in d.columns:
        prob = d["pred_home_win_prob"].clip(0.01, 0.99).values
        win = (act > 0).astype(float)
        out["brier"] = float(((prob - win) ** 2).mean())
        out["logloss"] = float(-(win * np.log(prob) + (1 - win) * np.log(1 - prob)).mean())
    if {"pred_total", "total_line", "actual_total"} <= set(d.columns):
        t = d.dropna(subset=["pred_total", "total_line", "actual_total"])
        out["total_mae"] = float(np.abs(t["pred_total"] - t["actual_total"]).mean())
        out["market_total_mae"] = float(np.abs(t["total_line"] - t["actual_total"]).mean())
        lean = np.sign(t["pred_total"] - t["total_line"])
        res = np.sign(t["actual_total"] - t["total_line"])
        ok = (lean != 0) & (res != 0)
        out["ou_acc"], out["ou_n"] = float((lean[ok] == res[ok]).mean()), int(ok.sum())

    if verbose:
        print(f"\n== {name}  (n={out['n_games']}) ==")
        print(f"MAE {out['mae']:.3f} (market {out['market_mae']:.3f}) | win acc {out['win_acc']:.1%} "
              f"(market {out['market_win_acc']:.1%}) | ATS {out['ats']:.1%} (n={out['ats_n']})")
        print("ATS by |model-market| edge: " + ", ".join(
            f"{k}: {v['ats']:.1%} (n={v['n']})" for k, v in buckets.items()))
        if "logloss" in out:
            print(f"Brier {out['brier']:.4f} | logloss {out['logloss']:.4f}")
    return out


def paired_mae_diff(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> tuple[float, float]:
    """Mean and standard error of |err_a| - |err_b| over shared games (negative = A better)."""
    m = preds_a[["game_id", "pred_margin", "actual_margin"]].merge(
        preds_b[["game_id", "pred_margin"]], on="game_id", suffixes=("_a", "_b"))
    diff = (m["pred_margin_a"] - m["actual_margin"]).abs() - (m["pred_margin_b"] - m["actual_margin"]).abs()
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
