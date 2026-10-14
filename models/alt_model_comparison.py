"""
models/alt_model_comparison.py
------------------------------
Scoped experiment: can NON-TREE models match the production XGBoost margin
model on the same walk-forward setup? Groundwork for a possible 2027 rebuild.

Same data, same features (get_feature_columns), same residual target
(actual_margin - spread_line, via get_margin_target), same walk-forward
(train on all seasons before the test season, test 2014..2025) as the
production path. Only the model class changes:

    xgb          production XGBoost via train_margin_model (baseline)
    ridge        Ridge on standardized features, alpha picked by inner
                 time-series CV on the training window only
    bayes_ridge  sklearn BayesianRidge on standardized features
    mlp          MLPRegressor (64, 32), early stopping, 3 seeds averaged
    blend        50/50 xgb + the best of ridge/bayes_ridge/mlp, where "best"
                 is chosen causally: lowest MAE over test seasons BEFORE the
                 one being predicted (ridge for the first test season). The
                 three fixed blends (blend_xgb_<model>) are also reported.

Nothing is tuned on test seasons. Outputs go to outputs/alt_models/.

Usage:
    python models/alt_model_comparison.py
"""

import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.feature_ablation import _ats_hit_rate
from models.train_model import (
    get_feature_columns, get_margin_target, reconstruct_margin, train_margin_model,
)

FIRST_TEST_SEASON = 2014
LAST_TEST_SEASON = 2025
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
MLP_SEEDS = [0, 1, 2]
EDGE_BINS = [0, 1, 2, 3, 5, np.inf]
EDGE_LABELS = ["0-1", "1-2", "2-3", "3-5", "5+"]
BASE_MODELS = ["xgb", "ridge", "bayes_ridge", "mlp"]
ALT_MODELS = ["ridge", "bayes_ridge", "mlp"]
OUT_DIR = config.PROJECT_ROOT / "outputs" / "alt_models" if hasattr(config, "PROJECT_ROOT") \
    else Path(__file__).resolve().parent.parent / "outputs" / "alt_models"
BURDEN_COLS = ["home_injury_burden", "away_injury_burden"]


# ---------------------------------------------------------------------------
# MODEL WRAPPERS -- each takes (train_df, feature_cols) and returns an object
# with .predict(df) -> raw residual prediction (same scale as the target)
# ---------------------------------------------------------------------------
class _Scaled:
    """StandardScaler (fit on train only) + estimator(s); predict averages
    the estimators (one for ridge/bayes, several seeds for the MLP)."""

    def __init__(self, scaler, estimators, cols):
        self.scaler, self.estimators, self.cols = scaler, estimators, cols

    def predict(self, df):
        x = self.scaler.transform(df[self.cols].values)
        return np.mean([m.predict(x) for m in self.estimators], axis=0)


def _scaled_train(train, cols):
    scaler = StandardScaler().fit(train[cols].values)
    return scaler, scaler.transform(train[cols].values), get_margin_target(train)


def fit_ridge(train, cols):
    """Alpha chosen by 4-split expanding-window CV inside the training data
    (rows are time-ordered), then refit on all training data."""
    train = train.sort_values(["season", "week"])
    scaler, x, y = _scaled_train(train, cols)
    best_alpha, best_mae = RIDGE_ALPHAS[0], np.inf
    for alpha in RIDGE_ALPHAS:
        maes = []
        for tr, va in TimeSeriesSplit(n_splits=4).split(x):
            m = Ridge(alpha=alpha).fit(x[tr], y[tr])
            maes.append(np.mean(np.abs(y[va] - m.predict(x[va]))))
        if np.mean(maes) < best_mae:
            best_alpha, best_mae = alpha, float(np.mean(maes))
    model = _Scaled(scaler, [Ridge(alpha=best_alpha).fit(x, y)], cols)
    model.alpha = best_alpha
    return model


def fit_bayes_ridge(train, cols):
    scaler, x, y = _scaled_train(train, cols)
    return _Scaled(scaler, [BayesianRidge().fit(x, y)], cols)


def fit_mlp(train, cols):
    scaler, x, y = _scaled_train(train, cols)
    ests = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for seed in MLP_SEEDS:
            ests.append(MLPRegressor(
                hidden_layer_sizes=(64, 32), early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=15, max_iter=300,
                random_state=seed).fit(x, y))
    return _Scaled(scaler, ests, cols)


FITTERS = {
    "xgb": lambda train, cols: train_margin_model(train, cols),
    "ridge": fit_ridge,
    "bayes_ridge": fit_bayes_ridge,
    "mlp": fit_mlp,
}


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def _metrics(pred, actual, spread, home_win):
    return {
        "n": len(pred),
        "mae": float(np.mean(np.abs(actual - pred))),
        "win_acc": float(((pred > 0).astype(int) == home_win).mean()),
        "ats": _ats_hit_rate(pred, spread, actual),
        "ats_n": int(((np.sign(pred - spread) != 0) & (np.sign(actual - spread) != 0)).sum()),
    }


def run():
    t0 = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")
    feature_cols = get_feature_columns(df)
    required = feature_cols + ["actual_margin", "home_win", "spread_line"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    clean = clean.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    print(f"{len(clean)} games, {len(feature_cols)} features, "
          f"seasons {clean.season.min()}-{clean.season.max()}")

    test_seasons = [s for s in sorted(clean.season.unique())
                    if FIRST_TEST_SEASON <= s <= LAST_TEST_SEASON]

    # preds[model] -> list of arrays aligned with test_frames
    test_frames, preds, preds_zero = [], {m: [] for m in BASE_MODELS}, {m: [] for m in BASE_MODELS}
    zero_masks = []
    chosen_alpha = {}

    for season in test_seasons:
        train = clean[clean.season < season]
        test = clean[clean.season == season]
        if len(train) < 200 or test.empty:
            continue
        # injury-zero counterfactual: replace literal-zero burden with the
        # training-set median burden (per column, only where it reads 0)
        test_cf = test.copy()
        zmask = np.zeros(len(test), dtype=bool)
        for c in BURDEN_COLS:
            is_zero = (test[c] == 0).values
            zmask |= is_zero
            test_cf.loc[is_zero, c] = train[c].median()
        zero_masks.append(zmask)

        for name in BASE_MODELS:
            model = FITTERS[name](train, feature_cols)
            if name == "ridge":
                chosen_alpha[int(season)] = model.alpha
            preds[name].append(reconstruct_margin(test, model.predict(test[feature_cols])))
            preds_zero[name].append(reconstruct_margin(test_cf, model.predict(test_cf[feature_cols])))
        test_frames.append(test)
        print(f"  season {season}: train n={len(train)}, test n={len(test)}, "
              f"{time.time() - t0:.0f}s elapsed")

    # ---- blends (50/50 xgb + alt). causal "best": lowest MAE on prior test seasons
    n_seasons = len(test_frames)
    for alt in ALT_MODELS:
        preds[f"blend_xgb_{alt}"] = [0.5 * preds["xgb"][i] + 0.5 * preds[alt][i] for i in range(n_seasons)]
        preds_zero[f"blend_xgb_{alt}"] = [0.5 * preds_zero["xgb"][i] + 0.5 * preds_zero[alt][i]
                                          for i in range(n_seasons)]
    blend_choice = []
    preds["blend"], preds_zero["blend"] = [], []
    for i in range(n_seasons):
        if i == 0:
            pick = "ridge"
        else:
            prior_mae = {a: np.mean(np.concatenate([np.abs(test_frames[j]["actual_margin"].values - preds[a][j])
                                                    for j in range(i)])) for a in ALT_MODELS}
            pick = min(prior_mae, key=prior_mae.get)
        blend_choice.append(pick)
        preds["blend"].append(preds[f"blend_xgb_{pick}"][i])
        preds_zero["blend"].append(preds_zero[f"blend_xgb_{pick}"][i])

    model_names = BASE_MODELS + ["blend"] + [f"blend_xgb_{a}" for a in ALT_MODELS]

    # ---- assemble long frame
    all_test = pd.concat(test_frames, ignore_index=True)
    actual = all_test["actual_margin"].values
    spread = all_test["spread_line"].values
    home_win = all_test["home_win"].values.astype(int)
    seasons_arr = all_test["season"].values
    P = {m: np.concatenate(preds[m]) for m in model_names}
    PZ = {m: np.concatenate(preds_zero[m]) for m in model_names}
    zero_mask = np.concatenate(zero_masks)

    # ---- results.csv: per model per season + ALL
    rows = []
    market_pred = spread
    for m in ["market"] + model_names:
        p = market_pred if m == "market" else P[m]
        for season in list(test_seasons) + ["ALL"]:
            sel = np.ones(len(p), bool) if season == "ALL" else seasons_arr == season
            if sel.sum() == 0:
                continue
            r = _metrics(p[sel], actual[sel], spread[sel], home_win[sel])
            if m == "market":
                r["ats"], r["ats_n"] = np.nan, 0
            err_x = np.abs(actual[sel] - P["xgb"][sel])
            err_m = np.abs(actual[sel] - p[sel])
            d = err_m - err_x
            r["mae_diff_vs_xgb"] = float(d.mean())
            r["mae_diff_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
            rows.append({"model": m, "season": season, **r})
    results = pd.DataFrame(rows)

    # ---- edge buckets (overall, all test seasons)
    eb = []
    for m in model_names:
        edge = np.abs(P[m] - spread)
        bucket = pd.cut(edge, bins=EDGE_BINS, labels=EDGE_LABELS, right=False)
        for lab in EDGE_LABELS:
            sel = np.asarray(bucket == lab)
            n_games = int(sel.sum())
            if n_games == 0:
                eb.append({"model": m, "edge_bucket": lab, "n_games": 0, "ats_n": 0,
                           "ats": np.nan, "mae": np.nan, "market_mae": np.nan})
                continue
            lean = np.sign(P[m][sel] - spread[sel])
            cover = np.sign(actual[sel] - spread[sel])
            valid = (lean != 0) & (cover != 0)
            eb.append({"model": m, "edge_bucket": lab, "n_games": n_games, "ats_n": int(valid.sum()),
                       "ats": float((lean[valid] == cover[valid]).mean()) if valid.any() else np.nan,
                       "mae": float(np.mean(np.abs(actual[sel] - P[m][sel]))),
                       "market_mae": float(np.mean(np.abs(actual[sel] - spread[sel])))})
    edge_buckets = pd.DataFrame(eb)

    # ---- injury-zero sensitivity
    iz = []
    for m in model_names:
        diff = PZ[m][zero_mask] - P[m][zero_mask]
        iz.append({
            "model": m,
            "n_zero_games": int(zero_mask.sum()),
            "n_test_games": int(len(zero_mask)),
            "mean_abs_diff_pts": float(np.mean(np.abs(diff))),
            "mean_signed_diff_pts": float(np.mean(diff)),
            "p90_abs_diff_pts": float(np.percentile(np.abs(diff), 90)),
            "max_abs_diff_pts": float(np.max(np.abs(diff))),
        })
    inj = pd.DataFrame(iz)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.round(5).to_csv(OUT_DIR / "results.csv", index=False)
    edge_buckets.round(5).to_csv(OUT_DIR / "edge_buckets.csv", index=False)
    inj.round(5).to_csv(OUT_DIR / "injury_zero_sensitivity.csv", index=False)
    elapsed = time.time() - t0
    info = {
        "elapsed_seconds": round(elapsed, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "n_games_total": int(len(clean)),
        "n_test_games": int(len(all_test)),
        "n_features": len(feature_cols),
        "test_seasons": [int(s) for s in test_seasons],
        "ridge_alpha_by_season": chosen_alpha,
        "blend_pick_by_season": {int(s): p for s, p in zip(test_seasons, blend_choice)},
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2))

    overall = results[results.season == "ALL"].set_index("model")
    print("\nOVERALL")
    print(overall[["n", "mae", "win_acc", "ats", "ats_n", "mae_diff_vs_xgb", "mae_diff_se"]]
          .round(4).to_string())
    print("\nINJURY-ZERO SENSITIVITY")
    print(inj.round(3).to_string(index=False))
    print(f"\nelapsed {elapsed:.0f}s; wrote {OUT_DIR}")


if __name__ == "__main__":
    run()
