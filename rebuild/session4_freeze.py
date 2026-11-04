"""
rebuild/session4_freeze.py
--------------------------
Session 4, step 0 helper: freeze F1 (B2 ridge + shrink k) and L1 (A-lite fade) parameters using ONLY
DEV-era data (seasons <= 2023, read with a parquet row filter so holdout rows never enter memory) and the
values already recorded in rebuild/session3_results.json. Writes rebuild/session4_frozen.json.

    python rebuild/session4_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rebuild.models.a_kalman import kalman  # noqa: E402
from rebuild.models.common import base_features  # noqa: E402


def load_dev_filtered() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/processed/asof_features.parquet", filters=[("season", "<=", 2023)])
    assert df["season"].max() <= 2023
    df = df[df["spread_line"].notna() & df["actual_margin"].notna()]
    return df.sort_values(["gw", "kickoff", "game_id"]).reset_index(drop=True)


def main():
    s3 = json.loads((ROOT / "rebuild/session3_results.json").read_text())
    b2 = s3["meta"]["B2_ridge_shrink_k"]
    ak = s3["meta"]["A_lite_kalman"]
    df = load_dev_filtered()
    feats = base_features(df)
    _, pr, pv, _ = kalman(df, ak["rho"], ak["q"], ak["p0"], ak["r"])
    z = pr / np.sqrt(pv)
    dev = df[df.season.between(2016, 2023)].index
    absz = np.abs(z[dev])
    cut = float(np.quantile(absz, 0.8))
    frozen = {
        "F1": {"model": "B2 ridge + shrink k (session 3 'B2_ridge_shrink_k')",
               "pipeline": "SimpleImputer(mean) -> StandardScaler -> Ridge(alpha); target = actual_margin - spread_line; "
                           "pred_margin = spread_line + k * ridge_prediction",
               "alpha": b2["alpha"], "k": b2["k"], "features": feats, "n_features": len(feats)},
        "L1": {"model": "A-lite Kalman (session 3 'A_lite_kalman'), frozen parameters",
               "rho": ak["rho"], "q": ak["q"], "p0": ak["p0"], "r": ak["r"],
               "z_definition": "z = (pred_margin - spread_line) / posterior_sd, start-of-week state",
               "top_quintile_abs_z_cut": cut,
               "dev_n": int(len(absz)), "dev_top_quintile_n": int((absz >= cut).sum()),
               "bet": "AGAINST A-lite's side: if z > 0 bet the away team ATS, if z < 0 bet the home team ATS"},
        "source_hash_session3_results_json": hashlib.sha256((ROOT / "rebuild/session3_results.json").read_bytes()).hexdigest(),
    }
    (ROOT / "rebuild/session4_frozen.json").write_text(json.dumps(frozen, indent=2))
    print(json.dumps({k: v for k, v in frozen["F1"].items() if k != "features"}, indent=1))
    print(json.dumps(frozen["L1"], indent=1))


if __name__ == "__main__":
    main()
