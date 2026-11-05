"""
rebuild/shadow_tracker.py
-------------------------
Prospective 2026 shadow run of the frozen F1 model (rebuild/PREREGISTRATION.md) next to production, plus the
data the pre-registered leads L1b and L2 (rebuild/PREREGISTRATION_2.md) need. Touches nothing in production
(dashboard/, tracking/, outputs/): production logs are read from origin/master with `git show` into a temp dir.

    python rebuild/shadow_tracker.py run       # capture lines + predict upcoming games + grade finished ones

Each run:
  1. downloads fresh nflverse 2026 play-by-play, injuries and schedules into data/raw/shadow2026/dl/
     (gitignored), builds the as-of store with the unchanged session-2 code (calendar extended to 2026 at
     runtime) and asserts every 2010-2025 row equals the committed store;
  2. CAPTURE: appends the current nflverse line of every not-yet-started game in the current and next week
     to rebuild/shadow_log/lines_2026.csv (needed for L2's opening line);
  3. PREDICT: for every not-yet-started game in the current week whose two injury reports are both published
     (operating rule, see predict()), appends one row to
     rebuild/shadow_log/predictions_2026.csv with the frozen F1 spread and the frozen A-lite z. Rows are only
     ever appended; the prediction of record for a game is its LAST row logged before kickoff.
     Starter QB = nflverse's listed starter for that game (the pre-game projection), which is what the
     as-of store uses.
  4. GRADE: rewrites rebuild/shadow_log/graded_2026.csv and rebuild/shadow_log/SUMMARY.md from the logs, nflverse
     results/closing lines and production's prediction_log on origin/master.

F1 is one fit on 2010-2025 (deterministic; refitted identically each run, never on 2026 data).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rebuild.features.common as common  # noqa: E402
from rebuild.build_asof_store import build_store, feature_columns  # noqa: E402
from rebuild.models.a_kalman import kalman  # noqa: E402

SEASON = 2026
RAW = ROOT / "data/raw/shadow2026"
LOG = ROOT / "rebuild/shadow_log"
FROZEN = json.loads((ROOT / "rebuild/session4_frozen.json").read_text())
BASE = "https://github.com/nflverse/nflverse-data/releases/download/"
FILES = {"schedules.parquet": "schedules/games.parquet",
         f"pbp_{SEASON}.parquet": f"pbp/play_by_play_{SEASON}.parquet",
         f"injuries_{SEASON}.parquet": f"injuries/injuries_{SEASON}.parquet"}
PROSPECTIVE_FROM_WEEK = 9          # first week logged before kickoff (weeks 1-8 were a retrospective backfill)
L2_MIN_LEAD_H = 48                 # L2 opening line = first capture made >= 48 h before kickoff
L2_MIN_MOVE = 0.5
BREAK_EVEN = 0.524


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def to_utc(kickoff_et: pd.Series) -> pd.Series:
    return pd.to_datetime(kickoff_et).dt.tz_localize("America/New_York", ambiguous=False,
                                                     nonexistent="shift_forward").dt.tz_convert("UTC")


# ----------------------------------------------------------------------------- data + store
def refresh_raw():
    dl = RAW / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    for y in range(2010, SEASON):
        for kind in ("pbp", "injuries"):
            link = RAW / f"{kind}_{y}.parquet"
            if not link.exists():
                link.symlink_to(Path("..") / f"{kind}_{y}.parquet")
    for local, remote in FILES.items():
        r = requests.get(BASE + remote, timeout=120)
        r.raise_for_status()
        (dl / local).write_bytes(r.content)
        shutil.copyfile(dl / local, RAW / local)


def build() -> pd.DataFrame:
    common.LAST_SEASON = SEASON                                   # runtime calendar extension only
    store = build_store(common.load_raw(2010, SEASON, RAW))
    old = pd.read_parquet(ROOT / "data/processed/asof_features.parquet").set_index("game_id").sort_index()
    new = store[store.season <= 2025].set_index("game_id").sort_index()
    cols = feature_columns(old)
    assert old.index.equals(new.index), "2010-2025 rows changed"
    pd.testing.assert_frame_equal(old[cols], new[cols], check_exact=False, rtol=0, atol=1e-9)
    store["kickoff_utc"] = to_utc(store["kickoff"])
    return store


def fit_f1(store: pd.DataFrame):
    tr = store[(store.season <= 2025) & store.spread_line.notna() & store.actual_margin.notna()]
    m = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=FROZEN["F1"]["alpha"]))
    return m.fit(tr[FROZEN["F1"]["features"]], tr.actual_margin - tr.spread_line)


def alite_z(store: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Frozen A-lite run over all completed games plus the current week (whose predictions come from the
    start-of-week state, before any update with that week's results)."""
    L = FROZEN["L1"]
    done = store[store.spread_line.notna() & store.actual_margin.notna()]
    cur = store[store.gw == rows.gw.min()]
    d = pd.concat([done[done.gw < rows.gw.min()], cur]).sort_values(["gw", "kickoff", "game_id"])
    d = d.drop_duplicates("game_id").reset_index(drop=True)
    _, pr, pv, _ = kalman(d, L["rho"], L["q"], L["p0"], L["r"])
    out = d[["game_id"]].assign(alite_pred_resid=pr, alite_post_sd=np.sqrt(pv))
    return rows[["game_id"]].merge(out, on="game_id", how="left")


# ----------------------------------------------------------------------------- steps
def append(path: Path, rows: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def capture(store: pd.DataFrame, t: pd.Timestamp) -> int:
    s = store[(store.season == SEASON) & (store.kickoff_utc > t)]
    if s.empty:
        return 0
    s = s[s.week <= s.week.min() + 1]
    rows = pd.DataFrame({"captured_at_utc": t.isoformat(), "game_id": s.game_id, "week": s.week,
                         "kickoff_utc": s.kickoff_utc.map(lambda x: x.isoformat()),
                         "spread_line": s.spread_line, "total_line": s.total_line, "ml_home_prob": s.ml_home_prob})
    append(LOG / "lines_2026.csv", rows)
    return len(rows)


def predict(store: pd.DataFrame, t: pd.Timestamp) -> int:
    up = store[(store.season == SEASON) & (store.kickoff_utc > t) & store.spread_line.notna()]
    if up.empty:
        return 0
    up = up[up.gw == up.gw.min()].copy()
    # Operating rule (not a model change): predict only once BOTH teams' injury reports for the week exist.
    # F1 learned large coefficients for the rare inj_report_missing flag (9 training team-games: about
    # +0.54 pts home / -1.05 pts away after k), so an early-week prediction would carry an artifact.
    ready = (up.home_inj_report_missing == 0) & (up.away_inj_report_missing == 0)
    print(f"predict: {int(ready.sum())} of {len(up)} current-week games have both injury reports; "
          f"waiting on {sorted(up.loc[~ready, 'game_id'])}")
    up = up[ready]
    if up.empty:
        return 0
    m = fit_f1(store)
    k = FROZEN["F1"]["k"]
    up["f1_spread"] = up.spread_line + k * m.predict(up[FROZEN["F1"]["features"]])
    up = up.merge(alite_z(store, up), on="game_id", how="left")
    up["alite_z"] = up.alite_pred_resid / up.alite_post_sd
    top = up.alite_z.abs() >= FROZEN["L1"]["top_quintile_abs_z_cut"]
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rows = pd.DataFrame({
        "predicted_at_utc": t.isoformat(), "game_id": up.game_id, "week": up.week,
        "kickoff_utc": up.kickoff_utc.map(lambda x: x.isoformat()), "home_team": up.home_team,
        "away_team": up.away_team, "home_qb_id": up.home_qb_id, "away_qb_id": up.away_qb_id,
        "line_at_pred": up.spread_line, "total_at_pred": up.total_line,
        "f1_spread": up.f1_spread.round(3), "f1_edge": (up.f1_spread - up.spread_line).round(3),
        "alite_pred_resid": up.alite_pred_resid.round(4), "alite_post_sd": up.alite_post_sd.round(4),
        "alite_z": up.alite_z.round(4), "l1_top_quintile": top.astype(int),
        "l1_fade_side": np.where(top, np.where(up.alite_z > 0, "away", "home"), ""),
        "home_inj_report_missing": up.home_inj_report_missing, "away_inj_report_missing": up.away_inj_report_missing,
        "code_commit": sha})
    append(LOG / "predictions_2026.csv", rows)
    return len(rows)


def production_log() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["git", "fetch", "-q", "origin", "master"], cwd=ROOT, check=False)
        f = Path(td) / "p.parquet"
        with open(f, "wb") as fh:
            r = subprocess.run(["git", "show", "origin/master:outputs/prediction_log.parquet"], cwd=ROOT, stdout=fh)
        if r.returncode != 0:
            return pd.DataFrame(columns=["game_id", "prod_pred_margin", "prod_line"])
        p = pd.read_parquet(f)
    return p[["game_id", "pred_margin", "open_spread_line"]].rename(
        columns={"pred_margin": "prod_pred_margin", "open_spread_line": "prod_line"})


def cover(pred, line, act):
    lean, cov = np.sign(pred - line), np.sign(act - line)
    return np.where((lean == 0) | (cov == 0) | np.isnan(pred) | np.isnan(line), np.nan, (lean == cov).astype(float))


def rate(x) -> dict:
    x = pd.Series(x).dropna()
    n = len(x)
    if n == 0:
        return {"ats": None, "n": 0, "ci95": None}
    p = float(x.mean()); h = 1.96 * np.sqrt(p * (1 - p) / n)
    return {"ats": p, "n": n, "ci95": [p - h, p + h]}


def grade(store: pd.DataFrame) -> dict:
    pf = LOG / "predictions_2026.csv"
    if not pf.exists():
        return {}
    preds = pd.read_csv(pf, parse_dates=["predicted_at_utc", "kickoff_utc"])
    preds = preds[preds.predicted_at_utc < preds.kickoff_utc]                    # pre-kickoff rows only
    rec = preds.sort_values("predicted_at_utc").groupby("game_id").tail(1)     # prediction of record
    res = store[(store.season == SEASON) & store.actual_margin.notna()][["game_id", "spread_line", "actual_margin"]]
    g = rec.merge(res.rename(columns={"spread_line": "close_line"}), on="game_id", how="inner")
    g = g.merge(production_log(), on="game_id", how="left")
    g["f1_cover_vs_line_at_pred"] = cover(g.f1_spread, g.line_at_pred, g.actual_margin)
    g["f1_cover_vs_close"] = cover(g.close_line + g.f1_edge, g.close_line, g.actual_margin)  # F1's side, at the close
    g["prod_cover_vs_own_line"] = cover(g.prod_pred_margin, g.prod_line, g.actual_margin)
    g["prod_cover_vs_close"] = cover(g.prod_pred_margin, g.close_line, g.actual_margin)
    fade_dir = np.where(g.l1_fade_side == "home", 1.0, np.where(g.l1_fade_side == "away", -1.0, np.nan))
    g["l1_fade_cover_vs_close"] = cover(g.close_line + fade_dir, g.close_line, g.actual_margin)

    # L2: opening line = first capture >= 48 h before kickoff; closing = nflverse final line
    lf = LOG / "lines_2026.csv"
    if lf.exists():
        ln = pd.read_csv(lf, parse_dates=["captured_at_utc", "kickoff_utc"])
        ln = ln[(ln.kickoff_utc - ln.captured_at_utc) >= pd.Timedelta(hours=L2_MIN_LEAD_H)]
        op = ln.sort_values("captured_at_utc").groupby("game_id").head(1)[["game_id", "spread_line", "captured_at_utc"]]
        g = g.merge(op.rename(columns={"spread_line": "l2_open_line", "captured_at_utc": "l2_open_at"}),
                    on="game_id", how="left")
    else:
        g["l2_open_line"], g["l2_open_at"] = np.nan, pd.NaT
    g["l2_move"] = g.close_line - g.l2_open_line
    side = np.where(g.l2_move.abs() >= L2_MIN_MOVE, np.sign(g.l2_move), np.nan)
    g["l2_follow_cover_vs_close"] = cover(g.close_line + side, g.close_line, g.actual_margin)
    g = g.sort_values(["week", "kickoff_utc", "game_id"])
    g.to_csv(LOG / "graded_2026.csv", index=False)

    p = g[g.week >= PROSPECTIVE_FROM_WEEK]
    e_f1, e_mk = (p.close_line + p.f1_edge - p.actual_margin).abs(), (p.close_line - p.actual_margin).abs()
    both = p.prod_pred_margin.notna()
    e_pr = (p.prod_pred_margin - p.actual_margin).abs()
    d1, d2 = e_f1 - e_mk, (e_pr - e_mk)[both]
    se = lambda d: float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None  # noqa: E731
    s = {"graded_games_prospective": int(len(p)), "weeks": sorted(int(w) for w in p.week.unique()),
         "f1_minus_market_mae": {"diff": float(d1.mean()) if len(d1) else None, "se": se(d1)},
         "prod_minus_market_mae": {"diff": float(d2.mean()) if len(d2) else None, "se": se(d2), "n": int(both.sum())},
         "f1_ats_vs_close": rate(p.f1_cover_vs_close), "f1_ats_vs_line_at_pred": rate(p.f1_cover_vs_line_at_pred),
         "prod_ats_vs_own_line": rate(p.prod_cover_vs_own_line), "prod_ats_vs_close": rate(p.prod_cover_vs_close),
         "L1b_fade": rate(p.l1_fade_cover_vs_close), "L2_follow_move": rate(p.l2_follow_cover_vs_close),
         "L2_games_with_open_line": int(p.l2_open_line.notna().sum())}
    write_summary(s)
    return s


def fmt(r: dict) -> str:
    if not r or not r.get("n"):
        return "n = 0"
    return f"{r['ats']:.1%} (n = {r['n']}, 95% CI {r['ci95'][0]:.1%}-{r['ci95'][1]:.1%})"


def write_summary(s: dict):
    md = f"""# 2026 shadow run: running summary (auto-generated by rebuild/shadow_tracker.py)

Updated {now_utc():%Y-%m-%d %H:%M} UTC. Prospective games only (week >= {PROSPECTIVE_FROM_WEEK}, predicted before kickoff).
**Interim numbers are descriptive.** The pre-registered tests (rebuild/PREREGISTRATION_2.md) are evaluated
once, at their fixed analysis points; do not act on these running tallies.

| | Value |
|---|---|
| Graded games | {s['graded_games_prospective']} (weeks {s['weeks']}) |
| F1 MAE minus market (closing line) | {s['f1_minus_market_mae']['diff']} (SE {s['f1_minus_market_mae']['se']}) |
| Production MAE minus market | {s['prod_minus_market_mae']['diff']} (SE {s['prod_minus_market_mae']['se']}, n = {s['prod_minus_market_mae']['n']}) |
| F1 ATS vs close | {fmt(s['f1_ats_vs_close'])} |
| Production ATS vs its own logged line | {fmt(s['prod_ats_vs_own_line'])} |
| Production ATS vs close | {fmt(s['prod_ats_vs_close'])} |
| L1b fade (A-lite top-\\|z\\| quintile) vs close | {fmt(s['L1b_fade'])} |
| L2 follow the line move (\\|move\\| >= {L2_MIN_MOVE}) vs close | {fmt(s['L2_follow_move'])} (games with an opening capture: {s['L2_games_with_open_line']}) |
"""
    (LOG / "SUMMARY.md").write_text(md)


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        print(__doc__)
        sys.exit(2)
    t = now_utc()
    refresh_raw()
    store = build()
    n_cap, n_pred = capture(store, t), predict(store, t)
    s = grade(store)
    remaining = int(((store.season == SEASON) & store.actual_margin.isna()).sum())
    print(json.dumps({"run_at_utc": t.isoformat(), "lines_captured": n_cap, "games_predicted": n_pred,
                      "unplayed_2026_reg_games": remaining, "summary": s}, indent=1, default=str))
    if remaining == 0:
        print("2026 REGULAR SEASON COMPLETE: all games graded; the shadow schedule can be stopped.")


if __name__ == "__main__":
    main()
