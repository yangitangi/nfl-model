"""
tracking/flag_injury_artifact.py
----------------------------------
Automatically flags games where injury_burden's SHAP contribution is
large AND driven by a true-zero reading on either team -- the specific
pattern confirmed (via manual SHAP checks) as the dominant driver behind
three real misses in the 2026 season: GB@NYJ (Wk2, ~45% of the edge),
SEA@ARI (Wk2, ~56%), PIT@CLE (Wk4, ~70%, and the pick went on to lose
outright). A literal zero is rare (~1.6% of team-games) and the model
treats it as a strong signal rather than "nothing to report yet" -- see
config.FEATURES["injuries_bucketed"] / ["injuries_smoothed_early"] for
two fixes that were built and walk-forward tested, both rejected (close,
but didn't clear the bar of improving MAE, win accuracy, AND ATS
together -- see models/tune_injury_weight.py for a third angle, also
rejected).

This script is NOT a model fix -- it's a tracking tool. Since the
underlying issue can't currently be corrected in the model itself, the
practical approach is to flag it on the dashboard each week it recurs,
so the pattern stays visible and auditable. If this keeps happening
into 2027, the accumulated log here is exactly the evidence that would
justify a bigger architecture change (e.g. moving away from a tree
model for this feature) rather than another patch attempt.

Writes one note per flagged game into outputs/analyst_notes.csv, same
file/mechanism as every other analyst note, identified by a fixed
prefix (safe to re-run -- replaces, doesn't duplicate).

Usage:
    python tracking/flag_injury_artifact.py --season 2026 --week 5
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.build_features import build_upcoming_features
from models.predict import load_model_artifacts

# Minimum |SHAP contribution|, in points, for a team's zero injury_burden
# reading to be considered a meaningful driver of the prediction rather
# than noise. Smallest confirmed real case this season contributed 0.84
# pts (CLE's half of the PIT@CLE edge) -- set a bit below that so we
# don't miss borderline cases, while still skipping trivial ones.
ZERO_BURDEN_SHAP_THRESHOLD = 0.75

# Guards against the most common false-positive: running this too early in
# the week, before that week's official injury reports have posted at all
# (typically Wednesday). When that's true, EVERY team reads as zero burden
# -- not because any one game has a rare, meaningful true zero, but because
# the data simply isn't there yet league-wide. A real artifact case is an
# ISOLATED zero in a week where most other teams already have real reports
# in; if most of the week is zero, skip flagging entirely rather than
# flag nearly every game (confirmed: Week 5 checked 5+ days out had 15/15
# games at zero both sides -- a data-timing gap, not 15 real artifacts).
LEAGUE_WIDE_ZERO_RATE_SKIP_THRESHOLD = 0.40

FLAG_PREFIX = "KNOWN MODEL LIMITATION (auto-flagged)"


def compute_shap_contribs(games: pd.DataFrame, feature_cols: list[str]):
    model, _, _ = load_model_artifacts()
    dmat = xgb.DMatrix(games[feature_cols])
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    return contribs  # shape (n_games, n_features + 1), last col is base value


def flag_week(season: int, week: int) -> pd.DataFrame:
    _, _, feature_cols = load_model_artifacts()
    upcoming = build_upcoming_features(season=season)
    games = upcoming[upcoming["week"] == week].reset_index(drop=True)
    if games.empty:
        print(f"No upcoming games found for {season} Week {week}.")
        return pd.DataFrame()

    all_burdens = pd.concat([games["home_injury_burden"], games["away_injury_burden"]])
    zero_rate = (all_burdens == 0).mean()
    if zero_rate >= LEAGUE_WIDE_ZERO_RATE_SKIP_THRESHOLD:
        print(f"Skipping: {zero_rate:.0%} of Week {week} teams show zero injury_burden -- "
              f"this week's official reports likely haven't posted yet (normally posts "
              f"Wednesday). Re-run this closer to kickoff once real reports are in.")
        return pd.DataFrame()

    contribs = compute_shap_contribs(games, feature_cols)
    home_idx = feature_cols.index("home_injury_burden")
    away_idx = feature_cols.index("away_injury_burden")

    rows = []
    for i, g in games.iterrows():
        culprits = []
        if g["home_injury_burden"] == 0 and abs(contribs[i, home_idx]) >= ZERO_BURDEN_SHAP_THRESHOLD:
            culprits.append((g["home_team"], float(contribs[i, home_idx])))
        if g["away_injury_burden"] == 0 and abs(contribs[i, away_idx]) >= ZERO_BURDEN_SHAP_THRESHOLD:
            culprits.append((g["away_team"], float(contribs[i, away_idx])))
        if culprits:
            rows.append({
                "game_id": g["game_id"], "week": week,
                "away_team": g["away_team"], "home_team": g["home_team"],
                "culprits": culprits,
            })

    return pd.DataFrame(rows)


def build_note_text(row) -> str:
    parts = []
    for team, contrib in row["culprits"]:
        parts.append(f"{team}'s injury report reads as a true zero, contributing {contrib:+.2f} pts to the prediction")
    culprit_str = " and ".join(parts)
    return (
        f"{FLAG_PREFIX}: {culprit_str}. This exact pattern (a true-zero injury_burden reading "
        f"driving a large swing) has been the dominant driver behind 3 confirmed real misses this "
        f"season -- GB@NYJ, SEA@ARI, PIT@CLE (which went on to lose outright as a -7.3 favorite). "
        f"Two fixes were built and walk-forward tested (bucketing, week-dependent smoothing); both "
        f"rejected -- see config.FEATURES. Treat this game's edge with extra skepticism; it may not "
        f"be real signal."
    )


def sync_flags(season: int, week: int) -> pd.DataFrame:
    flagged = flag_week(season, week)

    path = config.OUTPUTS_DIR / "analyst_notes.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["game_id", "week", "note", "created_at"])

    is_flag_row = existing["note"].astype(str).str.startswith(FLAG_PREFIX)
    is_this_week = existing["week"] == week
    kept = existing[~(is_flag_row & is_this_week)]

    if flagged.empty:
        kept.to_csv(path, index=False)
        print(f"No injury_burden artifact flags for {season} Week {week}. "
              f"(Cleared any stale flag rows for this week, if present.)")
        return kept

    new_rows = pd.DataFrame({
        "game_id": flagged["game_id"],
        "week": week,
        "note": flagged.apply(build_note_text, axis=1),
        "created_at": datetime.now().strftime("%Y-%m-%d"),
    })
    combined = pd.concat([kept, new_rows], ignore_index=True)
    combined.to_csv(path, index=False)

    print(f"Flagged {len(new_rows)} game(s) for {season} Week {week}:")
    for _, r in flagged.iterrows():
        culprit_str = ", ".join(f"{t} ({c:+.2f})" for t, c in r["culprits"])
        print(f"  {r['away_team']} @ {r['home_team']}: {culprit_str}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flag games where injury_burden's true-zero artifact is likely driving the prediction")
    parser.add_argument("--season", type=int, default=config.CURRENT_SEASON)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    sync_flags(args.season, args.week)
