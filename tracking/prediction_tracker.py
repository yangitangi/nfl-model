"""
tracking/prediction_tracker.py
-------------------------------
Weekly prediction log + backtest. Snapshots the model's prediction and the
market line at prediction time, lets you refresh with the (approximate)
closing line right before kickoff, then grades against final scores once
games are over. Run the report any time to see model vs. Vegas accuracy on
graded games -- including whether bigger model/market disagreements are
actually more likely to be right, which is the point of tracking this at all.

Note on "closing" lines: nflverse's schedule file doesn't separately label
opening vs. closing -- it just reflects whatever the line was when the file
was last generated. `update-closing` re-downloads it, so run that as close
to kickoff as practical for the number to actually mean "closing."

Usage:
    python tracking/prediction_tracker.py snapshot --week 1
        Run once, early in the week, after that week's lines are posted.
        Safe to re-run -- won't overwrite an existing snapshot for a game
        (the point is to freeze the prediction, not let it drift).

    python tracking/prediction_tracker.py update-closing --week 1
        Run once, close to kickoff, to record the (approximate) closing line
        for already-snapshotted games.

    python tracking/prediction_tracker.py update-results --week 1
        Run after that week's games are final. Fills in actual scores and
        grades every logged game that now has one. Safe to re-run --
        only touches games that aren't graded yet.

    python tracking/prediction_tracker.py report [--week 1]
        Prints model vs. Vegas MAE/win accuracy on graded games, plus a
        breakdown by how big the model's disagreement with the market was.
        Omit --week for a season-to-date report across all graded games.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.build_features import build_upcoming_features
from data.fetch_data import fetch_schedules, standardize_team_abbrs
from models.predict import load_model_artifacts, predict, add_market_edges, fair_home_win_prob

LOG_PATH = config.OUTPUTS_DIR / "prediction_log.parquet"

LOG_COLUMNS = [
    "game_id", "season", "week", "gameday", "home_team", "away_team", "div_game",
    "snapshot_at",
    "open_spread_line", "open_home_moneyline", "open_away_moneyline", "open_total_line",
    "pred_margin", "pred_home_win_prob",
    "fair_home_ml_prob_open", "spread_edge_open", "ml_edge_open",
    "close_captured_at",
    "close_spread_line", "close_home_moneyline", "close_away_moneyline", "close_total_line",
    "fair_home_ml_prob_close", "spread_edge_close", "ml_edge_close",
    "graded_at", "actual_home_score", "actual_away_score", "actual_margin", "actual_winner",
]


def _load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_parquet(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def _save_log(df: pd.DataFrame):
    df.to_parquet(LOG_PATH, index=False)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


# ---------------------------------------------------------------------------
# SNAPSHOT — record this week's prediction + market line at prediction time
# ---------------------------------------------------------------------------
def snapshot(season: int, week: int):
    log = _load_log()
    already_logged = set(log.loc[log["week"] == week, "game_id"]) if not log.empty else set()

    upcoming = build_upcoming_features(season=season, force_refresh=True)
    week_games = upcoming[upcoming["week"] == week]
    if week_games.empty:
        print(f"No Week {week} games found for season {season}.")
        return

    new_games = week_games[~week_games["game_id"].isin(already_logged)]
    if new_games.empty:
        print(f"All {len(week_games)} Week {week} games are already snapshotted. "
              "Nothing new to add -- snapshot freezes the prediction at a point "
              "in time rather than letting it drift. Use update-closing to "
              "record the closing line instead.")
        return

    _, _, feature_cols = load_model_artifacts()
    predicted = predict(new_games, feature_cols)
    predicted = add_market_edges(predicted)

    now = _now()
    rows = pd.DataFrame({
        "game_id": predicted["game_id"],
        "season": predicted["season"],
        "week": predicted["week"],
        "gameday": predicted["gameday"],
        "home_team": predicted["home_team"],
        "away_team": predicted["away_team"],
        "div_game": predicted["div_game"],
        "snapshot_at": now,
        "open_spread_line": predicted["spread_line"],
        "open_home_moneyline": predicted["home_moneyline"],
        "open_away_moneyline": predicted["away_moneyline"],
        "open_total_line": predicted["total_line"],
        "pred_margin": predicted["pred_margin"],
        "pred_home_win_prob": predicted["pred_home_win_prob"],
        "fair_home_ml_prob_open": predicted["fair_home_ml_prob"].astype(float),
        "spread_edge_open": predicted["spread_edge"],
        "ml_edge_open": predicted["ml_edge"],
        "close_captured_at": pd.NaT,
        "close_spread_line": np.nan,
        "close_home_moneyline": np.nan,
        "close_away_moneyline": np.nan,
        "close_total_line": np.nan,
        "fair_home_ml_prob_close": np.nan,
        "spread_edge_close": np.nan,
        "ml_edge_close": np.nan,
        "graded_at": pd.NaT,
        "actual_home_score": np.nan,
        "actual_away_score": np.nan,
        "actual_margin": np.nan,
        "actual_winner": None,
    })

    log = pd.concat([log, rows], ignore_index=True)
    _save_log(log)
    print(f"Snapshotted {len(rows)} new Week {week} game(s) -> {LOG_PATH}")
    for _, r in rows.iterrows():
        print(f"  {r['away_team']} @ {r['home_team']}: pred {r['pred_margin']:+.1f}, "
              f"spread {r['open_spread_line']:+.1f}, edge {r['spread_edge_open']:+.1f}")


# ---------------------------------------------------------------------------
# UPDATE CLOSING LINE
# ---------------------------------------------------------------------------
def update_closing_lines(season: int, week: int):
    log = _load_log()
    mask = (log["week"] == week) & (log["season"] == season)
    if not mask.any():
        print(f"No logged games for season {season} Week {week}. Run `snapshot` first.")
        return

    schedules = fetch_schedules(max_season=season, force_refresh=True)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    schedules = schedules[(schedules["season"] == season) & (schedules["week"] == week)]

    updated = 0
    for idx in log[mask].index:
        match = schedules[schedules["game_id"] == log.at[idx, "game_id"]]
        if match.empty:
            continue
        m = match.iloc[0]
        home_ml, away_ml = m["home_moneyline"], m["away_moneyline"]

        log.at[idx, "close_spread_line"] = m["spread_line"]
        log.at[idx, "close_home_moneyline"] = home_ml
        log.at[idx, "close_away_moneyline"] = away_ml
        log.at[idx, "close_total_line"] = m["total_line"]
        log.at[idx, "close_captured_at"] = _now()

        if pd.notna(home_ml) and pd.notna(away_ml):
            fair = fair_home_win_prob(home_ml, away_ml)
            log.at[idx, "fair_home_ml_prob_close"] = fair
            log.at[idx, "ml_edge_close"] = log.at[idx, "pred_home_win_prob"] - fair
        if pd.notna(m["spread_line"]):
            log.at[idx, "spread_edge_close"] = log.at[idx, "pred_margin"] - m["spread_line"]
        updated += 1

    _save_log(log)
    print(f"Updated closing lines for {updated} Week {week} game(s).")


# ---------------------------------------------------------------------------
# UPDATE RESULTS
# ---------------------------------------------------------------------------
def update_results(season: int, week: int):
    log = _load_log()
    mask = (log["week"] == week) & (log["season"] == season) & log["graded_at"].isna()
    if not mask.any():
        print(f"Nothing ungraded for season {season} Week {week}.")
        return

    schedules = fetch_schedules(max_season=season, force_refresh=True)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    schedules = schedules[(schedules["season"] == season) & (schedules["week"] == week)]

    graded_n = 0
    for idx in log[mask].index:
        match = schedules[schedules["game_id"] == log.at[idx, "game_id"]]
        if match.empty:
            continue
        m = match.iloc[0]
        if pd.isna(m["home_score"]):
            continue  # not final yet

        home_score, away_score = m["home_score"], m["away_score"]
        log.at[idx, "actual_home_score"] = home_score
        log.at[idx, "actual_away_score"] = away_score
        log.at[idx, "actual_margin"] = home_score - away_score
        log.at[idx, "actual_winner"] = (
            "home" if home_score > away_score else "away" if away_score > home_score else "tie"
        )
        log.at[idx, "graded_at"] = _now()
        graded_n += 1

    _save_log(log)
    remaining = int(mask.sum()) - graded_n
    print(f"Graded {graded_n} Week {week} game(s).")
    if remaining:
        print(f"{remaining} still don't have a final score yet -- run this again later.")


# ---------------------------------------------------------------------------
# REPORT — model vs. Vegas on graded games, spread and moneyline kept
# separate throughout since a model can be sharper on one than the other.
# ---------------------------------------------------------------------------
def _add_grading_columns(graded: pd.DataFrame) -> pd.DataFrame:
    """
    Adds home_win, _ats_hit (did the side the model leaned toward vs. the
    OPENING spread actually cover?), and _ml_hit (did the side the model
    favored vs. the fair/vig-free moneyline actually win? NaN where no
    moneyline was captured) -- the two independent "did following the
    model's lean pay off" checks, used by both the per-week breakdown and
    the season-aggregate section below.
    """
    graded = graded.copy()
    graded["home_win"] = (graded["actual_winner"] == "home").astype(int)

    covered_home = (graded["actual_margin"] - graded["open_spread_line"]) > 0
    model_leans_home = graded["spread_edge_open"] > 0
    graded["_ats_hit"] = np.where(model_leans_home, covered_home, ~covered_home)

    has_ml = graded["ml_edge_open"].notna()
    ml_leans_home = graded["ml_edge_open"] > 0
    ml_hit = np.where(ml_leans_home, graded["home_win"] == 1, graded["home_win"] == 0)
    graded["_ml_hit"] = np.where(has_ml, ml_hit, np.nan)

    return graded


def _print_week_breakdown(graded: pd.DataFrame):
    """Per-week table so the spread vs. moneyline trend is visible as weeks
    accumulate toward Week 18, rather than only ever seeing one lump total."""
    print("\n--- Week-by-week: spread vs. moneyline ---")
    header = f"{'Week':<6}{'N':<5}{'Model MAE':<12}{'Spread ATS':<13}{'Moneyline':<14}"
    print(header)
    print("-" * len(header))
    for wk, g in sorted(graded.groupby("week")):
        mae = (g["pred_margin"] - g["actual_margin"]).abs().mean()
        ats = g["_ats_hit"].mean()
        ml_n = int(g["_ml_hit"].notna().sum())
        ml_str = f"{g['_ml_hit'].mean():.0%} (n={ml_n})" if ml_n else "n/a"
        print(f"{wk:<6}{len(g):<5}{mae:<12.2f}{ats:<13.1%}{ml_str:<14}")


def report(season: int = None, week: int = None):
    log = _load_log()
    graded = log[log["graded_at"].notna()].copy()
    if season is not None:
        graded = graded[graded["season"] == season]
    if week is not None:
        graded = graded[graded["week"] == week]

    scope = f"Week {week}" if week else "all weeks"
    if graded.empty:
        print(f"No graded games yet ({scope}). Run `update-results` after games are final.")
        return

    graded = _add_grading_columns(graded)
    n = len(graded)
    print(f"\n=== BACKTEST: {n} graded game(s), {scope} ===\n")

    model_mae = (graded["pred_margin"] - graded["actual_margin"]).abs().mean()
    open_mae = (graded["open_spread_line"] - graded["actual_margin"]).abs().mean()
    model_pred_win = (graded["pred_margin"] > 0).astype(int)
    open_pred_win = (graded["open_spread_line"] > 0).astype(int)

    print(f"Model MAE:             {model_mae:.2f} pts")
    print(f"Vegas (open) MAE:      {open_mae:.2f} pts")
    print(f"Model win accuracy:    {(model_pred_win == graded['home_win']).mean():.1%}")
    print(f"Vegas (open) win acc:  {(open_pred_win == graded['home_win']).mean():.1%}")

    has_close = graded["close_spread_line"].notna()
    if has_close.any():
        c = graded[has_close]
        close_mae = (c["close_spread_line"] - c["actual_margin"]).abs().mean()
        close_pred_win = (c["close_spread_line"] > 0).astype(int)
        print(f"\nVegas (close) MAE:     {close_mae:.2f} pts  ({has_close.sum()} games with a closing line)")
        print(f"Vegas (close) win acc: {(close_pred_win == c['home_win']).mean():.1%}")

    # SPREAD: did the side the model leaned toward (relative to the OPENING
    # line) actually cover? This is the real test of whether following the
    # model's spread edge would have been worth anything, as opposed to
    # just whether the raw margin prediction was numerically close.
    print("\n--- SPREAD performance (ATS) ---")
    print(f"Following the model's lean vs. the opening line: {graded['_ats_hit'].mean():.1%} ({n} games)")
    print("Does a bigger spread disagreement mean more likely right?")
    graded["_edge_bucket"] = pd.cut(
        graded["spread_edge_open"].abs(), bins=[0, 1, 2, 4, np.inf],
        labels=["0-1 pt", "1-2 pt", "2-4 pt", "4+ pt"],
    )
    bucket_stats = graded.groupby("_edge_bucket", observed=True)["_ats_hit"].agg(["mean", "count"])
    for label, row in bucket_stats.iterrows():
        print(f"  {label:8s}  hit rate {row['mean']:.1%}  (n={int(row['count'])})")

    # MONEYLINE: did the side the model favored (relative to the fair,
    # vig-removed market price) actually win outright? Independent from the
    # spread check above -- a model can be sharper on one than the other.
    if graded["_ml_hit"].notna().any():
        has_ml = graded["_ml_hit"].notna()
        print("\n--- MONEYLINE performance ---")
        print(f"Model's favored side (vs. fair market) won "
              f"{graded['_ml_hit'].mean():.1%} of the time ({has_ml.sum()} games)")

    if week is None and graded["week"].nunique() > 1:
        _print_week_breakdown(graded)

    print(
        "\nSample size caution: this is only meaningful once it accumulates "
        "across many weeks/seasons. A handful of games either way is noise, "
        "not a verdict on the model."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Weekly prediction tracker & backtest")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p, week_required):
        p.add_argument("--season", type=int, default=config.CURRENT_SEASON)
        p.add_argument("--week", type=int, required=week_required, default=None)

    add_common_args(sub.add_parser("snapshot", help="Record this week's prediction + market line"), True)
    add_common_args(sub.add_parser("update-closing", help="Refresh the market line for logged games"), True)
    add_common_args(sub.add_parser("update-results", help="Grade logged games against final scores"), True)
    add_common_args(sub.add_parser("report", help="Print backtest metrics for graded games"), False)

    args = parser.parse_args()

    if args.command == "snapshot":
        snapshot(args.season, args.week)
    elif args.command == "update-closing":
        update_closing_lines(args.season, args.week)
    elif args.command == "update-results":
        update_results(args.season, args.week)
    elif args.command == "report":
        report(args.season, args.week)


if __name__ == "__main__":
    main()
