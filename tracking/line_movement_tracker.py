"""
tracking/line_movement_tracker.py
-----------------------------------
Captures a snapshot of the market line (spread, moneyline, total, odds)
for a week's games every time it runs, appending to
outputs/line_movement_log.parquet. Meant to run daily through the week
(see the scheduled task), not just at open/close, so we get a real
trajectory -- and start accumulating the kind of opening-vs-closing data
a genuine market blend needs to be testable (see the note in config.py's
MARGIN_TARGET_MODE about why same-reference blending can't move ATS, but
an opening-vs-closing blend potentially could).

Honest limitation: this captures LINE movement (direction, magnitude,
timing) -- not confirmed "reverse" line movement, which requires knowing
which side the public is actually betting (bet-percentage/handle data,
a paid feed nflverse doesn't provide). What we can say is "the line moved
toward X" -- not "that movement was against the public money." Said
plainly in the report output so it isn't overclaimed as more than it is.

Usage:
    python tracking/line_movement_tracker.py capture --week 1
        Snapshot the current line for that week's games. Safe to run more
        than once a day -- each call adds a new timestamped row rather
        than overwriting, since the point is building a trajectory.

    python tracking/line_movement_tracker.py report --week 1
        Show each game's movement from first capture to most recent.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.fetch_data import fetch_schedules, standardize_team_abbrs

LOG_PATH = config.OUTPUTS_DIR / "line_movement_log.parquet"

LOG_COLUMNS = [
    "captured_at", "game_id", "season", "week", "gameday", "home_team", "away_team",
    "spread_line", "home_spread_odds", "away_spread_odds",
    "total_line", "over_odds", "under_odds",
    "home_moneyline", "away_moneyline",
]


def _load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_parquet(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def _save_log(df: pd.DataFrame):
    df.to_parquet(LOG_PATH, index=False)


def capture(season: int, week: int):
    schedules = fetch_schedules(max_season=season, force_refresh=True)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    week_games = schedules[(schedules["season"] == season) & (schedules["week"] == week)]

    if week_games.empty:
        print(f"No games found for season {season} Week {week}.")
        return

    now = pd.Timestamp.now(tz="UTC")
    rows = week_games[[
        "game_id", "season", "week", "gameday", "home_team", "away_team",
        "spread_line", "home_spread_odds", "away_spread_odds",
        "total_line", "over_odds", "under_odds",
        "home_moneyline", "away_moneyline",
    ]].copy()
    rows.insert(0, "captured_at", now)

    log = _load_log()
    log = pd.concat([log, rows], ignore_index=True)
    _save_log(log)
    print(f"Captured {len(rows)} games for Week {week} at {now.isoformat()} -> {LOG_PATH}")


def report(season: int, week: int):
    log = _load_log()
    sub = log[(log["season"] == season) & (log["week"] == week)]
    if sub.empty:
        print(f"No captures yet for season {season} Week {week}. Run `capture` first.")
        return

    print(f"\n=== Line movement: season {season} Week {week} ===")
    print(f"{sub['game_id'].nunique()} games, up to {sub.groupby('game_id').size().max()} captures/game\n")

    for game_id, g in sub.sort_values("captured_at").groupby("game_id"):
        g = g.reset_index(drop=True)
        first, last = g.iloc[0], g.iloc[-1]
        n = len(g)
        spread_delta = last["spread_line"] - first["spread_line"]
        total_delta = last["total_line"] - first["total_line"]
        has_ml = pd.notna(last["home_moneyline"]) and pd.notna(first["home_moneyline"])

        print(f"{first['away_team']} @ {first['home_team']}  ({n} capture{'s' if n != 1 else ''}, "
              f"{first['captured_at'].strftime('%m/%d')} -> {last['captured_at'].strftime('%m/%d')})")
        print(f"  Spread: {first['spread_line']:+.1f} -> {last['spread_line']:+.1f}  ({spread_delta:+.1f})")
        print(f"  Total:  {first['total_line']:.1f} -> {last['total_line']:.1f}  ({total_delta:+.1f})")
        if has_ml:
            ml_delta = last["home_moneyline"] - first["home_moneyline"]
            print(f"  Home ML: {first['home_moneyline']:+.0f} -> {last['home_moneyline']:+.0f}  ({ml_delta:+.0f})")
        if abs(spread_delta) >= 1.0:
            print(f"  ⚠ Notable spread move ({spread_delta:+.1f} pts)")
        print()

    print(
        'Note: this shows line MOVEMENT (direction/magnitude) -- not confirmed "reverse" line '
        "movement, which needs bet-percentage/handle data (which side the public is actually "
        "betting) that isn't available here. Treat a big move as \"something changed,\" not "
        '"the sharps did this."'
    )


def main():
    parser = argparse.ArgumentParser(description="Daily line movement capture & report")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--season", type=int, default=config.CURRENT_SEASON)
        p.add_argument("--week", type=int, required=True)

    add_common(sub.add_parser("capture", help="Snapshot the current line for a week's games"))
    add_common(sub.add_parser("report", help="Show movement trajectory for a week's games"))

    args = parser.parse_args()
    if args.command == "capture":
        capture(args.season, args.week)
    elif args.command == "report":
        report(args.season, args.week)


if __name__ == "__main__":
    main()
