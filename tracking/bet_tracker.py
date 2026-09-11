"""
tracking/bet_tracker.py
------------------------
Tracks YOUR OWN bets placed throughout the season -- separate from
prediction_tracker.py, which tracks the MODEL's own predictions against
Vegas. This file has nothing to do with what the model predicts; it's just
a record of real bets placed and whether they won, so a season's worth of
action can be reviewed in one place.

Each bet is graded against the actual final score using the BET'S OWN line
(what you actually got from your book), not our model's or nflverse's
market snapshot -- those can differ from the number you actually got.

Usage:
    python tracking/bet_tracker.py add --week 1 --team SEA --type moneyline
    python tracking/bet_tracker.py add --week 1 --team SEA --type spread --line -3
    python tracking/bet_tracker.py add --week 1 --team SF --type spread --line 4
        Records a new bet as "pending". game_id/opponent are looked up
        automatically from the schedule for that team/week.

    python tracking/bet_tracker.py grade [--week 1]
        Grades every pending bet whose game is now final. Safe to re-run.

    python tracking/bet_tracker.py report [--week 1]
        Prints your W-L-P record, overall and by week.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.fetch_data import fetch_schedules, standardize_team_abbrs

BETS_PATH = config.OUTPUTS_DIR / "bets.csv"

BET_COLUMNS = [
    "bet_id", "season", "week", "game_id", "team", "opponent", "is_home",
    "bet_type", "line", "odds", "stake", "placed_at",
    "result", "actual_margin", "graded_at", "notes",
]


def _load_bets() -> pd.DataFrame:
    if not BETS_PATH.exists():
        return pd.DataFrame(columns=BET_COLUMNS)
    # An all-empty column (e.g. every bet still "pending", so graded_at is
    # blank for all rows) round-trips through CSV as float64 NaN -- coerce
    # the datetime columns back explicitly so a later Timestamp assignment
    # doesn't hit a dtype mismatch.
    df = pd.read_csv(BETS_PATH)
    for col in ["placed_at", "graded_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).astype("datetime64[ns, UTC]")
    return df


def _save_bets(df: pd.DataFrame):
    df.to_csv(BETS_PATH, index=False)


def _find_game(season: int, week: int, team: str) -> pd.Series:
    schedules = fetch_schedules(max_season=season)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    games = schedules[
        (schedules["season"] == season) & (schedules["week"] == week)
        & ((schedules["home_team"] == team) | (schedules["away_team"] == team))
    ]
    if games.empty:
        raise ValueError(f"No season {season} Week {week} game found for {team}.")
    return games.iloc[0]


def add_bet(season: int, week: int, team: str, bet_type: str, line: float = None,
            odds: float = None, stake: float = None, notes: str = None) -> str:
    """
    Adds a new pending bet. team is the side you bet ON -- for a spread bet,
    `line` is written the way you'd say the bet out loud: negative if `team`
    is laying points (favorite, e.g. "SEA -3" -> line=-3), positive if
    `team` is getting points (underdog, e.g. "SF +4" -> line=4). For a
    total bet, `team` is still required (for game lookup) but is otherwise
    unused -- direction comes from bet_type ("over"/"under").
    """
    team = team.upper()
    game = _find_game(season, week, team)
    is_home = game["home_team"] == team
    opponent = game["away_team"] if is_home else game["home_team"]

    bets = _load_bets()
    bet_id = f"{game['game_id']}_{team}_{bet_type}_{line if line is not None else ''}".rstrip("_")
    if not bets.empty and bet_id in set(bets["bet_id"]):
        print(f"[skip] Bet already recorded: {bet_id}")
        return bet_id

    row = pd.DataFrame([{
        "bet_id": bet_id, "season": season, "week": week, "game_id": game["game_id"],
        "team": team, "opponent": opponent, "is_home": is_home,
        "bet_type": bet_type, "line": line, "odds": odds, "stake": stake,
        "placed_at": pd.Timestamp.now(tz="UTC"),
        "result": "pending", "actual_margin": np.nan, "graded_at": pd.NaT,
        "notes": notes,
    }])
    bets = pd.concat([bets, row], ignore_index=True)
    _save_bets(bets)
    line_str = f" {line:+g}" if line is not None else ""
    print(f"[added] {team}{line_str} ({bet_type}) -- {opponent} in Week {week}")
    return bet_id


def _grade_one(bet: pd.Series, home_score: float, away_score: float) -> tuple[str, float]:
    team_score = home_score if bet["is_home"] else away_score
    opp_score = away_score if bet["is_home"] else home_score
    team_margin = team_score - opp_score

    bet_type = bet["bet_type"]
    if bet_type == "moneyline":
        if team_score == opp_score:
            return "push", team_margin
        return ("win" if team_margin > 0 else "loss"), team_margin

    if bet_type == "spread":
        net = team_margin + float(bet["line"])
        if net == 0:
            return "push", team_margin
        return ("win" if net > 0 else "loss"), team_margin

    if bet_type in ("over", "under"):
        total = home_score + away_score
        line = float(bet["line"])
        if total == line:
            return "push", team_margin
        covered_over = total > line
        won = covered_over if bet_type == "over" else not covered_over
        return ("win" if won else "loss"), team_margin

    raise ValueError(f"Unknown bet_type: {bet_type}")


def grade_bets(season: int, week: int = None):
    bets = _load_bets()
    mask = (bets["season"] == season) & (bets["result"] == "pending")
    if week is not None:
        mask &= bets["week"] == week
    if not mask.any():
        print(f"Nothing pending for season {season}" + (f" Week {week}." if week else "."))
        return

    schedules = fetch_schedules(max_season=season, force_refresh=True)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])

    graded_n = 0
    for idx in bets[mask].index:
        match = schedules[schedules["game_id"] == bets.at[idx, "game_id"]]
        if match.empty or pd.isna(match.iloc[0]["home_score"]):
            continue
        m = match.iloc[0]
        result, team_margin = _grade_one(bets.loc[idx], m["home_score"], m["away_score"])
        bets.at[idx, "result"] = result
        bets.at[idx, "actual_margin"] = team_margin
        bets.at[idx, "graded_at"] = pd.Timestamp.now(tz="UTC")
        graded_n += 1

    _save_bets(bets)
    remaining = int(mask.sum()) - graded_n
    print(f"Graded {graded_n} bet(s).")
    if remaining:
        print(f"{remaining} still pending (game not final yet).")


def report(season: int = None, week: int = None):
    bets = _load_bets()
    if season is not None:
        bets = bets[bets["season"] == season]
    if week is not None:
        bets = bets[bets["week"] == week]
    if bets.empty:
        print("No bets recorded" + (f" for season {season}" if season else "") + ".")
        return

    icon = {"win": "W", "loss": "L", "push": "P", "pending": "?"}
    print(f"\n{'Week':<6}{'Team':<6}{'Type':<11}{'Line':<8}{'Opp':<6}{'Result':<8}")
    print("-" * 45)
    for _, r in bets.sort_values(["week", "team"]).iterrows():
        line_str = f"{r['line']:+g}" if pd.notna(r["line"]) else ""
        print(f"{r['week']:<6}{r['team']:<6}{r['bet_type']:<11}{line_str:<8}{r['opponent']:<6}"
              f"{icon.get(r['result'], r['result']):<8}")

    graded = bets[bets["result"].isin(["win", "loss", "push"])]
    if not graded.empty:
        w = (graded["result"] == "win").sum()
        l = (graded["result"] == "loss").sum()
        p = (graded["result"] == "push").sum()
        decided = w + l
        win_pct = w / decided if decided else 0.0
        print(f"\nRecord: {w}-{l}-{p}  ({win_pct:.1%} of decided bets)")


def main():
    parser = argparse.ArgumentParser(description="Personal bet tracker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Record a new bet")
    p_add.add_argument("--season", type=int, default=config.CURRENT_SEASON)
    p_add.add_argument("--week", type=int, required=True)
    p_add.add_argument("--team", required=True)
    p_add.add_argument("--type", dest="bet_type", required=True,
                        choices=["moneyline", "spread", "over", "under"])
    p_add.add_argument("--line", type=float, default=None)
    p_add.add_argument("--odds", type=float, default=None)
    p_add.add_argument("--stake", type=float, default=None)
    p_add.add_argument("--notes", default=None)

    p_grade = sub.add_parser("grade", help="Grade pending bets against final scores")
    p_grade.add_argument("--season", type=int, default=config.CURRENT_SEASON)
    p_grade.add_argument("--week", type=int, default=None)

    p_report = sub.add_parser("report", help="Print your bet record")
    p_report.add_argument("--season", type=int, default=None)
    p_report.add_argument("--week", type=int, default=None)

    args = parser.parse_args()
    if args.command == "add":
        add_bet(args.season, args.week, args.team, args.bet_type, args.line,
                args.odds, args.stake, args.notes)
    elif args.command == "grade":
        grade_bets(args.season, args.week)
    elif args.command == "report":
        report(args.season, args.week)


if __name__ == "__main__":
    main()
