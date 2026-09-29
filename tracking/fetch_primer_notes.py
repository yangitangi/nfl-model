"""
tracking/fetch_primer_notes.py
-------------------------------
Pulls Evan Abrams' weekly Action Network NFL Betting Primer (a public
roundup of betting trends/systems for every game -- see
https://www.actionnetwork.com/nfl/nfl-betting-primer-trends-stats-systems-for-every-game)
and turns the per-game trend notes into rows in outputs/analyst_notes.csv,
the same file used for manual injury/roster notes and rendered on the
dashboard's game cards.

The primer's own text explicitly invites reuse with attribution ("If you
want to use any of the notes or content below, please credit @EvanHAbrams
and @ActionNetworkHQ") -- every note we store is prefixed with that credit.

The primer page itself is a client-rendered widget (an iframe at
nfl-primer.pages.dev) that doesn't expose its content to a page scraper --
its data actually comes from a public JSON feed the widget calls directly,
which is what this script hits instead.

Usage:
    python tracking/fetch_primer_notes.py --season 2026 --week 4
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

NOTES_FEED_URL = "https://primer-feed.evanhabrams.workers.dev/feed/nfl-primer-notes/w{week}"

# The primer feed uses a couple of team codes that differ from nflverse's
# (and this project's) convention -- map them before building game_id.
PRIMER_TEAM_FIXES = {"JAC": "JAX"}

ATTRIBUTION = "Action Network Betting Primer (Evan Abrams, @EvanHAbrams / @ActionNetworkHQ)"


def fetch_primer_notes(week: int) -> list[dict]:
    url = NOTES_FEED_URL.format(week=week)
    resp = requests.get(url, params={"ts": int(datetime.now().timestamp() * 1000)}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("notes", [])


def _fix_team(code: str) -> str:
    return PRIMER_TEAM_FIXES.get(code, code)


def build_primer_note_rows(season: int, week: int) -> pd.DataFrame:
    """One row per game -- all of that game's trend notes bulleted together
    into a single note, tagged by which team each trend is about."""
    notes = fetch_primer_notes(week)

    by_game: dict[str, list[dict]] = {}
    for n in notes:
        game = n.get("game")
        if not game or game == "SLATE" or "@" not in game:
            continue  # SLATE = league-wide note, not tied to one matchup
        by_game.setdefault(game, []).append(n)

    rows = []
    for game, items in sorted(by_game.items()):
        away_raw, home_raw = game.split("@")
        away, home = _fix_team(away_raw), _fix_team(home_raw)
        game_id = f"{season}_{week:02d}_{away}_{home}"

        bullets = []
        for n in items:
            tag = f"[{_fix_team(n['team'])}]" if n.get("team") else "[Both]"
            text = " ".join(n["text"].split())  # collapse embedded newlines/whitespace
            bullets.append(f"{tag} {text}")

        note = f"{ATTRIBUTION}, Week {week}: " + " | ".join(bullets)
        rows.append({
            "game_id": game_id,
            "week": week,
            "note": note,
            "created_at": datetime.now().strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(rows, columns=["game_id", "week", "note", "created_at"])


def sync_primer_notes(season: int, week: int) -> pd.DataFrame:
    """Fetches this week's primer notes and writes them into
    outputs/analyst_notes.csv, replacing any primer notes previously synced
    for this same week (identified by the fixed attribution prefix) so
    re-running mid-week picks up updates without duplicating rows."""
    new_rows = build_primer_note_rows(season, week)

    path = config.OUTPUTS_DIR / "analyst_notes.csv"
    if path.exists():
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame(columns=["game_id", "week", "note", "created_at"])

    is_primer_row = existing["note"].astype(str).str.startswith(ATTRIBUTION)
    is_this_week = existing["week"] == week
    kept = existing[~(is_primer_row & is_this_week)]

    combined = pd.concat([kept, new_rows], ignore_index=True)
    combined.to_csv(path, index=False)

    print(f"Synced {len(new_rows)} primer game-notes for {season} Week {week} -> {path}")
    for _, r in new_rows.iterrows():
        print(f"  {r['game_id']}: {len(r['note'])} chars")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync the Action Network NFL Betting Primer into analyst_notes.csv")
    parser.add_argument("--season", type=int, default=config.END_SEASON)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    sync_primer_notes(args.season, args.week)
