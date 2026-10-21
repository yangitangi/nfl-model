"""
rebuild/bootstrap_data.py
-------------------------
Downloads the raw nflverse data the rebuild needs into data/raw/ (gitignored,
~300 MB, fully regenerable) and prints how long it took. Safe to re-run: files
already on disk are skipped.

    python rebuild/bootstrap_data.py            # download everything
    python rebuild/bootstrap_data.py --check    # just report what is on disk
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data.fetch_data import (  # noqa: E402
    fetch_depth_chart, fetch_injuries_season, fetch_pbp_season, fetch_schedules,
)


def on_disk() -> dict:
    d = config.RAW_DATA_DIR
    return {
        "pbp": sorted(int(p.stem.split("_")[1]) for p in d.glob("pbp_*.parquet")),
        "injuries": sorted(int(p.stem.split("_")[1]) for p in d.glob("injuries_*.parquet")),
        "depth_chart": sorted(int(p.stem.split("_")[2]) for p in d.glob("depth_chart_*.parquet")),
        "schedules": (d / "schedules.parquet").exists(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--first", type=int, default=config.START_SEASON)
    ap.add_argument("--last", type=int, default=config.END_SEASON)
    args = ap.parse_args()

    if args.check:
        print(on_disk())
        return

    t0 = time.time()
    seasons = list(range(args.first, args.last + 1))
    fetch_schedules(max_season=args.last + 1)
    for s in seasons:
        fetch_pbp_season(s, columns=["game_id"])        # saves the FULL file, loads one column
        fetch_injuries_season(s)
        if s >= 2018:                                   # depth charts: best effort
            fetch_depth_chart(s)
    state = on_disk()
    print(f"\nDone in {time.time() - t0:.0f}s: {state}")
    missing = [s for s in seasons if s not in state["pbp"]]
    if missing:
        print(f"MISSING PBP SEASONS (network blocked?): {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
