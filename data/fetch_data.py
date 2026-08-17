"""
fetch_data.py
-------------
Pulls raw play-by-play and schedule data from nflverse (free, public source)
and saves it locally as parquet files, so you only download each season once.

Run directly to fetch everything configured in config.py:
    python data/fetch_data.py

Or import individual functions elsewhere:
    from data.fetch_data import fetch_pbp_season, fetch_all_seasons, fetch_schedules
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

# Allow running this file directly (adds project root to path so `import config` works)
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def fetch_pbp_season(season: int, force_refresh: bool = False,
                      columns: list[str] = None) -> pd.DataFrame:
    """
    Fetch play-by-play data for a single season.

    If already downloaded, loads from local cache (data/raw/) instead of
    re-downloading, unless force_refresh=True.

    The FULL raw file (all 370+ columns) is always what's saved to disk, so
    nothing is lost. `columns` only controls what gets loaded into memory for
    this call — pass config.PBP_COLUMNS (the default) to avoid memory blowups
    when combining many seasons. Pass columns=None explicitly to load everything.
    """
    local_path = config.RAW_DATA_DIR / f"pbp_{season}.parquet"
    columns = config.PBP_COLUMNS if columns is None else columns

    if local_path.exists() and not force_refresh:
        print(f"  [cache] Season {season} already on disk -> {local_path.name}")
        return _read_parquet_safe(local_path, columns)

    url = config.NFLVERSE_PBP_URL_TEMPLATE.format(season=season)
    print(f"  [download] Season {season} from nflverse...")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Failed to fetch season {season}: {e}")
        return pd.DataFrame()

    local_path.write_bytes(response.content)
    df = _read_parquet_safe(local_path, columns)
    print(f"  [saved] {len(df):,} plays -> {local_path.name}")
    return df


def _read_parquet_safe(path: Path, columns: list[str] = None) -> pd.DataFrame:
    """
    Reads a parquet file, restricting to `columns` if given. Silently drops
    any requested columns that don't exist in that file (nflverse occasionally
    renames/adds columns between seasons) instead of erroring out.
    """
    if columns is None:
        return pd.read_parquet(path)

    # Read only the schema (column names), not the actual data, to check availability
    available = set(pq.ParquetFile(path).schema.names)
    valid_cols = [c for c in columns if c in available]
    missing = set(columns) - set(valid_cols)
    if missing:
        print(f"  [warn] {path.name}: columns not found, skipped: {sorted(missing)}")
    return pd.read_parquet(path, columns=valid_cols)


def fetch_all_seasons(seasons: list[int] = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch play-by-play data for every season in the given list
    (defaults to config.SEASONS) and combine into a single DataFrame.
    """
    seasons = seasons or config.SEASONS
    print(f"Fetching play-by-play data for {len(seasons)} seasons "
          f"({seasons[0]}-{seasons[-1]})...")

    all_seasons = []
    for season in seasons:
        df = fetch_pbp_season(season, force_refresh=force_refresh)
        if not df.empty:
            all_seasons.append(df)

    if not all_seasons:
        raise RuntimeError("No play-by-play data was fetched. Check your network/config.")

    combined = pd.concat(all_seasons, ignore_index=True)
    print(f"\nTotal: {len(combined):,} plays across {len(all_seasons)} seasons")
    return combined


def fetch_schedules(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch the full game schedule (all seasons, includes final scores,
    rest days, and Vegas lines when available).
    """
    local_path = config.RAW_DATA_DIR / "schedules.parquet"

    if local_path.exists() and not force_refresh:
        print(f"  [cache] Schedules already on disk -> {local_path.name}")
        return pd.read_parquet(local_path)

    print("  [download] Schedules from nflverse...")
    try:
        response = requests.get(config.NFLVERSE_SCHEDULE_URL, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Failed to fetch schedules: {e}")
        return pd.DataFrame()

    local_path.write_bytes(response.content)
    df = pd.read_parquet(local_path)

    # Filter to configured season range only
    df = df[df["season"].between(config.START_SEASON, config.END_SEASON)]
    print(f"  [saved] {len(df):,} games -> {local_path.name}")
    return df


def standardize_team_abbrs(df: pd.DataFrame, team_cols: list[str]) -> pd.DataFrame:
    """
    Applies config.TEAM_ABBR_FIXES to relocated/renamed franchises
    so a team's history stays connected across seasons (e.g. OAK/LV Raiders).
    """
    df = df.copy()
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].replace(config.TEAM_ABBR_FIXES)
    return df


if __name__ == "__main__":
    print(f"=== NFL Data Fetch: {config.START_SEASON}-{config.END_SEASON} ===\n")

    schedules = fetch_schedules()
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])

    pbp = fetch_all_seasons()
    pbp = standardize_team_abbrs(pbp, ["home_team", "away_team", "posteam", "defteam"])

    print("\n=== Fetch complete ===")
    print(f"Schedules: {schedules.shape}")
    print(f"Play-by-play: {pbp.shape}")
