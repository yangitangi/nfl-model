"""
config.py
---------
Single control panel for the NFL prediction model pipeline.

Change settings HERE only — fetch_data.py, build_features.py, train_model.py,
and app.py all read from this file. You should not need to edit those files
just to change a season range, a file path, or which features are turned on.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------
# BASE_DIR auto-detects the project root (folder this config.py lives in),
# so paths work whether you run this in Claude's sandbox, VS Code, or anywhere else.
BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories automatically if they don't exist yet
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SEASON RANGE
# ---------------------------------------------------------------------------
# Update START_SEASON / END_SEASON each year — nothing else needs to change.
START_SEASON = 2010
END_SEASON = 2025          # most recently COMPLETED season with full data
CURRENT_SEASON = 2026      # season you're actively predicting (2026-2027)

SEASONS = list(range(START_SEASON, END_SEASON + 1))

# ---------------------------------------------------------------------------
# DATA SOURCE (nflverse — free, public play-by-play data)
# ---------------------------------------------------------------------------
NFLVERSE_PBP_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.parquet"
)

NFLVERSE_SCHEDULE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/"
    "games.parquet"
)

# ---------------------------------------------------------------------------
# FEATURE FLAGS
# ---------------------------------------------------------------------------
# Turn feature categories on/off without touching build_features.py.
# Useful for staging v1 -> v2 -> v3 as discussed.
FEATURES = {
    "efficiency": True,          # EPA/play, success rate           (v1)
    "turnovers": True,           # turnover margin (regressed)      (v1)
    "qb_performance": True,      # QB EPA per dropback              (v1)
    "rest_travel": True,         # rest days, bye week, travel      (v1)
    "market": True,              # vegas spread/total (if available) (v1)

    "qb_aggressiveness": False,  # CPOE, air yards                  (v2)
    "injuries": False,           # injury report weighted count     (v2)
    "weather": False,            # wind/temp/precip                 (v2)
    "pressure_line_play": False, # sack rate proxies                (v2)

    "pass_block_win_rate": False,# requires PFF/NGS access           (v3)
    "special_teams_dvoa": False, # requires Football Outsiders data (v3)
    "line_movement": False,      # requires odds API history        (v3)
}

# ---------------------------------------------------------------------------
# GAME TYPE
# ---------------------------------------------------------------------------
# nflverse tags games REG (regular season), WC/DIV/CON/SB (playoff rounds).
# Playoff games behave differently (higher stakes, tighter rosters, no
# "resting starters" games) - keep them out of training data by default.
INCLUDE_PLAYOFFS = False

# ---------------------------------------------------------------------------
# ROLLING WINDOW SETTINGS
# ---------------------------------------------------------------------------
ROLLING_WINDOW_GAMES = 5   # how many past games to average for rolling features
MIN_GAMES_FOR_ROLLING = 3  # min games played before rolling stats are considered reliable

# ---------------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------------
TARGET_VARIABLE = "point_margin"   # "point_margin" (regression) or "win_flag" (classification)
MODEL_TYPE = "xgboost"             # "xgboost", "logistic", "linear"
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# PLAY-BY-PLAY COLUMN SELECTION
# ---------------------------------------------------------------------------
# The raw nflverse play-by-play file has 370+ columns per season. Loading all
# of them for 16 seasons at once will exceed memory on most machines. We only
# need a subset to build the v1 feature set, so we prune columns at load time.
#
# Add to this list any time you turn on a new FEATURES flag that needs a raw
# pbp column not already listed here (e.g. enabling "weather" needs weather
# columns from the SCHEDULE file, not pbp, so no change needed there).
PBP_COLUMNS = [
    # identity / context
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "home_team", "away_team", "play_type", "down", "ydstogo", "qtr",
    "game_seconds_remaining",

    # efficiency
    "epa", "success", "wpa",

    # turnovers
    "interception", "fumble_lost",

    # QB performance
    "qb_dropback", "qb_epa", "passer_player_id", "passer_player_name",
    "cpoe", "air_yards", "pass_location",

    # situational
    "yardline_100", "goal_to_go", "touchdown", "field_goal_result",
    "third_down_converted", "third_down_failed",
]

# ---------------------------------------------------------------------------
# TEAM ABBREVIATION FIXES
# ---------------------------------------------------------------------------
# Teams that relocated/renamed since 2010 — nflverse data uses different
# abbreviations across seasons for these franchises. Standardize to current names.
TEAM_ABBR_FIXES = {
    "OAK": "LV",    # Oakland Raiders -> Las Vegas Raiders
    "SD": "LAC",    # San Diego Chargers -> LA Chargers
    "STL": "LA",    # St. Louis Rams -> LA Rams
}
