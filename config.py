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

# Official weekly injury reports (team, position, report_status = Out/
# Doubtful/Questionable, practice_status), back to 2009. Published by each
# team before that week's games -- legitimate pre-game information, not
# leakage, same category as the market spread.
NFLVERSE_INJURIES_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/"
    "injuries_{season}.parquet"
)

# Team colors/logos reference (sibling nflverse repo, not the main
# nflverse-data releases -- maintained alongside it, same organization).
# Used only for dashboard display (team logo icons), not model features.
NFLVERSE_TEAM_LOGOS_URL = (
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/"
    "teams_colors_logos.csv"
)

# Weekly depth charts (team, player, position, pos_rank -- rank 1 = starter),
# updated throughout the season. Used ONLY to identify each team's CURRENT
# starting QB for upcoming games -- who actually threw passes in a team's
# most recent played game is not a reliable proxy for who's starting next
# (offseason trades/free agency, a benching, or a meaningless Week 18
# relief appearance all break that assumption; see add_current_qb_starters).
NFLVERSE_DEPTH_CHARTS_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/"
    "depth_charts_{season}.parquet"
)

# ---------------------------------------------------------------------------
# FEATURE FLAGS
# ---------------------------------------------------------------------------
# Turn feature categories on/off without touching build_features.py.
# Useful for staging v1 -> v2 -> v3 as discussed.
FEATURES = {
    "efficiency": True,          # EPA/play, success rate           (v1)
    "turnovers": True,           # turnover margin (regressed)      (v1)
    "qb_performance": False,     # QB EPA per dropback, TEAM-blended -- superseded by qb_specific_form (v1)
    "rest_travel": True,         # rest days, bye week, travel      (v1)
    "market": True,              # vegas spread/total (if available) (v1)

    "qb_aggressiveness": False,  # CPOE, air yards, TEAM-blended -- superseded by qb_specific_form (v2)
    # Weighted count of a team's Out/Doubtful/Questionable players on that
    # week's OFFICIAL injury report (nflverse, back to 2009) -- legitimate
    # pre-game info, not leakage, same category as the market spread. Only
    # meaningful close to game time (official reports post a few days
    # before kickoff); weeks further out fall back to 0 (no news yet).
    # Validated via the same 5-season walk-forward panel: improved all
    # three metrics (MAE 9.75->9.71, win accuracy 66.5%->67.4%,
    # ATS 50.2%->51.3%), and at ZERO training-row cost (every team-week
    # has a burden value, even if 0) unlike most other v2/v2.5 features.
    "injuries": True,            # (v2)
    "weather": True,             # wind/temp/dome flag              (v2)
    "pressure_line_play": False, # sack rate proxies                (v2)

    "special_teams": False,      # EPA/play on ST snaps, own data    (v2.5) -- our own metric, see special_teams_dvoa below
    "opponent_adjusted_efficiency": False,  # single-pass opponent-adjustment proxy (v2.5)
    # QB stats attributed to the CURRENT starter (see add_qb_starter_form),
    # not blended across whoever's played for the team over the window --
    # directly fixes the blind spot where a rolling team average takes
    # several games to reflect an in-season starter change (injury,
    # benching). Validated via the same 5-season walk-forward panel used
    # throughout this project: REPLACING qb_performance+qb_aggressiveness
    # with this beat the team-blended baseline on MAE (9.89->9.75), win
    # accuracy (66.2%->66.5%), and ATS (49.6%->50.2%) -- despite losing
    # ~9% of training rows to the stricter missing-data requirement a new
    # starter's thin track record creates. Keeping BOTH together tested
    # worse than this alone (the team-blended version becomes redundant
    # noise once this exists), which is why qb_performance/
    # qb_aggressiveness are off above rather than additionally on.
    "qb_specific_form": True,    # (v2.5)

    "pass_block_win_rate": False,# requires PFF/NGS access           (v3)
    "special_teams_dvoa": False, # requires real Football Outsiders/FTN data -- distinct from
                                  # "special_teams" above, which is our own EPA-based proxy   (v3)
    "line_movement": False,      # requires odds API history        (v3)
}

# Maps each rolling stat column (its base name, before the home_/away_ prefix
# and _rollN suffix) to the FEATURES flag that gates it. A rolling column with
# no entry here is always included (e.g. point_margin_roll5, a core signal not
# tied to any single flag). See train_model.get_feature_columns().
FEATURE_COLUMN_GROUPS = {
    "off_epa_per_play": "efficiency",
    "off_success_rate": "efficiency",
    "def_epa_per_play_allowed": "efficiency",
    "def_success_rate_allowed": "efficiency",
    "turnovers_lost": "turnovers",
    "turnovers_forced": "turnovers",
    "qb_epa_per_dropback": "qb_performance",
    "qb_cpoe": "qb_aggressiveness",
    "qb_air_yards_per_att": "qb_aggressiveness",
    "st_epa_per_play": "special_teams",
    "off_epa_vs_opp_baseline": "opponent_adjusted_efficiency",
    "def_epa_allowed_vs_opp_baseline": "opponent_adjusted_efficiency",
    "start_qb_epa_per_dropback": "qb_specific_form",
    "start_qb_cpoe": "qb_specific_form",
    "start_qb_air_yards_per_att": "qb_specific_form",
}

# Weather defaults for indoor/dome/closed-roof games — there's no real
# weather to record, so these stand in for "climate controlled, a non-factor"
# rather than leaving NaNs that would otherwise shrink the training set.
DOME_DEFAULT_TEMP_F = 70
DOME_DEFAULT_WIND_MPH = 0

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

# "raw": the model predicts actual_margin directly, from scratch.
# "residual": the model predicts (actual_margin - spread_line) -- a small
# correction to the market line -- and spread_line is added back at
# prediction time. spread_line is already the single strongest feature
# (~17-22% importance), so asking the model to only learn the leftover gap
# is a lower-variance problem than reconstructing the whole margin. Requires
# FEATURES["market"] on; missing spread_line at prediction time (a future
# game with no line posted yet) falls back to a 0-point offset (pick'em
# prior) rather than leaving the prediction undefined.
# "blend": the model is trained the same way as "raw" (predicts actual_margin
# from scratch -- spread_line is still available to it as an input feature,
# just not force-added back afterward), then the final prediction is a
# weighted average of that raw prediction and the market spread:
#   MARKET_BLEND_WEIGHT * spread_line + (1 - MARKET_BLEND_WEIGHT) * raw_pred
# This is the technique nfelo (a well-regarded public NFL model) reports
# using. IMPORTANT, found via models/tune_blend_weight.py: when the number
# you blend toward is the SAME number you grade ATS against (true here --
# we only have one spread_line per historical game), blending is
# mathematically incapable of changing which side of the spread you land
# on -- algebraically, blended - spread_line = (1-w)*(raw_pred - spread_line),
# which never changes sign for any w<1. It only shrinks margin error (MAE),
# confirmed empirically: identical ATS hit rate across the entire weight
# grid 0.00-0.95 on 4 walk-forward validation seasons, only MAE moved. So
# this mode is NOT a fix for ATS performance specifically, only for margin
# accuracy -- kept "residual" as the default for that reason. Blending
# would only help ATS if the market number blended toward differs from the
# one graded against (e.g. opening line blended, closing line graded) --
# that needs real opening/closing data, which is what
# tracking/line_movement_tracker.py starts collecting for the 2026 season.
MARGIN_TARGET_MODE = "residual"   # "raw", "residual", or "blend"

# Weight given to the market spread in "blend" mode; (1 - this) goes to the
# model's own raw prediction. Tuned via walk-forward validation across
# multiple training seasons (see models/tune_blend_weight.py) -- not on the
# final holdout season, which would leak. nfelo reported ~0.65 as their own
# fitted value; ours may differ since it depends on this model's own
# raw-mode accuracy relative to the market, not nfelo's.
MARKET_BLEND_WEIGHT = 0.65

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
