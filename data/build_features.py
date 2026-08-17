"""
build_features.py
------------------
Turns raw play-by-play + schedule data into a game-level feature table that
a model can train on: one row per game, with rolling team stats computed
ONLY from games prior to that matchup (no data leakage).

Run directly to build the full processed feature table:
    python data/build_features.py

Or import the main function elsewhere:
    from data.build_features import build_feature_table
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.fetch_data import fetch_all_seasons, fetch_schedules, standardize_team_abbrs

# Per-team-game stats that get rolled into trailing averages. Shared between
# add_rolling_features (historical training rows) and compute_team_form_snapshot
# (the same math, evaluated one game past the end of a team's played history,
# for upcoming-game predictions) so the two can't drift out of sync.
ROLLING_STAT_COLS = [
    "off_epa_per_play", "off_success_rate",
    "def_epa_per_play_allowed", "def_success_rate_allowed",
    "turnovers_lost", "turnovers_forced",
    "qb_epa_per_dropback", "qb_cpoe", "qb_air_yards_per_att",
    "point_margin",
]


def build_team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses play-by-play data down to one row per team per game:
    offensive efficiency (when team has the ball) and defensive
    efficiency (when team is on defense), plus turnovers and QB performance.
    """
    # Only live scrimmage plays count toward efficiency stats (drop kneels,
    # spikes, penalties-only plays, etc. by requiring a play_type)
    live_plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    # --- Offensive stats (grouped by the team that had the ball) ---
    offense = live_plays.groupby(["game_id", "posteam"]).agg(
        off_epa_per_play=("epa", "mean"),
        off_success_rate=("success", "mean"),
        off_plays=("epa", "count"),
    ).reset_index().rename(columns={"posteam": "team"})

    # --- Defensive stats (grouped by the team that was defending) ---
    defense = live_plays.groupby(["game_id", "defteam"]).agg(
        def_epa_per_play_allowed=("epa", "mean"),
        def_success_rate_allowed=("success", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    # --- Turnovers: lost (team had the ball) vs forced (team was defending) ---
    turnovers_lost = pbp.groupby(["game_id", "posteam"]).agg(
        turnovers_lost=("interception", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    fumbles_lost = pbp.groupby(["game_id", "posteam"])["fumble_lost"].sum().reset_index()
    fumbles_lost = fumbles_lost.rename(columns={"posteam": "team", "fumble_lost": "fumbles_lost"})
    turnovers_lost = turnovers_lost.merge(fumbles_lost, on=["game_id", "team"], how="outer")
    turnovers_lost["turnovers_lost"] = (
        turnovers_lost["turnovers_lost"].fillna(0) + turnovers_lost["fumbles_lost"].fillna(0)
    )
    turnovers_lost = turnovers_lost[["game_id", "team", "turnovers_lost"]]

    # --- QB performance (dropbacks only) ---
    dropbacks = live_plays[live_plays["qb_dropback"] == 1]
    qb_stats = dropbacks.groupby(["game_id", "posteam"]).agg(
        qb_epa_per_dropback=("qb_epa", "mean"),
        qb_cpoe=("cpoe", "mean"),
        # Mean intended air yards per dropback — how far downfield a QB
        # targets on average. Sacks/scrambles have no air_yards and are
        # skipped by mean() rather than counted as 0, so this only reflects
        # actual throws. Paired with CPOE under "qb_aggressiveness": CPOE
        # measures accuracy vs. expectation, this measures willingness to
        # push the ball downfield in the first place.
        qb_air_yards_per_att=("air_yards", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    # --- Combine into one team-game row ---
    team_game = offense.merge(defense, on=["game_id", "team"], how="outer")
    team_game = team_game.merge(turnovers_lost, on=["game_id", "team"], how="left")
    team_game = team_game.merge(qb_stats, on=["game_id", "team"], how="left")
    team_game["turnovers_lost"] = team_game["turnovers_lost"].fillna(0)

    return team_game


def attach_opponent_and_turnovers_forced(team_game: pd.DataFrame) -> pd.DataFrame:
    """
    Adds turnovers_forced by matching each team-game row to its opponent's
    turnovers_lost in the same game_id.
    """
    opp = team_game[["game_id", "team", "turnovers_lost"]].rename(
        columns={"team": "opponent", "turnovers_lost": "turnovers_forced"}
    )
    # Merge each row to the OTHER team in the same game
    merged = team_game.merge(opp, on="game_id", how="left")
    merged = merged[merged["team"] != merged["opponent"]].reset_index(drop=True)
    return merged


def add_schedule_context(team_game: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches season, week, date, home/away flag, rest days, and score
    to each team-game row using the schedule table.
    """
    home_side = schedules[[
        "game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
        "home_score", "away_score", "home_rest", "away_rest", "div_game",
        "spread_line", "total_line", "roof", "temp", "wind",
    ]].copy()
    home_side["team"] = home_side["home_team"]
    home_side["is_home"] = 1
    home_side["rest_days"] = home_side["home_rest"]
    home_side["team_score"] = home_side["home_score"]
    home_side["opp_score"] = home_side["away_score"]

    away_side = schedules[[
        "game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
        "home_score", "away_score", "home_rest", "away_rest", "div_game",
        "spread_line", "total_line", "roof", "temp", "wind",
    ]].copy()
    away_side["team"] = away_side["away_team"]
    away_side["is_home"] = 0
    away_side["rest_days"] = away_side["away_rest"]
    away_side["team_score"] = away_side["away_score"]
    away_side["opp_score"] = away_side["home_score"]

    schedule_long = pd.concat([home_side, away_side], ignore_index=True)

    # Weather: nflverse only records temp/wind for true outdoor games — dome,
    # closed-roof, and retractable-open games are climate controlled and come
    # through as NaN. Fill those with neutral "non-factor" defaults instead of
    # leaving NaN (which would otherwise drop those games from training via
    # the dropna in prepare_data). Genuinely missing outdoor readings (rare,
    # ~5% of outdoor games) get the outdoor median rather than the dome default.
    schedule_long["is_outdoor"] = schedule_long["roof"].isin(["outdoors", "open"]).astype(int)
    outdoor_mask = schedule_long["is_outdoor"] == 1
    outdoor_temp_median = schedule_long.loc[outdoor_mask, "temp"].median()
    outdoor_wind_median = schedule_long.loc[outdoor_mask, "wind"].median()

    schedule_long["temp"] = schedule_long["temp"].where(outdoor_mask, config.DOME_DEFAULT_TEMP_F)
    schedule_long["wind"] = schedule_long["wind"].where(outdoor_mask, config.DOME_DEFAULT_WIND_MPH)
    schedule_long["temp"] = schedule_long["temp"].fillna(outdoor_temp_median)
    schedule_long["wind"] = schedule_long["wind"].fillna(outdoor_wind_median)

    schedule_long = schedule_long[[
        "game_id", "season", "week", "game_type", "gameday", "team", "is_home",
        "rest_days", "div_game", "spread_line", "total_line",
        "is_outdoor", "temp", "wind", "team_score", "opp_score",
    ]]

    merged = team_game.merge(schedule_long, on=["game_id", "team"], how="inner")
    merged["gameday"] = pd.to_datetime(merged["gameday"])
    merged["point_margin"] = merged["team_score"] - merged["opp_score"]
    merged["win"] = (merged["point_margin"] > 0).astype(int)

    return merged


def add_rolling_features(team_game: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling averages of each stat using ONLY prior games for that
    team (shift(1) before rolling — the game being predicted is never
    included in its own rolling average). This is the key anti-leakage step.
    """
    team_game = team_game.sort_values(["team", "gameday"]).reset_index(drop=True)

    window = config.ROLLING_WINDOW_GAMES
    min_games = config.MIN_GAMES_FOR_ROLLING

    for col in ROLLING_STAT_COLS:
        rolled = (
            team_game.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_games).mean())
        )
        team_game[f"{col}_roll{window}"] = rolled

    # Games played so far this season (also shifted — doesn't count current game)
    team_game["games_played_this_season"] = (
        team_game.groupby(["team", "season"]).cumcount()
    )

    return team_game


def assemble_game_level_table(team_game_rolled: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the team-game (2 rows per game) table into the final
    game-level table (1 row per game) with home_/away_ prefixed features,
    ready for model training.
    """
    roll_suffix = f"_roll{config.ROLLING_WINDOW_GAMES}"
    rolling_cols = [c for c in team_game_rolled.columns if c.endswith(roll_suffix)]
    # Columns that describe the game itself (not a specific team) and should
    # stay unprefixed rather than becoming home_x/away_x duplicates.
    shared_game_cols = ["game_id", "season", "week", "game_type", "gameday",
                         "div_game", "spread_line", "total_line",
                         "is_outdoor", "temp", "wind"]
    keep_cols = shared_game_cols + ["team", "is_home", "rest_days",
                 "games_played_this_season", "point_margin", "win"] + rolling_cols

    slim = team_game_rolled[keep_cols]

    home = slim[slim["is_home"] == 1].copy()
    away = slim[slim["is_home"] == 0].copy()

    home = home.rename(columns={c: f"home_{c}" for c in home.columns
                                  if c not in shared_game_cols})
    away = away.rename(columns={c: f"away_{c}" for c in away.columns
                                  if c not in shared_game_cols})

    away_only_cols = ["game_id"] + [c for c in away.columns if c.startswith("away_")]

    game_level = home.merge(away[away_only_cols], on="game_id", how="inner")

    # Final target variables
    game_level["actual_margin"] = game_level["home_point_margin"]
    game_level["actual_winner"] = np.where(game_level["actual_margin"] > 0, "home", "away")
    game_level["covered_spread"] = np.where(
        game_level["spread_line"].notna(),
        (game_level["actual_margin"] - game_level["spread_line"]) > 0,
        np.nan,
    )

    return game_level.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def build_feature_table(seasons: list[int] = None, save: bool = True) -> pd.DataFrame:
    """
    Full pipeline: raw data -> team-game stats -> rolling features ->
    game-level table. This is the main entry point.
    """
    print("Loading raw data...")
    pbp = fetch_all_seasons(seasons)
    pbp = standardize_team_abbrs(pbp, ["home_team", "away_team", "posteam", "defteam"])

    schedules = fetch_schedules()
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    if not config.INCLUDE_PLAYOFFS:
        schedules = schedules[schedules["game_type"] == "REG"]
    # Only keep games that have actually been played (have a final score) —
    # future/unplayed games can't have historical features built from them here.
    # (Predicting FUTURE games uses a separate function — see predict_upcoming below.)
    played_schedules = schedules[schedules["home_score"].notna()]

    print("Aggregating plays to team-game level...")
    team_game = build_team_game_stats(pbp)
    team_game = attach_opponent_and_turnovers_forced(team_game)

    print("Attaching schedule context (rest, home/away, market lines)...")
    team_game = add_schedule_context(team_game, played_schedules)

    print("Computing rolling features (no data leakage)...")
    team_game_rolled = add_rolling_features(team_game)

    print("Assembling final game-level table...")
    game_level = assemble_game_level_table(team_game_rolled)

    if save:
        out_path = config.PROCESSED_DATA_DIR / "game_level_features.parquet"
        game_level.to_parquet(out_path, index=False)
        print(f"[saved] {len(game_level):,} games -> {out_path}")

    return game_level


def compute_team_form_snapshot(team_game: pd.DataFrame) -> pd.DataFrame:
    """
    One row per team: the rolling stat averages that team carries INTO their
    next (not yet played) game — the mean of their last ROLLING_WINDOW_GAMES
    played games. This is mathematically what add_rolling_features' shift+
    rolling would produce for a hypothetical next row, since both only ever
    look at prior games. Carries across season boundaries here too, matching
    the training pipeline (team strength doesn't reset to zero on Week 1).
    """
    window = config.ROLLING_WINDOW_GAMES
    roll_suffix = f"_roll{window}"

    team_game_sorted = team_game.sort_values(["team", "gameday"])
    snapshot = (
        team_game_sorted.groupby("team")[ROLLING_STAT_COLS]
        .apply(lambda g: g.tail(window).mean())
        .rename(columns={c: f"{c}{roll_suffix}" for c in ROLLING_STAT_COLS})
        .reset_index()
    )
    return snapshot


def build_upcoming_features(season: int = None) -> pd.DataFrame:
    """
    Builds a game-level feature row for each upcoming (unplayed) game in
    `season` (defaults to config.CURRENT_SEASON), using each team's rolling
    form as of their most recent PLAYED game — which may be from the prior
    season, since rolling stats intentionally carry over season boundaries.

    Uses the same feature column names as game_level_features.parquet, so
    the saved model + feature_columns.joblib can be applied directly. There
    are no outcome columns (actual_margin/win/covered_spread) since these
    games haven't been played yet.
    """
    season = config.CURRENT_SEASON if season is None else season
    window = config.ROLLING_WINDOW_GAMES
    roll_suffix = f"_roll{window}"
    snapshot_cols = [f"{c}{roll_suffix}" for c in ROLLING_STAT_COLS]

    print("Loading raw data...")
    pbp = fetch_all_seasons()
    pbp = standardize_team_abbrs(pbp, ["home_team", "away_team", "posteam", "defteam"])

    # max_season=season pulls in the upcoming season too — fetch_schedules'
    # default range stops at config.END_SEASON, which wouldn't include it.
    schedules = fetch_schedules(max_season=season)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])
    if not config.INCLUDE_PLAYOFFS:
        schedules = schedules[schedules["game_type"] == "REG"]

    played = schedules[schedules["home_score"].notna()]
    upcoming = schedules[
        (schedules["season"] == season) & (schedules["home_score"].isna())
    ].copy()

    if upcoming.empty:
        print(f"No upcoming games found for season {season}.")
        return pd.DataFrame()

    print("Aggregating historical plays to team-game level...")
    team_game = build_team_game_stats(pbp)
    team_game = attach_opponent_and_turnovers_forced(team_game)
    team_game = add_schedule_context(team_game, played)

    print(f"Computing team form snapshots (last {window} games played)...")
    snapshot = compute_team_form_snapshot(team_game)

    # Weather fill for upcoming games: forecasts don't exist this far out, so
    # temp/wind are ~always NaN even for outdoor stadiums (unlike training
    # data, where it's ~5% missing). Fall back to the historical outdoor
    # median; dome/closed/open stadiums get the same neutral defaults used
    # in training. Medians come from `played`, not `upcoming`, since the
    # latter's temp/wind columns are ~100% NaN and would produce NaN medians.
    is_outdoor_hist = played["roof"].isin(["outdoors", "open"])
    outdoor_temp_median = played.loc[is_outdoor_hist, "temp"].median()
    outdoor_wind_median = played.loc[is_outdoor_hist, "wind"].median()

    upcoming["is_outdoor"] = upcoming["roof"].isin(["outdoors", "open"]).astype(int)
    outdoor_mask = upcoming["is_outdoor"] == 1
    upcoming["temp"] = np.where(outdoor_mask, outdoor_temp_median, config.DOME_DEFAULT_TEMP_F)
    upcoming["wind"] = np.where(outdoor_mask, outdoor_wind_median, config.DOME_DEFAULT_WIND_MPH)

    print(f"Building feature rows for {len(upcoming):,} upcoming games...")
    home_snapshot = snapshot.rename(
        columns={**{"team": "home_team"}, **{c: f"home_{c}" for c in snapshot_cols}}
    )
    away_snapshot = snapshot.rename(
        columns={**{"team": "away_team"}, **{c: f"away_{c}" for c in snapshot_cols}}
    )

    game_level = upcoming.merge(home_snapshot, on="home_team", how="left")
    game_level = game_level.merge(away_snapshot, on="away_team", how="left")

    game_level["home_rest_days"] = game_level["home_rest"]
    game_level["away_rest_days"] = game_level["away_rest"]
    # No 2026 games have been played yet, so every team enters every
    # upcoming game — Week 1 or Week 18 — with 0 completed games this season.
    game_level["home_games_played_this_season"] = 0
    game_level["away_games_played_this_season"] = 0

    keep_cols = [
        "game_id", "season", "week", "game_type", "gameday",
        "home_team", "away_team", "div_game", "spread_line", "total_line",
        "home_spread_odds", "away_spread_odds", "home_moneyline", "away_moneyline",
        "over_odds", "under_odds",
        "is_outdoor", "temp", "wind",
        "home_rest_days", "away_rest_days",
        "home_games_played_this_season", "away_games_played_this_season",
    ] + [f"home_{c}" for c in snapshot_cols] + [f"away_{c}" for c in snapshot_cols]

    game_level = game_level[keep_cols].copy()
    game_level["gameday"] = pd.to_datetime(game_level["gameday"])
    game_level = game_level.sort_values(["week", "gameday"]).reset_index(drop=True)

    missing = game_level[f"home_{snapshot_cols[0]}"].isna() | game_level[f"away_{snapshot_cols[0]}"].isna()
    if missing.any():
        teams = pd.concat([
            game_level.loc[missing, "home_team"], game_level.loc[missing, "away_team"],
        ]).unique()
        print(f"[warn] {missing.sum()} upcoming games involve a team with no rolling "
              f"history in {config.START_SEASON}-{config.END_SEASON} data: {sorted(teams)}")

    return game_level


if __name__ == "__main__":
    print(f"=== Building Features: {config.START_SEASON}-{config.END_SEASON} ===\n")
    df = build_feature_table()
    print(f"\n=== Done: {df.shape[0]:,} games, {df.shape[1]} columns ===")
