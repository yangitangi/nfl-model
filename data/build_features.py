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
        "spread_line", "total_line",
    ]].copy()
    home_side["team"] = home_side["home_team"]
    home_side["is_home"] = 1
    home_side["rest_days"] = home_side["home_rest"]
    home_side["team_score"] = home_side["home_score"]
    home_side["opp_score"] = home_side["away_score"]

    away_side = schedules[[
        "game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
        "home_score", "away_score", "home_rest", "away_rest", "div_game",
        "spread_line", "total_line",
    ]].copy()
    away_side["team"] = away_side["away_team"]
    away_side["is_home"] = 0
    away_side["rest_days"] = away_side["away_rest"]
    away_side["team_score"] = away_side["away_score"]
    away_side["opp_score"] = away_side["home_score"]

    schedule_long = pd.concat([home_side, away_side], ignore_index=True)
    schedule_long = schedule_long[[
        "game_id", "season", "week", "game_type", "gameday", "team", "is_home",
        "rest_days", "div_game", "spread_line", "total_line", "team_score", "opp_score",
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

    stat_cols = [
        "off_epa_per_play", "off_success_rate",
        "def_epa_per_play_allowed", "def_success_rate_allowed",
        "turnovers_lost", "turnovers_forced",
        "qb_epa_per_dropback", "qb_cpoe",
        "point_margin",
    ]

    window = config.ROLLING_WINDOW_GAMES
    min_games = config.MIN_GAMES_FOR_ROLLING

    for col in stat_cols:
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
    keep_cols = ["game_id", "season", "week", "game_type", "gameday", "team",
                 "is_home", "rest_days", "div_game", "spread_line", "total_line",
                 "games_played_this_season", "point_margin", "win"] + rolling_cols

    slim = team_game_rolled[keep_cols]

    home = slim[slim["is_home"] == 1].copy()
    away = slim[slim["is_home"] == 0].copy()

    home = home.rename(columns={c: f"home_{c}" for c in home.columns
                                  if c not in ["game_id", "season", "week", "game_type",
                                               "gameday", "div_game", "spread_line", "total_line"]})
    away = away.rename(columns={c: f"away_{c}" for c in away.columns
                                  if c not in ["game_id", "season", "week", "game_type",
                                               "gameday", "div_game", "spread_line", "total_line"]})

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


if __name__ == "__main__":
    print(f"=== Building Features: {config.START_SEASON}-{config.END_SEASON} ===\n")
    df = build_feature_table()
    print(f"\n=== Done: {df.shape[0]:,} games, {df.shape[1]} columns ===")
