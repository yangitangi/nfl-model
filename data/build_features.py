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
from data.fetch_data import (
    fetch_all_injuries, fetch_all_seasons, fetch_depth_chart, fetch_injuries_season,
    fetch_pbp_season, fetch_schedules, standardize_team_abbrs,
)

# Weighted severity of each official injury-report status, for a simple
# "injury burden" score per team per week. Out counts fully, Doubtful
# mostly, Questionable partially; Probable is a deprecated (pre-~2016)
# designation for a minor, unlikely-to-affect-availability concern, weighted
# lightly rather than excluded outright since it still carries some signal
# in older seasons. Anything else (no designation, practice-only status) is 0.
INJURY_STATUS_WEIGHTS = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25, "Probable": 0.1}

# Per-team-game stats that get rolled into trailing averages. Shared between
# add_rolling_features (historical training rows) and compute_team_form_snapshot
# (the same math, evaluated one game past the end of a team's played history,
# for upcoming-game predictions) so the two can't drift out of sync.
ROLLING_STAT_COLS = [
    "off_epa_per_play", "off_success_rate",
    "def_epa_per_play_allowed", "def_success_rate_allowed",
    "turnovers_lost", "turnovers_forced",
    "qb_epa_per_dropback", "qb_cpoe", "qb_air_yards_per_att",
    "st_epa_per_play",
    "point_margin",
]

# Opponent-adjusted delta columns (see add_opponent_adjusted_features) --
# derived, not raw per-game stats, but rolled with the exact same
# shift+rolling math, so they're kept in their own list rather than mixed
# into ROLLING_STAT_COLS above.
OPPONENT_ADJUSTED_STAT_COLS = ["off_epa_vs_opp_baseline", "def_epa_allowed_vs_opp_baseline"]

# Same three QB stats as the team-blended version above, but attributed to
# the specific passer -- see add_qb_starter_form.
QB_STARTER_STAT_COLS = ["qb_epa_per_dropback", "qb_cpoe", "qb_air_yards_per_att"]


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

    # --- Special teams (punts, kickoffs, field goals, extra points) ---
    # nflverse computes EPA uniformly across every play type, including
    # these, so this reuses the exact same methodology as offensive/defensive
    # EPA above rather than introducing a different metric. posteam on a
    # special-teams play is the kicking/punting team (the team taking the
    # action), which is what this attributes the value to.
    st_plays = pbp[pbp["play_type"].isin(["punt", "kickoff", "field_goal", "extra_point"])]
    st_stats = st_plays.groupby(["game_id", "posteam"]).agg(
        st_epa_per_play=("epa", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    # --- Combine into one team-game row ---
    team_game = offense.merge(defense, on=["game_id", "team"], how="outer")
    team_game = team_game.merge(turnovers_lost, on=["game_id", "team"], how="left")
    team_game = team_game.merge(qb_stats, on=["game_id", "team"], how="left")
    team_game = team_game.merge(st_stats, on=["game_id", "team"], how="left")
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


def add_opponent_adjusted_features(team_game_rolled: pd.DataFrame) -> pd.DataFrame:
    """
    Approximates the core idea behind DVOA-style ratings -- the same raw
    numbers mean different things against different opponents -- without
    DVOA's full iterative simultaneous solve. For each game, this team's
    RAW (unrolled) offensive EPA/play is compared against the opponent's
    own entering-game defensive rolling average (their established
    baseline coming in, already anti-leakage-safe since it's the
    shift+rolling output of add_rolling_features, which this must run
    after). Outperforming a good defense's baseline counts for more than
    matching it against a bad one. Same idea in reverse for defense vs.
    the opponent's offensive baseline.

    The resulting per-game deltas are then rolled the same way as every
    other stat (shift+rolling over the last N games) to produce the final
    opponent-adjusted features -- so a single game's raw performance never
    leaks into its own adjusted rolling average, same discipline as
    everywhere else in this pipeline.

    This is a single-pass proxy, not a full iterative DVOA solve -- simpler
    to reason about and to keep leakage-free, at the cost of not capturing
    DVOA's opponent-of-opponent convergence effects.
    """
    window = config.ROLLING_WINDOW_GAMES
    min_games = config.MIN_GAMES_FOR_ROLLING
    roll_suffix = f"_roll{window}"

    opp_baseline = team_game_rolled[[
        "game_id", "team", f"off_epa_per_play{roll_suffix}", f"def_epa_per_play_allowed{roll_suffix}",
    ]].rename(columns={
        "team": "opponent",
        f"off_epa_per_play{roll_suffix}": "opp_off_epa_baseline",
        f"def_epa_per_play_allowed{roll_suffix}": "opp_def_epa_baseline",
    })

    merged = team_game_rolled.merge(opp_baseline, on=["game_id", "opponent"], how="left")

    merged["off_epa_vs_opp_baseline"] = merged["off_epa_per_play"] - merged["opp_def_epa_baseline"]
    merged["def_epa_allowed_vs_opp_baseline"] = (
        merged["def_epa_per_play_allowed"] - merged["opp_off_epa_baseline"]
    )

    merged = merged.sort_values(["team", "gameday"]).reset_index(drop=True)
    for col in OPPONENT_ADJUSTED_STAT_COLS:
        merged[f"{col}{roll_suffix}"] = (
            merged.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_games).mean())
        )

    return merged


def build_qb_player_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player, per-game QB stats -- the same three columns as the
    team-blended qb_* stats in build_team_game_stats, but split out by
    passer_player_id instead of pooled across whoever threw for the team
    that game. This is what lets a QB change (injury, benching) show up
    in the very next game's features instead of being smoothed away by a
    team-level rolling window that has no notion of who's actually playing.
    """
    live_plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    dropbacks = live_plays[live_plays["qb_dropback"] == 1]
    qb_player_game = dropbacks.groupby(
        ["game_id", "posteam", "passer_player_id", "passer_player_name"]
    ).agg(
        qb_epa_per_dropback=("qb_epa", "mean"),
        qb_cpoe=("cpoe", "mean"),
        qb_air_yards_per_att=("air_yards", "mean"),
        qb_dropbacks=("qb_epa", "count"),
    ).reset_index().rename(columns={"posteam": "team"})
    return qb_player_game.dropna(subset=["passer_player_id"])


def _qb_primary_with_asof(team_game_rolled: pd.DataFrame, qb_player_game: pd.DataFrame,
                           window: int) -> pd.DataFrame:
    """
    Shared by add_qb_starter_form (historical rows) and
    get_current_starter_form (upcoming-game snapshot): whoever had the most
    dropbacks for a team in a game is that game's "starter of record," and
    each PLAYER's own rolling form is computed inclusive of that start --
    grouped by passer_player_id, not team, so it carries seamlessly across
    a trade or a team change.
    """
    min_games = config.MIN_GAMES_FOR_ROLLING
    primary = (
        qb_player_game.sort_values("qb_dropbacks", ascending=False)
        .groupby(["game_id", "team"]).first().reset_index()
    )
    dates = team_game_rolled[["game_id", "team", "gameday"]].drop_duplicates()
    primary = primary.merge(dates, on=["game_id", "team"], how="inner")
    primary = primary.sort_values(["passer_player_id", "gameday"]).reset_index(drop=True)

    for col in QB_STARTER_STAT_COLS:
        primary[f"{col}_asof"] = (
            primary.groupby("passer_player_id")[col]
            .transform(lambda s: s.rolling(window=window, min_periods=min_games).mean())
        )
    return primary


def add_qb_starter_form(team_game_rolled: pd.DataFrame, qb_player_game: pd.DataFrame,
                         window: int = None) -> pd.DataFrame:
    """
    Attributes QB form to the CURRENT starter -- whoever had the most
    dropbacks in a team's most recent game -- rather than blending
    together whoever has thrown for the team over the last N games.
    Directly fixes the blind spot where a team-level rolling average takes
    several games to "catch up" after a starter is hurt or benched
    mid-season (e.g. an injury on play 5 of a game -- the backup's own
    limited-but-real track record should count, not get diluted by
    5 games' worth of the previous starter's numbers).

    "Current starter" is identified from the team's own most recently
    PLAYED game (shift(1) on the team's own game sequence -- the same
    anti-leakage pattern used everywhere else in this pipeline), so a
    starter change is reflected starting the very next game, not blended
    away over several. This "most recent game" heuristic is correct for
    historical rows (we know exactly who started each past game), but is
    NOT used for upcoming games -- see get_current_starter_form, which
    uses the real depth chart instead, since a team's actual next starter
    can differ from who played their last game (trade, free agency,
    benching, or a backup mopping up a meaningless finale).
    """
    window = config.ROLLING_WINDOW_GAMES if window is None else window
    roll_suffix = f"_roll{window}"

    primary = _qb_primary_with_asof(team_game_rolled, qb_player_game, window)
    asof_cols = [f"{c}_asof" for c in QB_STARTER_STAT_COLS]
    team_game_rolled = team_game_rolled.merge(
        primary[["game_id", "team"] + asof_cols], on=["game_id", "team"], how="left"
    )
    team_game_rolled = team_game_rolled.sort_values(["team", "gameday"]).reset_index(drop=True)

    # Shift by the TEAM's own game sequence: game G inherits the starter-
    # of-record's as-of form from G's immediately preceding game.
    for col in QB_STARTER_STAT_COLS:
        team_game_rolled[f"start_{col}{roll_suffix}"] = (
            team_game_rolled.groupby("team")[f"{col}_asof"].shift(1)
        )

    # _asof columns are intentionally kept (not dropped) -- they don't end
    # in _rollN so assemble_game_level_table won't pick them up as model
    # features, but snapshot_qb_starter_form needs them for playoff games.
    return team_game_rolled


def get_current_qb_starters(season: int, force_refresh: bool = True) -> pd.DataFrame:
    """
    Each team's CURRENT starting QB per the official depth chart (pos_rank
    1 at QB), using the most recent snapshot available -- depth charts get
    updated throughout the season, not just once at kickoff, so this stays
    current as injuries/benchings happen. This is what actually determines
    who plays next Sunday; a team's own most-recent-game passer (the
    fallback used elsewhere in this file) is only a proxy for that, and a
    wrong one whenever there's been a trade, a free-agent signing, a
    benching, or a backup mopping up a meaningless finale.
    """
    depth_chart = fetch_depth_chart(season, force_refresh=force_refresh)
    if depth_chart.empty:
        return pd.DataFrame(columns=["team", "passer_player_id"])
    qb1 = depth_chart[(depth_chart["pos_abb"] == "QB") & (depth_chart["pos_rank"] == 1)]
    if qb1.empty:
        return pd.DataFrame(columns=["team", "passer_player_id"])
    latest = qb1.sort_values("dt").groupby("team").last().reset_index()
    return latest[["team", "gsis_id"]].rename(columns={"gsis_id": "passer_player_id"})


def snapshot_qb_starter_form(team_game_rolled_with_asof: pd.DataFrame,
                              qb_player_game: pd.DataFrame = None, season: int = None,
                              window: int = None) -> pd.DataFrame:
    """
    For upcoming games: each team's CURRENT starter's rolling form, ready
    to be inherited by their next game.

    When qb_player_game + season are given, "current starter" comes from
    the real depth chart (get_current_qb_starters) and the form used is
    that SPECIFIC PLAYER's own rolling stats -- wherever/whichever team he
    last played for, which correctly follows him across a trade or a
    free-agent signing. Falls back to the team's own most-recent-game
    passer (the old heuristic) for any team where the depth-chart starter
    has no prior primary-passer history to compute a rolling average from
    (e.g. a true rookie making his first career start).

    Without qb_player_game/season (used by build_playoff_features, which
    is retrospective -- grading already-played games under the SAME
    conditions the model trained under), falls back to the team-based
    heuristic only, since "today's depth chart" would be a nonsensical,
    time-traveling input for a game that already happened.
    """
    window = config.ROLLING_WINDOW_GAMES if window is None else window
    asof_cols = [f"{c}_asof" for c in QB_STARTER_STAT_COLS]

    fallback = (
        team_game_rolled_with_asof.sort_values("gameday")
        .groupby("team")[asof_cols].last()
        .reset_index()
    )

    if qb_player_game is None or season is None:
        result = fallback
    else:
        starters = get_current_qb_starters(season)
        if starters.empty:
            result = fallback
        else:
            primary = _qb_primary_with_asof(team_game_rolled_with_asof, qb_player_game, window)
            player_asof = (
                primary.sort_values("gameday").groupby("passer_player_id")[asof_cols].last().reset_index()
            )
            resolved = starters.merge(player_asof, on="passer_player_id", how="left")
            result = fallback.merge(
                resolved[["team"] + asof_cols], on="team", how="left", suffixes=("_fallback", "")
            )
            for col in asof_cols:
                result[col] = result[col].fillna(result[f"{col}_fallback"])
            result = result[["team"] + asof_cols]

    return result.rename(columns={f"{c}_asof": f"start_{c}_roll{window}" for c in QB_STARTER_STAT_COLS})


def build_injury_burden(injuries: pd.DataFrame) -> pd.DataFrame:
    """
    A simple weighted count of a team's notable injuries entering a given
    week's game, from the OFFICIAL weekly injury report -- published
    before that week's games, so this is legitimate pre-game information,
    not leakage, same category as the market spread. Not a rolling stat
    (no shift+window needed): the report for week W is inherently about
    week W's game, not a trailing average of past weeks.
    """
    injuries = injuries.copy()
    injuries["weight"] = injuries["report_status"].map(INJURY_STATUS_WEIGHTS).fillna(0)
    burden = injuries.groupby(["season", "week", "team"])["weight"].sum().reset_index()
    return burden.rename(columns={"weight": "injury_burden"})


def attach_injury_burden(team_game: pd.DataFrame, injury_burden: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge so a team-week with NO qualifying injuries (not present in
    injury_burden at all, since it only has rows for actual designations)
    correctly becomes 0 burden rather than NaN -- an inner merge here would
    silently drop every healthy team's games from training.
    """
    merged = team_game.merge(injury_burden, on=["season", "week", "team"], how="left")
    merged["injury_burden"] = merged["injury_burden"].fillna(0)
    return merged


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
    if "injury_burden" in team_game_rolled.columns:
        keep_cols.append("injury_burden")

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

    print("Attaching weekly injury reports...")
    injuries = fetch_all_injuries(seasons)
    injuries = standardize_team_abbrs(injuries, ["team"])
    injury_burden = build_injury_burden(injuries)
    team_game = attach_injury_burden(team_game, injury_burden)

    print("Computing rolling features (no data leakage)...")
    team_game_rolled = add_rolling_features(team_game)

    print("Computing opponent-adjusted features...")
    team_game_rolled = add_opponent_adjusted_features(team_game_rolled)

    print("Attributing QB form to the current starter, not the team...")
    qb_player_game = build_qb_player_game_stats(pbp)
    team_game_rolled = add_qb_starter_form(team_game_rolled, qb_player_game)

    print("Assembling final game-level table...")
    game_level = assemble_game_level_table(team_game_rolled)

    if save:
        out_path = config.PROCESSED_DATA_DIR / "game_level_features.parquet"
        game_level.to_parquet(out_path, index=False)
        print(f"[saved] {len(game_level):,} games -> {out_path}")

    return game_level


def compute_team_form_snapshot(team_game: pd.DataFrame, stat_cols: list[str] = None) -> pd.DataFrame:
    """
    One row per team: the rolling stat averages that team carries INTO their
    next (not yet played) game — the mean of their last ROLLING_WINDOW_GAMES
    played games. This is mathematically what add_rolling_features' shift+
    rolling would produce for a hypothetical next row, since both only ever
    look at prior games. Carries across season boundaries here too, matching
    the training pipeline (team strength doesn't reset to zero on Week 1).

    stat_cols defaults to ROLLING_STAT_COLS (raw per-game stats). Pass
    OPPONENT_ADJUSTED_STAT_COLS to snapshot those instead -- the tail-mean
    math is identical either way, it doesn't matter whether the column is a
    raw stat or a derived one, as long as team_game already has it.
    """
    stat_cols = ROLLING_STAT_COLS if stat_cols is None else stat_cols
    window = config.ROLLING_WINDOW_GAMES
    roll_suffix = f"_roll{window}"

    team_game_sorted = team_game.sort_values(["team", "gameday"])
    snapshot = (
        team_game_sorted.groupby("team")[stat_cols]
        .apply(lambda g: g.tail(window).mean())
        .rename(columns={c: f"{c}{roll_suffix}" for c in stat_cols})
        .reset_index()
    )
    return snapshot


def build_upcoming_features(season: int = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Builds a game-level feature row for each upcoming (unplayed) game in
    `season` (defaults to config.CURRENT_SEASON), using each team's rolling
    form as of their most recent PLAYED game — which may be from the prior
    season, since rolling stats intentionally carry over season boundaries.

    Uses the same feature column names as game_level_features.parquet, so
    the saved model + feature_columns.joblib can be applied directly. There
    are no outcome columns (actual_margin/win/covered_spread) since these
    games haven't been played yet.

    force_refresh re-downloads the schedule instead of using the local cache
    -- needed to pick up market line movement or final scores as the week
    progresses (e.g. the weekly prediction tracker). Historical play-by-play
    is always read from cache regardless, since completed seasons don't change.
    """
    season = config.CURRENT_SEASON if season is None else season
    window = config.ROLLING_WINDOW_GAMES
    roll_suffix = f"_roll{window}"
    snapshot_cols = [f"{c}{roll_suffix}" for c in ROLLING_STAT_COLS]
    adj_snapshot_cols = [f"{c}{roll_suffix}" for c in OPPONENT_ADJUSTED_STAT_COLS]
    qb_snapshot_cols = [f"start_{c}{roll_suffix}" for c in QB_STARTER_STAT_COLS]
    all_snapshot_cols = snapshot_cols + adj_snapshot_cols + qb_snapshot_cols

    print("Loading raw data...")
    pbp = fetch_all_seasons()
    if season not in config.SEASONS:
        # The season being predicted is in progress (games already played
        # this week aren't in the historical 2010-END_SEASON range) --
        # pull its play-by-play too, so rolling stats (including which QB
        # is the current starter, see add_qb_starter_form) reflect games
        # already played this season, not just through END_SEASON.
        # force_refresh here since this file grows week to week.
        current_pbp = fetch_pbp_season(season, force_refresh=force_refresh)
        if not current_pbp.empty:
            pbp = pd.concat([pbp, current_pbp], ignore_index=True)
    pbp = standardize_team_abbrs(pbp, ["home_team", "away_team", "posteam", "defteam"])

    # max_season=season pulls in the upcoming season too — fetch_schedules'
    # default range stops at config.END_SEASON, which wouldn't include it.
    schedules = fetch_schedules(max_season=season, force_refresh=force_refresh)
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
    # Opponent-adjusted features need each team's entering-game rolling
    # baseline to compare opponents against, so the first-pass rolling has
    # to run here too even though the historical rolling *table* itself
    # isn't otherwise needed for the upcoming-game snapshot.
    team_game_rolled = add_rolling_features(team_game)
    team_game_rolled = add_opponent_adjusted_features(team_game_rolled)

    print("Attributing QB form to the current starter, not the team...")
    qb_player_game = build_qb_player_game_stats(pbp)
    team_game_rolled = add_qb_starter_form(team_game_rolled, qb_player_game, window=window)

    print(f"Computing team form snapshots (last {window} games played)...")
    snapshot = compute_team_form_snapshot(team_game_rolled, ROLLING_STAT_COLS)
    snapshot_adj = compute_team_form_snapshot(team_game_rolled, OPPONENT_ADJUSTED_STAT_COLS)
    snapshot_qb = snapshot_qb_starter_form(
        team_game_rolled, qb_player_game=qb_player_game, season=season, window=window
    )
    snapshot = snapshot.merge(snapshot_adj, on="team").merge(snapshot_qb, on="team")

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

    # Injury burden for upcoming games: this week's OWN report, not a
    # rolling snapshot -- fetched fresh (force_refresh, since the current
    # season's file grows weekly) and only meaningful for whichever week is
    # close enough to have a real report filed; weeks further out fall back
    # to 0 (no known injuries yet) via the same left-merge-and-fillna as
    # the historical pipeline, which is an honest "no news yet," not a bug.
    print("Attaching weekly injury reports (this week's own report, not rolling)...")
    current_injuries = fetch_injuries_season(season, force_refresh=True)
    current_injuries = standardize_team_abbrs(current_injuries, ["team"])
    if not current_injuries.empty:
        injury_burden_upcoming = build_injury_burden(current_injuries)
    else:
        injury_burden_upcoming = pd.DataFrame(columns=["season", "week", "team", "injury_burden"])

    upcoming = upcoming.merge(
        injury_burden_upcoming.rename(columns={"team": "home_team", "injury_burden": "home_injury_burden"}),
        on=["season", "week", "home_team"], how="left")
    upcoming = upcoming.merge(
        injury_burden_upcoming.rename(columns={"team": "away_team", "injury_burden": "away_injury_burden"}),
        on=["season", "week", "away_team"], how="left")
    upcoming["home_injury_burden"] = upcoming["home_injury_burden"].fillna(0)
    upcoming["away_injury_burden"] = upcoming["away_injury_burden"].fillna(0)

    print(f"Building feature rows for {len(upcoming):,} upcoming games...")
    home_snapshot = snapshot.rename(
        columns={**{"team": "home_team"}, **{c: f"home_{c}" for c in all_snapshot_cols}}
    )
    away_snapshot = snapshot.rename(
        columns={**{"team": "away_team"}, **{c: f"away_{c}" for c in all_snapshot_cols}}
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
        "home_injury_burden", "away_injury_burden",
    ] + [f"home_{c}" for c in all_snapshot_cols] + [f"away_{c}" for c in all_snapshot_cols]

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


def build_playoff_features(season: int = None) -> pd.DataFrame:
    """
    Builds a game-level feature row for each ALREADY-PLAYED playoff game in
    `season` -- for retrospective backtesting, not live prediction (see
    build_upcoming_features for that).

    Every playoff game for a team uses the SAME entering-form snapshot:
    their rolling stats as of the end of that team's regular season.
    Playoff games never update this snapshot (a divisional-round opponent
    doesn't get a "boosted" snapshot from winning their wild-card game) --
    deliberately, since the deployed model was itself trained with playoffs
    entirely excluded from every team's rolling history
    (config.INCLUDE_PLAYOFFS=False). Scoring playoff games any other way
    would feed the model a kind of rolling signal (playoff-influenced form)
    it never saw in training.

    Unlike build_upcoming_features, market lines and weather here are REAL
    (these games already happened, so there's no forecast-fallback need),
    and actual_margin/actual_winner are populated since the outcome is
    known -- this is for grading the model against history, not predicting
    the future.
    """
    season = config.CURRENT_SEASON if season is None else season
    window = config.ROLLING_WINDOW_GAMES
    roll_suffix = f"_roll{window}"
    snapshot_cols = [f"{c}{roll_suffix}" for c in ROLLING_STAT_COLS]
    adj_snapshot_cols = [f"{c}{roll_suffix}" for c in OPPONENT_ADJUSTED_STAT_COLS]
    qb_snapshot_cols = [f"start_{c}{roll_suffix}" for c in QB_STARTER_STAT_COLS]
    all_snapshot_cols = snapshot_cols + adj_snapshot_cols + qb_snapshot_cols

    print("Loading raw data...")
    pbp = fetch_all_seasons()
    if season not in config.SEASONS:
        current_pbp = fetch_pbp_season(season, force_refresh=True)
        if not current_pbp.empty:
            pbp = pd.concat([pbp, current_pbp], ignore_index=True)
    pbp = standardize_team_abbrs(pbp, ["home_team", "away_team", "posteam", "defteam"])

    schedules = fetch_schedules(max_season=season)
    schedules = standardize_team_abbrs(schedules, ["home_team", "away_team"])

    played_reg = schedules[(schedules["game_type"] == "REG") & (schedules["home_score"].notna())]
    playoff_games = schedules[
        (schedules["season"] == season) & (schedules["game_type"] != "REG")
        & (schedules["home_score"].notna())
    ].copy()

    if playoff_games.empty:
        print(f"No played playoff games found for season {season}.")
        return pd.DataFrame()

    print("Aggregating historical (regular-season-only) plays to team-game level...")
    team_game = build_team_game_stats(pbp)
    team_game = attach_opponent_and_turnovers_forced(team_game)
    team_game = add_schedule_context(team_game, played_reg)
    team_game_rolled = add_rolling_features(team_game)
    team_game_rolled = add_opponent_adjusted_features(team_game_rolled)

    print("Attributing QB form to the current starter, not the team...")
    qb_player_game = build_qb_player_game_stats(pbp)
    team_game_rolled = add_qb_starter_form(team_game_rolled, qb_player_game, window=window)

    print(f"Computing end-of-regular-season form snapshots for {season}...")
    snapshot = compute_team_form_snapshot(team_game_rolled, ROLLING_STAT_COLS)
    snapshot_adj = compute_team_form_snapshot(team_game_rolled, OPPONENT_ADJUSTED_STAT_COLS)
    snapshot_qb = snapshot_qb_starter_form(team_game_rolled, window=window)
    snapshot = snapshot.merge(snapshot_adj, on="team").merge(snapshot_qb, on="team")

    games_played = (
        team_game_rolled.loc[team_game_rolled["season"] == season]
        .groupby("team")["games_played_this_season"].max() + 1
    ).rename("games_played_entering_playoffs").reset_index()

    # Weather is real here (these games already happened) -- the same
    # is_outdoor/dome-default treatment as everywhere else, but the outdoor
    # median fallback should essentially never trigger since real readings
    # already exist for played games (unlike future/upcoming games).
    is_outdoor_hist = played_reg["roof"].isin(["outdoors", "open"])
    outdoor_temp_median = played_reg.loc[is_outdoor_hist, "temp"].median()
    outdoor_wind_median = played_reg.loc[is_outdoor_hist, "wind"].median()

    playoff_games["is_outdoor"] = playoff_games["roof"].isin(["outdoors", "open"]).astype(int)
    outdoor_mask = playoff_games["is_outdoor"] == 1
    playoff_games["temp"] = np.where(
        outdoor_mask, playoff_games["temp"].fillna(outdoor_temp_median), config.DOME_DEFAULT_TEMP_F)
    playoff_games["wind"] = np.where(
        outdoor_mask, playoff_games["wind"].fillna(outdoor_wind_median), config.DOME_DEFAULT_WIND_MPH)

    # Injury reports are real here too (these games already happened) --
    # the actual report filed for that playoff week, not an estimate.
    playoff_injuries = fetch_injuries_season(season)
    playoff_injuries = standardize_team_abbrs(playoff_injuries, ["team"])
    if not playoff_injuries.empty:
        playoff_injury_burden = build_injury_burden(playoff_injuries)
    else:
        playoff_injury_burden = pd.DataFrame(columns=["season", "week", "team", "injury_burden"])
    playoff_games = playoff_games.merge(
        playoff_injury_burden.rename(columns={"team": "home_team", "injury_burden": "home_injury_burden"}),
        on=["season", "week", "home_team"], how="left")
    playoff_games = playoff_games.merge(
        playoff_injury_burden.rename(columns={"team": "away_team", "injury_burden": "away_injury_burden"}),
        on=["season", "week", "away_team"], how="left")
    playoff_games["home_injury_burden"] = playoff_games["home_injury_burden"].fillna(0)
    playoff_games["away_injury_burden"] = playoff_games["away_injury_burden"].fillna(0)

    home_snapshot = snapshot.rename(
        columns={**{"team": "home_team"}, **{c: f"home_{c}" for c in all_snapshot_cols}})
    away_snapshot = snapshot.rename(
        columns={**{"team": "away_team"}, **{c: f"away_{c}" for c in all_snapshot_cols}})

    game_level = playoff_games.merge(home_snapshot, on="home_team", how="left")
    game_level = game_level.merge(away_snapshot, on="away_team", how="left")

    game_level = game_level.merge(
        games_played.rename(columns={"team": "home_team",
                                      "games_played_entering_playoffs": "home_games_played_this_season"}),
        on="home_team", how="left")
    game_level = game_level.merge(
        games_played.rename(columns={"team": "away_team",
                                      "games_played_entering_playoffs": "away_games_played_this_season"}),
        on="away_team", how="left")

    game_level["home_rest_days"] = game_level["home_rest"]
    game_level["away_rest_days"] = game_level["away_rest"]
    game_level["actual_margin"] = game_level["home_score"] - game_level["away_score"]
    game_level["actual_winner"] = np.where(game_level["actual_margin"] > 0, "home", "away")

    keep_cols = [
        "game_id", "season", "week", "game_type", "gameday",
        "home_team", "away_team", "div_game", "spread_line", "total_line",
        "home_spread_odds", "away_spread_odds", "home_moneyline", "away_moneyline",
        "over_odds", "under_odds",
        "is_outdoor", "temp", "wind",
        "home_rest_days", "away_rest_days",
        "home_games_played_this_season", "away_games_played_this_season",
        "home_injury_burden", "away_injury_burden",
        "home_score", "away_score", "actual_margin", "actual_winner",
    ] + [f"home_{c}" for c in all_snapshot_cols] + [f"away_{c}" for c in all_snapshot_cols]

    game_level = game_level[keep_cols].copy()
    game_level["gameday"] = pd.to_datetime(game_level["gameday"])
    game_level = game_level.sort_values(["week", "gameday"]).reset_index(drop=True)

    return game_level


if __name__ == "__main__":
    print(f"=== Building Features: {config.START_SEASON}-{config.END_SEASON} ===\n")
    df = build_feature_table()
    print(f"\n=== Done: {df.shape[0]:,} games, {df.shape[1]} columns ===")
