"""
dashboard/app.py
-----------------
Streamlit dashboard: Week 1 matchups with Vegas market odds (spread, moneyline,
total) next to the model's own prediction, plus a simple "edge" comparison
between the two. Model validation metrics (held-out season vs. Vegas) are
shown at the top for context.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.build_features import build_upcoming_features
from data.fetch_data import fetch_team_logos
from models.train_model import prepare_data, evaluate
from models.predict import (
    load_model_artifacts as _load_model_artifacts,
    predict as _predict,
    add_market_edges,
    format_spread,
    format_ml,
)

st.set_page_config(page_title="NFL Game Predictions", page_icon="\U0001F3C8", layout="wide")

# Which week of the upcoming season to show. Hardcoded rather than a
# selectbox for now — Streamlit's selectbox widget wasn't behaving in
# testing, and Week 1 is what's actually actionable right now anyway.
WEEK_TO_SHOW = 1

# "Notable edge" thresholds — just a display cue, not a betting signal.
# The model currently trails Vegas on the 2025 holdout (see metrics below),
# so a big edge is more likely model error than a real market gap. Treat
# these as a diagnostic view of where the model disagrees with the market,
# not a recommendation.
SPREAD_EDGE_THRESHOLD = 2.0    # points
ML_EDGE_THRESHOLD = 0.05       # 5 percentage points

# Approximate team primary/accent colors, picked for visibility as a border
# accent on a dark card (a few teams' true primary is near-black, so those
# use their brighter secondary color instead — this is a visual cue, not a
# brand-accuracy exercise).
TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#3B2A85", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#C83803", "CIN": "#FB4F14", "CLE": "#FF3C00",
    "DAL": "#3357A6", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#24693D",
    "HOU": "#A71930", "IND": "#254A97", "JAX": "#00A6B7", "KC": "#E31837",
    "LA": "#2C6FBB", "LAC": "#0080C6", "LV": "#C0C0C0", "MIA": "#00A8B0",
    "MIN": "#6B3FA0", "NE": "#C60C30", "NO": "#D3BC8D", "NYG": "#3E5EA8",
    "NYJ": "#1F7A52", "PHI": "#00676E", "PIT": "#FFB612", "SEA": "#69BE28",
    "SF": "#D2001C", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#8C2A2A",
}

CARD_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at top left, #2a3b66 0%, #202c48 55%, #1c2438 100%);
}
.hero-box {
    background: linear-gradient(135deg, #2a3552 0%, #242e48 100%);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.3rem 1.6rem 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 1.05rem; font-weight: 700; color: #f2f4f8; margin-bottom: 1rem;
}
.hero-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1rem;
}
@media (max-width: 700px) { .hero-grid { grid-template-columns: repeat(2, 1fr); } }
.hero-stat {
    background: rgba(255,255,255,0.07); border-radius: 12px; padding: 0.85rem 1rem;
}
.hero-label {
    font-size: 0.72rem; color: #8a93a6; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;
}
.hero-value { font-size: 1.65rem; font-weight: 800; color: #f2f4f8; }
.hero-value.accent { color: #3ac982; }
.hero-caption { font-size: 0.82rem; color: #8a93a6; line-height: 1.5; }
.game-card {
    background: linear-gradient(135deg, #2a3552 0%, #232d46 100%);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.09);
    padding: 1.25rem 1.5rem 1rem 1.5rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    position: relative;
    overflow: hidden;
}
.game-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(0,0,0,0.38);
}
.game-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
    background: linear-gradient(90deg, var(--away-color) 0%, var(--away-color) 49%, var(--home-color) 51%, var(--home-color) 100%);
}
.game-meta {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.76rem; color: #8a93a6; text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.div-badge {
    background: rgba(255,255,255,0.11); padding: 2px 10px; border-radius: 999px; font-size: 0.68rem;
}
.matchup-title {
    font-size: 1.6rem; font-weight: 800; margin-bottom: 1.1rem; letter-spacing: -0.01em;
    display: flex; align-items: center;
}
.team-logo {
    width: 26px; height: 26px; object-fit: contain; vertical-align: middle;
    margin: 0 0.3rem 0 0; border-radius: 4px;
}
.team-away { color: var(--away-color); }
.team-home { color: var(--home-color); }
.at-sep { color: #5b6478; font-weight: 400; margin: 0 0.4rem; font-size: 1.1rem; }
.stats-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.3rem;
}
@media (max-width: 900px) { .stats-grid { grid-template-columns: 1fr; } }
.col-title {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; color: #6d7690; margin-bottom: 0.55rem;
}
.stat-line { font-size: 0.9rem; color: #ccd2e0; margin-bottom: 0.32rem; }
.stat-line b { color: #f2f4f8; }
.sub { color: #6d7690; }
.pred-spread { font-size: 1.25rem; font-weight: 800; color: #f2f4f8; margin-bottom: 0.55rem; }
.win-bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
.win-bar {
    flex: 1; height: 9px; border-radius: 6px; background: rgba(255,255,255,0.11);
    overflow: hidden; display: flex;
}
/* Two segments so each team's fill LENGTH matches its own percentage --
   away on the left (under the away label), home on the right, split at
   the actual probability boundary. Colored by each team's own accent so
   it's unambiguous which segment belongs to which label. */
.win-bar-seg-away { background: var(--away-color); height: 100%; }
.win-bar-seg-home { background: var(--home-color); height: 100%; }
.win-pct { font-size: 0.76rem; color: #9aa2b6; white-space: nowrap; }
.market-compare { font-size: 0.76rem; color: #6d7690; margin-top: 0.4rem; }
.edge-badge {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-weight: 700; font-size: 0.82rem;
}
.edge-small { background: rgba(58, 201, 130, 0.15); color: #3ac982; }
.edge-medium { background: rgba(255, 176, 32, 0.16); color: #ffb020; }
.edge-large { background: rgba(255, 82, 82, 0.17); color: #ff6b6b; }
.game-footer {
    margin-top: 0.9rem; padding-top: 0.7rem; border-top: 1px solid rgba(255,255,255,0.09);
    font-size: 0.76rem; color: #8892a8;
}
</style>
"""


def render_hero(metrics: dict) -> str:
    vegas_acc = f"{metrics['vegas_win_accuracy']:.1%}" if "vegas_win_accuracy" in metrics else "n/a"
    vegas_mae = f"{metrics['vegas_mae_margin']:.2f}" if "vegas_mae_margin" in metrics else "n/a"
    return f"""
    <div class="hero-box">
      <div class="hero-title">Model vs. Vegas &mdash; {config.END_SEASON} holdout season</div>
      <div class="hero-grid">
        <div class="hero-stat"><div class="hero-label">Model win accuracy</div>
          <div class="hero-value accent">{metrics['model_win_accuracy']:.1%}</div></div>
        <div class="hero-stat"><div class="hero-label">Vegas win accuracy</div>
          <div class="hero-value">{vegas_acc}</div></div>
        <div class="hero-stat"><div class="hero-label">Model MAE (pts)</div>
          <div class="hero-value accent">{metrics['model_mae_margin']:.2f}</div></div>
        <div class="hero-stat"><div class="hero-label">Vegas MAE (pts)</div>
          <div class="hero-value">{vegas_mae}</div></div>
      </div>
      <div class="hero-caption">
        Evaluated on {metrics['n_test_games']:,} held-out {config.END_SEASON} games
        (walk-forward split &mdash; the model never trained on this season). The model
        currently trails Vegas on both metrics, so treat any "edge" below as a
        diagnostic view of where the model disagrees with the market &mdash; more likely
        model error than a real inefficiency. Not betting advice.
      </div>
    </div>
    """


def edge_tier(value: float, threshold: float) -> str:
    magnitude = abs(value)
    if magnitude >= threshold * 2:
        return "large"
    if magnitude >= threshold:
        return "medium"
    return "small"


def team_logo_img(team: str, team_logos: dict) -> str:
    url = team_logos.get(team)
    return f'<img class="team-logo" src="{url}" alt="{team}">' if url else ""


def render_game_card(row, team_logos: dict) -> str:
    away_color = TEAM_COLORS.get(row["away_team"], "#5b6478")
    home_color = TEAM_COLORS.get(row["home_team"], "#5b6478")
    div_badge = '<span class="div-badge">Division game</span>' if row["div_game"] else ""

    spread_tier = edge_tier(row["spread_edge"], SPREAD_EDGE_THRESHOLD)
    spread_lean = row["home_team"] if row["spread_edge"] > 0 else row["away_team"]
    spread_badge = (f'<span class="edge-badge edge-{spread_tier}">'
                     f'{row["spread_edge"]:+.1f} pts &middot; {spread_lean}</span>')

    if pd.notna(row["ml_edge"]):
        ml_tier = edge_tier(row["ml_edge"], ML_EDGE_THRESHOLD)
        ml_lean = row["home_team"] if row["ml_edge"] > 0 else row["away_team"]
        ml_badge = (f'<span class="edge-badge edge-{ml_tier}">'
                     f'{row["ml_edge"]:+.1%} &middot; {ml_lean}</span>')
        market_home_pct = row["fair_home_ml_prob"] * 100
        market_away_pct = 100 - market_home_pct
        market_line = (f'<div class="market-compare">Market (fair): {row["away_team"]} '
                        f'{market_away_pct:.0f}% / {row["home_team"]} {market_home_pct:.0f}%</div>')
    else:
        ml_badge = '<span class="edge-badge">n/a</span>'
        market_line = '<div class="market-compare">Market (fair): n/a</div>'

    home_pct = row["pred_home_win_prob"] * 100
    away_pct = 100 - home_pct

    return f"""
    <div class="game-card" style="--away-color:{away_color}; --home-color:{home_color};">
      <div class="game-meta">
        <span>{row['gameday'].strftime('%a %b %d, %Y')}</span>
        {div_badge}
      </div>
      <div class="matchup-title">
        {team_logo_img(row['away_team'], team_logos)}<span class="team-away">{row['away_team']}</span><span class="at-sep">@</span>{team_logo_img(row['home_team'], team_logos)}<span class="team-home">{row['home_team']}</span>
      </div>
      <div class="stats-grid">
        <div class="stat-col">
          <div class="col-title">VEGAS MARKET</div>
          <div class="stat-line">Spread: <b>{format_spread(row['home_team'], row['away_team'], row['spread_line'])}</b>
            <span class="sub">({format_ml(row['home_spread_odds'])}/{format_ml(row['away_spread_odds'])})</span></div>
          <div class="stat-line">ML: <b>{row['home_team']} {format_ml(row['home_moneyline'])}</b> / {row['away_team']} {format_ml(row['away_moneyline'])}</div>
          <div class="stat-line">Total: <b>{row['total_line']:.1f}</b>
            <span class="sub">(o {format_ml(row['over_odds'])}/u {format_ml(row['under_odds'])})</span></div>
        </div>
        <div class="stat-col">
          <div class="col-title">OUR MODEL</div>
          <div class="pred-spread">{format_spread(row['home_team'], row['away_team'], row['pred_margin'])}</div>
          <div class="win-bar-wrap">
            <span class="win-pct">{row['away_team']} {away_pct:.0f}%</span>
            <div class="win-bar">
              <div class="win-bar-seg-away" style="width:{away_pct:.1f}%;"></div>
              <div class="win-bar-seg-home" style="width:{home_pct:.1f}%;"></div>
            </div>
            <span class="win-pct">{row['home_team']} {home_pct:.0f}%</span>
          </div>
          {market_line}
        </div>
        <div class="stat-col">
          <div class="col-title">EDGE (MODEL &minus; MARKET)</div>
          <div class="stat-line">Spread: {spread_badge}</div>
          <div class="stat-line">Moneyline: {ml_badge}</div>
        </div>
      </div>
      <div class="game-footer">{format_weather(row)} &middot; Rest: {row['home_team']} {row['home_rest_days']}d / {row['away_team']} {row['away_rest_days']}d</div>
    </div>
    """


@st.cache_resource
def load_model_artifacts():
    return _load_model_artifacts()


@st.cache_data(ttl=3600)
def load_holdout_metrics(feature_cols):
    model, calibrator, _ = load_model_artifacts()
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")
    _, test, _ = prepare_data(df)
    return evaluate(model, calibrator, test, feature_cols)


@st.cache_data(ttl=3600)
def load_upcoming(season: int):
    return build_upcoming_features(season=season)


@st.cache_data(ttl=86400)
def load_team_logos() -> dict:
    """team_abbr -> ESPN logo URL. Cached a full day since this almost
    never changes; display-only, never touches the model."""
    df = fetch_team_logos()
    if df.empty:
        return {}
    return dict(zip(df["team_abbr"], df["team_logo_espn"]))


def predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return _predict(df, feature_cols)


def format_weather(row) -> str:
    if row["is_outdoor"] == 0:
        return "Dome / indoor (climate controlled)"
    return f"~{row['temp']:.0f}°F, {row['wind']:.0f} mph wind — season avg, no forecast yet"


def main():
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    st.title("\U0001F3C8 NFL Game Predictions")
    st.caption(
        "Predicted point margin & win probability vs. the Vegas closing line. "
        "**Positive margin/spread = home team favored** in the raw numbers "
        "throughout; spreads are also shown in standard betting notation "
        "(favorite negative) for easy comparison."
    )

    model, calibrator, feature_cols = load_model_artifacts()

    with st.spinner("Evaluating on held-out season..."):
        metrics = load_holdout_metrics(feature_cols)
    st.markdown(render_hero(metrics), unsafe_allow_html=True)

    with st.spinner(f"Building features for {config.CURRENT_SEASON} Week {WEEK_TO_SHOW}..."):
        upcoming = load_upcoming(config.CURRENT_SEASON)

    if upcoming.empty:
        st.warning(
            f"No upcoming games found for {config.CURRENT_SEASON}. Run "
            "`python data/fetch_data.py` to refresh the schedule cache."
        )
        return

    week_games = upcoming[upcoming["week"] == WEEK_TO_SHOW]
    if week_games.empty:
        st.warning(f"No Week {WEEK_TO_SHOW} games found in the {config.CURRENT_SEASON} schedule.")
        return

    predicted = predict(week_games, feature_cols)
    predicted = add_market_edges(predicted)
    team_logos = load_team_logos()

    st.subheader(f"{config.CURRENT_SEASON} Season — Week {WEEK_TO_SHOW}")

    teams = sorted(set(predicted["home_team"]) | set(predicted["away_team"]))
    team_choice = st.multiselect("Filter by team", options=teams)

    view = predicted.copy()
    if team_choice:
        view = view[view["home_team"].isin(team_choice) | view["away_team"].isin(team_choice)]
    view = view.sort_values("gameday")

    if view.empty:
        st.info("No games match the current filters.")
        return

    for _, row in view.iterrows():
        st.markdown(render_game_card(row, team_logos), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
