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

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.build_features import build_upcoming_features
from models.train_model import prepare_data, evaluate

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


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(config.MODELS_DIR / "margin_model.joblib")
    calibrator = joblib.load(config.MODELS_DIR / "win_calibrator.joblib")
    feature_cols = joblib.load(config.MODELS_DIR / "feature_columns.joblib")
    return model, calibrator, feature_cols


@st.cache_data(ttl=3600)
def load_holdout_metrics(feature_cols):
    model, calibrator, _ = load_model_artifacts()
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "game_level_features.parquet")
    _, test, _ = prepare_data(df)
    return evaluate(model, calibrator, test, feature_cols)


@st.cache_data(ttl=3600)
def load_upcoming(season: int):
    return build_upcoming_features(season=season)


def predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    model, calibrator, _ = load_model_artifacts()
    pred_margin = model.predict(df[feature_cols])
    pred_win_prob = calibrator.predict_proba(pred_margin.reshape(-1, 1))[:, 1]
    out = df.copy()
    out["pred_margin"] = pred_margin
    out["pred_home_win_prob"] = pred_win_prob
    return out


def american_to_implied_prob(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def fair_home_win_prob(home_ml: float, away_ml: float) -> float:
    """Vig-free home win probability: normalizes both sides' implied
    probabilities (which sum to >100% due to the sportsbook's cut) to sum
    to 1, so the model's probability is compared against a fair number
    rather than one inflated by the vig."""
    home_implied = american_to_implied_prob(home_ml)
    away_implied = american_to_implied_prob(away_ml)
    return home_implied / (home_implied + away_implied)


def format_spread(home_team: str, away_team: str, home_margin: float) -> str:
    """home_margin: positive = home favored by that many points (this
    project's sign convention throughout). Converted to standard betting
    notation, favorite shown negative, e.g. 'SEA -3.5'."""
    if pd.isna(home_margin):
        return "TBD"
    if home_margin > 0:
        return f"{home_team} -{home_margin:.1f}"
    if home_margin < 0:
        return f"{away_team} -{-home_margin:.1f}"
    return "Pick'em"


def format_ml(odds: float) -> str:
    return "TBD" if pd.isna(odds) else f"{odds:+.0f}"


def format_weather(row) -> str:
    if row["is_outdoor"] == 0:
        return "Dome / indoor (climate controlled)"
    return f"~{row['temp']:.0f}°F, {row['wind']:.0f} mph wind — season avg, no forecast yet"


def main():
    st.title("\U0001F3C8 NFL Game Predictions")
    st.caption(
        "Predicted point margin & win probability vs. the Vegas closing line. "
        "**Positive margin/spread = home team favored** in the raw numbers "
        "throughout; spreads are also shown in standard betting notation "
        "(favorite negative) for easy comparison."
    )

    model, calibrator, feature_cols = load_model_artifacts()

    st.subheader(f"Model vs. Vegas — {config.END_SEASON} holdout season")
    with st.spinner("Evaluating on held-out season..."):
        metrics = load_holdout_metrics(feature_cols)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model win accuracy", f"{metrics['model_win_accuracy']:.1%}")
    c2.metric("Vegas win accuracy",
              f"{metrics['vegas_win_accuracy']:.1%}" if "vegas_win_accuracy" in metrics else "n/a")
    c3.metric("Model MAE (pts)", f"{metrics['model_mae_margin']:.2f}")
    c4.metric("Vegas MAE (pts)",
              f"{metrics['vegas_mae_margin']:.2f}" if "vegas_mae_margin" in metrics else "n/a")
    st.caption(
        f"Evaluated on {metrics['n_test_games']:,} held-out {config.END_SEASON} games "
        "(walk-forward split — the model never trained on this season). The model "
        "currently trails Vegas on both metrics, so treat any \"edge\" below as a "
        "diagnostic view of where the model disagrees with the market — more likely "
        "model error than a real inefficiency. Not betting advice."
    )

    st.divider()

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
    predicted["fair_home_ml_prob"] = predicted.apply(
        lambda r: fair_home_win_prob(r["home_moneyline"], r["away_moneyline"])
        if pd.notna(r["home_moneyline"]) and pd.notna(r["away_moneyline"]) else pd.NA,
        axis=1,
    )
    predicted["spread_edge"] = predicted["pred_margin"] - predicted["spread_line"]
    predicted["ml_edge"] = predicted["pred_home_win_prob"] - predicted["fair_home_ml_prob"].astype(float)

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
        with st.container(border=True):
            st.markdown(f"**{row['gameday'].strftime('%a %b %d, %Y')}**"
                        + ("  ·  Division game" if row["div_game"] else ""))
            st.markdown(f"### {row['away_team']} @ {row['home_team']}")

            mcol, pcol, ecol = st.columns(3)

            with mcol:
                st.markdown("**Vegas Market**")
                st.write(f"Spread: {format_spread(row['home_team'], row['away_team'], row['spread_line'])} "
                         f"({format_ml(row['home_spread_odds'])}/{format_ml(row['away_spread_odds'])})")
                st.write(f"Moneyline: {row['home_team']} {format_ml(row['home_moneyline'])} "
                         f"/ {row['away_team']} {format_ml(row['away_moneyline'])}")
                st.write(f"Total: {row['total_line']:.1f} "
                         f"(o {format_ml(row['over_odds'])}/u {format_ml(row['under_odds'])})")

            with pcol:
                st.markdown("**Our Model**")
                st.write(f"Predicted spread: {format_spread(row['home_team'], row['away_team'], row['pred_margin'])}")
                st.write(f"Win prob: {row['home_team']} {row['pred_home_win_prob']:.0%} "
                         f"/ {row['away_team']} {1 - row['pred_home_win_prob']:.0%}")
                st.caption(f"Raw predicted margin (home): {row['pred_margin']:+.1f} pts")

            with ecol:
                st.markdown("**Edge (model − market)**")
                spread_lean = row["home_team"] if row["spread_edge"] > 0 else row["away_team"]
                spread_flag = " \U0001F53A" if abs(row["spread_edge"]) >= SPREAD_EDGE_THRESHOLD else ""
                st.write(f"Spread: {row['spread_edge']:+.1f} pts toward {spread_lean}{spread_flag}")

                if pd.notna(row["ml_edge"]):
                    ml_lean = row["home_team"] if row["ml_edge"] > 0 else row["away_team"]
                    ml_flag = " \U0001F53A" if abs(row["ml_edge"]) >= ML_EDGE_THRESHOLD else ""
                    st.write(f"Moneyline: {row['ml_edge']:+.1%} toward {ml_lean}{ml_flag}")
                    st.caption(f"Fair market win prob (vig removed): "
                               f"{row['home_team']} {row['fair_home_ml_prob']:.0%}")
                else:
                    st.write("Moneyline: n/a")

            st.caption(
                f"{format_weather(row)}  ·  Rest: {row['home_team']} {row['home_rest_days']}d / "
                f"{row['away_team']} {row['away_rest_days']}d"
            )


if __name__ == "__main__":
    main()
