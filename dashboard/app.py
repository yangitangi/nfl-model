"""
dashboard/app.py
-----------------
Streamlit dashboard: browse upcoming games and see the model's predicted
margin/win probability next to the Vegas line. Model validation metrics
(held-out season vs. Vegas) are shown at the top for context.

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


def format_weather(row) -> str:
    if row["is_outdoor"] == 0:
        return "Dome / indoor (climate controlled)"
    return f"~{row['temp']:.0f}°F, {row['wind']:.0f} mph wind — season avg, no forecast yet"


def main():
    st.title("\U0001F3C8 NFL Game Predictions")
    st.caption(
        "Predicted point margin & win probability vs. the Vegas closing line. "
        "**Positive margin/spread = home team favored** — same sign convention "
        "throughout, home team's perspective."
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
        "(walk-forward split — the model never trained on this season). Matching "
        "Vegas is already a solid result; consistently beating it is rare even for "
        "professional modelers — treat any edge shown below with caution."
    )

    st.divider()

    with st.spinner(f"Building features for {config.CURRENT_SEASON} upcoming games..."):
        upcoming = load_upcoming(config.CURRENT_SEASON)

    if upcoming.empty:
        st.warning(
            f"No upcoming games found for {config.CURRENT_SEASON}. Run "
            "`python data/fetch_data.py` to refresh the schedule cache."
        )
        return

    predicted = predict(upcoming, feature_cols)

    st.subheader(f"{config.CURRENT_SEASON} Season — Upcoming Matchups")

    weeks = sorted(predicted["week"].unique())
    teams = sorted(set(predicted["home_team"]) | set(predicted["away_team"]))
    n_no_line = predicted["spread_line"].isna().sum()
    if n_no_line:
        st.caption(
            f"ℹ️ {n_no_line} of {len(predicted)} games don't have a Vegas "
            "line posted yet (too far out) — shown as \"TBD\"."
        )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        week_choice = st.selectbox("Week", options=["All"] + [str(w) for w in weeks])
    with col_b:
        team_choice = st.multiselect("Filter by team", options=teams)

    view = predicted.copy()
    if week_choice != "All":
        view = view[view["week"] == int(week_choice)]
    if team_choice:
        view = view[view["home_team"].isin(team_choice) | view["away_team"].isin(team_choice)]
    view = view.sort_values(["week", "gameday"])

    if view.empty:
        st.info("No games match the current filters.")
        return

    for _, row in view.iterrows():
        with st.container(border=True):
            cols = st.columns([2.5, 1.5, 1.7, 1.5, 2.3])

            with cols[0]:
                st.markdown(f"**Week {row['week']} · {row['gameday'].strftime('%a %b %d, %Y')}**")
                st.markdown(f"### {row['away_team']} @ {row['home_team']}")
                if row["div_game"]:
                    st.caption("Division game")

            with cols[1]:
                spread_txt = f"{row['spread_line']:+.1f}" if pd.notna(row["spread_line"]) else "TBD"
                total_txt = f"{row['total_line']:.1f}" if pd.notna(row["total_line"]) else "TBD"
                st.metric("Vegas Spread (home)", spread_txt)
                st.caption(f"Total: {total_txt}")

            with cols[2]:
                edge = (row["pred_margin"] - row["spread_line"]
                        if pd.notna(row["spread_line"]) else None)
                st.metric(
                    "Model Margin (home)", f"{row['pred_margin']:+.1f}",
                    delta=f"{edge:+.1f} vs Vegas" if edge is not None else None,
                )

            with cols[3]:
                st.metric(f"{row['home_team']} Win Prob", f"{row['pred_home_win_prob']:.0%}")

            with cols[4]:
                st.caption("Conditions")
                st.write(format_weather(row))
                st.caption(
                    f"Rest: {row['home_team']} {row['home_rest_days']}d / "
                    f"{row['away_team']} {row['away_rest_days']}d"
                )


if __name__ == "__main__":
    main()
