# NFL Game Prediction Model

An XGBoost model that predicts NFL game margins and win probabilities against the Vegas closing line, with a Streamlit dashboard for browsing weekly predictions, tracking results, and logging personal bets.

## What's here

- **`data/`** — fetches and caches play-by-play, schedule, injury, and depth-chart data from [nflverse](https://github.com/nflverse) (free, public).
- **`models/`** — trains the margin model (`train_model.py`) and serves predictions (`predict.py`). A few experimental alternatives (`score_distribution_model.py`, `compare_simple_models.py`) live alongside it from earlier ablations.
- **`dashboard/app.py`** — the Streamlit dashboard: upcoming games with market odds vs. model predictions, completed-game grading, model-vs-market discrepancy tables, and a personal bet tracker.
- **`tracking/`** — CLI tools that snapshot predictions before kickoff and grade them after (`prediction_tracker.py`), and track your own placed bets (`bet_tracker.py`).
- **`config.py`** — the single control panel for feature flags, season ranges, and file paths.

## Running locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

First run will fetch and cache several seasons of play-by-play data (a few hundred MB, stored in `data/raw/` — gitignored, regenerated automatically). This takes a few minutes; subsequent runs use the local cache.

## Weekly workflow

```bash
# Early in the week, once that week's lines are posted:
python tracking/prediction_tracker.py snapshot --week N

# After games finish:
python tracking/prediction_tracker.py update-results --week N
python tracking/bet_tracker.py grade --week N

# Anytime:
python tracking/prediction_tracker.py report
python tracking/bet_tracker.py report
```

## Deploying the dashboard

Deployed via [Streamlit Community Cloud](https://share.streamlit.io) — point it at this repo and `dashboard/app.py`. Since `data/raw/` isn't committed (it's large and fully regenerable), the first run after a deploy or a cold start re-fetches play-by-play data from nflverse, which takes a few minutes.
