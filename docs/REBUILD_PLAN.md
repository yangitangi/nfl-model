# 2027 Model Rebuild: plan, protocol, and work breakdown

## Why
The 2026 production model (XGBoost on the residual vs. the market spread) is about a coin flip against the
spread (46/93 in 2026) and loses to the closing line on MAE over 2014-2025 (see docs/BASELINES.md). Known
structural flaws:
1. `injury_burden` counts injuries (not who is hurt) and has a zero-reading artifact; high counts can push the
   prediction the wrong way (CAR@PHI, Week 6).
2. Missing QB form (rookie/backup with little data) is treated by the trees as a large negative
   (PIT@TB, Week 6: -3.9 pts from NaNs).
3. The model has no uncertainty estimate, so small edges look as confident as large ones.
4. An experiment (branch `experiment/alt-models-walkforward`) showed smooth models (ridge / Bayesian ridge / MLP)
   match the market on MAE and are 5-14x less sensitive to the zero-injury artifact, but add little beyond the
   market. So the target is NOT "a better algorithm": it is information the line does not already price, handled
   with proper shrinkage.

## Targets and success criteria
- Primary target: home margin, modeled as a residual vs. the closing spread (market = prior).
- Secondary: total points, home win probability (calibrated).
- A candidate "wins" only if, on the DEV window, it (a) has lower MAE than the market line by a margin larger
  than ~2 standard errors (paired), OR (b) shows ATS > 52.4% (break-even at -110) in the |edge| >= 2 buckets with
  stable results across at least 6 of 8 folds, AND (c) is not more sensitive to missing/zero inputs than ridge.
  Expect honest "no edge found, but robust and calibrated" to be a legitimate outcome.

## Protocol (rebuild/protocol.py is the source of truth)
- Walk-forward: for test season T, train on seasons 2010..T-1. DEV window: T = 2016..2023.
- FINAL HOLDOUT 2024-2025: untouched until a model is frozen; score once and report separately.
- All features "as-of kickoff" (computed from games/reports available before the game). Any pipeline step that
  uses full-season or future information is a leak. (Lesson: the first backup-QB test used the season-leading QB
  and falsely showed a 44% cover rate; the prospective version showed none.)
- Metrics via `rebuild.protocol.evaluate`: MAE, win acc, ATS overall and by |model-market| edge buckets with n,
  Brier/logloss of win prob, total MAE and O/U accuracy. Paired MAE difference vs. a baseline via `paired_mae_diff`.
- Always report the market line and the current XGBoost as baselines on the same rows.

## Data (nflverse; data/raw is gitignored and regenerable with `python rebuild/bootstrap_data.py`)
Play-by-play 2010-2025, schedules with closing lines (spread/total/moneyline, weather, rest), weekly injury
reports with player identity (data/raw/injuries_*.parquet), depth charts (2018+).
Tracked and available offline: data/processed/game_level_features.parquet (current feature table, 4,175 games).

## Candidate designs (build in this order)
A. Bayesian / state-space team strength with the market as prior: team offense/defense ratings that update
   weekly; shrinkage toward the market; produces a posterior for the margin (so edges carry uncertainty).
B. Regularized GLM (ridge / elastic net) on a cleaner feature set: market + EPA/success/explosiveness rolling
   features + QB features with hierarchical shrinkage (rookie/backup priors instead of NaN) + player-level injury
   value (position-weighted by who is out, not how many) + rest/travel/weather.
C. Gradient boosting challenger WITH monotonic constraints and no raw injury counts; useful only as an ensemble
   member or a check on A/B.
D. Ensemble / stacking of A-C, scored with the same protocol.
Do NOT start with deep learning; the data (about 4,200 games) does not support it.

## Features to build (leak-free)
1. QB: career+recent EPA/dropback, CPOE, sack rate with empirical-Bayes shrinkage to a position prior; rookie and
   backup priors from draft capital / prior seasons; explicit "starter confirmed" flag from depth chart as-of date.
2. Injuries by player: for each Out/Doubtful/Questionable player, weight = position value x player usage (snap
   share / target share from prior weeks). Remove the zero-reading and count features.
3. Team strength priors carried across seasons (regress to mean in the offseason), schedule-adjusted.
4. Market features: spread, total, moneyline-implied probability (and line movement if a reliable open line exists).
5. Context: rest days, travel/time zone, dome/weather, division game.

## Work breakdown (each item = one cloud session or one local task; see rebuild/CLOUD_SESSION_1_PROMPT.md)
0. DONE locally: this plan, protocol, env check, bootstrap script, baselines.
1. Cloud session 1 (small, the gate): environment + data bootstrap + reproduce baselines with the protocol on
   the cloud machine. Report timings and any blocked steps. STOP and report.
2. Cloud session 2: leak-free as-of feature store (team-game and player-level), with unit tests that fail if any
   feature changes when future rows are deleted.
3. Cloud sessions 3-5 (parallel): candidates A, B, C on the DEV window.
4. Cloud session 6: ensemble + totals + calibration + flat/Kelly betting simulation; final holdout scored once.
5. Shadow mode: backfill 2026 weeks 1-N and log weekly predictions next to the production model for live comparison.

## Budget rules for cloud sessions
- Each session has a time cap (stated in its prompt) and a "stop and report, do not retry in a loop" rule.
- No hyperparameter grids beyond what the prompt names; no tuning on the holdout.
- Work only on branches named `rebuild/<topic>`; open a pull request; never touch master, `dashboard/`,
  `tracking/`, or `outputs/` (production state).
- Extra usage is OFF on the account, so credit exhaustion stops work; it cannot create a charge.
