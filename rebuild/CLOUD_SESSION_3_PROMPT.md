# Cloud session 3 prompt (copy everything below the line into a new cloud session on this repo, branch: rebuild/session2-features)

---

You are working in the repo yangitangi/nfl-model, starting from branch `rebuild/session2-features` (it holds the leak-free
as-of feature store; if the branch is missing, use `rebuild-2027` and `git merge origin/rebuild/session2-features`).
Read `CLAUDE.md`, `docs/REBUILD_PLAN.md`, `docs/BASELINES.md`, `rebuild/session1_report.md` and
`rebuild/session2_report.md` first. This is cloud SESSION 3 of a 2027 NFL model rebuild: run candidate models B, C and
A-lite on the DEV window only (walk-forward, test seasons 2016-2023) and report honestly. Work at most about 90 minutes.
If blocked, STOP and report; do not retry failing commands in a loop.

Setup in a fresh container: `pip install -r requirements.txt statsmodels lightgbm`, `python rebuild/bootstrap_data.py`,
`python rebuild/build_asof_store.py` (about 3 minutes total). Run `pytest rebuild/tests -q` first and confirm it passes.

## Context you must respect (from sessions 1-2)
- Session 2 result: ridge on the residual (margin minus closing spread) trails the market by +0.060 MAE (SE 0.025) and
  chose the maximum penalty in 6 of 8 folds; 2021 was the outlier (+0.37). Both ridge models beat the production XGBoost
  by about 0.19 MAE on shared games. Seed noise for XGBoost is about +/-0.05 MAE, so differences under 0.05 are noise.
- The market is the prior. A candidate is only interesting if it beats the market on paired MAE, or shows ATS above
  52.4% in |edge| >= 2 buckets in at least 6 of 8 folds. "No edge, but robust" is a legitimate result.
- Every hyperparameter is chosen INSIDE the training window (inner folds by season). The 2024-2025 holdout is never used.

## Candidates (one module each under `rebuild/models/`, one shared runner `rebuild/run_candidates.py`)

**B2: regularized linear residual, fixed shrinkage.** Fix the failure mode of session 2: instead of CV-picking alpha per
fold, (i) use elastic net / ridge with a SINGLE alpha chosen once on seasons 2010-2015 and reused for all folds, and
(ii) a variant that shrinks the prediction by a factor k in {0.25, 0.5, 1.0} (k chosen once on 2010-2015, reused). Add a
robust loss variant (Huber) to test whether outlier games (2020-2021) drive the instability. Also add the
interaction set: spread x (rest diff), spread x (QB shrunk rating diff), total x weather.

**C: gradient boosting challenger.** LightGBM on the residual with monotonic constraints (sign fixed from football logic:
better team EPA/QB -> higher home margin, more injury value out -> lower), max depth 3, 300 trees max with early stopping
on an inner season fold, strong min_child_samples (>= 100), no raw injury counts. Average 5 seeds and report the seed SD.

**A-lite: state-space team ratings with the market as prior.** Kalman-style weekly update: each team has latent
offense/defense strength, prior = last season's posterior regressed toward 0 with a shrink weight fitted once on
2010-2015; the observation per game is the margin minus the market-implied margin (so the market is the baseline and the
model learns only what the line misses). Output a predicted residual AND its posterior variance. Use closed-form updates
(numpy), no MCMC, no pymc. Report whether posterior variance is informative: ATS by predicted-edge / posterior-sd
(z-score) bucket.

## Reporting (all in `rebuild/session3_report.md`, with `rebuild/session3_results.json`)
For every candidate and for the market and the production XGBoost row from BASELINES.md: n, MAE, win acc, ATS overall,
ATS by edge bucket with n, paired MAE vs market with SE, per-season MAE-minus-market table, and a count of seasons
better than the market. Add a Brier / log-loss row for win probability using a simple logistic calibration of the
predicted margin (fit inside the training window). Add the same table for totals using one simple model (ridge on
total_line residual); one line on whether it beats the total line.

Finish with a verdict: which candidates (if any) are worth carrying into the ensemble session, which are dropped and why.
Do not run an ensemble in this session.

## Deliverables
Branch `rebuild/session3-candidates`, pushed, pull request against `rebuild-2027` titled "Rebuild session 3: candidates
B2, C, A-lite on DEV". Do not merge it.

Hard constraints: do not modify `dashboard/`, `tracking/`, `outputs/`, `config.py`, `models/train_model.py`, or the session 2
feature code (only add new files); never push to `master`; no hyperparameter grids beyond those named above; do not touch
2024-2025. Final reply: verdict and key numbers in under 250 words.
