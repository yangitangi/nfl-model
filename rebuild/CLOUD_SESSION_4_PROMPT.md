# Cloud session 4 prompt (copy everything below the line into a new cloud session on this repo, branch: rebuild/session3-candidates)

---

You are working in the repo yangitangi/nfl-model, starting from branch `rebuild/session3-candidates` (it carries sessions 2
and 3). Read `CLAUDE.md`, `docs/REBUILD_PLAN.md`, `docs/BASELINES.md` and `rebuild/session1_report.md` through
`rebuild/session3_report.md` first. This is cloud SESSION 4: FREEZE the chosen model, score the untouched 2024-2025
holdout exactly once, test one pre-registered lead on fresh data, and backfill 2026 weeks as a shadow run. Work at most
about 90 minutes. If blocked, STOP and report; do not retry failing commands in a loop.

Setup in a fresh container: `pip install -r requirements.txt statsmodels lightgbm`, `python rebuild/bootstrap_data.py`,
`python rebuild/build_asof_store.py`, then `pytest rebuild/tests -q` must pass.

## Step 0: pre-registration (write this BEFORE loading any 2024-2025 data; commit it as `rebuild/PREREGISTRATION.md`)
State, in plain text with the exact parameters:
1. **Frozen model F1** = session 3's B2 ridge + shrink factor, with the alpha and k chosen on 2010-2015 (alpha 1e4, k 0.25
   per the session 3 report; copy the exact values and feature list from `rebuild/session3_results.json`). Training for each
   holdout season T: all seasons before T (2010..T-1), same preprocessing. No retuning.
2. **Lead L1** = A-lite's "fade the largest edges": for each game compute the A-lite z-score exactly as in session 3; take
   games in the top |z| quintile (quintile cut points fixed from the 2016-2023 DEV z-distribution, not recomputed on the
   new data); the bet is AGAINST A-lite's side. State the prediction: ATS cover rate above 52.4% for the fade.
3. **Decision rules**: F1 "passes" only if paired MAE vs the closing line is <= 0 or ATS >= 52.4% in the |edge| >= 2
   buckets with n >= 100; L1 "passes" only if the fade ATS is >= 52.4% with n >= 150 and its 95% interval excludes 50%.
   Everything else is reported as "no edge".

## Step 1: holdout, once
Score F1, the market, and the production XGBoost row (BASELINES.md) on 2024-2025 with `rebuild.protocol.evaluate` and
`paired_mae_diff` (SE included): n, MAE, win acc, ATS overall and by edge bucket, win-probability Brier/log-loss with the
calibration fitted on training seasons only. Score L1 on the same games. Run each of these exactly ONCE and report
whatever comes out; do not re-run with changes. Write results to `rebuild/session4_holdout.json` before looking at them
in any other way.

## Step 2: 2026 shadow backfill (weeks 1-9, as far as nflverse data exist)
Extend the as-of store to the 2026 season using the same code (download 2026 play-by-play, schedule, injuries; if the 2026
files are unavailable, stop this step and say so). For each completed 2026 week, predict with the fixed F1 fit trained on
2010-2025 (no refit during 2026, to keep the shadow simple). Output
`rebuild/shadow_2026.csv` with game_id, week, market spread, F1 spread, F1 edge, A-lite z, result, and covers, and compare
to the production log: read `outputs/prediction_log.parquet` and `outputs/weekly_results.csv` from branch `master`
(`git show origin/master:outputs/prediction_log.parquet > /tmp/prod_log.parquet`; read-only, never modify or commit
anything under `outputs/`). Report 2026 MAE vs the market, F1 ATS, and production model ATS on the same games, with n and
a note that these weeks are too few for conclusions.

## Step 3: report
`rebuild/session4_report.md`: pre-registration copy, holdout tables, L1 result, shadow-run table, and a plain verdict:
(a) is F1 ready to replace the production XGBoost as a robust, calibrated near-market baseline (yes/no and the numbers),
(b) does any evidence of a real edge exist (yes/no), (c) recommended next work, ranked, with honest expected value.

## Deliverables
Branch `rebuild/session4-holdout`, pushed, pull request against `rebuild-2027` titled "Rebuild session 4: frozen model,
holdout once, shadow 2026". Do not merge it.

Hard constraints: do not modify `dashboard/`, `tracking/`, `outputs/`, `config.py`, `models/train_model.py`, or earlier
session code (only add new files); never push to `master`; do not touch the holdout before the pre-registration commit;
no tuning on the holdout or on 2026 results. Final reply: verdict and key numbers in under 250 words.
