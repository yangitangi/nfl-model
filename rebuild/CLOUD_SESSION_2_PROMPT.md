# Cloud session 2 prompt (copy everything below the line into a new cloud session on this repo, branch: rebuild-2027)

---

You are working in the repo yangitangi/nfl-model, starting from branch `rebuild-2027`. Read `CLAUDE.md`,
`docs/REBUILD_PLAN.md`, `docs/BASELINES.md` and `rebuild/session1_report.md` (on branch `rebuild/session1-env`; if it
is not on your branch, `git fetch origin rebuild/session1-env` and read it with `git show`). This is cloud SESSION 2
of a 2027 NFL model rebuild: build a LEAK-FREE, AS-OF-KICKOFF feature store, prove it is leak-free with tests, and run
one cheap smoke-test model on the DEV window. Work for at most about 90 minutes of wall time. If a step is blocked,
STOP and report; do not retry failing commands in a loop.

Setup each fresh container: `pip install -r requirements.txt statsmodels lightgbm` then
`python rebuild/bootstrap_data.py` (about 2.5 minutes total).

## What to build (new code under `rebuild/features/`, one module per group, plus `rebuild/build_asof_store.py`)

Output: `data/processed/asof_features.parquet`, one row per regular-season game 2010-2025 (home/away, game_id, season,
week, kickoff date, closing spread/total/moneyline, result margin and total), plus columns for the groups below. Every
feature must be computable using ONLY information available before that game's kickoff (earlier games, earlier
injury reports, prior seasons).

1. **Team strength (offense/defense)**: rolling EPA/play, success rate and explosive-play rate, for offense and defense,
   split pass/rush, with exponential decay over prior games (try half-life 6 games; one fixed value, no grid) and
   carry-over from the prior season regressed toward the league mean in week 1-4 (state the shrink weight you use).
   Adjust for opponent strength using only opponents' as-of ratings.
2. **QB**: for the QB who STARTS the game (identify from the game's own first offensive snaps only for historical
   rows, which is what the market knew; document that this is the as-of-announcement proxy), career+recent EPA per
   dropback, CPOE, sack rate with empirical-Bayes shrinkage to a position prior. Rookies and players with few
   dropbacks get the PRIOR (with a `qb_dropbacks_prior` count and an `is_rookie_or_unknown` flag), never NaN.
3. **Injuries by player**: from `data/raw/injuries_*.parquet`, for each Out/Doubtful (and Questionable at 0.25) player on
   the final pre-game report, weight = position value x player usage from PRIOR games only (snap share if available,
   otherwise targets/carries/dropbacks share from pbp). Output team-level `inj_off_value`, `inj_def_value`,
   `inj_qb_out`. Remove count-based burden and any zero-reading artifact: a missing report is an explicit
   `inj_report_missing` flag with value imputed from the team's prior-week level, not 0.
4. **Market**: spread, total, moneyline-implied win probability (vig removed).
5. **Context**: rest days, short week, bye-week-before, travel/time zone difference, dome/weather, division game.

## Leak-free tests (REQUIRED, `rebuild/tests/test_asof_no_leak.py`, run with pytest)

- **Truncation test**: for 5 randomly chosen (seed fixed) historical weeks across different seasons, rebuild features
  from raw data with every row dated on or after that week's first kickoff deleted, and assert that the features of
  that week's games equal those in the full store (within 1e-9). This must fail if you deliberately introduce a leak
  (add one sanity test that injects a leak, e.g. a feature using `.shift(-1)`, and asserts the test catches it).
- **Monotonic-information test**: appending later seasons never changes earlier rows.
- No feature may use the game's own result, the season's final stats, or the closing line of a later game.
- Row-count and NaN report per feature (document any structural NaN; QB features must have none).

## Smoke test (DEV window only, 2016-2023 walk-forward, train on 2010..T-1)

Score with `rebuild.protocol.evaluate` and `paired_mae_diff`, each reported next to the market line AND the
BASELINES.md XGBoost row:
- (a) market only (spread -> margin, i.e. the residual is zero) as the reference,
- (b) ridge on the residual (margin - market spread) using the new features, standardised, alpha chosen by
  walk-forward on the TRAINING seasons only (3-fold by season inside the training window),
- (c) the same ridge with the injury and QB groups dropped, to show what those groups add.

Report MAE, win acc, ATS overall and by edge bucket with n, plus the paired MAE difference vs the market with a
standard error. Honest "no edge" is an acceptable result. Do NOT touch 2024-2025 and do not tune on them.

## Deliverables

- `rebuild/features/*.py`, `rebuild/build_asof_store.py`, `rebuild/tests/test_asof_no_leak.py`,
  `data/processed/asof_features.parquet` (commit it if under 20 MB), `rebuild/session2_report.md` (feature list with
  definitions, test results, smoke-test tables, wall times, and a short verdict on whether the store is ready for
  candidates A/B/C).
- Branch `rebuild/session2-features`, pushed, pull request against `rebuild-2027` titled "Rebuild session 2: as-of
  feature store and leak tests". Do not merge it.

Hard constraints: do not modify `dashboard/`, `tracking/`, `outputs/`, `config.py`, `models/train_model.py`; never push
to `master`; no hyperparameter grids beyond the ones named above; keep compute modest. Final reply: verdict and the
key numbers in under 250 words.
