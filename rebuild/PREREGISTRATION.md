# Pre-registration: session 4 (frozen model, holdout once, one lead)

Written and committed **before any 2024-2025 row was loaded** for modeling or scoring. Machine-readable copy of
every parameter: `rebuild/session4_frozen.json`, produced by `rebuild/session4_freeze.py`.

That script reads the store with a parquet row filter `season <= 2023`, so holdout rows never entered memory.
The only other 2024-2025 contact so far is the session-2 leak tests, which build features for all seasons but
fit and score nothing.

Date: 2026-11-04. Branch: `rebuild/session4-holdout`.

## 1. Frozen model F1 = session 3 "B2 ridge + shrink k"

- Target: `actual_margin - spread_line`. Prediction: `pred_margin = spread_line + k * ridge_prediction`.
- Pipeline (fitted on the training seasons only): `SimpleImputer(strategy="mean")` → `StandardScaler()` →
  `Ridge(alpha)`.
- **alpha = 10000.0, k = 0.25.** Both were chosen once on 2010-2015 in session 3 (validation on 2013/14/15) and are
  copied from `rebuild/session3_results.json` (`meta.B2_ridge_shrink_k`). They are not retuned.
- Features (66, exactly session 3's `base_features`): every numeric column of `data/processed/asof_features.parquet`
  except IDs and labels. The list is in `session4_frozen.json` under `F1.features`:
  - team ratings `{home,away}_{off,def}_{pass,rush}_{epa,sr,expl}`;
  - QB `{home,away}_qb_{epa_db_career, epa_db_recent, sack_rate, cpoe, dropbacks_prior, rookie_or_unknown, changed, prior_epa}`;
  - injuries `{home,away}_inj_{off_value, def_value, qb_out, report_missing}`;
  - market `spread_line, total_line, ml_home_prob, ml_vig`;
  - context `home_rest, away_rest, rest_diff, {home,away}_short_week, {home,away}_bye_before, tz_diff, neutral_site,
    is_dome, temp, wind, weather_missing, div_game`.
- Training for holdout season T: all seasons 2010..T−1 with a closing spread and a result. 2024 is therefore
  predicted from 2010-2023, and 2025 from 2010-2024.
- Win probability: logistic regression of home win on `pred_margin`, fitted on the same training seasons' in-sample
  F1 predictions. The market row is calibrated the same way, on `spread_line`.
- 2026 shadow: one fit on 2010-2025, no refit during 2026.

## 2. Lead L1 = fade A-lite's largest edges

- A-lite as in session 3 (`rebuild/models/a_kalman.py`), with **frozen** parameters:
  - rho = 0.4644431114227297
  - q = 0.08157799978395731
  - p0 = 1.1689502686226917e-84
  - r = 88.48867549833595

  The filter runs continuously from 2010 with these parameters. Predictions use the start-of-week state, so
  same-week results are never used.
- z = (A-lite pred_margin − spread_line) / A-lite posterior SD, exactly as in session 3.
- **Top-quintile cut is fixed from DEV 2016-2023:** |z| ≥ **0.3666252761195482**, the 80th percentile of |z| over
  the 2,095 DEV games (419 DEV games at or above it). It is not recomputed on new data.
- **Bet:** AGAINST A-lite's side. If z > 0 (A-lite likes home vs the line), take the away team against the spread;
  if z < 0, take the home team. Pushes are excluded.
- **Prediction:** the fade covers more than 52.4%.

## 3. Decision rules (fixed now)

- **F1 passes** only if, on the 2024-2025 holdout, either:
  - paired MAE vs the closing line ≤ 0, or
  - ATS ≥ 52.4% in the |F1 − market| ≥ 2 buckets with n ≥ 100.
- **L1 passes** only if the fade's ATS is ≥ 52.4% with n ≥ 150 **and** its 95% interval excludes 50%. The interval
  is normal-approximation: p ± 1.96·√(p(1−p)/n).
- Everything else is reported as **"no edge"**.
- Each holdout computation is run **once**. Results are written to `rebuild/session4_holdout.json` and reported as
  they come out, with no re-runs or changes.

## 4. Power notes, stated before seeing any data

- **L1 is very unlikely to reach n ≥ 150 on the holdout.** 2024-2025 have about 544 games. If |z| keeps its DEV
  distribution, about 20% (around 109 games) fall in the top quintile. With n ≈ 110, the SE of a cover rate is
  about 4.8 points, so even a true 53.5% fade (the DEV estimate) could not exclude 50%. L1 will therefore most
  likely be reported as "no edge / insufficient n" regardless of its point estimate.
- Secondary analysis, labelled as such and not a pass/fail test: L1 pooled over holdout + 2026 shadow weeks with
  the same rule. 2026 shadow numbers are descriptive only.
- **F1's ATS route is effectively closed by construction.** With k = 0.25, F1 almost never differs from the line by
  2+ points (DEV: 0 of 2,095 games). So F1 can only pass through paired MAE ≤ 0. In DEV that difference was +0.0014
  (SE 0.0037). The holdout SE will be about 0.007, so a pass would be a coin flip around "identical to the
  market", not evidence of edge.
- What F1 is expected to show is robustness: MAE within about ±0.02 of the market, calibrated probabilities, and a
  large improvement over the production XGBoost.
