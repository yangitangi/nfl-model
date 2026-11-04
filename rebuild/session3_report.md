# Rebuild session 3: candidates B2, C, A-lite on DEV

Branch `rebuild/session3-candidates`. It is based on `rebuild/session2-features`, because PR 2 (the feature
store) is not merged into `rebuild-2027` yet.

Setup: the container was reused (data, libraries and store present). `pytest rebuild/tests -q` gave 12 passed in
27 s before any work. Only new files were added: `rebuild/models/*`, `rebuild/run_candidates.py`,
`rebuild/session3_results.json` and this report. Session-2 feature code and production files are untouched.

Holdout safeguard: `load_dev()` drops seasons > 2023 at load time and asserts it. 2024-2025 were never loaded.

`python rebuild/run_candidates.py` runs everything in 80 s; 73 s of that is A-lite's likelihood fit.

## Verdict

**No candidate beats the closing line.** Every model with a sensible amount of shrinkage collapses onto the
market. None reaches the bar in `docs/REBUILD_PLAN.md`:
- paired MAE better than the market by about 2 SE: best is +0.0002 (SE 0.0013), i.e. worse;
- ATS > 52.4% in |edge| ≥ 2 in at least 6 of 8 folds: no candidate even produces |edge| ≥ 2 picks in most folds.

Totals are the same: the ridge on the total-line residual is −0.004 (SE 0.015) vs the line, which is noise.

| Candidate | Carry into ensemble session? | Why |
|---|---|---|
| **B2 ridge + shrink k (fixed alpha)** | **Yes, as the "robust baseline" member** | It fixes the session-2 instability: 2021 went from +0.375 to +0.002 MAE vs market, and every season is within ±0.02. It is never worse than the market by more than noise, and its calibrated win probability is marginally better (Brier −0.00018, SE 0.00010, about 1.8 SE). It is cheap and interpretable. Expect it to add about 0 to MAE. |
| B2 ridge, fixed alpha (k = 1) | Fold into the above | Same model without the extra shrink. +0.014 (SE 0.015). Best Brier (−0.00046, SE 0.00038, about 1.2 SE, not significant). |
| B2 Huber + k | Drop | Identical to the market (+0.0002, SE 0.0013); with alpha 1e4 its coefficients are about 0. It answers the question asked (below), so nothing is left to carry. |
| B2 + interactions + k | Drop | Indistinguishable from B2 ridge + k (+0.0013 vs +0.0014). The interactions add nothing. |
| **C LightGBM monotone** | **Keep only as an ensemble check, low priority** | +0.008 (SE 0.006). Early stopping picks 1-21 trees, so it is almost a constant. The seed SD of MAE is 0.003, so it is stable, but it has no signal. It is a useful sanity check that a non-linear learner finds nothing either. |
| A-lite Kalman | **Drop as a predictor; keep the code** | Worse than the market in **8 of 8** seasons: +0.033 (SE 0.012, about 2.8 SE). Its posterior SD is not informative (corr with \|error\| = −0.004). It does supply a natural uncertainty framework for session 4 if a real signal appears. |

What this means for the plan:
- The public as-of information in this store (team EPA, QB, injuries, rest, weather) is already priced into
  closing lines. The ensemble session should not be expected to find an MAE edge from these inputs.
- The realistic remaining value is:
  1. calibrated win probabilities and uncertainty that are at least as good as the market's;
  2. a model that is robust to missing inputs, which B2 is;
  3. testing genuinely new information that the line may not fully price, e.g. **opening-to-closing line movement**,
     or **pre-registered** fade/overreaction hypotheses (see the A-lite note below).

## Candidates (exact specifications)

Common to all: residual target = margin − closing spread; walk-forward test seasons 2016-2023; train 2010..T−1.

"Chosen once on 2010-2015" means validating on 2013, 2014 and 2015, each fitted on the seasons before it, then
freezing the choice for all 8 DEV folds.

**B2 linear** (`rebuild/models/b2_linear.py`): mean-impute → standardise → Ridge or HuberRegressor (ε 1.35).

| Variant | Alpha (grid {0.1, 1, 10, 100, 1e3, 1e4}, chosen once) | k (grid {0.25, 0.5, 1}, chosen once) | Validation MAE 2013-15 |
|---|---|---|---|
| ridge fixed alpha | 1e4 | 1 (not tuned) | 10.459 |
| ridge + k | 1e4 | 0.25 | 10.448 |
| Huber + k | 1e4 | 0.25 | 10.449 |
| ridge + interactions + k | 1e4 | 0.25 | 10.449 |

Both alpha and k land at the **most-shrinking edge** of the named grids. I did not extend the grids, per the
session rules. The direction is unambiguous: validation MAE improves monotonically with more shrinkage.

Interactions: spread × rest_diff, spread × (home − away QB shrunk career EPA), total × wind, total × temp.

**Does Huber show that outlier games (2020-21) drive the instability? No.** The session-2 instability came from
re-picking alpha per fold. 2021 chose alpha = 100 there and lost 0.375 MAE to the market. With one alpha fixed on
2010-2015, plain ridge's 2021 is +0.019, and with k it is +0.002. Huber behaves the same but shrinks further.

**C LightGBM** (`rebuild/models/c_lgbm.py`):
- Settings: L2 objective on the residual; depth 3, 7 leaves, learning rate 0.05, min_child_samples 150,
  subsample/colsample 0.8.
- Trees: at most 300, with early stopping (L1, 30 rounds) on season T−1 after fitting on < T−1, then a refit on
  < T.
- Monotone constraints on 38 of 66 features. Home offense, QB EPA/CPOE and away defense-allowed are +1; QB sack
  rate and injury value out are −1 for the own side; away-side features get the mirrored signs.
- Seeds 0-4, predictions averaged.
- Trees chosen per fold and seed ranged from 1 to 21: the inner fold stops almost immediately.
- Seed SD: MAE 0.0028; paired-vs-market 0.0028 (per-seed range +0.007 to +0.013).

**A-lite** (`rebuild/models/a_kalman.py`):
- State: offense and defense rating for each of the 32 franchises.
- Observations: home and away points minus the market-implied team scores, (total ± spread)/2.
- Weekly random-walk drift q. Season start: m ← ρm, P ← ρ²P + (1−ρ²)p0·I.
- Predictions use the start-of-week state.
- Parameters fitted once by maximum predictive likelihood on 2010-2015 (Nelder-Mead):
  - **ρ = 0.46** (the shrink weight toward 0),
  - weekly drift **q = 0.082 pts²**,
  - observation noise **r = 88.5 pts²** (SD 9.4),
  - initial prior variance **p0 → 0** (boundary).
- The optimizer stopped at its iteration cap at that boundary. The likelihood is extremely flat, with
  negative log-likelihood 11274.3 at the optimum:
  - p0 = 0.1 adds +0.1; p0 = 1 adds +3.1;
  - ρ = 0 adds +1.7; ρ = 1 adds +17.4;
  - q = 0 (no drift at all) adds +3.1.

  The data barely support any persistent "what the market misses" team strength.
- Posterior SD of the predicted margin residual: mean 1.89 pts (range 1.10-2.48). It is driven mostly by how far
  into the season the game is.

## Results: margin (DEV 2016-2023)

Market and candidates are scored on all 2,095 DEV games with a closing line. XGBoost is the BASELINES.md
production walk-forward (its own 1,924 rows; "vs XGB shared" = paired MAE on those 1,924 games, negative = the
candidate is better).

| Model | n | MAE | Mkt MAE | Win acc | ATS (n) | Paired MAE vs mkt (SE) | Seasons better | Edge ≥ 2 folds > 52.4% | vs XGB shared (SE) |
|---|---|---|---|---|---|---|---|---|---|
| Market (closing spread) | 2,095 | 9.820 | 9.820 | 66.1% | — | 0 | — | — | −0.244 (0.060) |
| XGBoost prod (BASELINES.md) | 1,924 | 10.021 | 9.778 | 66.0% | 51.0% (1,867) | +0.2436 (0.0597) | 1/8 | 4/8 | — |
| B2 ridge, fixed alpha | 2,095 | 9.834 | 9.820 | 66.2% | 51.3% (2,035) | +0.0140 (0.0148) | 3/8 | 0/8 | −0.230 (0.056) |
| B2 ridge + shrink k | 2,095 | 9.821 | 9.820 | 66.1% | 51.3% (2,035) | +0.0014 (0.0037) | 4/8 | 0/8 | −0.242 (0.058) |
| B2 Huber + shrink k | 2,095 | 9.820 | 9.820 | 66.1% | 51.5% (2,035) | +0.0002 (0.0013) | 4/8 | 0/8 | −0.243 (0.059) |
| B2 ridge + interactions + k | 2,095 | 9.821 | 9.820 | 66.1% | 51.8% (2,035) | +0.0013 (0.0038) | 4/8 | 0/8 | −0.242 (0.058) |
| C LightGBM monotone (5 seeds) | 2,095 | 9.828 | 9.820 | 66.1% | 49.3% (2,035) | +0.0083 (0.0059) | 3/8 | 0/8 | −0.236 (0.058) |
| A-lite Kalman | 2,095 | 9.853 | 9.820 | 66.1% | 49.5% (2,035) | +0.0331 (0.0118) | 0/8 | 1/8 | −0.207 (0.058) |

The overall ATS figures are all within about 1 SE of 50% (SE about 1.1 points at n = 2,035). None is near 52.4%.

### ATS by |model − market| edge (n)

| Model | 0-1 | 1-2 | 2-3 | 3-5 | 5+ |
|---|---|---|---|---|---|
| XGBoost prod | 50.1% (585) | 50.4% (482) | 54.0% (346) | 51.6% (308) | 47.9% (146) |
| B2 ridge, fixed alpha | 51.5% (1,742) | 50.7% (278) | 28.6% (14) | 0.0% (1) | — (0) |
| B2 ridge + shrink k | 51.3% (2,035) | — | — | — | — |
| B2 Huber + shrink k | 51.5% (2,035) | — | — | — | — |
| B2 ridge + interactions + k | 51.8% (2,035) | — | — | — | — |
| C LightGBM monotone | 49.2% (2,023) | 66.7% (12) | — | — | — |
| A-lite Kalman | 49.9% (1,892) | 44.6% (139) | 50.0% (4) | — | — |

The shrunk candidates almost never disagree with the line by 2+ points, so the ATS-bucket criterion cannot be met
by construction. That is the honest consequence of the shrinkage the data asked for.

XGBoost's edge ≥ 2 picks clear 52.4% in only 4 of 8 folds, short of the 6-of-8 bar (its 2-3 bucket is 54.0%).

### MAE minus market MAE by season (negative = better than the market)

| Model | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|---|
| XGBoost prod | +0.546 | +0.205 | +0.468 | +0.008 | +0.543 | −0.013 | +0.044 | +0.172 |
| B2 ridge, fixed alpha | −0.060 | −0.000 | −0.030 | +0.017 | +0.075 | +0.019 | +0.051 | +0.036 |
| B2 ridge + shrink k | −0.017 | −0.002 | −0.010 | −0.001 | +0.017 | +0.002 | +0.013 | +0.008 |
| B2 Huber + shrink k | −0.002 | +0.002 | −0.002 | −0.002 | +0.001 | −0.002 | +0.003 | +0.003 |
| B2 ridge + interactions + k | −0.018 | −0.001 | −0.009 | −0.002 | +0.017 | +0.001 | +0.014 | +0.008 |
| C LightGBM monotone | −0.006 | −0.005 | +0.016 | +0.028 | +0.017 | −0.009 | +0.023 | +0.004 |
| A-lite Kalman | +0.027 | +0.031 | +0.041 | +0.022 | +0.037 | +0.019 | +0.063 | +0.025 |

The B2 variants beat the market slightly in 2016-2019 and lose slightly from 2020 onward. This is a mild
regime-shift pattern, not an edge.

### Win probability

Calibration: logistic regression of home win on pred_margin, fitted per fold on that fold's training-window
predictions. For the linear and boosted models these are in-sample fits; for A-lite they are one-step-ahead.

| Model | Brier | Log-loss | Paired Brier vs market (SE) |
|---|---|---|---|
| Market (closing spread, same calibration) | 0.2117 | 0.6111 | — |
| B2 ridge, fixed alpha | 0.2112 | 0.6101 | −0.00046 (0.00038) |
| B2 ridge + shrink k | 0.2115 | 0.6107 | −0.00018 (0.00010) |
| B2 Huber + shrink k | 0.2116 | 0.6110 | −0.00002 (0.00001) |
| B2 ridge + interactions + k | 0.2115 | 0.6106 | −0.00018 (0.00010) |
| C LightGBM monotone | 0.2115 | 0.6108 | −0.00011 (0.00012) |
| A-lite Kalman | 0.2122 | 0.6124 | +0.00056 (0.00026) |
| XGBoost prod | n/a: no training-window predictions available | | |

B2's probabilities are a hair better than the market's, at about 1.2-1.8 SE. That is suggestive at best, and too
small to matter for betting.

### A-lite: is the posterior variance informative? No.

ATS by quintile of |z| = |predicted edge| / posterior SD, compared with quintiles of |edge| alone. The quintile
thresholds come from the DEV predictions, so this is descriptive.

| Quintile | \|z\| range | ATS (n) | \|edge\| range (pts) | ATS (n) |
|---|---|---|---|---|
| Q1 | 0.00-0.06 | 48.9% (405) | 0.00-0.12 | 49.1% (405) |
| Q2 | 0.06-0.14 | 54.3% (407) | 0.12-0.25 | 52.3% (409) |
| Q3 | 0.14-0.23 | 52.5% (406) | 0.25-0.42 | 53.7% (406) |
| Q4 | 0.23-0.37 | 45.6% (406) | 0.42-0.68 | 47.8% (406) |
| Q5 | 0.37-1.14 | 46.5% (411) | 0.68-2.61 | 44.7% (409) |

- There is no monotone relationship; ATS does not rise with confidence.
- corr(posterior SD, |error|) = −0.004.
- Posterior SD is almost constant across seasons (1.87-1.92).
- Dividing by the SD adds nothing over the raw edge.

**Exploratory note, not a finding:** the *largest* A-lite edges cover only 44.7-46.5%. In the top |z| quintile,
ATS is ≤ 50% in all 8 seasons: 50.0, 49.0, 43.8, 47.1, 46.4, 43.5, 44.7 and 48.6%, with n of 37-69 per season.

Read as a fade, that is 53.5% pooled (about 1.4 SE above 50%). The likely story is that the closing line
*over*-adjusts to a team's recent results relative to the market. However, the thresholds were picked on this
same DEV data and this is one of many cuts, so it must be **pre-registered and tested on fresh folds** in a later
session before anyone acts on it.

## Results: totals (DEV 2016-2023)

| Totals model | n | MAE | Line MAE | Paired vs line (SE) | O/U acc (n) | Seasons better |
|---|---|---|---|---|---|---|
| Ridge on total_line residual (alpha 1e4, chosen once on 2010-15) | 2,095 | 10.505 | 10.509 | −0.0036 (0.0148) | 51.8% (2,077) | 5/8 |
| A-lite implied total (bonus, same filter) | 2,095 | 10.529 | 10.509 | +0.0209 (0.0114) | 50.0% (2,077) | 1/8 |

**The totals ridge does not beat the total line.** −0.004 is 0.2 SE, and 51.8% O/U is below break-even.

## Cost

| Step | Time |
|---|---|
| pytest rebuild/tests | 27 s |
| B2, 4 variants (including the selection on 2010-15) | 2.3 s |
| C LightGBM (8 folds × 5 seeds × early stop + refit) | 3.0 s |
| A-lite (likelihood fit, 509 evaluations, + one filter pass 2010-2023) | 73 s |
| Totals + scoring | about 2 s |
