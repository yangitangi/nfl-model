# Rebuild session 4: frozen model, holdout once, shadow 2026

Branch `rebuild/session4-holdout`. It is built on `rebuild/session3-candidates`, so it carries sessions 2-3 until
PRs 2 and 3 merge.

Setup: the container was reused, and `pytest rebuild/tests -q` gave 12 passed before any work. Only new files were
added; nothing under `outputs/`, `dashboard/`, `tracking/`, `config.py`, `models/` or earlier session code was
modified. The production files were read through `git show origin/master:...` into the scratchpad and were not
committed.

## Order of operations (auditable in git history)

| Commit | What | Holdout contact |
|---|---|---|
| `843df4a` 03:30 UTC | `rebuild/PREREGISTRATION.md` + `session4_frozen.json` (F1/L1 parameters frozen from DEV-only data read with a `season <= 2023` parquet filter) | none |
| next commit, 03:31 UTC | `session4_holdout.py` run **once**; raw `session4_holdout.json` committed before it was read | the single scoring run |
| this commit | 2026 shadow (`session4_shadow.py`, `shadow_2026.csv`, `session4_shadow.json`) and this report | none: no re-runs or changes |

## Pre-registration (copy; full text in `rebuild/PREREGISTRATION.md`)

**F1, the frozen model** (session 3's "B2 ridge + shrink k"):
- Mean-impute → standardise → Ridge(**alpha = 1e4**) on the margin residual vs the closing spread.
- Prediction = spread + **k = 0.25** × ridge prediction.
- 66 store features.
- Holdout season T is trained on 2010..T−1. Win probability comes from a logistic calibration on the
  training-window predictions.

**L1, the lead** (fade A-lite's largest edges):
- A-lite runs with frozen parameters: ρ = 0.4644, q = 0.0816, p0 ≈ 0, r = 88.49.
- z = A-lite edge / posterior SD.
- Top quintile = |z| ≥ **0.36663**, the cut fixed from DEV 2016-2023.
- Bet against A-lite's side. The prediction is that the fade covers more than 52.4%.

**Decision rules:**
- F1 passes if its paired MAE vs the line is ≤ 0, or its ATS is ≥ 52.4% on |edge| ≥ 2 with n ≥ 100.
- L1 passes if its ATS is ≥ 52.4% with n ≥ 150 *and* its 95% CI excludes 50%.
- Everything else is "no edge".

**Power notes written before the data:**
- L1 would probably not reach n = 150, about 110 expected.
- F1 can only pass through MAE, and that would be a coin flip around "identical to the market" (SE about 0.007).

## Step 1: holdout 2024-2025 (run once; `rebuild/session4_holdout.json`)

| Model | n | MAE | Market MAE | Win acc (market) | ATS (n) | Paired MAE vs market (SE) | Brier | Log-loss |
|---|---|---|---|---|---|---|---|---|
| Market (closing spread) | 544 | 9.666 | 9.666 | 68.3% | — | 0 | 0.20565 | 0.59691 |
| **F1 (frozen B2 ridge + k)** | 544 | **9.663** | 9.666 | 68.3% (68.3%) | 50.3% (539) | **−0.0034 (0.0072)** | 0.20567 | 0.59696 |
| Production XGBoost (BASELINES.md) | 509 | 9.617 | 9.474 | 68.7% (68.5%) | 51.7% (505) | +0.1429 (0.0975) | n/a | n/a |

- F1 vs XGBoost on the 509 shared games: **−0.148 MAE (SE 0.095)**, i.e. F1 is better by about 1.5 SE.
- F1's Brier vs the market: +0.00003 (SE 0.00022), identical.
- F1 per season vs the market: 2024 −0.007, 2025 −0.000.
- XGBoost per season vs the market: 2024 +0.072, 2025 +0.217.
- The XGBoost row reproduces BASELINES.md exactly (9.617 / 9.474 / 68.7% / 51.7%).

ATS by |model − market| edge (n):

| Model | 0-1 | 1-2 | 2-3 | 3-5 | 5+ |
|---|---|---|---|---|---|
| F1 | 50.2% (538) | 100% (1) | — (0) | — (0) | — (0) |
| Production XGBoost | 54.1% (207) | 51.9% (135) | 47.0% (83) | 49.2% (59) | 52.4% (21) |

F1's largest |edge| in the holdout is 1.78 points, so its edge ≥ 2 route had n = 0, as expected.

**F1 decision: "passes" on the letter of the rule (paired MAE −0.0034 ≤ 0), and that is NOT evidence of edge.**
The difference is 0.47 SE from zero, exactly the coin flip the pre-registration predicted. Read correctly, F1 is
indistinguishable from the closing line on MAE, win accuracy and calibration.

**L1, fade the top-|z| quintile:**

| Sample | Bets (excl. pushes) | Fade ATS | 95% CI | Rule (n ≥ 150, ≥ 52.4%, CI excludes 50%) |
|---|---|---|---|---|
| Holdout 2024-2025, **primary** | 130 (23.9% of games) | 54.6% | 46.1%-63.2% | **fails**: n < 150, CI includes 50% |
| 2024 only | 61 | 47.5% | — | — |
| 2025 only | 69 | 60.9% | — | — |
| Secondary: holdout + 2026 wk 1-8, pooled (pre-registered as descriptive) | 157 | 56.1% (88/157) | 48.3%-63.8% | fails: CI includes 50% |

**L1 decision: no edge.** The point estimate stays on the "right" side (54.6%; DEV was 53.5%), but the two
holdout seasons point in opposite directions, and no sample excludes chance.

## Step 2: 2026 shadow, weeks 1-8 (`rebuild/shadow_2026.csv`, `rebuild/session4_shadow.json`)

- Data: nflverse 2026 play-by-play, injuries and schedules, downloaded in 2 s. Week 9 has no results yet, so the
  backfill covers weeks 1-8 (121 games).
- The store was extended with the unchanged session-2 code (calendar extended at runtime). The script asserts that
  every 2010-2025 feature row is identical to the committed store, and that check passed.
- F1 is one fit on 2010-2025 with no refit. A-lite z comes from the frozen filter.

| 2026 weeks 1-8 | n | MAE | vs market (SE) | ATS (n) |
|---|---|---|---|---|
| Market (nflverse closing spread) | 121 | 9.570 | — | — |
| F1 (frozen) | 121 | 9.587 | +0.016 (0.017) | 52.2% (115) |
| Production model (from `outputs/` on master) | 121 | 9.916 | +0.345 (0.262) | 49.1% (116) vs its logged line; 47.8% (115) vs the nflverse close |
| L1 fade (top-\|z\| games, 24% of games) | 27 | — | — | 63.0% (27) |

Notes:
- F1's largest 2026 |edge| is 0.46 points. Its "picks" are therefore near-coin-flips around the line, and its 52.2%
  is ±4.7 points of SE.
- Production's logged line differs from the nflverse closing line in 40 of 121 games, because production logs an
  earlier snapshot (e.g. ATL@PIT: 3.0 logged vs 6.5 close). Production's own `spread_correct` column gives 49.6%
  (n = 121, pushes counted as misses).
- Per week, F1 ATS ranged 36-71% and production 43-54%. **Eight weeks and 121 games are far too few for any
  conclusion**: the SE on ATS is about 4.7 points and the SE on the MAE difference about 0.02-0.26.

## Verdict

**(a) Is F1 ready to replace the production XGBoost as a robust, calibrated near-market baseline? Yes.**

| | F1 | Production XGBoost |
|---|---|---|
| Holdout 2024-25 MAE vs market | −0.003 (SE 0.007), i.e. equal | +0.143 (SE 0.098) |
| DEV 2016-23 MAE vs market | +0.001 (SE 0.004) | +0.244 (SE 0.060) |
| 2026 wk 1-8 MAE vs market | +0.016 (SE 0.017) | +0.345 (SE 0.26) |
| Head-to-head paired MAE | holdout −0.148 (SE 0.095); DEV −0.242 (SE 0.058) | — |
| Calibrated win probability | yes; Brier equal to the market's (0.2057 vs 0.2057 holdout) | none from a training-window fit |
| Robustness | no NaN-driven swings (shrunk QB and injury features, k = 0.25); \|edge\| ≤ 1.8 pts | sensitive to the zero-injury artifact and NaN QB (REBUILD_PLAN.md) |

Two caveats:
1. F1 is close to "the market line plus noise". Replacing XGBoost improves accuracy and removes spurious confident
   picks, but it gives no betting signal.
2. Production currently records an earlier line snapshot, while F1 is defined relative to the closing line. In
   live use it needs the latest available line as its input.

**(b) Is there any evidence of a real edge? No.**
- F1 matches the line and does not beat it.
- L1 fails its pre-registered rule: 54.6% on n = 130, CI 46-63%, with 2024 and 2025 disagreeing.
- Nothing in sessions 2-4 beat the closing line beyond noise on any metric: MAE, ATS buckets, Brier or totals.

**(c) Recommended next work, ranked by honest expected value:**
1. **Swap production to F1 in shadow first, then live, as the baseline (high value, low risk).** Run F1 next to
   XGBoost in the weekly tracking for the rest of 2026 without changing what the dashboard shows. Switch after the
   season if F1 stays within ±0.05 of the market on MAE. Expected gain: about 0.15-0.25 MAE vs the current model,
   and calibrated probabilities. Expected betting value: about zero.
2. **Use the model for what it can do: probabilities and uncertainty for decision support, not picks (moderate
   value).** Present the calibrated win probability and the market's own implied probability side by side, and stop
   surfacing "edges" under about 2 points, which are noise.
3. **Test new information the closing line might not fully price (low prior, the only route to edge).** The best
   candidates are:
   - opening-to-closing line movement and timing (production already logs open and close snapshots, so 2026 data is
     accumulating);
   - weather forecasts at bet time vs kickoff;
   - late inactive or injury news between the snapshot and kickoff.

   Each would need to be pre-registered like L1, with power for n ≥ 300. Expect most to fail.
4. **Keep L1 parked, not dead.** Re-test it on forward data under the same frozen rule.
   **Correction (2026-11-05):** an earlier version of this item said that detecting a 53.5% effect needs "about
   400+ bets, about 2 seasons". That was wrong.
   - About 24% of games qualify, roughly 65 bets per season, so 400 bets is about 6 seasons.
   - 80% power at 53.5% needs about 1,260 bets, about 19 seasons.

   The forward re-test (L1b in `rebuild/PREREGISTRATION_2.md`) can therefore only confirm a large effect (an
   observed rate of about 60% or more at n ≈ 100 by the end of 2027). A real edge here is unlikely (prior ≤ 20%).
5. **Do not spend more sessions on new architectures** (deep learning, MCMC, larger GBMs) using the same public
   features. Sessions 2-3 showed the information is already in the line.

## Files added in this session

- `rebuild/PREREGISTRATION.md`, `rebuild/session4_freeze.py`, `rebuild/session4_frozen.json`: pre-registration
  and frozen parameters.
- `rebuild/session4_holdout.py`, `rebuild/session4_holdout.json`: the single holdout run.
- `rebuild/session4_shadow.py`, `rebuild/session4_shadow.json`, `rebuild/shadow_2026.csv`: the 2026 shadow
  backfill. To reproduce it, download into `data/raw/shadow2026/dl/` as described in the script docstring.

## Cost

| Step | Time |
|---|---|
| pytest | 56 s |
| freeze | about 3 s |
| holdout | about 5 s |
| 2026 download | 2 s |
| shadow build + scoring | about 15 s |

The total wall time was well under the 90-minute cap.
