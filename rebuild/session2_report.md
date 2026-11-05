# Rebuild session 2: as-of feature store and leak tests

Branch `rebuild/session2-features` (from `rebuild-2027`). Container reused from session 1, so raw data and libraries
were already present; pytest was installed. No production files were touched, and 2024-2025 were not used for any
model fitting or scoring. They are present in the store only as feature rows.

## Verdict

**The store is ready for candidates A/B/C, with caveats.** It is leak-free under the required tests: 12/12 pass,
including a sanity test showing that an injected `.shift(-1)` leak is caught. It is fast (5 s to build), has
complete QB and injury features (no NaN), and every NaN that remains is structural and documented.

**Smoke test: no edge.** On DEV 2016-2023, ridge on the market residual is slightly *worse* than the closing line:
- +0.060 MAE (SE 0.025) with all features.
- +0.052 (SE 0.027) without the QB and injury groups.

Both beat the production XGBoost on the shared games, by about 0.19 MAE (SE 0.06). That is the expected
"robust but no information beyond the line" result.

Caveats for the next sessions:
1. **Ridge alpha hit the top of the named grid (1e4) in 6/8 folds.** That means the data wants the residual
   shrunk to about 0. The one fold that chose alpha=100 (2021) lost 0.37 MAE to the market. Candidates should
   expect near-zero coefficients and should get the market prior from model structure (candidate A), not from a
   wide feature regression.
2. **OL injury usage is a constant 0.6.** There are no snap counts in `data/raw`. Adding nflverse
   `snap_counts` (2012+, needs a PFR-to-GSIS id map) is the obvious upgrade for session 3.
3. **QB starter = the game's actual starter** (nflverse `*_qb_id`). This is the as-of-announcement proxy and
   assumes late QB switches were known.
4. **The information set is week-granular.** Sunday games do not see that week's Thursday result. This is
   conservative and makes the truncation test exact.
5. **Shrinkage pseudo-counts and position values are fixed constants, not fitted.** Candidate B can estimate them
   (only inside training folds).

## Files

| File | Purpose |
|---|---|
| `rebuild/features/common.py` | raw loading, team-code normalisation (OAK→LV, SD→LAC, STL→LA), global week index, `truncate_raw` |
| `rebuild/features/team_strength.py` | decayed, opponent-adjusted offense/defense ratings with prior-season carry-over |
| `rebuild/features/qb.py` | starting-QB career/recent EPA, CPOE, sack rate with shrinkage to a replacement prior |
| `rebuild/features/injuries.py` | player-level injury value from the final pre-game report |
| `rebuild/features/market.py`, `context.py` | closing market, rest/travel/weather/division |
| `rebuild/build_asof_store.py` | `build_store(raw)` (pure function of raw tables) → `data/processed/asof_features.parquet` |
| `rebuild/tests/test_asof_no_leak.py` | leak tests (pytest) |
| `rebuild/session2_smoke.py` | DEV smoke test → `rebuild/session2_smoke.json` |
| `rebuild/asof_store_report.json` | row counts per season and NaN counts |
| `data/processed/asof_features.parquet` | 4,175 rows × 80 columns, 1.6 MB (committed) |

## As-of convention

Each regular-season game has a global week index `gw`. Features of a game in week `gw` use only:
- games in weeks `< gw`, including earlier seasons;
- the game's own pre-game schedule row: teams, kickoff, closing spread/total/moneylines, roof, rest, weather and
  the starting QB id;
- injury reports for weeks `<= gw`. The week-`gw` report is the final pre-game report, and all reports are
  timestamped before kickoff in nflverse; I checked this, 0 of about 40k status rows are dated at or after
  kickoff.

Only regular-season games are used, both as rows and as history.

## Feature list (per side `home_`/`away_` unless noted; 66 features)

**Team strength (24): `{off,def}_{pass,rush}_{epa,sr,expl}`.**
- Per game, from pbp: EPA/play, success rate and explosive rate (pass ≥ 20 yds, rush ≥ 10 yds). Pass = dropbacks
  (including sacks and scrambles); rush = designed runs. `def_*` is what the team allowed, so higher = worse.
- Rating = play-weighted EWM over the team's prior games this season, half-life 6 games (decay 0.5^(1/6) per game).
- Carry-over: prior = L + 0.5 × (last season's final EWM − L), i.e. shrink weight 0.5 toward the league mean L.
  - Team game 1: prior only.
  - Games 2-4: w = 1, ¾, ½, ¼ on the prior (w × prior + (1−w) × current-season EWM).
  - Game 5 onwards: current season only.
- L = league mean over the previous 17 global weeks.
- Opponent adjustment: one pass, each game's per-play value shifted by (opponent's unadjusted as-of rating − L),
  then re-rated.
- NaN only before each team's first 2010 game (16 games).

**QB (8): `qb_epa_db_career`, `qb_epa_db_recent`, `qb_sack_rate`, `qb_cpoe`, `qb_dropbacks_prior`,
`qb_rookie_or_unknown`, `qb_changed`, `qb_prior_epa`.**
- Starter: the schedule's `*_qb_id` (the game's first-snap starter; the as-of-announcement proxy).
- Shrinkage: (sum + K × prior) / (n + K), with K = 300 dropbacks (EPA), 200 dropbacks (sack rate) and 300 attempts
  (CPOE).
- Career = all prior QB games; recent = EWM with a 6-game half-life.
- Prior = replacement level: as-of mean over dropbacks by QBs with fewer than 300 career dropbacks, previous 51
  global weeks. Fallback constants are used only when that window has fewer than 500 dropbacks (early 2010).
- Rookies and unknown QBs get exactly the prior, flagged with `qb_rookie_or_unknown` (< 100 career dropbacks).
- `qb_changed` = starter differs from the team's previous game starter.
- **0 NaN.**

**Injuries (4): `inj_off_value`, `inj_def_value`, `inj_qb_out`, `inj_report_missing`.**
- Status weight: Out 1, Doubtful 1, Questionable 0.25, Probable 0.
- Usage = min(1, share / ref), where share is the player's share of team usage over the team's previous 6 games:
  - QB: dropbacks.
  - WR/TE/RB/FB: targets + carries.
  - Defenders: tackles, sacks, TFL, QB hits, passes defensed, INTs.
  - OL: constant 0.6.
  - K/P/LS: 0.
- Value = status weight × position value × usage, summed over the team's players.
- `inj_qb_out` = max over the team's QBs of status weight × usage.
- No counts and no zero-reading: a team-week with no report has `inj_report_missing = 1` (9 team-games) and values
  carried forward from the team's previous game.
- **0 NaN.**

**Market (4, game level): `spread_line`, `total_line`, `ml_home_prob` (vig removed: p_h/(p_h+p_a)), `ml_vig`.**
- 1 NaN: 2017_04_CHI_GB has no moneylines.

**Context (13, game level):**
- `home_rest`, `away_rest`, `rest_diff`
- `*_short_week` (rest ≤ 5) and `*_bye_before` (rest ≥ 13, excluding week 1)
- `tz_diff` (home stadium tz − away team tz, using historic codes; 0 at neutral sites) and `neutral_site`
- `is_dome`; `temp`, `wind` (domes: 70 / 0); `weather_missing`
- `div_game`
- Structural NaN: temp/wind for 196 outdoor games without weather.

**IDs and labels (not features):** `game_id`, `season`, `week`, `gw`, `kickoff`, teams, QB ids, `actual_margin`,
`actual_total`, `home_win`, scores.

Rows: 256 per season for 2010-2020, 272 for 2021-2025 (2022: 271, the cancelled BUF@CIN game). 4,175 in total.

## Leak tests (`python -m pytest rebuild/tests/test_asof_no_leak.py`): 12 passed in 28 s

| Test | What it checks | Result |
|---|---|---|
| truncation × 5 random weeks (seed 2027): 2014 wk 8, 2012 wk 12, 2015 wk 12, 2011 wk 12, 2016 wk 6 | Rebuild from raw with every game kicked off on/after the week's first kickoff deleted (the target week keeps only its pre-game schedule row, results blanked), later injury reports deleted, and report rows dated on/after kickoff deleted. All 66 features of that week's games equal the full store within 1e-9. | pass |
| truncation × 3 fixed weeks: 2019 wk 1, 2021 wk 3, 2023 wk 18 | The same, covering the week-1 carry-over, the early-season blend and the 17-game era | pass |
| injected-leak sanity | A feature built with `.shift(-1)` (the team's next-game rating) is flagged by the truncation check, and nothing else is | pass |
| monotonic information | Building with seasons ≤ 2018 only gives identical 2010-2018 rows | pass |
| row counts / NaN | 4,175 unique games; QB and injury columns have 0 NaN; other NaNs only as documented above | pass |
| disguised label | No feature has \|corr\| ≥ 0.9 with margin or total (DEV seasons) | pass |

The truncation test also asserts that the truncated pbp contains no play from the target week onward. Results,
later games' closing lines and the season's final stats are therefore physically absent when the target week is
rebuilt.

## Smoke test (DEV 2016-2023 walk-forward, train 2010..T-1; `rebuild/session2_smoke.json`)

Ridge setup:
- Mean-impute → standardise → Ridge on (margin − spread).
- Alpha grid {0.1, 1, 10, 100, 1e3, 1e4}, chosen inside the training window. There are 3 season folds: validate
  on T-3, T-2 and T-1, each fitted on the seasons before it.

All games with a closing line: n = 2,095. The XGBoost row is the BASELINES.md row (its 1,924 games).

| Model | n | MAE | Market MAE | Win acc (market) | ATS (n) | Paired MAE vs market (SE) | Paired vs XGB, 1,924 shared games (SE) |
|---|---|---|---|---|---|---|---|
| (a) market | 2,095 | 9.820 | 9.820 | 66.1% (66.2%) | n/a | 0 | **−0.244 (0.060)** |
| (b) ridge, all 66 features | 2,095 | 9.880 | 9.820 | 66.3% (66.2%) | 50.5% (2,035) | **+0.060 (0.025)** | −0.189 (0.059) |
| (c) ridge, no QB/injury (42) | 2,095 | 9.872 | 9.820 | 66.4% (66.2%) | 51.5% (2,035) | +0.052 (0.027) | −0.184 (0.059) |
| XGBoost (BASELINES.md) | 1,924 | 10.021 | 9.778 | 66.0% (66.1%) | 51.0% (1,867) | +0.243 | n/a |

ATS by |model − market| edge (n):

| Model | 0-1 | 1-2 | 2-3 | 3-5 | 5+ |
|---|---|---|---|---|---|
| (b) ridge all | 51.1% (1,621) | 49.5% (287) | 42.9% (49) | 52.6% (57) | 33.3% (21) |
| (c) ridge no QB/inj | 51.3% (1,553) | 54.4% (270) | 54.5% (123) | 45.1% (71) | 33.3% (18) |
| XGBoost | 50.1% (585) | 50.4% (482) | 54.0% (346) | 51.6% (308) | 47.9% (146) |

Ridge MAE minus market MAE by test season:

| Model | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | Seasons better |
|---|---|---|---|---|---|---|---|---|---|
| (b) all | −0.060 | −0.000 | −0.030 | +0.017 | +0.075 | +0.375 | +0.051 | +0.036 | 3 of 8 |
| (c) no QB/inj | −0.044 | −0.005 | −0.028 | −0.020 | +0.118 | +0.342 | +0.022 | +0.017 | 4 of 8 |

Chosen alpha:
- (b): 1e4 in every fold except 2021 (100).
- (c): 1e4 in 2016-2018 and 2022-2023, 1e3 in 2019-2020, 100 in 2021.

Reading:
- **No edge.** Neither ridge beats the market. (b) is about 2.4 SE worse; (c) is about 1.9 SE worse.
- No edge bucket clears the 52.4% break-even with any sample size. (c)'s 1-2 and 2-3 buckets (54.4%, 54.5%) are
  only about 1-1.5 SE above 50% (SE about 3 and 4.5 points); fold stability was not checked.
- **QB + injury groups add nothing measurable.** (b) − (c) = +0.008 MAE, inside noise. The market already prices
  starter identity and injuries. The groups' value is robustness (no NaN, no zero artifact), not edge.
- Losses concentrate in 2020-2021, the no-fans season and the season after it. The lower alpha chosen for 2021
  made it worse. Home-field and COVID-era shifts are a likely factor for candidate A to model explicitly.
- Both ridges beat the production XGBoost by about 0.19 MAE (about 3 SE) on the shared games, consistent with the
  earlier alt-models experiment. The current production model loses to the line by more than these do.

## Wall times

| Step | Time |
|---|---|
| Setup (container reused: libraries and data already present; pytest installed) | about 10 s |
| `build_asof_store.py` (load 1 s + build 5 s) | about 6 s |
| Leak tests (12 tests, 10 store rebuilds) | 28 s |
| Smoke test (2 models × 8 folds × 6 alphas × 3 inner folds) | 5 s |

Compute is negligible. In a fresh container, add about 2.5 min for pip install and bootstrap (session 1).
