# Rebuild session 1: environment and baseline reproduction (cloud)

Date: 2026-11-04. Branch `rebuild/session1-env` (from `rebuild-2027`). Total session time was about 20 minutes.
No production files were modified (`dashboard/`, `tracking/`, `outputs/`, `config.py`, `models/train_model.py`).

## Verdict

**Ready for larger sessions: YES.** Network, data bootstrap, core and optional libraries (statsmodels, lightgbm),
the protocol, and git push all work. The XGBoost baseline reproduces within tolerance on the DEV window.
Caveats, none of them blocking:
- 4 CPUs and about 30 GB of free disk. That is plenty for GLM, state-space or GBM work on about 4,200 games. MCMC
  (pymc) will be slow-ish but feasible with modest chains.
- pymc resolves but took **221 s just to resolve** and would upgrade scipy from 1.17.1 to 1.18.1. Budget about 5
  minutes for the install in session 3 (candidate A), and pin or check scipy afterwards.
- The environment is ephemeral: `pip install -r requirements.txt` (about 1.5 min) and `bootstrap_data.py`
  (about 0.5 min) must be rerun at the start of every session.
- The `gh` CLI is present but not authenticated. PRs are opened through the session's GitHub integration rather
  than `gh` (see step 6).

## 1. Environment (`python rebuild/env_check.py`; full output in `rebuild/env_report_cloud.json`)

Core libraries were not preinstalled (xgboost was missing). `pip install -r requirements.txt` took 98 s and
installed the exact pinned versions. `rebuild/env_report.json` (the local Windows report) was left unchanged; the
cloud report is saved separately.

| Item | Cloud | Local (reference) |
|---|---|---|
| Python | 3.13.18 | 3.11.5 |
| Platform | Linux x86_64 (glibc 2.39) | Windows 10 |
| CPUs | 4 | 24 |
| RAM | 15 GB | n/a |
| Free disk | 30.3 GB | 212.7 GB |
| pandas / numpy / pyarrow / sklearn / xgboost / scipy | 3.0.5 / 2.4.6 / 24.0.0 / 1.9.0 / 3.2.0 / 1.17.1 | same |

| Optional lib | At env check | After step 2 |
|---|---|---|
| lightgbm | missing | 4.7.0 installed |
| statsmodels | missing | 0.15.0 installed |
| pymc | missing | missing (resolves, not installed) |
| arviz | missing | missing (comes with pymc) |
| cmdstanpy | missing | missing (not tested) |
| torch | missing | missing (not needed; the plan rules out deep learning) |

| Network check (HEAD) | Result |
|---|---|
| nflverse schedules | 200 OK, 0.38 s |
| nflverse pbp 2025 | 200 OK, 0.28 s |
| PyPI | 200 OK, 0.03 s |

## 2. Optional library feasibility (`pip install --dry-run`)

| Package | Resolves? | Resolve time | Would install | Action |
|---|---|---|---|---|
| statsmodels | yes | 4 s | statsmodels 0.15.0, patsy 1.0.3, formulaic 1.2.2, interface_meta 2.0.1, wrapt 2.6.0.dev1 | installed (5 s for both) |
| lightgbm | yes | 1 s | lightgbm 4.7.0 | installed |
| pymc | yes | **221 s** | pymc 6.4.0, pytensor 3.3.7, numba 0.68.0, llvmlite 0.50.0, arviz 1.3.0, xarray, **scipy 1.18.1 (upgrade)** | not installed |

Note: statsmodels pulled in a **pre-release** `wrapt 2.6.0.dev1`. It imports fine. If it causes trouble later,
pin `wrapt<2.6`.

## 3. Data (`python rebuild/bootstrap_data.py`)

| Item | Result |
|---|---|
| Wall time | **33 s** (it ran at the same time as step 4) |
| Disk used by `data/raw` | **308 MB** (41 files; the largest are pbp at about 20 MB per season) |
| pbp seasons | 2010-2025, all 16 present |
| injuries seasons | 2010-2025, all 16 present |
| depth charts | 2018-2025, all 8 present |
| schedules | present |
| Failed seasons | none |

## 4. Baseline reproduction (`python rebuild/session1_repro.py`; results in `rebuild/session1_repro.json`)

Method: I used `models/train_model.py` without changes (`get_feature_columns`, which gave 32 features;
`train_margin_model`; `reconstruct_margin`) on `data/processed/game_level_features.parquet`. Rows were filtered
the same way as `prepare_data`. I ran walk-forward for T = 2016..2023, training on all seasons before T, and scored
with `rebuild.protocol.evaluate(name="xgb_repro", seasons=DEV_TEST_SEASONS)`. No 2024-2025 rows were used for
training or scoring.

| DEV 2016-2023 | n | MAE model | MAE market | Win acc model | Win acc market | ATS (n) |
|---|---|---|---|---|---|---|
| BASELINES.md (local) | 1,924 | 10.021 | 9.778 | 66.0% | 66.1% | 51.0% (1,867) |
| xgb_repro (cloud, seed 42) | 1,924 | **10.049** | 9.778 | **65.4%** | 66.1% | **50.2%** (1,867) |
| Difference | 0 | +0.028 | 0 | -0.6 pt | 0 | -0.8 pt |

ATS by |model - market| edge (cloud): 0-1: 51.1% (595) · 1-2: 47.7% (470) · 2-3: 51.6% (322) · 3-5: 50.2% (331)
· 5+: 52.3% (149). The local run had 2-3: 54.0% (346). That bucket does not reproduce, which supports the existing
note that no bucket is reliably above break-even.

**Both differences are within tolerance** (MAE diff 0.028 < 0.05; ATS diff 0.8 < 1 pt). The rows are identical
(same n and same market MAE), but the per-game predictions differ: correlation with
`outputs/walk_forward_preds_2014_2025.pkl` is 0.981, mean |diff| is 1.07 pts, and max is 9.3 pts.

Diagnosis:
- The code and data are unchanged since the reference predictions were generated: there is no git diff in
  `config.py`, `models/`, or `data/` between commit `9ed9e5b` (which wrote the pkl) and HEAD. The reference
  script (`models/backup_qb_test.py`) also drops rows with a NaN `spread_line`, but there are 0 such rows before
  2024, so that is not the cause.
- Thread count is not the cause either: `n_jobs=1` and `n_jobs=4` give identical results on this machine.
- **Most likely cause: XGBoost's row/column subsampling RNG (`subsample=0.8`, `colsample_bytree=0.8`) differs
  between Windows and Linux builds, even with the same seed and version.** Seed noise is about the same size as
  the gap. Over 10 seeds (0-9) on the cloud, results were:
  - MAE: mean 10.020, SD 0.024, range 9.966-10.048
  - win acc: mean 65.9%, SD 0.4 pt
  - ATS: mean 50.9%, SD 0.4 pt, range 50.6-51.5%

  The local baseline (10.021 / 66.0% / 51.0%) is almost exactly the cloud 10-seed mean. Cloud seed 42 happens to
  sit at the unlucky end of the range.
- Implication for later sessions: **the seed alone moves MAE by about ±0.05.** Any candidate's claimed
  improvement over XGBoost smaller than that is noise. Use `paired_mae_diff` against the market, and for
  stochastic models, average over several seeds.

## 5. Cost note

| Step | Wall time | Notes |
|---|---|---|
| pip install requirements.txt | 98 s | needed in every fresh container |
| env_check | about 2 s | |
| dry-run statsmodels / lightgbm / pymc | 4 s / 1 s / **221 s** | pymc's resolver is the only slow item |
| install statsmodels + lightgbm | 5 s | |
| bootstrap_data (step 3) | 33 s | 308 MB |
| walk-forward repro (step 4) | 1.5 s compute (4 s wall) | 8 folds × 200 trees; trivially cheap |
| 10-seed sensitivity check | about 15 s | |

Nothing was expensive. Setup costs about 2.5 minutes per fresh session, plus about 5 minutes if pymc is needed.
Model fitting at this data size is seconds per walk-forward, so compute is not the constraint for sessions 2-5.
MCMC is the only likely exception.

## 6. Capabilities

| Capability | Result |
|---|---|
| `git push` | **works**: pushed `rebuild/session1-env` via the preconfigured proxy credential |
| `gh` CLI | installed (2.89.0) but **not authenticated** ("Failed to log in using token (GH_TOKEN)"). Usable only for search. |
| Open a PR | through the session's GitHub integration (MCP tools), not `gh`. The PR link is reported in the session's final reply. |

## Files added
- `rebuild/session1_report.md` (this report)
- `rebuild/session1_repro.py`: the walk-forward reproduction script (re-runnable, about 5 s)
- `rebuild/session1_repro.json`: metrics from `evaluate` for xgb_repro
- `rebuild/env_report_cloud.json`: env_check output on the cloud machine
