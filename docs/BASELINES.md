# Baselines (reference numbers every candidate must be compared against)

Source: walk-forward predictions of the production XGBoost margin model (outputs/walk_forward_preds_2014_2025.pkl),
scored with `rebuild.protocol.evaluate`. Closing spread = "market". ATS ignores pushes; break-even at -110 is 52.4%.

| Window | n | MAE model | MAE market | Win acc model | Win acc market | ATS |
|---|---|---|---|---|---|---|
| DEV 2016-2023 | 1,924 | 10.021 | 9.778 | 66.0% | 66.1% | 51.0% (n=1,867) |
| HOLDOUT 2024-2025 (do not tune on) | 509 | 9.617 | 9.474 | 68.7% | 68.5% | 51.7% (n=505) |

ATS by |model - market| edge, DEV 2016-2023: 0-1: 50.1% (585) | 1-2: 50.4% (482) | 2-3: 54.0% (346) | 3-5: 51.6% (308) | 5+: 47.9% (146)
ATS by edge, HOLDOUT 2024-2025: 0-1: 54.1% | 1-2: 51.9% | 2-3: 47.0% | 3-5: 49.2% | 5+: 52.4% (n=21)

Takeaways: the market beats the current model on MAE in both windows; the current model's win accuracy ties the
market; no edge bucket is reliably above break-even (the earlier "3-5 pt bucket" figure was a sample artifact).

Other reference (branch `experiment/alt-models-walkforward`, pooled 2014-2025, MAE): market 9.910, XGBoost 10.148,
ridge 9.930, Bayesian ridge 9.912, MLP 9.917. Injury-zero sensitivity (mean |change| in pts): XGBoost 1.10,
ridge 0.23, MLP 0.17, Bayesian ridge 0.08.

Per-game numbers for 2026 (live): 46/93 ATS, 58/92 winners (Weeks 1-6), see outputs/weekly_results.csv.
