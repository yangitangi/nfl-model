# Cloud session 1 prompt (copy everything below the line into a new cloud session on this repo, branch: rebuild-2027)

---

You are working in the repo yangitangi/nfl-model, starting from branch `rebuild-2027`. Read `CLAUDE.md`,
`docs/REBUILD_PLAN.md` and `docs/BASELINES.md` first. This is cloud SESSION 1 of a 2027 NFL model rebuild: a
small, cheap GATE session to find out whether this environment can support the larger work. Keep total compute
modest, work for at most about 30 minutes, and STOP and report (do not retry failing commands in a loop).

Do these steps in order and record the result of each in `rebuild/session1_report.md`:

1. Environment: run `python rebuild/env_check.py` (installs missing core libraries from requirements.txt first if
   needed). Record Python version, CPU count, free disk, which of lightgbm / statsmodels / pymc / arviz / cmdstanpy
   / torch are installed, and the outbound-network results.
2. Optional library feasibility (do NOT run heavy installs): for `statsmodels`, `lightgbm`, `pymc`, run
   `pip install --dry-run <pkg>` and record whether they resolve. Install only `statsmodels` and `lightgbm` if they
   resolve quickly.
3. Data: run `python rebuild/bootstrap_data.py` (downloads nflverse 2010-2025 into data/raw). Record wall time,
   total disk used by data/raw, and any season that failed. If the network blocks it, say so and skip to step 6.
4. Baseline reproduction: using `models/train_model.py` functions (`get_feature_columns`, `train_margin_model`,
   `reconstruct_margin`) on `data/processed/game_level_features.parquet`, run a walk-forward over test seasons
   2016-2023 (train on all seasons before each test season) and score it with `rebuild.protocol.evaluate`
   (named "xgb_repro", seasons=DEV_TEST_SEASONS). Compare MAE / win accuracy / ATS with the DEV row in
   `docs/BASELINES.md` (expected MAE about 10.02, win acc about 66.0%, ATS about 51.0%). Report any difference
   larger than 0.05 MAE or 1 point of ATS and your best guess why (library versions, randomness, etc.).
5. Cost note: record how long steps 3 and 4 took, and anything that looked slow or expensive.
6. Capabilities: record whether you can push a branch and open a pull request from this environment (is `gh`
   available, is a git credential available). If you cannot open a PR, push the branch and say so.

Deliverables: `rebuild/session1_report.md` (tables of the above, plus a short verdict: "ready for larger
sessions: yes/no and what blocks it"), committed on a new branch `rebuild/session1-env` and pushed; open a pull
request against `rebuild-2027` titled "Rebuild session 1: environment and baseline reproduction" if possible.

Hard constraints: do not modify `dashboard/`, `tracking/`, `outputs/`, `config.py`, `models/train_model.py`;
never push to `master`; do not touch the 2024-2025 holdout except through the protocol's evaluate call on the DEV
seasons only. Final reply: the verdict and the key numbers in under 250 words.
