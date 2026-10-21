# Project guidelines for AI sessions (NFL prediction model)

- This repo has a LIVE production system on `master` (Streamlit dashboard + weekly tracking). Do not modify
  `dashboard/`, `tracking/`, `outputs/`, `models/train_model.py`, `config.py`, or anything the dashboard imports,
  and never push to `master`. Rebuild work lives under `rebuild/` and `docs/` on `rebuild/*` branches.
- Use `python` (not `python3`). Raw data is not in the repo: run `python rebuild/bootstrap_data.py`.
- Every experiment is scored with `rebuild/protocol.py` (walk-forward DEV 2016-2023; HOLDOUT 2024-2025 untouched).
- Features must be as-of kickoff. If a step could use future information, treat it as a bug and add a test.
- Report honestly: sample sizes, standard errors, and "no edge" when that is the result. Do not tune on the holdout.
- Keep compute modest and stop-and-report when blocked; do not retry failing commands in a loop.
- Open a pull request for review; do not merge it yourself.
