"""
Leak tests for the as-of feature store (rebuild/build_asof_store.py).

    python -m pytest rebuild/tests/test_asof_no_leak.py -v      (about 2 minutes; needs data/raw)

* Truncation test: for randomly chosen (fixed seed) historical weeks, rebuild the store from raw data
  with every game kicked off on/after that week's first kickoff deleted (the target week's own games keep
  only their pre-game schedule row) and with later injury reports deleted. The target week's features must
  equal the full store's within 1e-9.
* Leak sanity test: the same check must FAIL for a deliberately leaky feature (.shift(-1)).
* Monotonic-information test: building with seasons up to 2018 only gives identical 2010-2018 rows.
* NaN / row-count report: QB features have no NaN; other NaNs only where documented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rebuild.build_asof_store import build_store, feature_columns  # noqa: E402
from rebuild.features.common import load_raw, truncate_raw  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data/raw/pbp_2010.parquet").exists(),
                                reason="raw data missing: run python rebuild/bootstrap_data.py")

rng = np.random.default_rng(2027)
_SEASONS = rng.choice(np.arange(2011, 2024), size=5, replace=False)
TRUNC_WEEKS = [(int(s), int(w)) for s, w in zip(_SEASONS, rng.integers(1, 18, size=5))]
# fixed extra cases: week 1 (prior-season carry-over only), early-season blend, 17-game-era last week
EXTRA_WEEKS = [(2019, 1), (2021, 3), (2023, 18)]

# documented structural NaNs: team ratings before a team's first 2010 game, outdoor games without
# weather, and the one game with no moneylines (2017_04_CHI_GB)
ALLOWED_NAN = {"temp": 250, "wind": 250, "ml_home_prob": 1, "ml_vig": 1}
TEAM_RATING_MAX_NAN = 16


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def full(raw):
    return build_store(raw)


@pytest.fixture(scope="module")
def full_leaky(raw):
    return build_store(raw, inject_leak=True)


def mismatches(full_store: pd.DataFrame, trunc_store: pd.DataFrame, season: int, week: int) -> list[str]:
    a = full_store[(full_store.season == season) & (full_store.week == week)].set_index("game_id").sort_index()
    b = trunc_store[(trunc_store.season == season) & (trunc_store.week == week)].set_index("game_id").sort_index()
    assert len(a) > 0 and a.index.equals(b.index)
    bad = []
    for c in feature_columns(full_store):
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x):
            if not np.allclose(x.values.astype(float), y.values.astype(float), rtol=0, atol=1e-9, equal_nan=True):
                bad.append(c)
        elif not x.equals(y):
            bad.append(c)
    return bad


@pytest.mark.parametrize("season,week", TRUNC_WEEKS + EXTRA_WEEKS)
def test_truncation(raw, full, season, week):
    t = truncate_raw(raw, season, week)
    # the truncated raw data really contains nothing from the target week onwards
    assert t["sched"].loc[t["sched"].season.eq(season) & t["sched"].week.eq(week), "result"].isna().all()
    assert not t["pbp"]["game_id"].isin(full.loc[full.gw >= full.loc[full.season.eq(season) & full.week.eq(week), "gw"].min(), "game_id"]).any()
    bad = mismatches(full, build_store(t), season, week)
    assert bad == [], f"features changed when future rows were deleted ({season} wk {week}): {bad}"


def test_truncation_catches_injected_leak(raw, full_leaky):
    season, week = TRUNC_WEEKS[0]
    bad = mismatches(full_leaky, build_store(truncate_raw(raw, season, week), inject_leak=True), season, week)
    assert "home_leak_next_off_pass_epa" in bad and "away_leak_next_off_pass_epa" in bad
    assert all("leak" in c for c in bad), bad   # only the injected column trips the test


def test_monotonic_information(raw, full):
    cut = 2018
    early = {"sched": raw["sched"][raw["sched"].season <= cut], "pbp": raw["pbp"][raw["pbp"].season <= cut],
             "inj": raw["inj"][raw["inj"].season <= cut]}
    e = build_store(early).set_index("game_id").sort_index()
    f = full[full.season <= cut].set_index("game_id").sort_index()
    assert e.index.equals(f.index)
    cols = feature_columns(full)
    pd.testing.assert_frame_equal(e[cols], f[cols], check_exact=False, rtol=0, atol=1e-9)


def test_row_counts_and_nans(full):
    assert len(full) == full["game_id"].nunique() == 4175
    assert full.groupby("season").size().between(256, 272).all()
    nan = full[feature_columns(full)].isna().sum()
    qb_cols = [c for c in nan.index if "_qb_" in c]
    assert len(qb_cols) >= 14 and nan[qb_cols].sum() == 0, nan[qb_cols][nan[qb_cols] > 0]
    inj_cols = [c for c in nan.index if "_inj_" in c]
    assert nan[inj_cols].sum() == 0
    for c, n in nan[nan > 0].items():
        if c.startswith(("home_off_", "home_def_", "away_off_", "away_def_")):
            assert n <= TEAM_RATING_MAX_NAN, (c, n)
            assert full.loc[full[c].isna(), "season"].eq(2010).all()
        else:
            assert c in ALLOWED_NAN and n <= ALLOWED_NAN[c], (c, n)


def test_no_feature_is_a_disguised_label(full):
    d = full[full.season <= 2023]
    for c in feature_columns(full):
        if pd.api.types.is_numeric_dtype(d[c]) and d[c].std() > 0:
            for lab in ("actual_margin", "actual_total"):
                assert abs(d[c].corr(d[lab])) < 0.9, (c, lab)
