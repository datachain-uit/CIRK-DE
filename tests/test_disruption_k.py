"""Tests of E3 and of the extension to k = 20 (amendment A2 of PREREG-v2)."""
from __future__ import annotations

import pandas as pd

from scripts import disruption_k as DK


def test_disruption_k_is_deterministic_and_gap_inserts_almost_nothing():
    a = DK.simulate_k(reps=3, p0=400, ks=(2, 5))
    b = DK.simulate_k(reps=3, p0=400, ks=(2, 5))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 8
    assert (a[a.eps == 0.01].random_share <= 0.01).all()
    assert (a[a.eps == 0.5].random_share > 0.1).all()
    assert (a.random_genes_per_trial == a.random_share * a.k).all()


def test_new_budget_appended_last_leaves_earlier_rows_unchanged():
    """k = 20 placed last in --ks: the rows of the k values already run (seed 3, same order) must match value by value."""
    base = DK.simulate_k(reps=3, p0=400, ks=(2, 5, 10))
    ext = DK.simulate_k(reps=3, p0=400, ks=(2, 5, 10, 20))
    pd.testing.assert_frame_equal(ext.iloc[: len(base)].reset_index(drop=True), base)
    assert len(ext) == len(base) + 4 and set(ext.iloc[len(base):].k) == {20}


def test_cli_rejects_repeated_budgets(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        DK.main(["--reps", "1", "--ks", "2", "2", "--out", str(tmp_path / "x.csv")])
