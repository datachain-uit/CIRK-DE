"""Tests of the two exploratory analyses E1, E2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mpqda.encoding import decode_topk
from scripts import disruption_plain as DP
from scripts import explore_deg_vs_de as EX


def test_plain_keys_put_the_panel_on_top():
    rng = np.random.default_rng(0)
    S = np.array([3, 17, 42, 99, 250])
    x = DP.plain_keys(rng, S, 400, 5)
    assert set(decode_topk(x, 5).tolist()) == set(S.tolist())
    assert x.min() >= 0 and x.max() < 1


def test_plain_simulation_runs_and_is_deterministic():
    a = DP.simulate_plain(reps=3, p0=300, k=5)
    b = DP.simulate_plain(reps=3, p0=300, k=5)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 16 and a.random_genes.between(0, 1).all()


def test_deg_minus_de_is_paired_by_fold():
    rows = []
    for c in ("gse10245", "gse30219", "gse37745"):
        for r in range(3):
            for f in range(5):
                for k in (2, 5, 10, 50):
                    rows.append({"cohort": c, "repeat": r, "fold": f, "k": k, "method": "de_rand1",
                                 "best_fitness": -0.2 + 0.001 * f})
                    rows.append({"cohort": c, "repeat": r, "fold": f, "k": k, "method": "de_g",
                                 "best_fitness": -0.2 + 0.001 * f + 0.01})
    out = EX.compare(pd.DataFrame(rows))
    assert np.allclose(out.mean_diff, 0.01) and (out.n == 45).all()
    assert (out.share_deg_better == 1).all() and (out.p_one_sided < 0.001).all()
