"""scripts/make_taimma_optimism.py on SYNTHETIC results: optimism = outer log-loss − (−output_fitness) per unit, every
method, differences against v1 and GA with the right sign, byte-identical rerun, without T5C1/T1 (job v1b not run) only B-LL remains."""
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import make_taimma_optimism as O
from test_tables import COHORTS, fake_results
from test_v2_tables import write_v2


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    fake_results(root)
    write_v2(root)
    for c in COHORTS:                                  # job v1b: B-C1 for the T1 panels of v1 and the seven baselines
        dst = root / "T5C1" / "T1" / c
        dst.mkdir(parents=True, exist_ok=True)
        ev = pd.read_csv(root / "T5LL" / "T1" / c / "classification_per_fold.csv", encoding="utf-8-sig")
        ev.assign(logloss=ev.logloss + 0.05).to_csv(dst / "classification_per_fold.csv", index=False, encoding="utf-8-sig")
    return root


def unit_table(root, phase_dir):
    sel = pd.concat([pd.read_csv(root / s / c / "selection_per_fold.csv") for s in ("T1", "T1v2") for c in COHORTS])
    ev = pd.concat([pd.read_csv(root / phase_dir / s / c / "classification_per_fold.csv", encoding="utf-8-sig")
                    for s in ("T1", "T1v2") for c in COHORTS])
    return sel.merge(ev[ev.model == "lr_l2"], on=O.KEY)


def test_optimism_matches_definition_and_is_reproducible(results, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    info = O.build(results, a)
    O.build(results, b)
    assert info["phase_b"] == ["B-C1", "B-LL"]
    for name in ("optimism_v2.csv", "optimism_v2_gain.csv", "optimism_v2.tex", "claims_optimism.csv", "optimism_inputs_sha256.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    s = pd.read_csv(a / "optimism_v2.csv")
    assert set(s.method) <= set(O.METHODS) and s.n.eq(45).all()
    assert {"mpqda_v2", "mpqda_v2_best", "de_g", "mpqda_de", "mpqda_de_best", "ga", "bpso"} <= set(s.method)
    u = unit_table(results, "T5C1")
    u = u[u.method.isin(O.METHODS)]
    want = (u.logloss + u.output_fitness).groupby([u.method, u.k]).mean()
    got = s[s.phase_b == "B-C1"].set_index(["method", "k"]).optimism
    assert np.allclose(got.loc[want.index].to_numpy(), want.to_numpy(), atol=1e-12)
    assert np.allclose(s.optimism, s.outer_logloss - s.inner_logloss, atol=1e-12)


def test_gain_sign_and_claims(results, tmp_path):
    O.build(results, tmp_path / "o")
    g = pd.read_csv(tmp_path / "o" / "optimism_v2_gain.csv")
    assert len(g) == 8 and set(g.reference) == {"mpqda_de", "ga"} and g.n.eq(45).all()
    u = unit_table(results, "T5LL")
    one = g[(g.phase_b == "B-LL") & (g.k == 5) & (g.reference == "ga")].iloc[0]
    pa = u[(u.method == "mpqda_v2") & (u.k == 5)].set_index(["cohort", "repeat", "fold"])
    pb = u[(u.method == "ga") & (u.k == 5)].set_index(["cohort", "repeat", "fold"])
    assert np.isclose(one.inner_logloss_diff, (pb.output_fitness - pa.output_fitness).mean(), atol=1e-12)
    assert np.isclose(one.outer_logloss_diff, (pa.logloss - pb.logloss).mean(), atol=1e-12)
    assert (g.outer_nb_half_width > 0).all()
    claims = pd.read_csv(tmp_path / "o" / "claims_optimism.csv")
    assert claims.claim_id.is_unique and set(claims.phase_b) == {"B-C1", "B-LL"}


def test_missing_v1b_keeps_only_b_ll(results, tmp_path):
    shutil.rmtree(results / "T5C1" / "T1")
    info = O.build(results, tmp_path / "o")
    assert info["phase_b"] == ["B-LL"] and any(m.startswith("T5C1/T1/") for m in info["missing"])
    shutil.rmtree(results / "T5LL" / "T1")
    with pytest.raises(SystemExit):
        O.build(results, tmp_path / "p")
