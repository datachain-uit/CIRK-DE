"""Test cho phân tích lại và phân tích sửa đổi A2 (Review-2026-11-20, TQĐ50, TQĐ51). Dữ liệu giả, chỉ để kiểm mã."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st

from scripts import make_taimma_corrected as K
from scripts import make_taimma_tables as T
from scripts import make_taimma_v2_tables as V

COH = list(T.COHORTS)


def _sel(methods_k, seed=0, shift=None):
    """selection_per_fold giả: 3 cohort × 3 lần lặp × 5 fold; best_fitness = nền theo fold + độ lệch theo phương pháp + nhiễu."""
    rng = np.random.default_rng(seed)
    shift = shift or {}
    rows = []
    for ci, c in enumerate(COH):
        for r in range(3):
            for f in range(5):
                base = -0.2 + 0.01 * ci + 0.002 * f
                for m, ks in methods_k.items():
                    for k in ks:
                        bf = base + shift.get((m, k), shift.get(m, 0.0)) + rng.normal(0, 0.003)
                        for name in ((m, m + "_best") if m.startswith("mpqda") else (m,)):
                            rows.append({"cohort": c, "repeat": r, "fold": f, "method": name, "k": k, "best_fitness": bf,
                                         "n_eval": 1531, "selection_time_s": 20.0 + k})
    return pd.DataFrame(rows)


def test_describe_uses_the_pooled_nadeau_bengio_formula():
    rng = np.random.default_rng(1)
    by_c = [rng.normal(0.01 * (i + 1), 0.02, 15) for i in range(3)]
    out = K.describe(by_c)
    mean = np.mean([d.mean() for d in by_c])
    se = np.sqrt(sum((1 / 15 + 0.25) * d.var(ddof=1) for d in by_c)) / 3
    q = st.t.ppf(0.975, 14)
    assert np.isclose(out["nb_ci_low"], mean - q * se) and np.isclose(out["nb_ci_high"], mean + q * se)
    assert out["n"] == 45 and out["n_cohorts_positive"] == sum(d.mean() > 0 for d in by_c)
    assert out["boot_ci_low"] < out["mean_diff"] < out["boot_ci_high"]
    # CI hiệu chỉnh rộng hơn CI bootstrap (phương sai nhân (1/J + ρ)·J = 4,75 khi J = 15)
    assert out["nb_ci_high"] - out["nb_ci_low"] > out["boot_ci_high"] - out["boot_ci_low"]


def test_hv1_rows_reproduce_the_registered_bootstrap_ci():
    sel = _sel({"mpqda_v2": (5, 10), "mpqda_de": (5, 10), "ga": (5, 10)}, shift={"mpqda_v2": 0.006})
    mine = K.inner_table(sel)
    reg = V.hv1(sel)
    for (k, ref), label in (((5, "mpqda_de"), "v2 - v1"), ((10, "ga"), "v2 - GA")):
        r = reg[(reg.k == k) & (reg.reference == ref)].iloc[0]
        m = mine[(mine.comparison == label) & (mine.k == k)].iloc[0]
        assert np.isclose(m.mean_diff, r.mean_diff) and np.isclose(m.boot_ci_low, r.ci_low) and np.isclose(m.boot_ci_high, r.ci_high)
        assert np.isclose(m.p_wilcoxon_one_sided, r.p)


def test_decomposition_adds_up():
    sel = _sel({"mpqda_v2": K.ALL_K, "mpqda_de": K.ALL_K, "de_g": K.ALL_K, "de_rand1": K.ALL_K},
               shift={"mpqda_v2": 0.009, "de_g": 0.006, "de_rand1": 0.001})
    inner = K.inner_table(sel)
    chk = K.decomposition_check(inner)
    assert len(chk) == 4 and chk.ok.all()


def test_outer_auroc_ci_matches_hv2():
    rows = []
    rng = np.random.default_rng(2)
    for c in COH:
        for r in range(3):
            for f in range(5):
                for m in ("mpqda_v2", "bpso", "mpqda_de"):
                    for k in (5, 10):
                        rows.append({"cohort": c, "repeat": r, "fold": f, "method": m, "k": k, "model": "lr_l2",
                                     "logloss": 0.2 + rng.normal(0, 0.05), "auroc": 0.97 + rng.normal(0, 0.01)})
    ev = pd.DataFrame(rows)
    mine = K.outer_table(ev)
    reg = V.hv2(ev)
    q = st.t.ppf(0.975, 14)
    for _, r in reg.iterrows():
        m = mine[(mine.k == r.k) & (mine.reference == r.reference) & (mine.metric == "auroc")].iloc[0]
        assert np.isclose(m["diff"], r.auroc_diff)
        assert np.isclose(m.nb_ci_low, r.auroc_diff - q * r.auroc_se) and np.isclose(m.nb_ci_high, r.auroc_diff + q * r.auroc_se)
        ll = mine[(mine.k == r.k) & (mine.reference == r.reference) & (mine.metric == "logloss")].iloc[0]
        assert np.isclose(ll.nb_ci_low, r.logloss_ci_low) and np.isclose(ll.nb_ci_high, r.logloss_ci_high)


def _hv6_frames(gain):
    small = _sel({"mpqda_v2": (2, 5, 10), "mpqda_v2_no_gap": (2, 5, 10)}, seed=3,
                 shift={("mpqda_v2", k): gain[k] for k in (2, 5, 10)})
    c4 = _sel({"mpqda_v2": (20, 50), "mpqda_v2_no_gap": (20, 50)}, seed=4,
              shift={("mpqda_v2", k): gain[k] for k in (20, 50)})
    return small, c4


def test_hv6_detects_an_increasing_contribution():
    small, c4 = _hv6_frames({2: 0.0, 5: 0.002, 10: 0.005, 20: 0.009, 50: 0.015})
    a, page = K.hv6(K.hv6_diffs(small, c4))
    assert list(a.k) == [20, 50] and a.significant.all() and (a.family_size == 2).all()
    assert page.significant.iloc[0] and page.n_blocks.iloc[0] == 45
    assert K.hv6_reading(a, page) == "supported"


def test_hv6_reading_without_a_trend():
    small, c4 = _hv6_frames({2: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0})
    a, page = K.hv6(K.hv6_diffs(small, c4))
    assert K.hv6_reading(a, page) == "not supported"
    small, c4 = _hv6_frames({2: 0.015, 5: 0.009, 10: 0.005, 20: 0.002, 50: 0.0})     # giảm theo k
    a, page = K.hv6(K.hv6_diffs(small, c4))
    assert not page.significant.iloc[0]


def test_gate_r50_passes_only_on_identical_runs():
    c1 = _sel({"mpqda_v2": (2, 5, 10, 50)}, seed=5)
    c4 = c1[c1.k == 50].copy()
    genes = pd.DataFrame([{"cohort": r.cohort, "repeat": r.repeat, "fold": r.fold, "method": r.method, "k": 50, "gene": f"G{j}"}
                          for r in c4.itertuples() for j in range(3)])
    assert K.gate_r50(c1, c4, genes, genes)["pass"].all()
    bad = c4.copy()
    bad.loc[bad.index[0], "best_fitness"] += 1e-9
    assert not K.gate_r50(c1, bad, genes, genes)["pass"].all()
    assert not K.gate_r50(c1, c4.iloc[1:], genes, genes)["pass"].all()
    other = genes.copy()
    other.loc[other.index[0], "gene"] = "X"
    assert not K.gate_r50(c1, c4, genes, other)["pass"].all()


def test_build_end_to_end(tmp_path):
    res = tmp_path / "taimma"
    sel_t1 = _sel({"mpqda_de": K.ALL_K, "ga": K.ALL_K, "de_rand1": K.ALL_K}, seed=6)
    sel_v2 = _sel({"mpqda_v2": K.ALL_K, "de_g": K.ALL_K}, seed=7, shift={"mpqda_v2": 0.006, "de_g": 0.004})
    sel_ab = _sel({"mpqda_v2_no_gap": (2, 5, 10), "mpqda_v2_no_new": (2, 5, 10), "mpqda_v2_no_tour": (2, 5, 10)}, seed=8)
    for scen, frame in (("T1", sel_t1), ("T1v2", sel_v2), ("T2v2", sel_ab)):
        for c in COH:
            (res / scen / c).mkdir(parents=True, exist_ok=True)
            frame[frame.cohort == c].to_csv(res / scen / c / "selection_per_fold.csv", index=False)
    out = tmp_path / "out"
    info = K.build(res, out)
    assert {"corrected_inner", "decomposition_check", "runtime"} <= set(info["written"])
    inner = pd.read_csv(out / "corrected_inner.csv")
    assert set(inner.group) == {"HV1", "HV4", "E2", "D"}
    assert pd.read_csv(out / "decomposition_check.csv").ok.all()
    claims = pd.read_csv(out / "claims_corrected.csv")
    assert len(claims) and claims.claim_id.str.startswith("CORR:").any()
    assert "C4 chưa có" in info["status"]
