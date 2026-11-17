"""Test của scripts/dev_diagnostic.py (Review-2026-11-17, TQĐ31) trên kết quả DEV GIẢ có cài sẵn đảo chiều: M1 có log-loss
nội bộ thấp hơn v1 nhưng log-loss ngoài cao hơn."""

from pathlib import Path

import numpy as np
import pandas as pd

from mpqda import config as M  # noqa: E402

from scripts import dev_diagnostic as D  # noqa: E402

INNER = {"mpqda_v2_m1": 0.130, "mpqda_v2_m2": 0.132, "mpqda_v2_m3": 0.137, "mpqda_de": 0.150, "de_g": 0.138, "de_rand1": 0.147,
         "ga": 0.128}
OUTER = {"mpqda_v2_m1": 0.42, "mpqda_v2_m2": 0.39, "mpqda_v2_m3": 0.37, "mpqda_de": 0.35, "de_g": 0.41, "de_rand1": 0.38, "ga": 0.47}


def write_dev(taimma: Path, seed=0, with_wide=True):
    rng = np.random.default_rng(seed)
    units = {d: ([(0, f) for f in range(5)] if d.startswith("tcga") else [(r, f) for r in (0, 1) for f in range(5)])
             for d in M.DEV_DATASETS}
    for d, us in units.items():
        sel, genes, ev = [], [], {pb: [] for pb in D.PHASE_B}
        for r, f in us:
            for base, inner in INNER.items():
                for k in D.KS:
                    for out in M.output_names(base):
                        best = out.endswith(M.BEST_SUFFIX)
                        fit = -(inner - 0.002 * best + 0.003 * rng.random())
                        sel.append({"cohort": d, "repeat": r, "fold": f, "method": out, "k": k, "best_fitness": fit,
                                    "output_fitness": fit, "n_eval": 1300})
                        panel = sorted(rng.choice(40, size=k, replace=False))
                        genes += [{"cohort": d, "repeat": r, "fold": f, "method": out, "k": k, "rank": i + 1, "gene": f"G{g:03d}"}
                                  for i, g in enumerate(panel)]
                        for pb in D.PHASE_B:
                            ll = OUTER[base] + 0.03 * best + 0.01 * rng.random() + (0.02 if pb == "B-C1" else 0.0)
                            C = {"B-LL": 10.0, "B-C1": 1.0, "B-LL-wide": 1000.0 if base.startswith("mpqda_v2") else 100.0}[pb]
                            ev[pb].append({"cohort": d, "repeat": r, "seed": 42 + r, "fold": f, "method": out, "k": k, "model": "lr_l2",
                                           "auroc": 0.98 - ll / 5, "logloss": ll, "C": C})
        a = taimma / "DEV" / d
        a.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sel).to_csv(a / "selection_per_fold.csv", index=False)
        pd.DataFrame(genes).to_csv(a / "selected_genes.csv", index=False)
        for pb, sub in D.PHASE_B.items():
            if pb == "B-LL-wide" and not with_wide:
                continue
            e = taimma / sub / "DEV" / d
            e.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(ev[pb]).to_csv(e / "classification_per_fold.csv", index=False, encoding="utf-8-sig")


def test_diagnostic_finds_the_planted_inversion_and_is_reproducible(tmp_path):
    taimma = tmp_path / "taimma"
    write_dev(taimma)
    a, b = tmp_path / "a", tmp_path / "b"
    info = D.build(taimma, a)
    D.build(taimma, b)
    names = sorted(p.name for p in a.iterdir())
    assert all((a / n).read_bytes() == (b / n).read_bytes() for n in names)
    assert info["phase_b"] == ["B-C1", "B-LL", "B-LL-wide"]
    pairs = pd.read_csv(a / "dev_diag_pairs.csv")
    row = lambda q, pb="B-LL", pair=("mpqda_v2_m1", "mpqda_de"): pairs[(pairs.phase_b == pb) & (pairs.a == pair[0]) & (pairs.b == pair[1]) & (pairs.k == "5+10") & (pairs.quantity == q)].iloc[0]  # noqa: E731,E501
    assert row("inner_fitness").n == 50 and row("inner_fitness").mean_diff > 0 and row("inner_fitness").share_a_better == 1.0
    assert row("outer_logloss").mean_diff > 0 and row("outer_logloss").share_a_better < 0.5        # M1 kém hơn ở ngoài
    assert row("optimism", "B-C1").mean_diff > 0 and row("optimism", "B-C1").p_two_sided_exploratory < 0.05
    io = pd.read_csv(a / "dev_diag_inner_outer.csv")
    within = io[(io.phase_b == "B-LL") & (io.analysis == "within_unit_across_methods") & (io.cohort == "all")].iloc[0]
    assert within.n == 50 and within.rho_mean < 0 and within.share_rho_negative > 0.5
    means = io[(io.phase_b == "B-LL") & (io.analysis == "method_means") & (io.cohort == "all")].iloc[0]
    assert means.n == 7 and means.rho_mean < 0
    cc = pd.read_csv(a / "dev_diag_c_choice.csv")
    m1 = cc[(cc.method == "mpqda_v2_m1") & (cc.k == 5)].set_index("phase_b")
    assert m1.share_at_upper["B-LL"] == 1.0 and m1.share_at_upper["B-LL-wide"] == 1.0
    assert cc[(cc.method == "mpqda_de") & (cc.phase_b == "B-LL-wide")].share_at_upper.eq(0.0).all()
    ov = pd.read_csv(a / "dev_diag_overlap.csv")
    assert ov.jaccard_mean.between(0, 1).all() and set(ov.k) == {5, 10}
    by = pd.read_csv(a / "dev_diag_by_dataset.csv")
    assert set(by.cohort) == {*M.DEV_DATASETS, "all"} and by[by.cohort == "all"].n.isin([50 // 2]).all()


def test_diagnostic_runs_without_b_ll_wide(tmp_path):
    taimma = tmp_path / "taimma"
    write_dev(taimma, with_wide=False)
    info = D.build(taimma, tmp_path / "out")
    assert info["phase_b"] == ["B-C1", "B-LL"]
    cc = pd.read_csv(tmp_path / "out" / "dev_diag_c_choice.csv")
    assert set(cc.phase_b) == {"B-LL"}
