"""scripts/make_taimma_v2_tables.py on SYNTHETIC results (random numbers): every table HV1–HV4, the right number of tests, byte-identical rerun,
and the tables of v1 (make_taimma_tables.py) unchanged when v2 results are added."""
import numpy as np
import pandas as pd

from mpqda import config as M
from scripts import make_taimma_tables as T
from scripts import make_taimma_v2_tables as V
from test_tables import COHORTS, fake_results

B7 = list(M.B7_INCLUDED)


def write_v2(root, seed=7):
    rng = np.random.default_rng(seed)

    def scen(name, cohorts, methods, ks, phase_dirs=("T5LL", "T5", "T5C1", "T5LLW"), history=False):
        for c in cohorts:
            sel, genes, ev, hist, sc = [], [], [], [], []
            for repeat in range(3):
                for fold in range(5):
                    for m in methods:
                        for k in ks:
                            best = float(-rng.uniform(0.15, 0.5))
                            for out in M.output_names(m):
                                sel.append({"cohort": c, "repeat": repeat, "fold": fold, "method": out, "k": k, "best_fitness": best,
                                            "output_fitness": best, "n_eval": 1400})
                                panel = rng.choice(300, size=k, replace=False)
                                genes += [{"cohort": c, "repeat": repeat, "fold": fold, "method": out, "k": k, "rank": r,
                                           "gene": f"G{g:04d}"} for r, g in enumerate(sorted(panel), 1)]
                                ev.append({"cohort": c, "repeat": repeat, "seed": 42 + repeat, "fold": fold, "method": out, "k": k,
                                           "model": "lr_l2", "auroc": float(rng.uniform(0.85, 1.0)),
                                           "logloss": float(rng.uniform(0.1, 0.5)), "C": 10.0})
                                y = rng.integers(0, 2, 12)
                                y[:2] = [0, 1]
                                sc += [{"cohort": c, "repeat": repeat, "fold": fold, "method": out, "k": k, "sample_id": f"S{i}",
                                        "label": int(v), "score": float(rng.random())} for i, v in enumerate(y)]
                            if history:
                                hist += [{"cohort": c, "repeat": repeat, "fold": fold, "method": m, "k": k, "t": t,
                                          "best_fitness": best, "leader_overlap": 0.7, "success_rate": 0.2, "n_new_genes": 1.0,
                                          "diversity": 0.5, "exploration_rate": 0.8} for t in range(3)]
            d = root / name / c
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(sel).to_csv(d / "selection_per_fold.csv", index=False)
            pd.DataFrame(genes).to_csv(d / "selected_genes.csv", index=False)
            if history:
                pd.DataFrame(hist).to_csv(d / "history.csv", index=False)
            for pd_ in phase_dirs:
                e = root / pd_ / name / c
                e.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(ev).to_csv(e / "classification_per_fold.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(sc).to_csv(e / "e1_scores.csv", index=False)

    scen("T1v2", COHORTS, ["mpqda_v2", "de_g"], (2, 5, 10, 50), history=True)
    scen("T1v2", B7, ["mpqda_v2", "de_g"], (5, 10))
    scen("T2v2", COHORTS, ["mpqda_v2_no_gap", "mpqda_v2_no_new", "mpqda_v2_no_tour"], (2, 5, 10))
    for s in (1, 2, 3, 4):
        scen(f"T3v2_s{s}", ["gse37745"], ["mpqda_v2"], (5, 10), phase_dirs=("T5LL",))


def test_v2_tables_complete_and_reproducible(tmp_path):
    root = tmp_path / "results"
    fake_results(root)
    write_v2(root)
    for c in B7:                                   # v1 on B7 (the B7 table needs both v1 and v2)
        for sub in ("T1",):
            src = root / "T5LL" / sub / COHORTS[0]
            dst = root / "T5LL" / sub / c
            dst.mkdir(parents=True, exist_ok=True)
            ev = pd.read_csv(src / "classification_per_fold.csv", encoding="utf-8-sig").assign(cohort=c)
            ev.to_csv(dst / "classification_per_fold.csv", index=False, encoding="utf-8-sig")
    a, b = tmp_path / "a", tmp_path / "b"
    info = V.build(root, a)
    V.build(root, b)
    names = sorted(p.name for p in a.iterdir())
    assert all((a / n).read_bytes() == (b / n).read_bytes() for n in names)
    assert {"stats_hv1_search", "stats_hv2_generalization", "stats_hv2_generalization_bauc", "stats_hv3_stability",
            "table_v2_ablation", "table_v2_main", "table_v2_main_bc1", "stats_v2_friedman_ranks", "diagnostics_v2_behaviour",
            "table_b7_v2", "table_v2_main_bllw", "stats_hv5_b7", "diagnostics_v2_c_choice"} <= set(info["written"])
    assert not {"stats_hv2_generalization_bc1", "stats_hv2_generalization_bllw"} & set(info["written"])   # job v1b not run
    hv1 = pd.read_csv(a / "stats_hv1_search.csv")
    assert len(hv1) == 4 and set(hv1.reference) == {"mpqda_de", "ga"} and hv1.n.eq(45).all() and (hv1.p_holm >= hv1.p).all()
    assert (hv1.ci_low <= hv1.mean_diff).all() and (hv1.mean_diff <= hv1.ci_high).all()
    hv2 = pd.read_csv(a / "stats_hv2_generalization.csv")
    assert len(hv2) == 4 and set(hv2.reference) == {"bpso", "mpqda_de"}
    hv3 = pd.read_csv(a / "stats_hv3_stability.csv")
    assert len(hv3) == 4 and hv3.n.eq(15).all()
    hv4 = pd.read_csv(a / "table_v2_ablation.csv")
    assert len(hv4) == 9 and hv4.n.eq(45).all() and "outer_logloss_ablation_minus_v2" in hv4
    fr = pd.read_csv(a / "stats_v2_friedman_ranks.csv")
    assert set(fr.method) >= {"mpqda_v2", "de_g", "mpqda_de"}
    claims = pd.read_csv(a / "claims_v2.csv")
    assert claims.claim_id.is_unique and claims.phase_b.notna().all()


def test_v1_tables_unchanged_by_v2_results(tmp_path):
    root = tmp_path / "results"
    fake_results(root)
    before = tmp_path / "before"
    T.build(root, before)
    write_v2(root)
    after = tmp_path / "after"
    T.build(root, after)
    for p in before.iterdir():
        assert (after / p.name).read_bytes() == p.read_bytes(), p.name
