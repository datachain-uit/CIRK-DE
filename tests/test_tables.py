"""make_taimma_tables.py on SYNTHETIC results (random numbers, not experimental results)."""
import shutil

import numpy as np
import pandas as pd

from mpqda import config as M
from scripts import make_taimma_tables as T
from survey_v2.stats import nadeau_bengio_ttest

COHORTS = ["gse10245", "gse30219", "gse37745"]
T1_METHODS = ["mpqda_de", "ga", "bpso", "de_rand1", "shade", "qlde"]


def fake_results(root, seed=0):
    """Files with exactly the names and columns read by the tables: selection_per_fold, selected_genes, history (phase A) and classification_per_fold (phase B).
    T5LL (B-LL) and T5 (B-AUC) are the same copy: the main tables and the *_bauc tables must be identical."""
    rng = np.random.default_rng(seed)

    def write(scenario, cohort, methods, ks, metric="logloss", with_history=False):
        sel, genes, ev, hist = [], [], [], []
        for repeat in range(3):
            for fold in range(5):
                for m in methods:
                    for k in ks:
                        best = 1.0 if metric == "auroc" and rng.random() < 0.95 else float(-rng.uniform(0.2, 0.6))
                        for out in M.output_names(m):
                            sel.append({"cohort": cohort, "repeat": repeat, "fold": fold, "method": out, "k": k,
                                        "best_fitness": best, "output_fitness": best, "n_eval": 1500})
                            panel = rng.choice(300, size=k, replace=False)
                            genes += [{"cohort": cohort, "repeat": repeat, "fold": fold, "method": out, "k": k, "rank": r,
                                       "gene": f"G{g:04d}"} for r, g in enumerate(sorted(panel), 1)]
                            ev.append({"cohort": cohort, "repeat": repeat, "seed": 42 + repeat, "fold": fold, "method": out, "k": k,
                                       "model": "lr_l2", "auroc": float(rng.uniform(0.9, 1.0)), "logloss": float(rng.uniform(0.1, 0.4)),
                                       "C": float(rng.choice([0.01, 0.1, 1.0, 10.0]))})
                        if with_history:
                            hist += [{"cohort": cohort, "repeat": repeat, "fold": fold, "method": m, "k": k, "t": t,
                                      "n_eval": 30 * (t + 1), "best_fitness": best - 0.1 / (t + 1)} for t in range(4)]
        d = root / scenario / cohort
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sel).to_csv(d / "selection_per_fold.csv", index=False)
        pd.DataFrame(genes).to_csv(d / "selected_genes.csv", index=False)
        if with_history:
            pd.DataFrame(hist).to_csv(d / "history.csv", index=False)
        for phase_dir in ("T5LL", "T5"):
            e = root / phase_dir / scenario / cohort
            e.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(ev).to_csv(e / "classification_per_fold.csv", index=False, encoding="utf-8-sig")

    for c in COHORTS:
        write("T1", c, T1_METHODS, (2, 5, 10, 50), with_history=True)
        write("T1b", c, T1_METHODS, (5, 10, 50), metric="auroc")
        write("T2", c, list(M.VARIANTS), (2, 5, 10))
    for s in (1, 2, 3, 4):
        write(f"T3_s{s}", "gse37745", T1_METHODS, (5, 10))


def test_pooled_nb_single_cohort_matches_survey():
    d = np.random.default_rng(12).normal(0.01, 0.03, size=15)
    a, b = T.pooled_nb([d]), nadeau_bengio_ttest(d, alternative="two-sided")
    assert np.isclose(a["ci_high"], b["ci_high"]) and np.isclose(a["ci_low"], b["ci_low"]) and np.isclose(a["se"], b["se_nb"])
    assert a["df"] == 14 and np.isclose(a["mean"], b["mean"])
    assert np.isclose(T.pooled_p_greater(a), nadeau_bengio_ttest(d, alternative="greater")["p"])
    three = T.pooled_nb([d, d + 0.01, d - 0.01])
    assert np.isclose(three["mean"], d.mean()) and np.isclose(three["se"], a["se"] / np.sqrt(3)) and three["df"] == 14


def test_wilcoxon_all_zero_differences():
    assert T.wilcoxon_p(np.zeros(15), "greater") == 1.0
    assert T.wilcoxon_p(np.arange(1, 16, dtype=float), "greater") < 0.001


def test_tables_are_byte_identical_and_complete(tmp_path):
    fake_results(tmp_path / "results")
    first, second = tmp_path / "a", tmp_path / "b"
    info = T.build(tmp_path / "results", first)
    T.build(tmp_path / "results", second)
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    assert all((first / n).read_bytes() == (second / n).read_bytes() for n in names)       # the two PDF figures included
    assert {"table3_main.tex", "table4_ablation.tex", "table5_seed_stability.tex", "fig2_anytime.pdf",
            "fig3_critical_difference.pdf", "claims.csv", "inputs_sha256.csv", "stats_h1_saturation.csv",
            "stats_h2_inner_fitness.csv", "stats_h3_noninferiority.csv", "stats_h4_stability.csv", "table3_main_bauc.csv",
            "stats_h3_noninferiority_bauc.csv", "stats_secondary_pairwise_bauc.csv", "table_s_c_choice.csv"} <= set(names)
    assert info["missing"] == []
    inputs = pd.read_csv(first / "inputs_sha256.csv")
    # phase A of T1, T1b, T2; B-LL of T1, T1b, T2; B-AUC of T1; history T1; selected_genes T1 (GSE37745) and T3_s1..4
    assert len(inputs) == 3 * 3 + 3 * 3 + 3 + 3 + 4 + 1 and inputs.sha256.str.len().eq(64).all()
    claims = pd.read_csv(first / "claims.csv")
    assert list(claims.columns) == ["claim_id", "value", "file", "column", "filter", "phase_b"] and claims.claim_id.is_unique
    h1 = pd.read_csv(first / "stats_h1_saturation.csv")
    assert h1.n_runs.tolist() == [270, 270, 270, 540]          # 15 folds × 6 methods × 3 k: each MPQDA run counted once
    h2 = pd.read_csv(first / "stats_h2_inner_fitness.csv")
    assert len(h2) == 6 and h2.n.eq(15).all() and (h2.p_holm >= h2.p).all()
    h3 = pd.read_csv(first / "stats_h3_noninferiority.csv")
    assert h3.k.tolist() == [5, 10] and h3.comparator.isin(T.BASELINE_ORDER).all()
    h5 = pd.read_csv(first / "table4_ablation.csv")
    assert (h5.family == "main").sum() == 39 and set(h5.family) == {"main", "descriptive"}
    assert (h5[h5.variant == "mpqda_a0_noband"].component == "band re-encoding").all()
    assert h5[h5.family == "main"].p_holm.notna().all() and h5[h5.family == "descriptive"].p_holm.isna().all()
    h4 = pd.read_csv(first / "stats_h4_stability.csv")
    assert len(h4) == 6 and h4.n.eq(15).all()


def test_bauc_tables_match_main_tables_when_phase_b_is_identical(tmp_path):
    """T5LL is a copy of T5: the main tables and the *_bauc tables are identical; claims.csv records the phase B of each number."""
    fake_results(tmp_path / "results")
    out = tmp_path / "out"
    T.build(tmp_path / "results", out)
    for name in ("table3_main", "stats_h3_noninferiority", "stats_secondary_pairwise"):
        assert (out / f"{name}.csv").read_bytes() == (out / f"{name}_bauc.csv").read_bytes()
    claims = pd.read_csv(out / "claims.csv")
    assert claims.phase_b.notna().all()
    tag = claims.claim_id.str.split(":").str[0]
    assert (claims[tag.isin(["T3", "H3", "FRIEDMAN"])].phase_b == "B-LL").all()
    assert (claims[tag.isin(["T3_BAUC", "H3_BAUC"])].phase_b == "B-AUC").all() and tag.isin(["T3_BAUC", "H3_BAUC"]).any()
    assert (claims[tag.isin(["H1", "H2", "T5"])].phase_b == "phase A").all()
    h5 = claims[tag == "H5"]
    assert set(h5.phase_b) == {"B-LL", "phase A"}
    assert (h5[h5["filter"].str.contains("question=Q2")].phase_b == "B-LL").all()
    cc = pd.read_csv(out / "table_s_c_choice.csv")
    assert set(cc.phase_b) == {"B-LL", "B-AUC"} and set(cc.C) == {0.01, 0.1, 1.0, 10.0}
    sums = cc.groupby(["phase_b", "scenario", "method", "k"]).n_with_C.sum()
    n = cc.groupby(["phase_b", "scenario", "method", "k"]).n_panels.first()
    assert (sums == n).all() and (n == 45).all()


def test_t4_c_choice_is_read_when_present(tmp_path):
    fake_results(tmp_path / "results")
    for phase_dir in ("T5LL", "T5"):
        for c in COHORTS:
            src = tmp_path / "results" / phase_dir / "T1" / c
            shutil.copytree(src, tmp_path / "results" / phase_dir / "T4" / c)
    T.build(tmp_path / "results", tmp_path / "out")
    cc = pd.read_csv(tmp_path / "out" / "table_s_c_choice.csv")
    assert set(cc.scenario) == {"T1", "T4"}


def fake_b7(root, seed=1):
    """Synthetic phase B-LL for the B7 datasets: classification_per_fold.csv and e1_scores.csv (per-sample scores for AUPRC)."""
    rng = np.random.default_rng(seed)
    for cohort in M.B7_INCLUDED:
        ev, sc = [], []
        for repeat in range(3):
            for fold in range(5):
                for m in T1_METHODS:
                    for out in M.output_names(m):
                        for k in (5, 10):
                            ev.append({"cohort": cohort, "repeat": repeat, "seed": 42 + repeat, "fold": fold, "method": out,
                                       "k": k, "model": "lr_l2", "auroc": float(rng.uniform(0.7, 0.9)),
                                       "logloss": float(rng.uniform(0.3, 0.6)), "C": 1.0})
                            y = rng.integers(0, 2, 20)
                            y[:2] = [0, 1]
                            sc += [{"cohort": cohort, "repeat": repeat, "fold": fold, "method": out, "k": k,
                                    "sample_id": f"S{i}", "label": int(l), "score": float(rng.random())} for i, l in enumerate(y)]
        d = root / "T5LL" / "T1" / cohort
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ev).to_csv(d / "classification_per_fold.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(sc).to_csv(d / "e1_scores.csv", index=False)


def test_b7_table_is_separate_and_has_auprc(tmp_path):
    """PREREG, Section 8: separate table for B7 (secondary comparison, Holm over the baselines, AUPRC); not in H1–H5 and Table 3."""
    from sklearn.metrics import average_precision_score
    fake_results(tmp_path / "results")
    fake_b7(tmp_path / "results")
    out = tmp_path / "out"
    info = T.build(tmp_path / "results", out)
    assert {"table_b7_main", "stats_b7_secondary"} <= set(info["written"]) and info["missing"] == []
    t = pd.read_csv(out / "table_b7_main.csv")
    assert set(t.cohort) == set(M.B7_INCLUDED) and set(t.k) == {5, 10} and t.auprc_mean.between(0, 1).all()
    sc = pd.read_csv(tmp_path / "results" / "T5LL" / "T1" / M.B7_INCLUDED[0] / "e1_scores.csv")
    g = sc[(sc.method == "ga") & (sc.k == 5)]
    want = np.mean([average_precision_score(f.label, f.score) for _, f in g.groupby(["repeat", "fold"])])
    got = t[(t.cohort == M.B7_INCLUDED[0]) & (t.method == "ga") & (t.k == 5)].auprc_mean.iloc[0]
    assert np.isclose(got, want)
    s = pd.read_csv(out / "stats_b7_secondary.csv")
    assert len(s) == len(M.B7_INCLUDED) * 2 * (len(T1_METHODS) - 1) and (s.p_nb_holm >= s.p_nb).all()
    assert not set(pd.read_csv(out / "table3_main.csv").cohort) & set(M.B7_INCLUDED)
    claims = pd.read_csv(out / "claims.csv")
    tag = claims.claim_id.str.split(":").str[0]
    assert (claims[tag.isin(["B7", "B7_SEC"])].phase_b == "B-LL").all() and tag.isin(["B7", "B7_SEC"]).any()


def test_missing_scenarios_are_reported_not_fatal(tmp_path):
    info = T.build(tmp_path / "empty", tmp_path / "out")
    assert info["written"] == [] and info["missing"]
    assert (tmp_path / "out" / "missing_inputs.txt").read_text(encoding="utf-8").strip()
