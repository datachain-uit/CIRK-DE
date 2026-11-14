"""make_taimma_tables.py trên kết quả GIẢ (số ngẫu nhiên, không phải kết quả thực nghiệm)."""
import numpy as np
import pandas as pd

from mpqda import config as M
from scripts import make_taimma_tables as T
from survey_v2.stats import nadeau_bengio_ttest

COHORTS = ["gse10245", "gse30219", "gse37745"]
T1_METHODS = ["mpqda_de", "ga", "bpso", "de_rand1", "shade", "qlde"]


def fake_results(root, seed=0):
    """Tệp theo đúng tên và cột mà bảng đọc: selection_per_fold, selected_genes, history (pha A) và classification_per_fold (pha B)."""
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
                                       "model": "lr_l2", "auroc": float(rng.uniform(0.9, 1.0)), "logloss": float(rng.uniform(0.1, 0.4))})
                        if with_history:
                            hist += [{"cohort": cohort, "repeat": repeat, "fold": fold, "method": m, "k": k, "t": t,
                                      "n_eval": 30 * (t + 1), "best_fitness": best - 0.1 / (t + 1)} for t in range(4)]
        d = root / scenario / cohort
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sel).to_csv(d / "selection_per_fold.csv", index=False)
        pd.DataFrame(genes).to_csv(d / "selected_genes.csv", index=False)
        if with_history:
            pd.DataFrame(hist).to_csv(d / "history.csv", index=False)
        e = root / "T5" / scenario / cohort
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
    assert all((first / n).read_bytes() == (second / n).read_bytes() for n in names)       # kể cả hai hình PDF
    assert {"table3_main.tex", "table4_ablation.tex", "table5_seed_stability.tex", "fig2_anytime.pdf",
            "fig3_critical_difference.pdf", "claims.csv", "inputs_sha256.csv", "stats_h1_saturation.csv",
            "stats_h2_inner_fitness.csv", "stats_h3_noninferiority.csv", "stats_h4_stability.csv"} <= set(names)
    assert info["missing"] == []
    inputs = pd.read_csv(first / "inputs_sha256.csv")
    assert len(inputs) == 3 * 3 + 3 * 3 + 3 + 4 + 1 and inputs.sha256.str.len().eq(64).all()
    claims = pd.read_csv(first / "claims.csv")
    assert list(claims.columns) == ["claim_id", "value", "file", "column", "filter"] and claims.claim_id.is_unique
    h1 = pd.read_csv(first / "stats_h1_saturation.csv")
    assert h1.n_runs.tolist() == [270, 270, 270, 540]          # 15 fold × 6 phương pháp × 3 k: mỗi lần chạy MPQDA tính một lần
    h2 = pd.read_csv(first / "stats_h2_inner_fitness.csv")
    assert len(h2) == 6 and h2.n.eq(15).all() and (h2.p_holm >= h2.p).all()
    h3 = pd.read_csv(first / "stats_h3_noninferiority.csv")
    assert h3.k.tolist() == [5, 10] and h3.comparator.isin(T.BASELINE_ORDER).all()
    h5 = pd.read_csv(first / "table4_ablation.csv")
    assert (h5.family == "main").sum() == 33 and (h5.family == "a0").sum() == 6
    h4 = pd.read_csv(first / "stats_h4_stability.csv")
    assert len(h4) == 6 and h4.n.eq(15).all()


def test_missing_scenarios_are_reported_not_fatal(tmp_path):
    info = T.build(tmp_path / "empty", tmp_path / "out")
    assert info["written"] == [] and info["missing"]
    assert (tmp_path / "out" / "missing_inputs.txt").read_text(encoding="utf-8").strip()
