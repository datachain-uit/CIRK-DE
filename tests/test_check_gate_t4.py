"""Test của scripts/check_gate_t4.py (Review-2026-11-16, TQĐ27) trên tệp giả."""

import numpy as np
import pandas as pd

from scripts import check_gate_t4 as G  # noqa: E402


def fake(dataset, repeats=range(5), seed=0):
    rng = np.random.default_rng(seed)
    genes, metrics = [], []
    for r in repeats:
        for f in range(5):
            for m in (*G.T4_METHODS, "relieff"):
                for k in (2, 5, 10, 50, 25):
                    panel = rng.choice(500, size=k, replace=False)
                    genes += [{"cohort": dataset, "repeat": r, "fold": f, "method": m, "k": k, "rank": i + 1,
                               "gene": f"G{g}"} for i, g in enumerate(panel)]
                    metrics.append({"cohort": dataset, "repeat": r, "seed": 42 + r, "fold": f, "method": m, "k": k,
                                    "model": "lr_l2", "auroc": rng.random(), "bacc": rng.random(), "mcc": rng.random(),
                                    "logloss": rng.random(), "brier": rng.random(), "threshold": rng.random(),
                                    "C": 0.01, "fit_time_s": rng.random()})
            metrics.append({"cohort": dataset, "repeat": r, "seed": 42 + r, "fold": f, "method": "b0", "k": 16479,
                            "model": "lr_l2", "auroc": rng.random(), "bacc": 0.5, "mcc": 0.0, "logloss": 0.69,
                            "brier": 0.25, "threshold": 0.5, "C": 0.01, "fit_time_s": 2.0})
            for k in (2, 5, 10, 50):
                metrics.append({"cohort": dataset, "repeat": r, "seed": 42 + r, "fold": f, "method": "random_k", "k": k,
                                "model": "lr_l2", "auroc": rng.random(), "bacc": 0.5, "mcc": 0.0, "logloss": 0.6,
                                "brier": 0.2, "threshold": 0.5, "C": np.nan, "fit_time_s": 1.0})
    return pd.DataFrame(genes), pd.DataFrame(metrics)


def write(root_t, root_s, dataset, genes, metrics, t_genes=None, t_metrics=None):
    sel_s, clf_s = G.survey_dirs(root_s, dataset)
    for p in (sel_s, clf_s, root_t / "T4" / dataset, root_t / "T5" / "T4" / dataset):
        p.mkdir(parents=True, exist_ok=True)
    genes.to_csv(sel_s / "selected_genes.csv", index=False)
    metrics.to_csv(clf_s / "classification_per_fold.csv", index=False)
    tg = genes[genes["repeat"].isin(G.REPEATS)] if t_genes is None else t_genes
    tm = metrics[metrics["repeat"].isin(G.REPEATS)] if t_metrics is None else t_metrics
    tg.to_csv(root_t / "T4" / dataset / "selected_genes.csv", index=False)
    tm.assign(fit_time_s=tm["fit_time_s"] * 3).to_csv(root_t / "T5" / "T4" / dataset / "classification_per_fold.csv",
                                                      index=False)


def test_identical_runs_pass_for_gpl570_and_b7(tmp_path):
    for d in ("gse10245", "gse31210_egfr"):
        genes, metrics = fake(d)
        write(tmp_path / "taimma", tmp_path / "survey", d, genes, metrics)
        row, diff = G.check_dataset(tmp_path / "taimma", tmp_path / "survey", d)
        assert row["pass"] and diff.empty and row["n_panels"] == 3 * 5 * 6 * 4
        assert row["n_metric_rows"] == 3 * 5 * (6 * 4 + 1 + 4)


def test_one_gene_or_one_metric_differs(tmp_path):
    d = "tcga_luad_egfr"
    genes, metrics = fake(d, seed=1)
    tg = genes[genes["repeat"].isin(G.REPEATS)].copy()
    tg.iloc[0, tg.columns.get_loc("gene")] = "OTHER"
    tm = metrics[metrics["repeat"].isin(G.REPEATS)].copy()
    tm.iloc[5, tm.columns.get_loc("auroc")] += 1e-9
    write(tmp_path / "taimma", tmp_path / "survey", d, genes, metrics, tg, tm)
    row, diff = G.check_dataset(tmp_path / "taimma", tmp_path / "survey", d)
    assert not row["pass"] and set(diff["part"]) == {"genes", "metrics"} and len(diff) == 2


def test_missing_rows_fail(tmp_path):
    d = "gse37745"
    genes, metrics = fake(d, seed=2)
    tm = metrics[metrics["repeat"].isin(G.REPEATS)].iloc[1:]
    write(tmp_path / "taimma", tmp_path / "survey", d, genes, metrics, None, tm)
    row, _ = G.check_dataset(tmp_path / "taimma", tmp_path / "survey", d)
    assert not row["pass"] and row["n_missing_metric_rows"] == 1


def test_other_models_of_the_survey_are_out_of_scope(tmp_path):
    d = "gse30219"
    genes, metrics = fake(d, seed=3)
    extra = metrics[metrics["method"] == "l1_lr"].assign(model="rf")
    write(tmp_path / "taimma", tmp_path / "survey", d, genes, pd.concat([metrics, extra]), None,
          metrics[metrics["repeat"].isin(G.REPEATS)])
    row, diff = G.check_dataset(tmp_path / "taimma", tmp_path / "survey", d)
    assert row["pass"] and diff.empty
