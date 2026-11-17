"""Test của scripts/b7_confounder_check.py (Review-2026-11-16, TQĐ27) trên dữ liệu giả."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survey_v2 import config as C  # noqa: E402

from scripts import b7_confounder_check as B  # noqa: E402


def write_genes(path: Path, cohort: str, panels: dict):
    rows = []
    for (method, k, rep, fold), genes in panels.items():
        rows += [{"cohort": cohort, "repeat": rep, "fold": fold, "method": method, "k": k, "rank": r, "gene": g}
                 for r, g in enumerate(genes, 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_counts_sex_genes_and_egfr(tmp_path):
    res = tmp_path / "taimma"
    write_genes(res / "T1" / "tcga_luad_egfr" / "selected_genes.csv", "tcga_luad_egfr", {
        ("ga", 5, 0, 0): ["EGFR", "A", "B", "C", "D"],
        ("ga", 5, 0, 1): ["XIST", "A", "B", "C", "D"],
        ("ga", 10, 0, 0): [f"G{i}" for i in range(10)],
        ("mpqda_de", 5, 0, 0): ["EGFR", "RPS4Y1", "B", "C", "D"],
        ("mpqda_de", 2, 0, 0): ["XIST", "EGFR"],                  # k = 2 không kiểm
    })
    write_genes(res / "T1" / "gse31210_egfr" / "selected_genes.csv", "gse31210_egfr", {
        ("ga", 5, 0, 0): ["A", "B", "C", "D", "E"]})
    table, info = B.run(res, ["T1"], None)
    assert info == {}
    row = table.set_index(["cohort", "method", "k"])
    ga5 = row.loc[("tcga_luad_egfr", "ga", 5)]
    assert ga5.n_panels == 2 and ga5.n_with_sex_gene == 1 and ga5.n_with_egfr == 1 and ga5.sex_genes_seen == "XIST"
    m5 = row.loc[("tcga_luad_egfr", "mpqda_de", 5)]
    assert m5.n_with_sex_gene == 1 and m5.n_with_egfr == 1 and m5.sex_genes_seen == "RPS4Y1"
    assert ("tcga_luad_egfr", "mpqda_de", 2) not in row.index
    assert row.loc[("tcga_luad_egfr", "ga", 10)].n_with_sex_gene == 0
    assert row.loc[("gse31210_egfr", "ga", 5)].share_with_egfr == 0.0


def test_dataset_checks_on_fake_metadata(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.15).astype(int)
    sex = np.where(rng.random(n) < np.where(y == 1, 0.8, 0.45), "female", "male")
    tss = rng.choice(["05", "38", "44", "55", "64"], size=n)
    meta = pd.DataFrame({"sample_id": [f"TCGA-{t}-{i:04d}-01A" for i, t in enumerate(tss)], "label": y, "sex": sex})
    d = tmp_path / C.dataset_dir("tcga_luad_egfr")
    d.mkdir(parents=True)
    meta.to_csv(d / "metadata.csv", index=False)
    res = tmp_path / "taimma"
    write_genes(res / "T1" / "tcga_luad_egfr" / "selected_genes.csv", "tcga_luad_egfr", {("ga", 5, 0, 0): list("ABCDE")})
    _, info = B.run(res, ["T1"], tmp_path)
    assert info["n"] == n and info["n_positive"] == int(y.sum())
    assert 0.6 < info["sex_only_auroc"] < 0.8                      # tín hiệu giới tính cài sẵn
    assert info["tss_folds"] == 25 and 0.3 < info["tss_only_auroc_mean"] < 0.7     # TSS ngẫu nhiên


def test_dataset_checks_also_on_gse31210(tmp_path):
    rng = np.random.default_rng(1)
    n = 224
    y = (rng.random(n) < 0.56).astype(int)
    sex = np.where(rng.random(n) < np.where(y == 1, 0.75, 0.35), "female", "male")
    meta = pd.DataFrame({"sample_id": [f"GSM{773000 + i}" for i in range(n)], "label": y, "sex": sex})
    d = tmp_path / C.dataset_dir("gse31210_egfr")
    d.mkdir(parents=True)
    meta.to_csv(d / "metadata.csv", index=False)
    res = tmp_path / "taimma"
    write_genes(res / "T1v2" / "gse31210_egfr" / "selected_genes.csv", "gse31210_egfr", {("mpqda_v2", 5, 0, 0): list("ABCDE")})
    table, info = B.run(res, ["T1v2"], tmp_path)
    assert set(table.scenario) == {"T1v2"} and "n" not in info              # không có metadata của tcga_luad_egfr
    g = info["gse31210_egfr"]
    assert g["n"] == n and 0.6 < g["sex_only_auroc"] < 0.85 and "tss_folds" not in g   # mã GSM không có TSS


def test_missing_results_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        B.run(tmp_path, ["T1"], None)
