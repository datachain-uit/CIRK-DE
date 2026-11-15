import shutil

import pandas as pd

from conftest import ROOT
from scripts import check_gate_gt1 as G

REF = ROOT / "vendor" / "reference" / "E1_wrapper_feature_selection"
COHORT = "gse10245"


def copy_reference(tmp_path):
    run = tmp_path / "run" / COHORT
    run.mkdir(parents=True)
    for name in ("selected_genes.csv", "selection_per_fold.csv"):
        shutil.copy(REF / COHORT / name, run / name)
    return tmp_path / "run"


def test_returns_0_when_files_are_identical(tmp_path):
    root = copy_reference(tmp_path)
    table, n = G.compare(root / COHORT, REF / COHORT, COHORT)
    assert table.empty and n == 15 * 2 * 3                     # 15 fold × (GA, BPSO) × k ∈ {5, 10, 50}
    assert G.main(["--run-root", str(root), "--datasets", COHORT]) == 0


def test_returns_2_when_one_gene_differs(tmp_path):
    root = copy_reference(tmp_path)
    path = root / COHORT / "selected_genes.csv"
    genes = pd.read_csv(path)
    genes.loc[0, "gene"] = "NOT_A_GENE"
    genes.to_csv(path, index=False)
    table, _ = G.compare(root / COHORT, REF / COHORT, COHORT)
    assert len(table) == 1 and table.field.tolist() == ["genes"] and table.run.tolist() == ["NOT_A_GENE"]
    report = tmp_path / "gate.csv"
    assert G.main(["--run-root", str(root), "--datasets", COHORT, "--report", str(report)]) == 2
    assert len(pd.read_csv(report)) == 1


def test_n_eval_exact_and_fitness_beyond_tolerance_fail(tmp_path):
    root = copy_reference(tmp_path)
    path = root / COHORT / "selection_per_fold.csv"
    sel = pd.read_csv(path, float_precision="round_trip")
    sel.loc[0, "best_fitness"] = float(sel.loc[0, "best_fitness"]) - 1e-9
    sel.loc[1, "n_eval"] = int(sel.loc[1, "n_eval"]) + 1
    sel.to_csv(path, index=False)
    table, _ = G.compare(root / COHORT, REF / COHORT, COHORT)
    assert sorted(table.field) == ["best_fitness", "n_eval"]
    assert G.main(["--run-root", str(root), "--datasets", COHORT]) == 2


def test_last_bit_fitness_difference_is_reported_but_passes(tmp_path):
    """Trường hợp gặp trên Kaggle ngày 15/11/2026: cùng tập gen và n_eval, best_fitness khác ở bit cuối."""
    import numpy as np
    root = copy_reference(tmp_path)
    path = root / COHORT / "selection_per_fold.csv"
    sel = pd.read_csv(path, float_precision="round_trip")
    i = int(np.flatnonzero(sel.method.isin(["ga", "bpso"]).to_numpy())[0])
    sel.loc[i, "best_fitness"] = float(np.nextafter(float(sel.loc[i, "best_fitness"]), -1.0))
    sel.to_csv(path, index=False)
    near = []
    table, _ = G.compare(root / COHORT, REF / COHORT, COHORT, within_tol=near)
    assert table.empty and len(near) == 1 and 0 < near[0]["abs_diff"] <= G.FITNESS_TOL
    report = tmp_path / "gate.csv"
    assert G.main(["--run-root", str(root), "--datasets", COHORT, "--report", str(report)]) == 0
    assert pd.read_csv(report).empty and len(pd.read_csv(tmp_path / "gate_within_tol.csv")) == 1
    from mpqda import config as M
    assert G.FITNESS_TOL == M.SATURATION_EPS


def test_subset_of_units_passes_and_empty_run_fails(tmp_path):
    """Phiên S0 chỉ chạy lần lặp 0: chỉ các đơn vị đã chạy được so. Không có đơn vị nào thì cổng không đạt."""
    root = copy_reference(tmp_path)
    for name in ("selected_genes.csv", "selection_per_fold.csv"):
        frame = pd.read_csv(root / COHORT / name, float_precision="round_trip")
        frame[frame.repeat == 0].to_csv(root / COHORT / name, index=False)
    table, n = G.compare(root / COHORT, REF / COHORT, COHORT)
    assert table.empty and n == 5 * 2 * 3
    assert G.main(["--run-root", str(root), "--datasets", COHORT]) == 0
    assert G.main(["--run-root", str(tmp_path / "nothing"), "--datasets", COHORT]) == 2


def test_other_methods_in_run_are_ignored_and_unknown_unit_fails(tmp_path):
    root = copy_reference(tmp_path)
    path = root / COHORT / "selection_per_fold.csv"
    sel = pd.read_csv(path, float_precision="round_trip")
    extra = sel.iloc[[0]].assign(method="shade")
    unknown = sel.iloc[[0]].assign(k=7)
    pd.concat([sel, extra], ignore_index=True).to_csv(path, index=False)
    assert G.compare(root / COHORT, REF / COHORT, COHORT)[0].empty
    pd.concat([sel, unknown], ignore_index=True).to_csv(path, index=False)
    assert G.compare(root / COHORT, REF / COHORT, COHORT)[0].field.tolist() == ["reference"]
