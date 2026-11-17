"""Cổng G-T4 (Review-2026-11-16, TQĐ27): mốc không wrapper của bài TAIMMA (T4: sáu selector, B0, random-k) phải trùng kết
quả E1 của survey ở tag survey-v3-dmkd.2 trên lần lặp 0–2, cho ba cohort GPL570 và hai bộ B7. T4 gọi nguyên script survey trong
vendor/ (c1828d2), cùng bản chia (SHA-256 trùng data card) và cùng seed, nên panel phải trùng tuyệt đối và chỉ số của pha B-AUC
(pha B của survey) lệch không quá 1e-12.

    python scripts/check_gate_t4.py --taimma <KLTN_DATA_ROOT>/results/survey_v2/taimma \
        --survey <bản clone survey ở tag survey-v3-dmkd.2>/results/survey_v2 --out gate_t4

Đọc taimma/T4/<bộ>/selected_genes.csv và taimma/T5/T4/<bộ>/classification_per_fold.csv; phía survey đọc
E1_feature_selection[_b7]/<bộ>/selected_genes.csv và E1_classification[_b7]/<bộ>/classification_per_fold.csv. Ghi
<out>.csv (một dòng mỗi bộ) và <out>_diff.csv (mọi ô lệch). Trả 0 khi mọi bộ đạt, 2 khi có bộ không đạt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GPL570 = ("gse10245", "gse30219", "gse37745")
B7 = ("tcga_luad_egfr", "gse31210_egfr")
T4_METHODS = ("l1_lr", "stability_selection", "f_test", "mrmr", "svm_rfe", "abess")
REFERENCES = ("b0", "random_k")
REPEATS = (0, 1, 2)
K = (2, 5, 10, 50)
METRICS = ("auroc", "bacc", "mcc", "logloss", "brier", "threshold", "C")
TOL = 1e-12


def survey_dirs(survey: Path, dataset: str) -> tuple[Path, Path]:
    suffix = "_b7" if dataset in B7 else ""
    return survey / f"E1_feature_selection{suffix}" / dataset, survey / f"E1_classification{suffix}" / dataset


def _genes(path: Path) -> pd.Series:
    g = pd.read_csv(path)
    g = g[g["repeat"].isin(REPEATS) & g["method"].isin(T4_METHODS) & g["k"].isin(K)]
    return g.sort_values(["repeat", "fold", "method", "k", "rank"]).groupby(["repeat", "fold", "method", "k"])["gene"].apply(
        lambda s: tuple(map(str, s)))


def _metrics(path: Path) -> pd.DataFrame:
    """Chỉ LR-L2 (bộ phân loại duy nhất của bài TAIMMA); survey còn có rf, svm_linear trên GPL570."""
    m = pd.read_csv(path, float_precision="round_trip")
    keep = m["repeat"].isin(REPEATS) & (m["model"] == "lr_l2") & (
        m["method"].isin(REFERENCES) & ((m["method"] == "b0") | m["k"].isin(K)) | m["method"].isin(T4_METHODS) & m["k"].isin(K))
    return m[keep].set_index(["repeat", "fold", "method", "k", "model"]).sort_index()


def check_dataset(taimma: Path, survey: Path, dataset: str) -> tuple[dict, pd.DataFrame]:
    sel_s, clf_s = survey_dirs(survey, dataset)
    gt, gs = _genes(taimma / "T4" / dataset / "selected_genes.csv"), _genes(sel_s / "selected_genes.csv")
    mt, ms = _metrics(taimma / "T5" / "T4" / dataset / "classification_per_fold.csv"), _metrics(clf_s / "classification_per_fold.csv")
    diffs = []
    missing_g = sorted(set(gt.index) ^ set(gs.index))
    for key in sorted(set(gt.index) & set(gs.index)):
        if gt[key] != gs[key]:
            diffs.append({"dataset": dataset, "part": "genes", "key": str(key), "column": "gene",
                          "taimma": ";".join(gt[key]), "survey": ";".join(gs[key])})
    missing_m = sorted(set(mt.index) ^ set(ms.index))
    common = mt.index.intersection(ms.index)
    max_diff = {}
    for col in METRICS:
        if col not in mt.columns or col not in ms.columns:
            continue
        a, b = mt.loc[common, col].astype(float).to_numpy(), ms.loc[common, col].astype(float).to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)
        max_diff[col] = float(d.max()) if len(d) else 0.0
        for i in np.flatnonzero(d > TOL):
            diffs.append({"dataset": dataset, "part": "metrics", "key": str(common[i]), "column": col,
                          "taimma": a[i], "survey": b[i]})
    row = {"dataset": dataset, "n_panels": int(len(gt)), "n_metric_rows": int(len(common)),
           "n_missing_panels": len(missing_g), "n_missing_metric_rows": len(missing_m),
           "n_cells_different": len(diffs), **{f"max_abs_diff_{c}": v for c, v in max_diff.items()}}
    row["pass"] = bool(len(gt) > 0 and len(common) > 0 and not missing_g and not missing_m and not diffs)
    return row, pd.DataFrame(diffs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taimma", type=Path, required=True, help="thư mục taimma/ của kết quả")
    ap.add_argument("--survey", type=Path, required=True, help="results/survey_v2 của bản clone survey ở tag survey-v3-dmkd.2")
    ap.add_argument("--datasets", nargs="+", default=[*GPL570, *B7])
    ap.add_argument("--out", default="gate_t4")
    a = ap.parse_args(argv)
    rows, diffs = [], []
    for d in a.datasets:
        row, diff = check_dataset(a.taimma, a.survey, d)
        rows.append(row)
        diffs.append(diff)
    table = pd.DataFrame(rows)
    table.to_csv(f"{a.out}.csv", index=False)
    pd.concat(diffs, ignore_index=True).to_csv(f"{a.out}_diff.csv", index=False)
    print(table.to_string(index=False))
    return 0 if table["pass"].all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
