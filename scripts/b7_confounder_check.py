"""B7: kiểm yếu tố nhiễu trên panel của wrapper (Review-2026-11-16, TQĐ27).

    python scripts/b7_confounder_check.py --results <KLTN_DATA_ROOT>/results/survey_v2/taimma --scenarios T1 T1v2 \
        --data-root <KLTN_DATA_ROOT> --out b7_confounders
(--scenarios nhận tên thư mục pha A: T1 cho v1 và baseline, T1v2 cho v2 và DE-G.)

Cùng cách kiểm với advisor_recheck/b7_confounder_check.py của survey (16/11/2026), áp cho panel pha A của bài TAIMMA:
  * với mỗi (kịch bản, bộ B7, phương pháp, k ∈ {5, 10}): số panel, số panel chứa gen đánh dấu giới tính (SEX_GENES: XIST, TSIX
    và 12 gen trên nhiễm sắc thể Y, đúng danh sách của survey), số panel chứa chính EGFR;
  * trên tcga_luad_egfr: AUROC của giới tính đơn lẻ (nữ = 1) và của nơi lấy mẫu (TSS, mã thứ hai của barcode) đơn lẻ bằng
    LR trên CV 5 × 5 phân tầng, seed 42–46. Giá trị của survey: 0,649 và 0,577. Khoá ở mức đầu của <out>_dataset.json.
  * trên gse31210_egfr (bộ của HV5, Review-2026-11-17): cùng phép kiểm khi metadata có cột sex; ghi dưới khoá "gse31210_egfr".
Chỉ để mô tả trong bảng B7; không dùng để chọn, loại hay xếp hạng phương pháp. Ghi <out>_panels.csv và <out>_dataset.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

B7 = ("tcga_luad_egfr", "gse31210_egfr")
K_CHECK = (5, 10)
SEX_GENES = frozenset({"XIST", "TSIX", "RPS4Y1", "RPS4Y2", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY", "TXLNGY",
                       "NLGN4Y", "PRKY", "TTTY15"})
PANEL = ["method", "k", "repeat", "fold"]


def panel_table(genes: pd.DataFrame, scenario: str, cohort: str) -> pd.DataFrame:
    g = genes[genes["k"].isin(K_CHECK)]
    rows = []
    for (method, k), part in g.groupby(["method", "k"], sort=True):
        sets = part.groupby(["repeat", "fold"])["gene"].apply(lambda s: set(map(str, s)))
        n = len(sets)
        n_sex = int(sum(bool(s & SEX_GENES) for s in sets))
        n_egfr = int(sum("EGFR" in s for s in sets))
        sex_genes = sorted(set().union(*sets) & SEX_GENES) if n else []
        rows.append({"scenario": scenario, "cohort": cohort, "method": method, "k": int(k), "n_panels": n,
                     "n_with_sex_gene": n_sex, "share_with_sex_gene": n_sex / n if n else np.nan,
                     "sex_genes_seen": ";".join(sex_genes), "n_with_egfr": n_egfr,
                     "share_with_egfr": n_egfr / n if n else np.nan})
    return pd.DataFrame(rows)


def dataset_checks(meta: pd.DataFrame) -> dict:
    y = meta["label"].astype(int).to_numpy()
    out = {"n": int(len(y)), "n_positive": int(y.sum())}
    if "sex" in meta.columns:
        female = (meta["sex"].astype(str).str.lower() == "female").astype(int).to_numpy()
        out["sex_only_auroc"] = float(roc_auc_score(y, female))
        out["positive_female"] = int(((female == 1) & (y == 1)).sum())
    tss = meta["sample_id"].astype(str).str.split("-").str[1]
    if tss.notna().all():
        X = pd.get_dummies(tss).to_numpy(float)
        aucs = []
        for seed in (42, 43, 44, 45, 46):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        out.update({"tss_only_auroc_mean": float(np.mean(aucs)), "tss_only_auroc_sd": float(np.std(aucs, ddof=1)),
                    "tss_folds": len(aucs)})
    return out


def run(results: Path, scenarios, data_root: Path | None) -> tuple[pd.DataFrame, dict]:
    frames = []
    for sc in scenarios:
        for co in B7:
            path = Path(results) / sc / co / "selected_genes.csv"
            if path.is_file():
                frames.append(panel_table(pd.read_csv(path), sc, co))
    if not frames:
        raise FileNotFoundError(f"không thấy selected_genes.csv của B7 trong {results} / {list(scenarios)}")
    info = {}
    if data_root is not None:
        from survey_v2.data import resolve_dataset_dir
        for co in B7:
            meta_path = resolve_dataset_dir(Path(data_root), co) / "metadata.csv"
            if not meta_path.is_file():
                continue
            checks = dataset_checks(pd.read_csv(meta_path))
            if co == "tcga_luad_egfr":
                info.update(checks)                       # khoá như bản của survey
            else:
                info[co] = checks
    return pd.concat(frames, ignore_index=True), info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="thư mục taimma/ chứa các kịch bản")
    ap.add_argument("--scenarios", nargs="+", default=["T1"])
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out", default="b7_confounders")
    a = ap.parse_args(argv)
    table, info = run(a.results, a.scenarios, a.data_root)
    table.to_csv(f"{a.out}_panels.csv", index=False)
    Path(f"{a.out}_dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
