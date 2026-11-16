"""Dựng ba tập con phát triển tcga_dev_a, tcga_dev_b, tcga_dev_c từ cohort TCGA LUAD/LUSC (Review-2026-11-16, TQĐ23).
Chép từ taimma_ref/build_dev_datasets_ref.py.

    python scripts/build_dev_datasets.py --data-root <gốc dữ liệu có TCGA> [--out-root <gốc ghi được>]

Quy tắc (đặt trước, không chỉnh sau khi xem kết quả):
  1. Mỗi bệnh nhân giữ một mẫu (sample_id nhỏ nhất theo thứ tự chuỗi); mỗi bệnh nhân chỉ có một nhãn.
  2. Trong từng lớp (lớp 0 trước), hoán vị bệnh nhân đã sắp xếp bằng một default_rng(DRAW_SEED) dùng chung, rồi cắt thành
     ba khối liên tiếp cho a, b, c. Số bệnh nhân mỗi lớp trong một tập = round(170 · tỉ lệ lớp), nên với 515/501 mỗi tập có
     86 LUAD và 84 LUSC. Ba tập rời nhau theo bệnh nhân.
  3. Mỗi tập là một bộ dữ liệu riêng trong experiment-data-perm/<tên>/: expression.h5 (key "expression", mẫu × gen),
     metadata.csv (sample_id, patient_id, class, label, source_dataset) và repeated_splits.csv (một lần lặp, 5 fold).
     Mỗi tập có seed chia riêng, vì survey gán cùng mã cohort 99 cho mọi bộ ngoài registry: nếu chung seed chia thì seed
     tìm kiếm của ba tập trùng nhau.
  4. Ghi dev_datasets_card.json: SHA-256 của các tệp, SHA-256 của ma trận giá trị, số mẫu mỗi lớp, mẫu bị bỏ do trùng
     bệnh nhân. SHA-256 của ba repeated_splits.csv chép vào mpqda.config.EXPECTED_SPLIT_SHA256 trước khi chạy DEV.
Không đọc ba cohort GPL570 hay hai bộ B7. Mẫu của tcga_luad_egfr là tập con của cohort này (cùng biểu hiện, khác nhãn).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.e1_splits import build_repeated_split_manifest  # noqa: E402

SOURCE = "tcga_luad_lusc_primary_tumor"
SUBSETS = {"tcga_dev_a": 1171, "tcga_dev_b": 1172, "tcga_dev_c": 1173}     # tên -> seed chia 5 fold
N_PER_SUBSET, DRAW_SEED = 170, 20261117
CLASS_NAME = {0: "LUAD", 1: "LUSC"}
CARD = "dev_datasets_card.json"


def one_sample_per_patient(sample_ids, patient_ids, labels) -> tuple[pd.DataFrame, list[str]]:
    """Bảng (sample_id, patient_id, label) còn một mẫu mỗi bệnh nhân, và danh sách mẫu bị bỏ."""
    df = pd.DataFrame({"sample_id": np.asarray(sample_ids, dtype=str), "patient_id": np.asarray(patient_ids, dtype=str),
                       "label": np.asarray(labels, dtype=int)}).sort_values("sample_id", kind="stable")
    if (df.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("có bệnh nhân mang hai nhãn")
    dropped = df.loc[df.duplicated("patient_id", keep="first"), "sample_id"].tolist()
    return df.drop_duplicates("patient_id", keep="first").reset_index(drop=True), dropped


def assign(df: pd.DataFrame, n_per_subset: int = N_PER_SUBSET, seed: int = DRAW_SEED) -> dict[str, list[str]]:
    """Tên tập -> danh sách patient_id (đã sắp xếp)."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {name: [] for name in SUBSETS}
    for label in (0, 1):
        pats = np.sort(df.loc[df["label"] == label, "patient_id"].to_numpy(dtype=str))
        n = int(round(n_per_subset * len(pats) / len(df)))
        if n < 5 or len(SUBSETS) * n > len(pats):
            raise ValueError(f"lớp {label}: {len(pats)} bệnh nhân không đủ cho {len(SUBSETS)} tập × {n}")
        perm = rng.permutation(pats)
        for j, name in enumerate(SUBSETS):
            out[name].extend(perm[j * n:(j + 1) * n].tolist())
    return {name: sorted(v) for name, v in out.items()}


def values_sha256(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype="<f8").tobytes()).hexdigest()


def write_subset(ds, rows: pd.DataFrame, name: str, split_seed: int, out_root: Path) -> dict:
    d = Path(out_root) / C.dataset_dir(name)
    if d.exists():
        raise FileExistsError(f"{d} đã có; xoá thư mục cũ trước khi dựng lại")
    d.mkdir(parents=True)
    index = {s: i for i, s in enumerate(np.asarray(ds.sample_ids, dtype=str))}
    rows = rows.sort_values("sample_id", kind="stable").reset_index(drop=True)
    X = ds.X[[index[s] for s in rows["sample_id"]]]
    pd.DataFrame(X, index=pd.Index(rows["sample_id"], name="sample_id"), columns=np.asarray(ds.genes, dtype=str)).to_hdf(
        d / "expression.h5", key="expression", mode="w")
    y = rows["label"].to_numpy(dtype=int)
    meta = pd.DataFrame({"sample_id": rows["sample_id"], "patient_id": rows["patient_id"],
                         "class": [CLASS_NAME[v] for v in y], "label": y, "source_dataset": SOURCE})
    meta.to_csv(d / "metadata.csv", index=False)
    manifest = build_repeated_split_manifest(rows["sample_id"], rows["patient_id"], y, seeds=(split_seed,),
                                             n_folds=C.N_FOLDS)
    manifest.to_csv(d / "repeated_splits.csv", index=False)
    return {"dir": str(Path(C.dataset_dir(name))), "split_seed": split_seed, "n": int(len(y)),
            "n_per_class": [int((y == 0).sum()), int((y == 1).sum())],
            "sha256": {f: sha256(d / f) for f in ("expression.h5", "metadata.csv", "repeated_splits.csv")},
            "values_sha256": values_sha256(X)}


def build(data_root: Path, out_root: Path) -> dict:
    ds = load_dataset(data_root, SOURCE, gene_space=None)
    table, dropped = one_sample_per_patient(ds.sample_ids, ds.groups, ds.y)
    groups = assign(table)
    card = {"source": SOURCE, "source_files_sha256": ds.files_sha256, "draw_seed": DRAW_SEED,
            "n_per_subset": N_PER_SUBSET, "dropped_duplicate_patient_samples": dropped, "subsets": {}}
    for name, split_seed in SUBSETS.items():
        rows = table[table["patient_id"].isin(groups[name])]
        card["subsets"][name] = write_subset(ds, rows, name, split_seed, out_root)
    Path(out_root, C.PERM_DATA_DIR, CARD).write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return card


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="gốc dữ liệu có experiment-data-perm/" + SOURCE)
    ap.add_argument("--out-root", type=Path, default=None, help="gốc ghi được (mặc định: --data-root)")
    a = ap.parse_args(argv)
    card = build(a.data_root, a.out_root or a.data_root)
    for name, info in card["subsets"].items():
        print(f"{name}: n = {info['n']} {tuple(info['n_per_class'])}; repeated_splits.csv "
              f"{info['sha256']['repeated_splits.csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
