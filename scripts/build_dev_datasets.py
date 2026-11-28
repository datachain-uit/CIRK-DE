"""Build the three development subsets tcga_dev_a, tcga_dev_b, tcga_dev_c from the TCGA LUAD/LUSC cohort (PREREG-v2, Section 2.2).


    python scripts/build_dev_datasets.py --data-root <data root holding TCGA> [--out-root <writable root>]

Rules (fixed in advance, not changed after seeing results):
  1. Each patient keeps one sample (smallest sample_id in string order); each patient has a single label.
  2. Within each class (class 0 first), the sorted patients are permuted with one shared default_rng(DRAW_SEED), then cut into
     three consecutive blocks for a, b, c. The number of patients per class in a subset = round(170 · class share), so with 515/501 each subset has
     86 LUAD and 84 LUSC. The three subsets are disjoint by patient.
  3. Each subset is a separate dataset in experiment-data-perm/<name>/: expression.h5 (key "expression", samples × genes),
     metadata.csv (sample_id, patient_id, class, label, source_dataset) and repeated_splits.csv (one repeat, 5 folds).
     Each subset has its own split seed, because the vendored code gives cohort code 99 to every dataset outside its registry: with a shared split seed
     the search seeds of the three subsets would coincide.
  4. Write dev_datasets_card.json: SHA-256 of the files, SHA-256 of the value matrix, number of samples per class, samples dropped as duplicate
     patients. The SHA-256 of the three repeated_splits.csv are copied into mpqda.config.EXPECTED_SPLIT_SHA256 before the DEV run.
CSV files are written with LF line endings (added at integration: pandas on Windows writes CRLF by default, which changes the SHA-256 against the build on Kaggle).
Never reads the three GPL570 cohorts or the two B7 datasets. The samples of tcga_luad_egfr are a subset of this cohort (same expression, other labels).
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
SUBSETS = {"tcga_dev_a": 1171, "tcga_dev_b": 1172, "tcga_dev_c": 1173}     # name -> seed of the 5-fold split
N_PER_SUBSET, DRAW_SEED = 170, 20261117
CLASS_NAME = {0: "LUAD", 1: "LUSC"}
CARD = "dev_datasets_card.json"


def one_sample_per_patient(sample_ids, patient_ids, labels) -> tuple[pd.DataFrame, list[str]]:
    """Table (sample_id, patient_id, label) with one sample per patient, and the list of dropped samples."""
    df = pd.DataFrame({"sample_id": np.asarray(sample_ids, dtype=str), "patient_id": np.asarray(patient_ids, dtype=str),
                       "label": np.asarray(labels, dtype=int)}).sort_values("sample_id", kind="stable")
    if (df.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("a patient carries two labels")
    dropped = df.loc[df.duplicated("patient_id", keep="first"), "sample_id"].tolist()
    return df.drop_duplicates("patient_id", keep="first").reset_index(drop=True), dropped


def assign(df: pd.DataFrame, n_per_subset: int = N_PER_SUBSET, seed: int = DRAW_SEED) -> dict[str, list[str]]:
    """Subset name -> list of patient_id (sorted)."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {name: [] for name in SUBSETS}
    for label in (0, 1):
        pats = np.sort(df.loc[df["label"] == label, "patient_id"].to_numpy(dtype=str))
        n = int(round(n_per_subset * len(pats) / len(df)))
        if n < 5 or len(SUBSETS) * n > len(pats):
            raise ValueError(f"class {label}: {len(pats)} patients are not enough for {len(SUBSETS)} subsets × {n}")
        perm = rng.permutation(pats)
        for j, name in enumerate(SUBSETS):
            out[name].extend(perm[j * n:(j + 1) * n].tolist())
    return {name: sorted(v) for name, v in out.items()}


def values_sha256(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype="<f8").tobytes()).hexdigest()


def write_subset(ds, rows: pd.DataFrame, name: str, split_seed: int, out_root: Path) -> dict:
    d = Path(out_root) / C.dataset_dir(name)
    if d.exists():
        raise FileExistsError(f"{d} already exists; delete the old folder before rebuilding")
    d.mkdir(parents=True)
    index = {s: i for i, s in enumerate(np.asarray(ds.sample_ids, dtype=str))}
    rows = rows.sort_values("sample_id", kind="stable").reset_index(drop=True)
    X = ds.X[[index[s] for s in rows["sample_id"]]]
    pd.DataFrame(X, index=pd.Index(rows["sample_id"], name="sample_id"), columns=np.asarray(ds.genes, dtype=str)).to_hdf(
        d / "expression.h5", key="expression", mode="w")
    y = rows["label"].to_numpy(dtype=int)
    meta = pd.DataFrame({"sample_id": rows["sample_id"], "patient_id": rows["patient_id"],
                         "class": [CLASS_NAME[v] for v in y], "label": y, "source_dataset": SOURCE})
    meta.to_csv(d / "metadata.csv", index=False, lineterminator="\n")   # LF: same bytes on Windows and Linux
    manifest = build_repeated_split_manifest(rows["sample_id"], rows["patient_id"], y, seeds=(split_seed,),
                                             n_folds=C.N_FOLDS)
    manifest.to_csv(d / "repeated_splits.csv", index=False, lineterminator="\n")
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
    ap.add_argument("--data-root", type=Path, required=True, help="data root holding experiment-data-perm/" + SOURCE)
    ap.add_argument("--out-root", type=Path, default=None, help="writable root (default: --data-root)")
    a = ap.parse_args(argv)
    card = build(a.data_root, a.out_root or a.data_root)
    for name, info in card["subsets"].items():
        print(f"{name}: n = {info['n']} {tuple(info['n_per_class'])}; repeated_splits.csv "
              f"{info['sha256']['repeated_splits.csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
