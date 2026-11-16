"""scripts/build_dev_datasets.py (Review-2026-11-16, TQĐ23) trên dữ liệu giả; chép từ taimma_ref, chỉ đổi import."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.e1_splits import validate_repeated_split_manifest  # noqa: E402

from scripts import build_dev_datasets as B  # noqa: E402


@pytest.fixture(autouse=True)
def _not_strict(monkeypatch):
    """Thêm khi tích hợp: bộ nguồn giả không có số mẫu của TCGA thật, nên tắt KLTN_STRICT_N mà notebook Kaggle đặt."""
    monkeypatch.delenv("KLTN_STRICT_N", raising=False)


def fake_source(root: Path, n0=304, n1=296, p=40, n_dup=4, seed=0) -> pd.DataFrame:
    """Bộ nguồn giả: n0 + n1 bệnh nhân, n_dup bệnh nhân có thêm một mẫu thứ hai."""
    rng = np.random.default_rng(seed)
    pats = [f"P{i:04d}" for i in range(n0 + n1)]
    labels = [0] * n0 + [1] * n1
    rows = [(f"{p_}-01A", p_, y) for p_, y in zip(pats, labels)]
    rows += [(f"{pats[i]}-01B", pats[i], labels[i]) for i in range(n_dup)]
    meta = pd.DataFrame(rows, columns=["sample_id", "patient_id", "label"])
    meta["class"] = np.where(meta.label == 1, "LUSC", "LUAD")
    d = root / C.dataset_dir(B.SOURCE)
    d.mkdir(parents=True)
    X = pd.DataFrame(rng.normal(size=(len(meta), p)), index=pd.Index(meta.sample_id, name="sample_id"),
                     columns=[f"G{j}" for j in range(p)])
    X.to_hdf(d / "expression.h5", key="expression", mode="w")
    meta.to_csv(d / "metadata.csv", index=False)
    return X


@pytest.fixture()
def built(tmp_path):
    X = fake_source(tmp_path)
    card = B.build(tmp_path, tmp_path)
    return tmp_path, X, card


def test_sizes_classes_and_disjoint_patients(built):
    root, _, card = built
    expected = (int(round(170 * 304 / 600)), int(round(170 * 296 / 600)))
    seen = set()
    for name in B.SUBSETS:
        ds = load_dataset(root, name)
        assert (int((ds.y == 0).sum()), int((ds.y == 1).sum())) == expected
        assert card["subsets"][name]["n_per_class"] == list(expected)
        pats = set(ds.groups.tolist())
        assert len(pats) == len(ds.y)                      # một mẫu mỗi bệnh nhân
        assert not (pats & seen)                           # ba tập rời nhau theo bệnh nhân
        seen |= pats
    assert len(card["dropped_duplicate_patient_samples"]) == 4


def test_values_come_from_source_rows(built):
    root, X, _ = built
    ds = load_dataset(root, "tcga_dev_b")
    assert list(ds.genes) == list(X.columns)
    assert np.array_equal(ds.X, X.loc[ds.sample_ids].to_numpy())


def test_manifests_are_valid_one_repeat_and_have_distinct_seeds(built):
    root, _, _ = built
    seeds = set()
    for name, seed in B.SUBSETS.items():
        man = pd.read_csv(root / C.dataset_dir(name) / "repeated_splits.csv")
        assert validate_repeated_split_manifest(man, n_folds=C.N_FOLDS) == []
        assert set(man["repeat"]) == {0} and set(man["seed"]) == {seed}
        seeds.add(seed)
        assert C.cohort_id(name) == C.UNREGISTERED_COHORT_ID     # vì vậy seed chia phải khác nhau
    assert len(seeds) == len(B.SUBSETS)


def test_build_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fake_source(a)
    fake_source(b)
    ca, cb = B.build(a, a), B.build(b, b)
    for name in B.SUBSETS:
        for f in ("metadata.csv", "repeated_splits.csv"):
            assert ca["subsets"][name]["sha256"][f] == cb["subsets"][name]["sha256"][f]
        assert ca["subsets"][name]["values_sha256"] == cb["subsets"][name]["values_sha256"]


def test_refuses_to_overwrite(built):
    root, _, _ = built
    with pytest.raises(FileExistsError):
        B.build(root, root)


def test_card_records_split_sha(built):
    root, _, card = built
    for name in B.SUBSETS:
        path = root / C.dataset_dir(name) / "repeated_splits.csv"
        assert card["subsets"][name]["sha256"]["repeated_splits.csv"] == sha256(path)


def test_csv_files_use_lf_line_endings(built):
    """Thêm khi tích hợp: SHA-256 bản chia phải giống nhau trên Windows và Kaggle."""
    root, _, _ = built
    for name in B.SUBSETS:
        for f in ("metadata.csv", "repeated_splits.csv"):
            assert b"\r\n" not in (root / C.dataset_dir(name) / f).read_bytes()
