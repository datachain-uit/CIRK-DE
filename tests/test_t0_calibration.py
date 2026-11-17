"""scripts/t0_calibration.py (Review-2026-11-17b, TQĐ34) trên dữ liệu giả: b = 0 dùng đúng hoán vị và phần train của T0,
b ≥ 1 cho hoán vị mới; bảng tóm tắt đủ dòng; chạy lại trùng giá trị. Dữ liệu giả chỉ để kiểm mã."""
import numpy as np
import pandas as pd
from survey_v2.data import variance_prefilter

from scripts import run_taimma as R
from scripts import t0_calibration as K

CID = 31


def _setup(seed=5, n=50, p=40):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(31, int), np.ones(19, int)]
    rng.shuffle(y)
    X = rng.normal(size=(n, p))
    X[:, :8] += 1.2 * y[:, None]
    genes = np.array([f"G{j:02d}" for j in range(p)])
    folds = []
    for fold in range(2):
        te = np.arange(fold * 10, fold * 10 + 10)
        folds.append({"repeat": 0, "split_seed": 4242, "fold": fold, "tr": np.setdiff1d(np.arange(n), te), "te": te})
    return X, y, genes, folds


def test_b0_is_the_registered_t0_permutation():
    X, y, genes, folds = _setup()
    cells = K.prepare(X, y, genes, CID, folds, p0=30, perms=[0, 1])
    f = folds[0]
    c0 = next(c for c in cells if c["fold"] == 0 and c["b"] == 0)
    c1 = next(c for c in cells if c["fold"] == 0 and c["b"] == 1)
    pre = variance_prefilter(X[f["tr"]], 30)
    assert np.array_equal(c0["y_tr"], y[f["tr"]][R.shuffle_permutation(CID, 4242, 0, len(f["tr"]))])
    assert np.array_equal(c0["X_tr"], X[f["tr"]][:, pre]) and np.array_equal(c0["y_te"], y[f["te"]])
    assert not np.array_equal(c0["y_tr"], c1["y_tr"]) and np.isclose(c0["r"], np.corrcoef(c0["y_tr"], y[f["tr"]])[0, 1])


def test_calibration_tables_are_complete_and_reproducible():
    X, y, genes, folds = _setup()
    kw = dict(perms=[0, 1], methods=["mpqda_v2"], k=3, p0=30, draws=8, P=6, T=2, workers=1, cohort="smoke_synth")
    units, rand, summary = K.calibrate(X, y, genes, CID, folds, **kw)
    units2, _, summary2 = K.calibrate(X, y, genes, CID, folds, **kw)
    assert units.equals(units2) and summary.equals(summary2)
    assert len(units) == 2 * 2 * 2 and set(units.output) == {"mpqda_v2", "mpqda_v2_best"}      # 2 fold × 2 b × 2 đầu ra
    assert units.pit.between(0, 1).all() and len(rand) == 2 * 2 * 8
    allrows = summary[summary.b == "all"].set_index("output")
    assert set(allrows.index) == {"mpqda_v2", "mpqda_v2_best"} and allrows.n_folds.eq(4).all()
    assert allrows.share_b_outside_range.between(0, 1).all()
    assert set(summary[summary.output == "random_panel"].b) == {"0", "1"}
