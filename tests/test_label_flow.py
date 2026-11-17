"""Đường đi của nhãn trong T0 nhãn xáo trộn (Review-2026-11-17b, TQĐ33). Ba điều phải đúng thì AUROC ngoài của T0 không thể
đến từ rò rỉ nhãn thật: (a) mọi phương pháp nhận đúng một vector nhãn train đã hoán vị; (b) tìm kiếm của v2 và DE-G chỉ đọc
giá trị fitness, không đọc dữ liệu hay nhãn; (c) nhãn test thật không ảnh hưởng tới panel. Dữ liệu giả chỉ để kiểm mã."""
import numpy as np
import pandas as pd
import pytest

from mpqda import config as M
from mpqda.de_v2 import run_de_gapped, run_mpqda_v2
from mpqda.fitness import InnerCVFitness
from scripts import run_taimma as R

METHODS = ("mpqda_v2", "de_g", "mpqda_de", "ga")
CID, SPLIT_SEED = 31, 4242


def _data(seed=3, n=48, p0=40):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(30, int), np.ones(18, int)]
    rng.shuffle(y)
    X = rng.normal(size=(n, p0))
    X[:, :6] += 1.5 * y[:, None]                                  # sáu gen mang nhãn thật
    te = np.sort(rng.choice(n, 10, replace=False))
    return X, y, np.setdiff1d(np.arange(n), te), te


def _unit(out, method, X, y, tr, te, y_te=None):
    perm = R.shuffle_permutation(CID, SPLIT_SEED, 0, len(tr))
    parts = dict(R.PARTS, **R.SHUFFLE_PART)
    paths = R.checkpoint_paths(out, 0, 0, method, parts)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    return {"cohort": "smoke_synth", "cid": CID, "repeat": 0, "split_seed": SPLIT_SEED, "fold": 0, "method": method,
            "seed_index": 0, "k_list": [3], "metric": "logloss", "X_tr": X[tr], "y_tr": y[tr][perm],
            "genes": np.array([f"G{j:02d}" for j in range(X.shape[1])]), "P": 6, "T": 2, "n_jobs_eval": 1,
            "label_shuffled": True, "paths": paths, "X_te": X[te], "y_te": y[te] if y_te is None else y_te}


def test_every_method_fits_the_same_permuted_training_labels(tmp_path, monkeypatch):
    X, y, tr, te = _data()
    seen = {}

    class Spy(InnerCVFitness):
        def __init__(self, X_tr, y_tr, *a, **kw):
            seen.setdefault(current[0], []).append(np.array(y_tr, copy=True))
            super().__init__(X_tr, y_tr, *a, **kw)

    monkeypatch.setattr(R, "InnerCVFitness", Spy)
    current = [None]
    for m in METHODS:
        current[0] = m
        unit = _unit(tmp_path / m, m, X, y, tr, te)
        R.run_unit(unit)
        assert len(seen[m]) == 1 and np.array_equal(seen[m][0], unit["y_tr"])
    assert not np.array_equal(seen["mpqda_v2"][0], y[tr])         # nhãn đã hoán vị, không phải nhãn thật
    assert all(np.array_equal(seen[m][0], seen["mpqda_v2"][0]) for m in METHODS)


class ValuesOnly:
    """Chỉ cho đọc evaluate_many và n_eval; đọc thuộc tính khác (X, y, folds, …) là lỗi."""

    def __init__(self, inner):
        self._inner = inner

    def evaluate_many(self, panels, n_jobs=1):
        return self._inner.evaluate_many(panels, n_jobs)

    @property
    def n_eval(self):
        return self._inner.n_eval

    def __getattr__(self, name):
        raise AssertionError(f"tìm kiếm đọc thuộc tính {name} của fitness")


@pytest.mark.parametrize("search", ["mpqda_v2", "de_g"])
def test_v2_search_reads_only_fitness_values(search):
    X, y, tr, _ = _data()
    p0, k = X.shape[1], 3

    def run(fitness):
        rng = np.random.default_rng([7, k])
        if search == "de_g":
            return run_de_gapped(fitness, p0, k, rng, P=6, T=3)
        return run_mpqda_v2(fitness, p0, k, rng, P=6, T=3, **M.v2_config("mpqda_v2"))

    a = run(InnerCVFitness(X[tr], y[tr], fold_seed=11))
    b = run(ValuesOnly(InnerCVFitness(X[tr], y[tr], fold_seed=11)))
    assert np.array_equal(np.sort(a["best"]), np.sort(b["best"])) and a["best_fitness"] == b["best_fitness"]
    if search == "mpqda_v2":
        assert np.array_equal(np.sort(a["consensus"]), np.sort(b["consensus"]))


@pytest.mark.parametrize("method", ["mpqda_v2", "de_g"])
def test_test_labels_do_not_change_the_panel(tmp_path, method):
    X, y, tr, te = _data()
    R.run_unit(_unit(tmp_path / "a", method, X, y, tr, te))
    R.run_unit(_unit(tmp_path / "b", method, X, y, tr, te, y_te=1 - y[te]))
    for part in ("genes", "history", "metrics"):
        fa = pd.read_csv(R.checkpoint_paths(tmp_path / "a", 0, 0, method, R.PARTS)[part])
        fb = pd.read_csv(R.checkpoint_paths(tmp_path / "b", 0, 0, method, R.PARTS)[part])
        drop = [c for c in ("selection_time_s", "cpu_time_s") if c in fa]
        assert fa.drop(columns=drop).equals(fb.drop(columns=drop)), part
    sa = pd.read_csv(R.checkpoint_paths(tmp_path / "a", 0, 0, method, R.SHUFFLE_PART)["shuffle_eval"])
    sb = pd.read_csv(R.checkpoint_paths(tmp_path / "b", 0, 0, method, R.SHUFFLE_PART)["shuffle_eval"])
    assert len(sa) and np.allclose(sa.auroc + sb.auroc, 1.0, atol=1e-12) and np.allclose(sa.bacc + sb.bacc, 1.0, atol=1e-12)
