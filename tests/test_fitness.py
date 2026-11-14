import inspect

import numpy as np

from conftest import synthetic_xy
from mpqda.fitness import InnerCVFitness
from survey_v2 import wrappers as W


def test_auroc_mode_identical_to_survey(xy):
    X, y = xy
    seed = 12345
    ours, ref = InnerCVFitness(X, y, fold_seed=seed, metric="auroc"), W.InnerCVFitness(X, y, seed=seed)
    rng = np.random.default_rng(5)
    for _ in range(20):
        S = rng.choice(X.shape[1], size=int(rng.integers(2, 30)), replace=False)
        assert ours(S) == ref(S)                       # bằng nhau tuyệt đối, không xấp xỉ
    assert ours.n_eval == ref.n_eval == 20
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(ours.folds, ref.folds))


def test_cache_n_eval_and_log(xy):
    X, y = xy
    fit = InnerCVFitness(X, y, fold_seed=7, metric="logloss")
    v1, v2 = fit([3, 1, 2]), fit(np.array([2, 3, 1]))
    assert v1 == v2 and fit.n_eval == 1 and v1 < 0                                   # log-loss âm, cache theo panel đã sắp
    vals = fit.evaluate_many([[1, 2, 3], [4, 5], [5, 4], [6, 7]])
    assert fit.n_eval == 3 and vals[0] == v1 and vals[1] == vals[2]
    assert [entry[0] for entry in fit.log] == [(1, 2, 3), (1, 2, 3), (1, 2, 3), (4, 5), (4, 5), (6, 7)]
    assert [entry[2] for entry in fit.log] == [1, 1, 3, 3, 3, 3]
    assert len(fit.folds) == 3


def test_logloss_is_clipped_and_finite(xy):
    X, y = xy
    Xs = X.copy()
    Xs[:, 0] = 50.0 * y                                                              # tách hoàn toàn: xác suất sát 0 hoặc 1
    v = InnerCVFitness(Xs, y, fold_seed=1, metric="logloss")([0])
    assert np.isfinite(v) and -np.log(1 / 1e-6) <= v < 0


def test_parallel_evaluation_is_bit_identical(xy):
    X, y = xy
    rng = np.random.default_rng(6)
    panels = [rng.choice(400, 10, replace=False) for _ in range(13)] + [[3, 1, 2], [2, 1, 3]]
    a = InnerCVFitness(X, y, fold_seed=7, metric="logloss")
    b = InnerCVFitness(X, y, fold_seed=7, metric="logloss")
    assert a.evaluate_many(panels, n_jobs=1) == b.evaluate_many(panels, n_jobs=4)
    assert a.n_eval == b.n_eval == 14 and a.log == b.log


def test_shuffled_labels_inner_auroc_near_half():
    X, y = synthetic_xy(seed=2)
    fit = InnerCVFitness(X, np.random.default_rng(8).permutation(y), fold_seed=9, metric="auroc")
    rng = np.random.default_rng(10)
    vals = [fit(rng.choice(400, 5, replace=False)) for _ in range(200)]
    assert abs(np.mean(vals) - 0.5) < 0.05


def test_constructor_takes_no_test_data():
    params = list(inspect.signature(InnerCVFitness.__init__).parameters)
    assert params == ["self", "X_tr", "y_tr", "fold_seed", "metric", "n_folds", "C_lr"]
