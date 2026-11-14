import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_xy
from mpqda import config as M
from mpqda.de import HISTORY_COLUMNS, run_mpqda
from mpqda.fitness import InnerCVFitness

P, T, K = 8, 4, 5


def run(kw, n_jobs=1, seed=(99, 5), X=None, y=None):
    if X is None:
        X, y = synthetic_xy(p=300)
    fit = InnerCVFitness(X, y, fold_seed=21, metric="logloss")
    return run_mpqda(fit, X.shape[1], K, np.random.default_rng(list(seed)), P=P, T=T, n_jobs=n_jobs, **kw), fit


def frame(res):
    return pd.DataFrame(res["history"], columns=list(HISTORY_COLUMNS))


@pytest.mark.parametrize("name", ["mpqda_de", *M.VARIANTS])
def test_deterministic_parallel_identical_and_budget(name):
    kw = M.VARIANTS.get(name, {})
    (a, fa), (b, _), (c, _) = run(kw, 1), run(kw, 4), run(kw, 1)
    for other in (b, c):                                       # trùng giữa n_jobs_eval 1 và 4, và giữa hai lần chạy cùng seed
        assert a["consensus"].tolist() == other["consensus"].tolist() and a["best"].tolist() == other["best"].tolist()
        assert a["consensus_fitness"] == other["consensus_fitness"] and a["best_fitness"] == other["best_fitness"]
        assert a["archive"] == other["archive"] and a["pi"] == other["pi"] and a["n_eval"] == other["n_eval"]
        assert frame(a).to_csv() == frame(other).to_csv()
    assert len(a["consensus"]) == K == len(set(a["consensus"].tolist())) and len(a["best"]) == K == len(set(a["best"].tolist()))
    assert a["n_eval"] <= P * (T + 1) + 1 and a["n_eval"] == fa.n_eval
    h = frame(a)
    assert h.t.tolist() == list(range(T + 1)) and h.n_eval.is_monotonic_increasing and h.best_fitness.is_monotonic_increasing
    assert h.best_fitness.iloc[-1] == a["best_fitness"] and (h.archive_size <= kw.get("A_max", M.A_MAX)).all()
    assert len(a["archive"]) == h.archive_size.iloc[-1]
    assert [f for _, f in a["archive"]] == sorted((f for _, f in a["archive"]), reverse=True)


def test_different_seed_changes_the_run():
    assert frame(run({})[0]).to_csv() != frame(run({}, seed=(100, 5))[0]).to_csv()


def test_each_variant_flag_changes_only_its_part():
    full, _ = run({})
    h = frame(full)
    assert h.q_00.notna().all() and h.mean_F.between(0.1, 1.0).all() and full["Q"].shape == (2, 3)
    a3, _ = run(M.VARIANTS["mpqda_a3_fixedF"])                 # A3: không có bộ điều khiển, F luôn 0,5
    assert a3["Q"] is None and (frame(a3).mean_F == 0.5).all() and (frame(a3).sd_F == 0).all() and frame(a3).q_00.isna().all()
    a51, _ = run(M.VARIANTS["mpqda_a5_A1"])                    # A5_A1: archive luôn là panel tốt nhất
    assert a51["consensus_fitness"] == a51["best_fitness"] and (frame(a51).archive_size == 1).all()
    assert a51["consensus"].tolist() == a51["best"].tolist() and (frame(a51).leader_is_archive_best.iloc[1:] == 1.0).all()
    a2a, _ = run(M.VARIANTS["mpqda_a2a_bestleader"])           # A2a: leader là cá thể tốt nhất, không lấy từ archive
    assert frame(a2a).leader_is_archive_best.isna().all() and frame(a2a).q_00.notna().all()
    a1, _ = run(M.VARIANTS["mpqda_a1_uniform"])                # A1: chỉ đổi khoá khởi tạo
    assert frame(a1).best_fitness.iloc[0] != h.best_fitness.iloc[0]
    # quần thể đầu (t = 0) của các biến thể không đổi khởi tạo trùng với bản đầy đủ: cùng seed, cùng thứ tự rút số
    for name in ("mpqda_a0_noband", "mpqda_a2a_bestleader", "mpqda_a2b_randleader", "mpqda_a5_A30"):
        other = frame(run(M.VARIANTS[name])[0])
        assert other.loc[0, ["best_fitness", "mean_fitness", "diversity", "mean_F"]].tolist() == \
            h.loc[0, ["best_fitness", "mean_fitness", "diversity", "mean_F"]].tolist()
    a3_first = frame(a3).loc[0, ["best_fitness", "mean_fitness", "diversity"]].tolist()
    assert a3_first == h.loc[0, ["best_fitness", "mean_fitness", "diversity"]].tolist()


def test_invalid_flags_are_rejected():
    X, y = synthetic_xy(p=50)
    with pytest.raises(ValueError):
        run_mpqda(InnerCVFitness(X, y, 1), 50, 3, np.random.default_rng(0), P=4, T=1, leader="nearest")


def test_leader_overlap_at_first_generation():
    """p0 = 5.000, P = 30, t = 1 trên dữ liệu giả: leader_overlap thuộc [0,3; 0,7], và dưới 0,1 khi reencode = False (Đ2)."""
    rng = np.random.default_rng(3)
    X = rng.normal(8, 1, size=(60, 5000))
    y = np.r_[np.zeros(40, int), np.ones(20, int)]
    for reencode, lo, hi in ((True, 0.3, 0.7), (False, 0.0, 0.1)):
        fit = InnerCVFitness(X, y, fold_seed=1, metric="logloss")
        r = run_mpqda(fit, 5000, 10, np.random.default_rng([1, 10]), T=1, reencode=reencode)
        assert lo <= r["history"][1]["leader_overlap"] <= hi
        assert r["n_eval"] <= 30 * 2 + 1
