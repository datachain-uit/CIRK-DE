import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_xy
from mpqda import baselines as B
from mpqda import config as M
from mpqda.de import HISTORY_COLUMNS
from mpqda.fitness import InnerCVFitness
from survey_v2 import wrappers as W

P, T, K = 6, 3, 5


def run(method, n_jobs=1, seed=3, metric="logloss", P=P, T=T):
    X, y = synthetic_xy(p=200)
    fit = InnerCVFitness(X, y, fold_seed=21, metric=metric)
    return B.run_method(method, fit, X.shape[1], K, B.make_rng(method, seed, K), P=P, T=T, n_jobs=n_jobs), fit


@pytest.mark.parametrize("method", ["de_rand1", "shade", "jde", "jade", "ga", "bpso"])
def test_deterministic_exact_k_and_budget(method):
    (a, fit), (b, _) = run(method), run(method, n_jobs=1 if method in ("ga", "bpso") else 3)
    assert a["best"].tolist() == b["best"].tolist() and a["best_fitness"] == b["best_fitness"] and a["n_eval"] == b["n_eval"]
    ha, hb = (pd.DataFrame(r["history"], columns=list(HISTORY_COLUMNS)) for r in (a, b))
    assert ha.to_csv() == hb.to_csv()
    assert len(a["best"]) == K == len(set(a["best"].tolist())) and (np.diff(a["best"]) > 0).all()
    assert a["n_eval"] <= P * (T + 1) and a["n_eval"] == fit.n_eval
    assert ha.t.tolist() == list(range(T + 1)) and ha.best_fitness.is_monotonic_increasing
    assert ha.best_fitness.iloc[-1] == a["best_fitness"] == max(v for _, v, _ in fit.log)
    assert run(method, seed=4)[0]["history"][0]["mean_fitness"] != a["history"][0]["mean_fitness"]


def test_full_budget_bound_is_1530():
    assert M.P * (M.T + 1) == 1530


def test_rng_streams_follow_the_spec():
    for method, tag in (("ga", [9, 5, 0]), ("bpso", [9, 5, 1]), ("shade", [9, 5]), ("mpqda_a3_fixedF", [9, 5])):
        assert B.make_rng(method, 9, 5).random() == np.random.default_rng(tag).random()


def test_shade_memory_and_external_archive_sizes():
    r, _ = run("shade", P=8, T=6)
    assert r["M_CR"].shape == r["M_F"].shape == (8,) and r["ext_archive_size"] <= 8      # H = P, archive ngoài cỡ P
    assert ((r["M_CR"] >= 0) & (r["M_CR"] <= 1)).all() and ((r["M_F"] > 0) & (r["M_F"] <= 1)).all()
    h = pd.DataFrame(r["history"])
    assert (h.archive_size <= 8).all() and h.archive_size.is_monotonic_increasing and h.mean_F.between(0, 1).all()
    n_updates = int((h.success_rate.iloc[1:] > 0).sum())                                  # mỗi thế hệ có thành công ghi một ô
    assert int((r["M_F"] != 0.5).sum()) <= min(n_updates, 8)


def test_jade_parameters():
    r, _ = run("jade", P=8, T=6)
    assert 0 <= r["mu_CR"] <= 1 and 0 < r["mu_F"] <= 1 and r["ext_archive_size"] <= 8
    assert max(M.JADE_P_MIN, int(np.ceil(M.JADE_P * M.P))) == 2


def test_jde_parameters_follow_the_rule():
    r, _ = run("jde", P=8, T=6)
    F, CR = r["F"], r["CR"]
    assert ((F == M.JDE_INIT[0]) | ((F >= 0.1) & (F <= 1.0))).all() and ((CR >= 0) & (CR <= 1)).all()
    # với tau1 = tau2 = 0 không tham số nào đổi; với tau = 1 mọi tham số được nhận đều đã rút lại
    X, y = synthetic_xy(p=200)
    frozen = B.run_jde(InnerCVFitness(X, y, 21), 200, K, np.random.default_rng([3, K]), P=8, T=4, tau=(0.0, 0.0))
    assert (frozen["F"] == 0.5).all() and (frozen["CR"] == 0.9).all()
    always = B.run_jde(InnerCVFitness(X, y, 21), 200, K, np.random.default_rng([3, K]), P=8, T=4, tau=(1.0, 1.0))
    changed = always["F"] != 0.5
    assert changed.any() and ((always["F"][changed] >= 0.1) & (always["F"][changed] <= 1.0)).all()


def test_qlde_stops_until_configured():
    assert M.QLDE is None
    with pytest.raises(NotImplementedError):
        run("qlde")


@pytest.mark.parametrize("method", ["ga", "bpso"])
def test_survey_wrappers_reproduced_in_auroc_mode(method):
    """G-T1 ở mức đơn vị: GA, BPSO của survey với fitness mới ở chế độ AUROC trùng run_wrapper của survey."""
    X, y = synthetic_xy(p=200)
    seed = 777
    ref = W.run_wrapper(method, X, y, [K], seed=seed, pop=P, generations=T)
    fit = InnerCVFitness(X, y, fold_seed=seed, metric="auroc")
    r = B.run_method(method, fit, X.shape[1], K, B.make_rng(method, seed, K), P=P, T=T)
    assert r["best"].tolist() == np.sort(ref.panels[K]).tolist()
    assert r["best_fitness"] == ref.info["best_fitness"][K] and r["n_eval"] == ref.info["n_eval"][K]
    assert [h["best_fitness"] for h in r["history"]] == ref.info["history"][K]


def test_all_wrappers_share_one_fitness_function():
    """T-R3: mọi wrapper trên cùng fold ngoài gọi cùng một hàm fitness (cùng fold nội bộ, cùng LR-L2)."""
    X, y = synthetic_xy(p=200)
    probe = InnerCVFitness(X, y, fold_seed=21, metric="logloss")
    for method in ("mpqda_de", "de_rand1", "shade", "jde", "jade", "ga", "bpso"):
        fit = InnerCVFitness(X, y, fold_seed=21, metric="logloss")
        r = B.run_method(method, fit, X.shape[1], K, B.make_rng(method, 3, K), P=P, T=2)
        assert probe(r["best"]) == r["best_fitness"]
