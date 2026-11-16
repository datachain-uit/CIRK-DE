"""Tích hợp MPQDA-DE v2 vào nhánh taimma-2027 (Review-2026-11-16, Mục 6.1): cấu hình, seed, runner, B-C1 và v1 không đổi."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from survey_v2 import config as C
from survey_v2.classify import evaluate_panel

from conftest import ROOT, SMOKE_COHORT, run_script, synthetic_xy
from mpqda import config as M
from mpqda import de_v2, registration
from mpqda.baselines import run_method
from mpqda.de import HISTORY_COLUMNS
from mpqda.fitness import InnerCVFitness
from mpqda.phase_b import evaluate_panel_ll

REFERENCE = ROOT / "tests" / "data" / "v1_reference_4fea32f.json"


# ---------------------------------------------------------------- test 1: hai bảng cấu hình khớp de_v2.py
def test_config_tables_match_de_v2():
    assert M.V2_CANDIDATES == de_v2.CANDIDATES
    renamed = {"mpqda_" + name: cfg for name, cfg in de_v2.ABLATIONS.items()}
    assert {m: c for m, c in M.V2_ABLATIONS.items() if m != "mpqda_v2_no_adapt"} == renamed
    assert M.V2_ABLATIONS["mpqda_v2_no_adapt"] == {"adapt": False}
    assert set(registration.CANDIDATES) == set(M.V2_CANDIDATES)
    assert set(registration.REGISTERED) == {"mpqda_v2", *M.V2_ABLATIONS, "de_g"}
    assert set(M.DEV_DATASETS) & set(registration.CONFIRMATORY_DATASETS) == set()
    assert set(M.DEV_DATASETS) <= set(M.EXPECTED_SPLIT_SHA256)


# ---------------------------------------------------------------- test 2: METHOD_ID cũ không đổi, seed chung cho họ v2
def test_method_ids_and_seed_family():
    old = {"mpqda_de": 13, "de_rand1": 14, "shade": 15, "qlde": 16, "jde": 17, "jade": 18}
    assert all(M.METHOD_ID[m] == v for m, v in old.items())
    assert all(M.METHOD_ID[m] == C.METHOD_ID[m] for m in C.METHOD_ID)
    assert M.METHOD_ID["mpqda_v2"] == 19 and M.METHOD_ID["de_g"] == 20
    assert all(M.base_method(m) == "mpqda_v2" for m in M.V2_FAMILY)
    assert all(M.base_method(m) == "mpqda_de" for m in M.VARIANTS) and M.base_method("ga") == "ga"
    assert all(M.output_names(m) == (m, m + "_best") for m in M.V2_FAMILY) and M.output_names("de_g") == ("de_g",)
    assert M.v2_config("mpqda_v2_m2") == {"adapt": True}
    if M.V2_CHOSEN is None:
        with pytest.raises(ValueError):
            M.v2_config("mpqda_v2")


def test_ablation_configs_follow_the_chosen_candidate(monkeypatch):
    monkeypatch.setattr(M, "V2_CHOSEN", "mpqda_v2_m2")
    assert M.v2_config("mpqda_v2") == {"adapt": True}
    assert M.v2_config("mpqda_v2_no_tour") == {"adapt": True, "tour": 1, "r1_from": "population"}
    assert M.v2_config("mpqda_v2_no_adapt") == {"adapt": False}
    X, y = synthetic_xy(n=60, p=200, seed=2)
    for m in ("mpqda_v2", "mpqda_v2_no_gap", "mpqda_v2_no_new"):
        r = run_method(m, InnerCVFitness(X, y, fold_seed=3), X.shape[1], 5, np.random.default_rng([1, 5]), P=6, T=2)
        assert len(r["consensus"]) == 5 and r["n_eval"] <= 6 * 3 + 1


# ---------------------------------------------------------------- test 3: smoke test của runner với v2 và de_g
@pytest.mark.slow
def test_runner_smoke_v2_on_development_name(smoke_run):
    """Dữ liệu giả đặt tên tcga_dev_a (dữ liệu phát triển, cổng đăng ký không chặn): ứng viên M1–M3 và de_g."""
    data, repo, env = smoke_run["data"], smoke_run["repo"], smoke_run["env"]
    dst = data / C.dataset_dir("tcga_dev_a")
    if not dst.exists():
        shutil.copytree(data / C.dataset_dir(SMOKE_COHORT), dst)
    methods = ["mpqda_v2_m1", "mpqda_v2_m2", "mpqda_v2_m3", "de_g"]
    args = ["--scenario", "SMOKE_v2", "--metric", "logloss", "--datasets", "tcga_dev_a", "--methods", *methods,
            "--k", "2", "5", "--repeats", "0", "--folds", "0", "--pop", "6", "--generations", "3", "--workers", "2"]
    run_script(repo, "run_taimma.py", args, env)
    out = data / "results" / "survey_v2" / "taimma" / "SMOKE_v2" / "tcga_dev_a"
    sel = pd.read_csv(out / "selection_per_fold.csv")
    hist = pd.read_csv(out / "history.csv")
    assert set(sel.method) == {*(n for m in methods[:3] for n in (m, m + "_best")), "de_g"}
    assert sel.exact_k.all() and (sel.n_selected == sel.k).all() and (sel.n_eval <= 6 * 4 + 1).all()
    assert (sel[sel.method.str.startswith("mpqda_v2")].base_method == "mpqda_v2").all()
    assert sel[sel.method.isin(methods[:3])].search_seed.nunique() == 1          # cùng seed cho họ v2
    assert list(hist.columns[-2:]) == ["mean_CR", "n_new_genes"]
    v2 = hist[hist.method.isin(methods[:3])]
    assert v2.mean_CR.notna().all() and v2[v2.t > 0].n_new_genes.notna().all()
    assert hist[hist.method == "de_g"].mean_CR.isna().all()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["v2_chosen"] is None and cfg["v2_config"]["mpqda_v2_m3"] == {"tour": 2} and "head_tags" in cfg


@pytest.mark.slow
def test_runner_registration_gate_blocks_confirmatory_names(smoke_run):
    """Tên bộ GPL570 (SMOKE_COHORT) là dữ liệu xác nhận: ứng viên bị chặn (mã 2) kể cả ở chế độ smoke; mpqda_v2 khi chưa
    chốt V2_CHOSEN bị từ chối ngay ở dòng lệnh; v1 không bị chặn."""
    base = ["--scenario", "SMOKE_gate", "--metric", "logloss", "--datasets", SMOKE_COHORT, "--k", "5", "--repeats", "0",
            "--folds", "0", "--pop", "6", "--generations", "2"]
    res = run_script(smoke_run["repo"], "run_taimma.py", [*base, "--methods", "mpqda_v2_m1"], smoke_run["env"], check=False)
    assert res.returncode == 2 and "cổng đăng ký" in res.stderr
    res = run_script(smoke_run["repo"], "run_taimma.py", [*base, "--methods", "de_g"], smoke_run["env"], check=False)
    assert res.returncode == 2 and "V2_CHOSEN" in res.stderr
    res = run_script(smoke_run["repo"], "run_taimma.py", [*base, "--methods", "mpqda_v2"], smoke_run["env"], check=False)
    assert res.returncode == 2 and "V2_CHOSEN" in res.stderr
    assert not (smoke_run["data"] / "results" / "survey_v2" / "taimma" / "SMOKE_gate").exists()


# ---------------------------------------------------------------- test 4: v1 và bảy baseline trùng kết quả ở 4fea32f
def _j(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else repr(float(v))
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def test_v1_and_baselines_unchanged_since_4fea32f():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    old_cols = ref["history_columns"]
    assert list(HISTORY_COLUMNS[:len(old_cols)]) == old_cols and list(HISTORY_COLUMNS[len(old_cols):]) == ["mean_CR", "n_new_genes"]
    X, y = synthetic_xy(n=60, p=300, seed=4)
    for key, want in ref["runs"].items():
        metric, method, k = key.split("|")
        fit = InnerCVFitness(X, y, fold_seed=17, metric=metric)
        r = run_method(method, fit, X.shape[1], int(k), np.random.default_rng([31, int(k)]), P=6, T=3)
        got = {"best": sorted(int(g) for g in r["best"]), "best_fitness": repr(float(r["best_fitness"])), "n_eval": int(r["n_eval"]),
               "history": [{c: _j(h.get(c, np.nan)) for c in old_cols} for h in r["history"]]}
        if "consensus" in want:
            got["consensus"] = sorted(int(g) for g in r["consensus"])
            got["consensus_fitness"] = repr(float(r["consensus_fitness"]))
        assert got == want, key
        assert all(np.isnan(h.get("mean_CR", np.nan)) and np.isnan(h.get("n_new_genes", np.nan)) for h in r["history"]), key


# ---------------------------------------------------------------- B-C1
def test_phase_b_c1_is_survey_lr_with_c_fixed_at_one():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(70, int), np.ones(50, int)]
    rng.shuffle(y)
    X = rng.normal(size=(120, 8)) + 0.9 * y[:, None]
    Xtr, ytr, Xte, yte = X[:95], y[:95], X[95:], y[95:]
    a = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=4, criterion="c1")
    b = evaluate_panel(Xtr, ytr, Xte, yte, classifier="lr_l2", seed=4, c_grid=(1.0,))
    assert a.hyperparam == 1.0 and a.threshold == b.threshold and a.logloss == b.logloss
    assert np.array_equal(a.scores_test, b.scores_test)
