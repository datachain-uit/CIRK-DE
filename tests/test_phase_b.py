"""Pha B-LL (TQĐ11, Review-2026-11-14b): khác biệt duy nhất so với pha B của survey là tiêu chí chọn C.

Test 1–4 chép từ taimma_ref/test_phase_b_ll_ref.py; test 5 chạy hai script trên dữ liệu giả của smoke test; test 6 (bảng) nằm ở
tests/test_tables.py. Dữ liệu giả chỉ để kiểm mã.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from survey_v2.classify import evaluate_panel

from conftest import SMOKE_COHORT
from mpqda.phase_b import CRITERIA, evaluate_panel_ll, make_evaluator

FIELDS = ("hyperparam", "auroc", "bacc", "mcc", "logloss", "brier", "threshold")


def data(seed, effect, n=141, n1=60, k=10, n_test=28):
    """Cỡ mẫu và tỉ lệ lớp như GSE30219 (Bảng 4 của review)."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n - n1, int), np.ones(n1, int)]
    rng.shuffle(y)
    X = rng.normal(0, 1, size=(n, k)) + effect * y[:, None] * np.where(np.arange(k) % 2 == 0, 1, -1)
    te = rng.choice(n, n_test, replace=False)
    tr = np.setdiff1d(np.arange(n), te)
    return X[tr], y[tr], X[te], y[te]


def same(a, b):
    return all(getattr(a, f) == getattr(b, f) for f in FIELDS) and np.array_equal(a.scores_test, b.scores_test)


def test_auroc_criterion_reproduces_survey_exactly():
    for seed in range(8):
        Xtr, ytr, Xte, yte = data(seed, 0.6)
        assert same(evaluate_panel(Xtr, ytr, Xte, yte, classifier="lr_l2", seed=seed),
                    evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=seed, criterion="auroc"))


def test_logloss_criterion_removes_the_tie_artifact():
    c_auc, c_ll, ll_auc, ll_ll = [], [], [], []
    for seed in range(12):
        Xtr, ytr, Xte, yte = data(seed, 1.2)
        a = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=seed, criterion="auroc")
        b = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=seed, criterion="logloss")
        c_auc.append(a.hyperparam); c_ll.append(b.hyperparam); ll_auc.append(a.logloss); ll_ll.append(b.logloss)
    assert np.mean(np.array(c_auc) == 0.01) > 0.5          # tham chiếu: 7/12
    assert np.mean(np.array(c_ll) == 0.01) == 0.0           # tham chiếu: 0/12
    assert np.mean(ll_ll) < 0.5 * np.mean(ll_auc)           # tham chiếu: 0,266 -> 0,086


def test_small_n_rule_is_unchanged():
    Xtr, ytr, Xte, yte = data(3, 1.0, n=40, n1=12, n_test=10)    # lớp nhỏ còn <= 10 mẫu train
    assert min(np.sum(ytr == 0), np.sum(ytr == 1)) <= 10
    a = evaluate_panel(Xtr, ytr, Xte, yte, classifier="lr_l2", seed=3)
    b = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=3)
    assert same(a, b) and b.hyperparam == 1.0 and b.threshold == 0.5


def test_evaluator_delegates_other_classifiers():
    Xtr, ytr, Xte, yte = data(5, 0.8)
    ev = make_evaluator("logloss")
    a = evaluate_panel(Xtr, ytr, Xte, yte, classifier="svm_linear", seed=5)
    b = ev(Xtr, ytr, Xte, yte, classifier="svm_linear", seed=5)
    assert a.hyperparam == b.hyperparam and a.auroc == b.auroc and np.array_equal(a.scores_test, b.scores_test)
    assert same(ev(Xtr, ytr, Xte, yte, seed=5), evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=5))
    assert ev.criterion == "logloss" and CRITERIA == ("logloss", "auroc")
    with pytest.raises(ValueError):
        make_evaluator("brier")


def _classify(repo, script, out, env, extra=()):
    args = ["--datasets", SMOKE_COHORT, "--selection-roots", str(env["_SEL_ROOT"]), "--out", out, "--repeats", "0",
            "--k", "2", "5", "--random-draws", "5", *extra]
    clean = {k: v for k, v in env.items() if not k.startswith("_")}
    res = subprocess.run([sys.executable, str(repo / script), *args], cwd=repo, env=clean, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"{script} trả mã {res.returncode}\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}"


@pytest.mark.slow
def test_script_auroc_mode_matches_survey_and_logloss_mode_records_rule(smoke_run):
    """Mức script: --criterion auroc trùng script survey (mọi cột trừ thời gian fit), config.json có c_selection, vendor/
    không đổi. --criterion logloss ghi đúng quy tắc và không chọn C theo hoà AUROC."""
    from scripts.vendor_survey import check
    repo, env = smoke_run["repo"], dict(smoke_run["env"], _SEL_ROOT=smoke_run["out"].parent)
    results = smoke_run["data"] / "results" / "survey_v2"
    _classify(repo, "vendor/scripts/run_e1_classification.py", "taimma/PB_survey", env)
    _classify(repo, "scripts/run_taimma_phase_b.py", "taimma/PB_auc", env, ("--criterion", "auroc"))
    _classify(repo, "scripts/run_taimma_phase_b.py", "taimma/PB_ll", env, ("--criterion", "logloss"))
    a, b, ll = (results / "taimma" / d / SMOKE_COHORT for d in ("PB_survey", "PB_auc", "PB_ll"))
    pa = pd.read_csv(a / "classification_per_fold.csv", float_precision="round_trip").drop(columns="fit_time_s")
    pb = pd.read_csv(b / "classification_per_fold.csv", float_precision="round_trip").drop(columns="fit_time_s")
    assert len(pa) and set(pa.method) >= {"b0", "random_k", "mpqda_de", "ga"}
    assert pa.to_csv(index=False) == pb.to_csv(index=False)
    assert (a / "e1_random_k.csv").read_bytes() == (b / "e1_random_k.csv").read_bytes()
    cfg_a = json.loads((a / "config.json").read_text(encoding="utf-8"))
    cfg_b = json.loads((b / "config.json").read_text(encoding="utf-8"))
    cfg_ll = json.loads((ll / "config.json").read_text(encoding="utf-8"))
    assert "c_selection" not in cfg_a
    assert cfg_b["c_selection"]["criterion"] == "auroc" and cfg_ll["c_selection"]["criterion"] == "logloss"
    assert cfg_ll["c_selection"]["grid"] == [0.01, 0.1, 1.0, 10.0] and cfg_ll["c_selection"]["decision"] == "TQĐ11"
    assert cfg_ll["provenance"]["script"].replace("\\", "/").endswith("scripts/run_taimma_phase_b.py")
    assert not cfg_a["provenance"]["script"].replace("\\", "/").endswith("scripts/run_taimma_phase_b.py")
    # bản sao kho trong test đã commit nhưng chưa push: lỗi truy vết duy nhất được phép là "chưa push"; script ngoài vendor/
    # phải được nhận là đã theo dõi (lỗi gặp trên Kaggle ngày 15/11/2026 ở commit 38bc3fe)
    for cfg in (cfg_b, cfg_ll):
        errors = cfg.get("traceable_errors") or cfg["provenance"].get("traceable_errors")
        assert errors is not None and len(errors) == 1 and "remote" in errors[0], errors
    pll = pd.read_csv(ll / "classification_per_fold.csv", float_precision="round_trip")
    assert len(pll) == len(pa) and set(pll.method) == set(pa.method)
    assert check() == []                                                    # vendor/ của kho đang test không đổi
