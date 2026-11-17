"""Bản vá của Review-2026-11-17 (taimma_ref_v3): B-LL-wide (TQĐ27), việc v1b và lệnh check-registered (TQĐ30), HV5 và cột
family_size của các bảng HV (TQĐ27, TQĐ29). Dữ liệu giả chỉ để kiểm mã, không phải kết quả thực nghiệm."""
import json

import numpy as np
import pandas as pd
import pytest
from survey_v2.classify import evaluate_panel

from mpqda import config as M
from mpqda.phase_b import C_FIXED, C_GRID_WIDE, evaluate_panel_ll, grid_for, make_evaluator
from scripts import make_taimma_v2_tables as V
from scripts import run_taimma_eval as E
from scripts import run_taimma_phase_b as PB
from scripts import taimma_jobs as J
from test_phase_b import data
from test_tables import COHORTS, fake_results
from test_v2_tables import write_v2

B7 = list(M.B7_INCLUDED)


# ---------------------------------------------------------------- B-LL-wide
def test_grid_for_each_criterion():
    assert grid_for("c1") == (C_FIXED,) and grid_for("llwide", (1.0,)) == C_GRID_WIDE
    assert grid_for("logloss", (0.1, 1.0)) == (0.1, 1.0) and grid_for("auroc") == (0.01, 0.1, 1.0, 10.0)
    assert C_GRID_WIDE == (1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0) and list(C_GRID_WIDE) == sorted(C_GRID_WIDE)


def test_llwide_is_logloss_on_the_wide_grid_whatever_grid_the_caller_passes():
    Xtr, ytr, Xte, yte = data(8, 0.6)
    a = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=8, criterion="llwide")
    b = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=8, criterion="logloss", c_grid=C_GRID_WIDE)
    c = make_evaluator("llwide")(Xtr, ytr, Xte, yte, seed=8, c_grid=(1.0,))
    for r in (b, c):
        assert r.hyperparam == a.hyperparam and r.threshold == a.threshold and r.logloss == a.logloss
        assert np.array_equal(r.scores_test, a.scores_test)


def test_llwide_reaches_below_the_original_grid_on_noise():
    Xtr, ytr, Xte, yte = data(3, 0.0)                   # không có tín hiệu: log-loss nhỏ nhất ở C nhỏ nhất
    a = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=3, criterion="llwide")
    b = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=3, criterion="logloss")
    assert a.hyperparam in C_GRID_WIDE and a.hyperparam < 0.01 and b.hyperparam == 0.01


def test_llwide_keeps_the_small_n_rule():
    Xtr, ytr, Xte, yte = data(3, 1.0, n=40, n1=12, n_test=10)
    a = evaluate_panel(Xtr, ytr, Xte, yte, classifier="lr_l2", seed=3)
    b = evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=3, criterion="llwide")
    assert b.hyperparam == 1.0 and b.threshold == 0.5 and np.array_equal(a.scores_test, b.scores_test)


def test_phase_b_script_records_the_wide_grid():
    assert PB.GRIDS["llwide"] == list(C_GRID_WIDE) and PB.DECISIONS.get("llwide") == "TQĐ27"
    assert set(PB.RULES) == set(PB.GRIDS) == {"logloss", "auroc", "c1", "llwide"}


# ---------------------------------------------------------------- run_taimma_eval
def test_eval_llw_writes_to_t5llw_and_checks_its_criterion(tmp_path, monkeypatch):
    assert E.PHASE_B["llw"] == "T5LLW" and E.classify_script("llw") == [E.CLF_LL, "--criterion", "llwide"]
    assert E.classify_script("auc") == [E.CLF] and E.classify_script("c1") == [E.CLF_LL, "--criterion", "c1"]
    monkeypatch.setattr(E, "results_root", lambda: tmp_path)
    for sub, crit in (("T5LLW", "llwide"), ("T5LL", "llwide"), ("T5C1", "c1"), ("T5", None)):
        d = tmp_path / M.RESULT_SUBDIR / sub / "T1" / "gse10245"
        d.mkdir(parents=True)
        cfg = {"traceable": True, "commit_hash": "abc", **({"c_selection": {"criterion": crit}} if crit else {})}
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    ok = lambda sub: E.check_configs(f"{M.RESULT_SUBDIR}/{sub}/T1", ["gse10245"], "abc")  # noqa: E731
    assert ok("T5LLW") == [] and ok("T5C1") == [] and ok("T5") == []
    assert len(ok("T5LL")) == 1 and "chờ logloss" in ok("T5LL")[0]          # T5LL không nhận nhầm tiêu chí của T5LLW


def test_eval_selection_root_overrides_only_the_panel_source(tmp_path):
    cmd, out = E.phase_b_command("T1", phase_b="llw", selection_root=tmp_path)
    assert cmd[cmd.index("--selection-roots") + 1] == str(tmp_path) and out == f"{M.RESULT_SUBDIR}/T5LLW/T1"
    cmd0, _ = E.phase_b_command("T1", phase_b="llw")
    assert cmd0[cmd0.index("--selection-roots") + 1].endswith("T1") and cmd0[cmd0.index("--selection-roots") + 1] != str(tmp_path)
    assert E.main(["--scenario", "T1", "--phase-b", "c1", "--selection-root", str(tmp_path), "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        E.main(["--scenario", "T4", "--selection-root", str(tmp_path), "--dry-run"])
    with pytest.raises(SystemExit):
        E.main(["--scenario", "T1", "--selection-root", str(tmp_path / "missing"), "--dry-run"])


# ---------------------------------------------------------------- taimma_jobs
def test_v2_jobs_run_four_phase_b_criteria():
    for name in ("c1", "c2", "c3"):
        job = J.JOBS[name]
        crits = [s[s.index("--phase-b") + 1] for s in job.steps if s[0] == J.EVAL]
        assert crits and set(crits) == {"ll", "auc", "c1", "llw"} and crits[:4] == ["ll", "auc", "c1", "llw"]
        assert any("/T5LLW/" in rel for rel, _ in job.outputs)


def test_v1b_job_reads_s1_panels_after_the_registration_check(monkeypatch):
    job = J.JOBS["v1b"]
    assert job.session == "S0v2" and J.SESSIONS["S0v2"][-1] == "v1b" and job.steps[0] == [J.SELF, "check-registered"]
    evals = job.steps[1:]
    want = [("T1", "c1"), ("T1", "llw")] + ([("T1_b7", "c1"), ("T1_b7", "llw")] if B7 else [])
    assert [(s[s.index("--scenario") + 1], s[s.index("--phase-b") + 1]) for s in evals] == want
    assert all(s[s.index("--selection-root") + 1] == "{V1_T1}" for s in evals)
    monkeypatch.setenv("KLTN_V1_T1", "/kaggle/working/_v1/results/survey_v2/taimma/T1")
    assert J.render(evals[0])[-1] == "/kaggle/working/_v1/results/survey_v2/taimma/T1"
    rels = {rel for rel, _ in job.outputs}
    assert rels == {f"{M.RESULT_SUBDIR}/{pb}/T1/{d}" for pb in ("T5C1", "T5LLW") for d in [*COHORTS, *B7]}


def test_check_registered(capsys):
    def fake(tags, dirty):
        def git(*args):
            return tags if args[0] == "tag" else dirty
        return git
    assert J.cmd_check_registered(fake("taimma-2027-prereg-v2\n", "")) == 0
    assert J.cmd_check_registered(fake("taimma-2027-v1-results\n", "")) == 1
    assert J.cmd_check_registered(fake("taimma-2027-prereg-v2\n", " M mpqda/config.py\n")) == 1
    assert "REGISTERED_OK" in capsys.readouterr().out


# ---------------------------------------------------------------- bảng v2: họ Holm, HV2 dưới B-C1 và B-LL-wide, HV5
def _results(tmp_path, v1b=False):
    root = tmp_path / "results"
    fake_results(root)
    write_v2(root)
    for c in B7:
        src = root / "T5LL" / "T1" / COHORTS[0]
        dst = root / "T5LL" / "T1" / c
        dst.mkdir(parents=True, exist_ok=True)
        ev = pd.read_csv(src / "classification_per_fold.csv", encoding="utf-8-sig").assign(cohort=c)
        ev.to_csv(dst / "classification_per_fold.csv", index=False, encoding="utf-8-sig")
    if v1b:                                       # việc v1b: B-C1, B-LL-wide của T1 v1 (ở đây chép từ T5LL)
        for sub in ("T5C1", "T5LLW"):
            for c in COHORTS:
                dst = root / sub / "T1" / c
                dst.mkdir(parents=True, exist_ok=True)
                (dst / "classification_per_fold.csv").write_bytes((root / "T5LL" / "T1" / c / "classification_per_fold.csv").read_bytes())
    return root


def test_every_hv_table_carries_its_registered_family_size(tmp_path):
    out = tmp_path / "out"
    info = V.build(_results(tmp_path), out)
    files = {"HV1": "stats_hv1_search", "HV2": "stats_hv2_generalization", "HV2_BAUC": "stats_hv2_generalization_bauc",
             "HV3": "stats_hv3_stability", "HV4": "table_v2_ablation", "HV5": "stats_hv5_b7"}
    for family, name in files.items():
        assert name in info["written"], name
        t = pd.read_csv(out / f"{name}.csv")
        size = V.FAMILY_SIZE[family.split("_")[0]]
        assert t.holm_family.eq(family).all() and t.family_size.eq(size).all() and len(t) == size, name
        p_col = "p_two_sided" if family.startswith("HV2") else "p"
        i = t[p_col].idxmin()                    # Holm trên cả họ: p_holm nhỏ nhất bằng cỡ họ nhân p nhỏ nhất
        assert np.isclose(t.p_holm[i], min(1.0, size * t[p_col][i])), name


def test_hv2_under_bc1_and_bllw_once_v1b_has_run(tmp_path):
    out = tmp_path / "out"
    info = V.build(_results(tmp_path, v1b=True), out)
    for sfx in ("_bc1", "_bllw"):
        t = pd.read_csv(out / f"stats_hv2_generalization{sfx}.csv")
        assert len(t) == 4 and t.holm_family.eq(f"HV2{sfx.upper()}").all() and t.n.eq(45).all()
        main = pd.read_csv(out / f"table_v2_main{sfx}.csv")
        assert {"mpqda_de", "bpso", "mpqda_v2", "de_g"} <= set(main.method)
    claims = pd.read_csv(out / "claims_v2.csv")
    assert set(claims.phase_b) >= {"B-LL", "B-AUC", "B-C1", "B-LL-wide"} and claims.claim_id.is_unique
    cc = pd.read_csv(out / "diagnostics_v2_c_choice.csv")
    assert set(cc.phase_b) == {"B-LL", "B-AUC", "B-LL-wide"} and cc.share_at_upper.between(0, 1).all()
    assert "stats_hv5_b7" in info["written"]


def _b7_frame(shift5, shift10, n_rep=3, n_fold=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, shift in ((5, shift5), (10, shift10)):
        for r in range(n_rep):
            for f in range(n_fold):
                base, j = float(rng.uniform(0.70, 0.85)), 1e-4 * (n_fold * r + f)   # hiệu khác nhau từng fold: Wilcoxon chính xác
                for m, au, ll in (("mpqda_de", base, 0.55), ("mpqda_v2", base - shift - np.sign(shift) * j, 0.60),
                                  ("ga", base - 0.01, 0.65)):
                    rows.append({"cohort": V.HV5_COHORT, "repeat": r, "fold": f, "method": m, "k": k, "model": "lr_l2",
                                 "auroc": au, "logloss": ll + 0.01 * rng.random()})
    return pd.DataFrame(rows)


def _b7_sel(n_rep=3, n_fold=5):
    rows = [{"cohort": V.HV5_COHORT, "repeat": r, "fold": f, "method": m, "k": k, "best_fitness": fit}
            for k in (5, 10) for r in range(n_rep) for f in range(n_fold)
            for m, fit in (("mpqda_de", -0.50 - 0.001 * f), ("mpqda_v2", -0.48 - 0.001 * f))]
    return pd.DataFrame(rows)


def test_hv5_one_sided_wilcoxon_over_15_folds_with_holm_over_k():
    t = V.hv5(_b7_frame(shift5=0.02, shift10=-0.01), _b7_sel()).set_index("k")
    assert np.isclose(t.inner_fitness_v2_minus_v1, 0.02).all() and t.share_v2_inner_better.eq(1.0).all()
    assert t.n.eq(15).all() and t.family_size.eq(2).all()
    assert np.isclose(t.auroc_v1_minus_v2[5], 0.02 + 7e-4) and t.share_v2_lower[5] == 1.0 and t.share_v2_lower[10] == 0.0
    assert np.isclose(t.p[5], 2.0 ** -15) and np.isclose(t.p_holm[5], 2 * 2.0 ** -15) and bool(t.significant[5])
    assert t.p[10] > 0.99 and not bool(t.significant[10])
    assert (t.logloss_v2_minus_v1 > 0).all() and np.isclose(t.auroc_ga_minus_v1, -0.01).all()
    assert (t.logloss_ci_low <= t.logloss_v2_minus_v1).all() and (t.logloss_v2_minus_v1 <= t.logloss_ci_high).all()


def test_hv5_without_selection_leaves_inner_columns_empty():
    t = V.hv5(_b7_frame(0.02, 0.02))
    assert t.inner_fitness_v2_minus_v1.isna().all() and len(t) == 2


def test_hv5_refuses_a_missing_cohort():
    with pytest.raises(ValueError):
        V.hv5(_b7_frame(0.02, 0.02).assign(cohort="tcga_luad_egfr"))


@pytest.mark.slow
def test_script_llwide_records_and_uses_the_wide_grid(smoke_run):
    """Mức script: --criterion llwide ghi lưới rộng và TQĐ27 vào config.json; mọi C của LR-L2 nằm trong lưới rộng."""
    from conftest import SMOKE_COHORT
    from test_phase_b import _classify
    repo, env = smoke_run["repo"], dict(smoke_run["env"], _SEL_ROOT=smoke_run["out"].parent)
    _classify(repo, "scripts/run_taimma_phase_b.py", "taimma/PB_llw", env, ("--criterion", "llwide"))
    d = smoke_run["data"] / "results" / "survey_v2" / "taimma" / "PB_llw" / SMOKE_COHORT
    sel = json.loads((d / "config.json").read_text(encoding="utf-8"))["c_selection"]
    assert sel["criterion"] == "llwide" and sel["grid"] == list(C_GRID_WIDE) and sel["decision"] == "TQĐ27"
    ev = pd.read_csv(d / "classification_per_fold.csv", float_precision="round_trip")
    lr = ev[(ev.model == "lr_l2") & ev.C.notna()]
    assert len(lr) and lr.C.isin(list(C_GRID_WIDE)).all()
