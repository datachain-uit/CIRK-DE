"""Waiver A1 of T0-v2: only item (4) is waived for mpqda_v2, lower bound, exactly the per-fold AUROC of the
S0v2 run at d5e72e4 (tests/data/t0v2_shuffle_d5e72e4, copied from S0v2_results.zip); every other difference still fails the gate."""
import json
import shutil

import pandas as pd
import pytest

from conftest import ROOT
from mpqda import config as M
from scripts import taimma_jobs as J

KAGGLE = ROOT / "tests" / "data" / "t0v2_shuffle_d5e72e4"
WAIVER = json.loads((ROOT / J.T0V2_WAIVER).read_text(encoding="utf-8"))


def _det(d, rows=3):
    d.mkdir(parents=True, exist_ok=True)
    genes = pd.DataFrame([{"cohort": "gse10245", "repeat": 0, "fold": f, "method": m, "k": 5, "rank": r, "gene": f"G{f}{r}"}
                          for f in range(rows) for m in ("mpqda_v2", "mpqda_v2_best") for r in range(1, 6)])
    hist = pd.DataFrame([{"cohort": "gse10245", "repeat": 0, "fold": f, "method": "mpqda_v2", "k": 5, "t": t,
                          "best_fitness": -0.3 + 0.01 * t, "leader_overlap": 0.7} for f in range(rows) for t in range(3)])
    sel = pd.DataFrame([{"cohort": "gse10245", "repeat": 0, "fold": f, "method": m, "k": 5, "n_selected": 5, "exact_k": True,
                         "n_eval": 900} for f in range(rows) for m in ("mpqda_v2", "mpqda_v2_best")])
    genes.to_csv(d / "selected_genes.csv", index=False)
    hist.to_csv(d / "history.csv", index=False)
    sel.to_csv(d / "selection_per_fold.csv", index=False)


def _root(tmp_path):
    for tag in ("a", "b", "j4"):
        _det(tmp_path / f"T0v2_det_{tag}" / "gse10245")
    sh = tmp_path / "T0v2_shuffle" / "gse37745"
    sh.mkdir(parents=True)
    for name in ("t0_shuffle_eval.csv", "selection_per_fold.csv"):
        shutil.copy(KAGGLE / name, sh / name)
    return tmp_path


def _check(root, waiver=None):
    return J.check_t0(root=root, prefix="T0v2", main="mpqda_v2", waiver=waiver)


def test_kaggle_run_fails_without_waiver(tmp_path):
    problems, info = _check(_root(tmp_path))
    assert len(problems) == 1 and problems[0].startswith("(4)") and "mpqda_v2 = 0.3915" in problems[0]
    assert info["waived"] == [] and abs(info["shuffle_mean_outer_auroc"]["mpqda_v2"] - 0.39153846153846156) < 1e-12


def test_committed_waiver_covers_exactly_this_run(tmp_path):
    problems, info = _check(_root(tmp_path), WAIVER)
    # the decision code is kept verbatim: it is recorded in docs/taimma/T0v2_waiver.json
    assert problems == [] and [w["method"] for w in info["waived"]] == ["mpqda_v2"] and info["waived"][0]["decision"] == "TQĐ33"
    assert WAIVER["items"][0]["bound"] == "lower" and len(WAIVER["items"][0]["per_fold_auroc"]) == M.N_OUTER_FOLDS


def test_waiver_does_not_cover_a_different_run(tmp_path):
    root = _root(tmp_path)
    path = root / "T0v2_shuffle" / "gse37745" / "t0_shuffle_eval.csv"
    ev = pd.read_csv(path, float_precision="round_trip")
    i = ev.index[(ev.method == "mpqda_v2") & (ev.repeat == 0) & (ev.fold == 0)][0]
    ev.loc[i, "auroc"] += 1e-6                                     # one fold differs: no longer the waived run
    ev.to_csv(path, index=False)
    problems, info = _check(root, WAIVER)
    assert len(problems) == 1 and "mpqda_v2" in problems[0] and info["waived"] == []


@pytest.mark.parametrize("change", ["upper", "other_method", "det"])
def test_waiver_never_covers_other_failures(tmp_path, change):
    root = _root(tmp_path)
    path = root / "T0v2_shuffle" / "gse37745" / "t0_shuffle_eval.csv"
    ev = pd.read_csv(path, float_precision="round_trip")
    if change == "upper":                                          # same method but above the upper bound
        ev.loc[ev.method == "mpqda_v2", "auroc"] = 1 - ev.loc[ev.method == "mpqda_v2", "auroc"]
    elif change == "other_method":                                 # de_g below the lower bound: not in the waiver
        ev.loc[ev.method == "de_g", "auroc"] = 0.30
    else:                                                          # (1) determinism broken
        (root / "T0v2_det_b" / "gse10245" / "history.csv").write_text("x\n", encoding="utf-8")
    ev.to_csv(path, index=False)
    problems, _ = _check(root, WAIVER)
    assert problems and not (change != "det" and problems == [])
    if change == "det":
        assert any(p.startswith("(1)") for p in problems)
    else:
        assert any(p.startswith("(4)") for p in problems)


def test_cmd_check_t0_records_the_waiver(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path / "results" / M.RESULT_SUBDIR)
    monkeypatch.setattr(J, "results_root", lambda: tmp_path / "results")
    monkeypatch.setattr(J, "pip_freeze_md5", lambda: "md5")
    assert J.cmd_check_t0(v2=True, waiver_path=J.T0V2_WAIVER) == 0
    report = json.loads((root / "T0v2_report.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["waived"][0]["method"] == "mpqda_v2" and len(report["waiver_sha256"]) == 64
    assert "waived under PREREG-v2, Section 12" in capsys.readouterr().out
    assert J.cmd_check_t0(v2=True) == 3                            # no waiver: fails as registered


def test_t0v2_shuffle_job_uses_the_committed_waiver():
    step = J.JOBS["t0v2_shuffle"].steps[-1]
    assert step == [J.SELF, "check-t0", "--v2", "--waiver", J.T0V2_WAIVER] and (ROOT / J.T0V2_WAIVER).is_file()
    assert "--waiver" not in J.JOBS["t0_shuffle"].steps[-1]        # T0 of v1 unchanged
