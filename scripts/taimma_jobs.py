"""Kaggle jobs of the TAIMMA study, written after the pattern of scripts/kaggle_jobs.py of the vendored code.

    python scripts/taimma_jobs.py plan                 # job list by session S0–S3
    python scripts/taimma_jobs.py show <job>           # the commands of a job
    python scripts/taimma_jobs.py run <job> [--workers 4] [--log-dir logs]
    python scripts/taimma_jobs.py verify <job>         # traceable, commit, SHA-256 of the splits, expected row counts
    python scripts/taimma_jobs.py status
    python scripts/taimma_jobs.py check-t0             # the four conditions of T0 (PREREG, Section 9) on the results of S0
    python scripts/taimma_jobs.py build-dev            # build the three DEV subsets of v2 if absent, then check SHA-256
    python scripts/taimma_jobs.py check-registered     # HEAD carries a registration tag (original or amendment) and the tree is clean (job v1b)
    python scripts/taimma_jobs.py check-registered --require-tag taimma-2027-prereg-v2.2   # HEAD carries exactly this tag (job c4)
    python scripts/taimma_jobs.py check-t0 --v2 --waiver docs/taimma/T0v2_waiver.json   # T0-v2 with the approved waiver

Each job is a sequence of commands run one after another; phase A commands run in parallel according to --workers. Each `run` records `pip freeze`.
S1, S2, S3 open only after every job of S0 has passed. Priority order when running late: T0 → T1 → T1b → T3 → T2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from survey_v2 import config as C  # noqa: E402

from mpqda import config as M  # noqa: E402
from mpqda.runlog import head_commit, utf8_console  # noqa: E402

RUN, EVAL, GATE, SELF = "scripts/run_taimma.py", "scripts/run_taimma_eval.py", "scripts/check_gate_gt1.py", "scripts/taimma_jobs.py"
CALIB = "scripts/t0_calibration.py"
GPL570 = list(C.GPL570_COHORTS)
T0_SHUFFLE_RANGE = (0.40, 0.60)
T0_MIN_LEADER_OVERLAP = 0.2
T0_WAIVER_TOL = 1e-9                  # waiver A1: the per-fold AUROC must match the record to 1e-9
N_EVAL_MAX = M.P * (M.T + 1) + 1


@dataclass
class Job:
    name: str
    session: str
    what: str
    steps: list                                   # each element: list of arguments (after "python"), run in order
    outputs: list = field(default_factory=list)   # (relative folder under results/survey_v2, expected row count of selection_per_fold.csv or None)
    hours: str = ""


def _run(scenario, metric, datasets, methods, k, *extra, out=None, repeats=("0", "1", "2")):
    return [RUN, "--scenario", scenario, "--metric", metric, "--datasets", *datasets, "--methods", *methods,
            "--k", *map(str, k), "--repeats", *repeats, *(["--out", out] if out else []), *extra]


def _rows(methods, k, n_folds=M.N_OUTER_FOLDS) -> int:
    return n_folds * M.n_outputs(methods) * len(k)


def _phase_a_outputs(sub, datasets, methods, k, n_folds=M.N_OUTER_FOLDS):
    return [(f"{M.RESULT_SUBDIR}/{sub}/{d}", _rows(methods, k, n_folds)) for d in datasets]


def _eval(scenario, *extra):
    """Phase B of one scenario: B-LL first (main), then the phase B of the vendored code (B-AUC, sensitivity)."""
    return [[EVAL, "--scenario", scenario, "--phase-b", "ll", *extra], [EVAL, "--scenario", scenario, "--phase-b", "auc", *extra]]


def _b_outputs(sub, datasets):
    return [(f"{M.RESULT_SUBDIR}/{pb}/{sub}/{d}", None) for pb in ("T5LL", "T5") for d in datasets]


def build_jobs() -> dict[str, Job]:
    W = ["--workers", "{WORKERS}"]
    t1m, b7 = list(M.t1_methods()), list(M.B7_INCLUDED)
    variants = list(M.VARIANTS)
    det = ["--repeats", "0", "--folds", "0", "1", "2"]
    jobs = [
        Job("t0_det", "S0", "T0 determinism and n_jobs: GSE10245, repeat 0, folds 0–2, k = 5; QRK-DE and GA run twice; "
            "QRK-DE with n_jobs_eval 1 and 4",
            [_run("T0", "logloss", ["gse10245"], ["mpqda_de", "ga"], [5], "--folds", "0", "1", "2", "--workers", "1",
                  out=f"{M.RESULT_SUBDIR}/T0_det_{tag}", repeats=("0",)) for tag in ("a", "b")]
            + [_run("T0", "logloss", ["gse10245"], ["mpqda_de"], [5], "--folds", "0", "1", "2", "--workers", "1",
                    "--n-jobs-eval", "4", out=f"{M.RESULT_SUBDIR}/T0_det_j4", repeats=("0",))],
            [(f"{M.RESULT_SUBDIR}/T0_det_{tag}/gse10245", _rows(["mpqda_de", "ga"], [5], 3)) for tag in ("a", "b")]
            + [(f"{M.RESULT_SUBDIR}/T0_det_j4/gse10245", _rows(["mpqda_de"], [5], 3))], "0,1"),
        Job("t0_shuffle", "S0", "T0 shuffled labels: GSE37745, 15 folds, k = 5, QRK-DE and GA; then check the four conditions of T0",
            [_run("T0", "logloss", ["gse37745"], ["mpqda_de", "ga"], [5], "--shuffle-train-labels", *W,
                  out=f"{M.RESULT_SUBDIR}/T0_shuffle"), [SELF, "check-t0"]],
            _phase_a_outputs("T0_shuffle", ["gse37745"], ["mpqda_de", "ga"], [5]), "0,15"),
        Job("gt1", "S0", "G-T1 (session S0): GA, BPSO in AUROC mode on GSE10245, repeat 0, k ∈ {5, 10, 50}; then check_gate_gt1",
            [_run("T1b", "auroc", ["gse10245"], ["ga", "bpso"], [5, 10, 50], *W, out=f"{M.RESULT_SUBDIR}/GT1_s0",
                  repeats=("0",)),
             [GATE, "--run-root", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/GT1_s0", "--datasets", "gse10245",
              "--report", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/GT1_s0/gate_gt1.csv"]],
            _phase_a_outputs("GT1_s0", ["gse10245"], ["ga", "bpso"], [5, 10, 50], 5), "0,15"),
        Job("t1", "S1", "T1 main comparison (log-loss) on the three GPL570 cohorts, then phases B-LL and B-AUC",
            [_run("T1", "logloss", GPL570, t1m, [2, 5, 10, 50], *W), *_eval("T1")],
            _phase_a_outputs("T1", GPL570, t1m, [2, 5, 10, 50]) + _b_outputs("T1", GPL570), "3,5"),
        Job("t1_b7", "S1", "T1 on the B7 datasets that passed gates G1, G2 (k ∈ {5, 10}), then phases B-LL and B-AUC; skipped if none passed",
            [_run("T1", "logloss", b7, t1m, [5, 10], *W), *_eval("T1_b7")] if b7 else [],
            _phase_a_outputs("T1", b7, t1m, [5, 10]) + _b_outputs("T1", b7), "1,9"),
        Job("t1b", "S2", "T1b saturation (AUROC) in full, check_gate_gt1 on all of it, then phases B-LL and B-AUC",
            [_run("T1b", "auroc", GPL570, t1m, [5, 10, 50], *W),
             [GATE, "--run-root", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/T1b", "--report", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/T1b/gate_gt1.csv"],
             *_eval("T1b")],
            _phase_a_outputs("T1b", GPL570, t1m, [5, 10, 50]) + _b_outputs("T1b", GPL570), "3,0"),
        Job("t4", "S2", "T4 non-wrapper references (six selectors) once, then phases B-LL and B-AUC with B0, random-k",
            [[EVAL, "--scenario", "T4", "--phase-b", "ll"], [EVAL, "--scenario", "T4", "--phase-b", "auc", "--no-phase-a"]],
            [(f"{M.RESULT_SUBDIR}/T4/{d}", None) for d in [*GPL570, *b7]] + _b_outputs("T4", [*GPL570, *b7]), "1,5"),
        Job("t6", "S2", "T6 cost: GSE37745, repeat 0, folds 0–4, k = 10, QRK-DE with n_jobs_eval 1 and 4; run alone, at the end of the session",
            [_run("T6", "logloss", ["gse37745"], ["mpqda_de"], [10], "--workers", "1", "--n-jobs-eval", n,
                  out=f"{M.RESULT_SUBDIR}/T6_j{n}", repeats=("0",)) for n in ("1", "4")],
            [(f"{M.RESULT_SUBDIR}/T6_j{n}/gse37745", _rows(["mpqda_de"], [10], 5)) for n in ("1", "4")], "0,2"),
        Job("t2", "S3", "T2 ablation: seven variants, k ∈ {2, 5, 10}, three GPL570 cohorts, then phases B-LL and B-AUC",
            [_run("T2", "logloss", GPL570, variants, [2, 5, 10], *W), *_eval("T2")],
            _phase_a_outputs("T2", GPL570, variants, [2, 5, 10]) + _b_outputs("T2", GPL570), "1,6"),
        Job("t3", "S3", "T3 seed stability: GSE37745, s = 1..4, QRK-DE and the five tier-1 baselines, k ∈ {5, 10}, then phases B-LL "
            "and B-AUC",
            [step for s in (1, 2, 3, 4) for step in (
                _run(f"T3_s{s}", "logloss", ["gse37745"], ["mpqda_de", *M.TIER1], [5, 10], "--seed-index", str(s), *W),
                *_eval(f"T3_s{s}"))],
            [out for s in (1, 2, 3, 4) for out in (
                (f"{M.RESULT_SUBDIR}/T3_s{s}/gse37745", _rows(["mpqda_de", *M.TIER1], [5, 10])),
                *_b_outputs(f"T3_s{s}", ["gse37745"]))], "1,6"),
    ]
    return {j.name: j for j in jobs}


DEV_METHODS = ["mpqda_v2_m1", "mpqda_v2_m2", "mpqda_v2_m3", "mpqda_de", "de_g", "de_rand1", "ga"]
DEV_TCGA = [d for d in M.DEV_DATASETS if d.startswith("tcga_dev_")]


def build_dev_jobs() -> list[Job]:
    """Development phase of v2 (PREREG-v2, Section 2.2): development data only; never touches GPL570 or B7."""
    W = ["--workers", "{WORKERS}"]
    return [Job(
        "dev", "DEV", "DEV: build the three TCGA subsets, phase A (candidates M1–M3, v1, DE-G, DE/rand/1, GA; k ∈ {5, 10}), descriptive B-LL, "
        "dev_select (selection rule and gate G-DEV)",
        [[SELF, "build-dev"],
         _run("DEV", "logloss", DEV_TCGA, DEV_METHODS, [5, 10], *W, repeats=("0",)),
         _run("DEV", "logloss", ["gse42127"], DEV_METHODS, [5, 10], *W, repeats=("0", "1")),
         [EVAL, "--scenario", "DEV", "--phase-b", "ll"],
         ["scripts/dev_select.py", "--results", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/DEV", "--eval",
          "{RESULTS}/" + f"{M.RESULT_SUBDIR}/T5LL/DEV", "--out", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/DEV/dev_selection"]],
        _phase_a_outputs("DEV", DEV_TCGA, DEV_METHODS, [5, 10], 5) + _phase_a_outputs("DEV", ["gse42127"], DEV_METHODS, [5, 10], 10)
        + [(f"{M.RESULT_SUBDIR}/T5LL/DEV/{d}", None) for d in M.DEV_DATASETS], "2,5")]


V2_PAIR = ["mpqda_v2", "de_g"]
A2_TAG = "taimma-2027-prereg-v2.2"                                # amendment A2 of PREREG-v2: HV6
A2_METHODS = ["mpqda_v2", "mpqda_v2_no_gap"]
A2_K = [20, 50]
T0V2_WAIVER = "docs/taimma/T0v2_waiver.json"                     # amendment A1 of PREREG-v2
V1_METHODS = ["mpqda_de", *M.TIER1, *M.TIER2]                     # the eight methods of v1 reused (G-R)
PHASE_B_V2 = ("ll", "auc", "c1", "llw")                           # B-LL main; B-AUC, B-C1, B-LL-wide sensitivity
PHASE_B_DIRS4 = ("T5LL", "T5", "T5C1", "T5LLW")


def _eval4(scenario):
    """Phase B of v2: B-LL (main), then B-AUC, B-C1 and B-LL-wide (sensitivity), in that order (PREREG-v2, Section 5)."""
    return [[EVAL, "--scenario", scenario, "--phase-b", pb] for pb in PHASE_B_V2]


def _b_outputs4(sub, datasets):
    return [(f"{M.RESULT_SUBDIR}/{pb}/{sub}/{d}", None) for pb in PHASE_B_DIRS4 for d in datasets]


def v2_ablations() -> list[str]:
    """Ablations of C2: no gap, no new genes, no tournament; no adaptation added only when the chosen configuration is M2."""
    return [m for m in M.V2_ABLATIONS if m != "mpqda_v2_no_adapt" or M.V2_CHOSEN == "mpqda_v2_m2"]


def build_v2_jobs() -> list[Job]:
    """Confirmatory phase of v2 (PREREG-v2): can run only after the tag taimma-2027-prereg-v2 (registration gate in run_taimma.py)."""
    W = ["--workers", "{WORKERS}"]
    b7, abl = list(M.B7_INCLUDED), v2_ablations()
    det = [_run("T0v2", "logloss", ["gse10245"], V2_PAIR, [5], "--folds", "0", "1", "2", "--workers", "1",
                out=f"{M.RESULT_SUBDIR}/T0v2_det_{tag}", repeats=("0",)) for tag in ("a", "b")]
    det += [_run("T0v2", "logloss", ["gse10245"], ["mpqda_v2"], [5], "--folds", "0", "1", "2", "--workers", "1",
                 "--n-jobs-eval", "4", out=f"{M.RESULT_SUBDIR}/T0v2_det_j4", repeats=("0",))]
    gr_out = f"{M.RESULT_SUBDIR}/GR"
    v1_outputs = [n for m in V1_METHODS for n in M.output_names(m)]
    return [
        Job("t0v2_det", "S0v2", "T0-v2 determinism and n_jobs: GSE10245, repeat 0, folds 0–2, k = 5; mpqda_v2 and de_g twice; "
            "mpqda_v2 with n_jobs_eval 4", det,
            [(f"{M.RESULT_SUBDIR}/T0v2_det_{tag}/gse10245", _rows(V2_PAIR, [5], 3)) for tag in ("a", "b")]
            + [(f"{M.RESULT_SUBDIR}/T0v2_det_j4/gse10245", _rows(["mpqda_v2"], [5], 3))], "0,05"),
        Job("t0v2_shuffle", "S0v2", "T0-v2 shuffled labels: GSE37745, 15 folds, k = 5, mpqda_v2 and de_g; then check-t0 --v2",
            [_run("T0v2", "logloss", ["gse37745"], V2_PAIR, [5], "--shuffle-train-labels", *W,
                  out=f"{M.RESULT_SUBDIR}/T0v2_shuffle"), [SELF, "check-t0", "--v2", "--waiver", T0V2_WAIVER]],
            _phase_a_outputs("T0v2_shuffle", ["gse37745"], V2_PAIR, [5]), "0,1"),
        Job("gr", "S0v2", "G-R: rerun 16 units of v1 and the seven baselines (gse10245 repeat 0 fold 0 k = 5; gse37745 repeat 2 "
            "fold 4 k = 10) and compare with T1 of S1 (gene set, N_eval identical; best_fitness difference ≤ 1e-12)",
            [_run("GR", "logloss", ["gse10245"], V1_METHODS, [5], "--folds", "0", *W, out=gr_out, repeats=("0",)),
             _run("GR", "logloss", ["gse37745"], V1_METHODS, [10], "--folds", "4", *W, out=gr_out, repeats=("2",)),
             [GATE, "--run-root", "{RESULTS}/" + gr_out, "--reference", "{V1_T1}", "--datasets", "gse10245", "gse37745",
              "--methods", *v1_outputs, "--report", "{RESULTS}/" + gr_out + "/gate_gr.csv"]],
            [(f"{gr_out}/gse10245", _rows(V1_METHODS, [5], 1)), (f"{gr_out}/gse37745", _rows(V1_METHODS, [10], 1))], "0,15"),
        Job("v1b", "S0v2", "B-C1 and B-LL-wide for T1 of v1 and the seven baselines (panels of S1, read from {V1_T1}) on the three GPL570 cohorts "
            "and B7, so that HV2 has a reference under each criterion; runs only at a commit carrying a registration tag",
            [[SELF, "check-registered"]]
            + [[EVAL, "--scenario", sc, "--phase-b", pb, "--selection-root", "{V1_T1}"]
               for sc in ("T1", *(("T1_b7",) if b7 else ())) for pb in ("c1", "llw")],
            [(f"{M.RESULT_SUBDIR}/{pb}/T1/{d}", None) for pb in ("T5C1", "T5LLW") for d in [*GPL570, *b7]], "0,3"),
        Job("c1", "C1", "C1: mpqda_v2 and de_g on the three GPL570 cohorts (k ∈ {2, 5, 10, 50}) and the two B7 datasets (k ∈ {5, 10}); phase B with four "
            "criteria",
            [_run("T1v2", "logloss", GPL570, V2_PAIR, [2, 5, 10, 50], *W),
             *([_run("T1v2", "logloss", b7, V2_PAIR, [5, 10], *W)] if b7 else []),
             *_eval4("T1v2"), *(_eval4("T1v2_b7") if b7 else [])],
            _phase_a_outputs("T1v2", GPL570, V2_PAIR, [2, 5, 10, 50]) + _phase_a_outputs("T1v2", b7, V2_PAIR, [5, 10])
            + _b_outputs4("T1v2", [*GPL570, *b7]), "2,0"),
        Job("c2", "C2", "C2: ablations of v2 (" + ", ".join(abl) + "), k ∈ {2, 5, 10}, three GPL570 cohorts; phase B with four criteria",
            [_run("T2v2", "logloss", GPL570, abl, [2, 5, 10], *W), *_eval4("T2v2")],
            _phase_a_outputs("T2v2", GPL570, abl, [2, 5, 10]) + _b_outputs4("T2v2", GPL570), "1,5"),
        Job("c3", "C3", "C3: mpqda_v2 with seeds 1–4 on GSE37745, k ∈ {5, 10}; phase B with four criteria",
            [step for s_ in (1, 2, 3, 4) for step in (
                _run(f"T3v2_s{s_}", "logloss", ["gse37745"], ["mpqda_v2"], [5, 10], "--seed-index", str(s_), *W),
                *_eval4(f"T3v2_s{s_}"))],
            [out for s_ in (1, 2, 3, 4) for out in (
                (f"{M.RESULT_SUBDIR}/T3v2_s{s_}/gse37745", _rows(["mpqda_v2"], [5, 10])),
                *_b_outputs4(f"T3v2_s{s_}", ["gse37745"]))], "0,6"),
        Job("c4", "C4", "C4 (amendment A2, HV6): mpqda_v2 and mpqda_v2_no_gap on the three GPL570 cohorts, k ∈ {20, 50}, seed 0; "
            "phase A only. mpqda_v2 at k = 50 repeats C1 to check reproduction (gate R50, scripts/make_taimma_corrected.py)",
            [[SELF, "check-registered", "--require-tag", A2_TAG], _run("T4v2", "logloss", GPL570, A2_METHODS, A2_K, *W)],
            _phase_a_outputs("T4v2", GPL570, A2_METHODS, A2_K), "0,6"),
        Job("t0v2_calib", "CAL", "Permutation calibration of T0-v2 (descriptive only): GSE37745, 15 folds, k = 5, permutations b = 0..10 "
            "(b = 0 is the registered permutation), mpqda_v2 and GA, 50 random panels per (fold, b); in parallel with C1–C3",
            [[CALIB, "--workers", "{WORKERS}"]], [], "1,0"),
    ]


JOBS = {**build_jobs(), **{j.name: j for j in build_dev_jobs()}, **{j.name: j for j in build_v2_jobs()}}
SESSIONS = {"S0": ["t0_det", "t0_shuffle", "gt1"], "S1": ["t1", "t1_b7"], "S2": ["t1b", "t4", "t6"], "S3": ["t2", "t3"],
            "DEV": ["dev"], "S0v2": ["t0v2_det", "t0v2_shuffle", "gr", "v1b"], "C1": ["c1"], "C2": ["c2"], "C3": ["c3"],
            "CAL": ["t0v2_calib"], "C4": ["c4"]}


# ------------------------------------------------------------------------------------------------- helpers
def data_root() -> Path:
    root = os.environ.get("KLTN_DATA_ROOT")
    if not root:
        raise SystemExit("set KLTN_DATA_ROOT")
    return Path(root)


def results_root() -> Path:
    return data_root().joinpath(*C.RESULT_DIR)


def render(argv, workers: int = 4) -> list[str]:
    res = results_root().as_posix() if os.environ.get("KLTN_DATA_ROOT") else "$KLTN_DATA_ROOT/results/survey_v2"
    v1_t1 = os.environ.get("KLTN_V1_T1", "$KLTN_V1_T1")          # T1 of v1 (S1_results.zip) for gate G-R
    return [a.replace("{RESULTS}", res).replace("{WORKERS}", str(workers)).replace("{V1_T1}", v1_t1) for a in argv]


def pip_freeze_md5() -> str | None:
    """MD5 of `python -m pip freeze`, computed as in GATE2 of the notebook."""
    try:
        txt = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=180,
                             check=True).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"[pip freeze] cannot compute the MD5: {exc}", flush=True)
        return None
    return hashlib.md5(txt.encode("utf-8")).hexdigest()


def write_pip_freeze(job: Job, log_dir: Path) -> Path | None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / f"pip_freeze_{job.name}.txt"
        txt = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=180).stdout
        out.write_text(f"# python {sys.version.split()[0]}\n" + txt, encoding="utf-8")
        return out
    except Exception as exc:  # noqa: BLE001 — an auxiliary write step must not break the session
        print(f"[pip freeze] skipped: {exc}", flush=True)
        return None


def verify(job: Job, commit: str | None) -> list[str]:
    """traceable = true, commit_hash = HEAD, SHA-256 of the splits equal to the expected value, expected row counts reached."""
    problems, root = [], results_root()
    for rel, expected_rows in job.outputs:
        cfg_path = root / rel / "config.json"
        if not cfg_path.exists():
            problems.append(f"{rel}: missing config.json")
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("traceable") is not True:
            problems.append(f"{rel}: traceable = {cfg.get('traceable')} {cfg.get('traceable_errors', '')}")
        if commit and cfg.get("commit_hash") != commit:
            problems.append(f"{rel}: commit_hash {cfg.get('commit_hash')} differs from HEAD {commit}")
        if cfg.get("resumed_from"):
            problems.append(f"{rel}: checkpoints from another commit ({[r.get('commit_hash') for r in cfg['resumed_from']]})")
        if "split_sha256_match" in cfg and cfg["split_sha256_match"] is not True:
            # kept verbatim: this message is recorded in the archived session logs
            problems.append(f"{rel}: split_sha256 {cfg.get('split_sha256')} khác giá trị chờ {cfg.get('split_sha256_expected')}")
        if expected_rows is not None:
            path = root / rel / "selection_per_fold.csv"
            n = len(pd.read_csv(path)) if path.exists() else 0
            if n != expected_rows:
                problems.append(f"{rel}: selection_per_fold.csv has {n} rows, expected {expected_rows}")
    return problems


# ------------------------------------------------------------------------------------------------- T0
def _frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def same_values(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Two tables equal value by value (no tolerance), independent of the numeric type: an int column in a file holding only QRK-DE becomes float
    when the file also has GA rows that leave that column empty (0 against 0.0). NaN equals NaN."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.array_equal(x.to_numpy(dtype=float), y.to_numpy(dtype=float), equal_nan=True):
                return False
        elif not (x.astype(str) == y.astype(str)).all():
            return False
    return True


def waiver_covers(waiver: dict | None, method: str, ev: pd.DataFrame, lo: float, hi: float) -> bool:
    """The waiver (PREREG-v2, Section 12, A1) applies only to item (4), the exact method, the exact bound exceeded and the exact recorded per-fold AUROC
    (difference at most T0_WAIVER_TOL). A run giving other numbers is not waived."""
    if not waiver:
        return False
    part = ev[ev.method == method]
    mean = float(part.auroc.mean())
    side = "lower" if mean < lo else "upper" if mean > hi else None
    for item in waiver.get("items", []):
        if item.get("method") != method or item.get("bound") != side:
            continue
        if abs(mean - float(item["mean_auroc"])) > T0_WAIVER_TOL:
            return False
        want = {(int(r["repeat"]), int(r["fold"])): float(r["auroc"]) for r in item["per_fold_auroc"]}
        got = {(int(r.repeat), int(r.fold)): float(r.auroc) for r in part.itertuples(index=False)}
        return set(want) == set(got) and all(abs(got[key] - want[key]) <= T0_WAIVER_TOL for key in want)
    return False


def check_t0(root: Path | None = None, prefix: str = "T0", main: str = "mpqda_de",
             waiver: dict | None = None) -> tuple[list[str], dict]:
    """The four conditions of T0 (PREREG, Section 9) on the results of t0_det and t0_shuffle (prefix T0, main method mpqda_de), or
    of T0-v2 (prefix T0v2, mpqda_v2; PREREG-v2, Section 9). Returns (list of failures, recorded numbers). waiver: the approved
    waiver for item (4) (waiver_covers); items (1)–(3) and the fold count of item (4) are never waived."""
    root = results_root() / M.RESULT_SUBDIR if root is None else Path(root)
    a, b, j4 = (root / f"{prefix}_det_{t}" / "gse10245" for t in ("a", "b", "j4"))
    sh = root / f"{prefix}_shuffle" / "gse37745"
    problems, info = [], {}
    need = [d / n for d in (a, b, j4) for n in ("selected_genes.csv", "history.csv", "selection_per_fold.csv")]
    need += [sh / "t0_shuffle_eval.csv", sh / "selection_per_fold.csv"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        return [f"missing file: {m}" for m in missing], info
    for name in ("selected_genes.csv", "history.csv"):                       # (1) determinism
        same = (a / name).read_bytes() == (b / name).read_bytes()
        info[f"det_identical_{name}"] = same
        if not same:
            problems.append(f"(1) determinism: {name} of two runs with the same seed differ")
    for name in ("selected_genes.csv", "history.csv"):                       # (2) n_jobs_eval 1 and 4
        one = _frame(a / name)
        one = one[one.method.isin(M.output_names(main))].reset_index(drop=True)
        same = same_values(one, _frame(j4 / name))
        info[f"njobs_identical_{name}"] = same
        if not same:
            problems.append(f"(2) parallelism: {name} of QRK-DE differs between n_jobs_eval 1 and 4")
    sel = pd.concat([_frame(d / "selection_per_fold.csv") for d in (a, b, j4, sh)], ignore_index=True)   # (3)
    bad = sel[(sel.n_selected != sel.k) | (~sel.exact_k.astype(bool)) | (sel.n_eval > N_EVAL_MAX)]
    info["n_rows_checked"], info["n_eval_max"] = int(len(sel)), int(sel.n_eval.max())
    if len(bad):
        problems.append(f"(3) {len(bad)} rows with |S| ≠ k, exact_k = False or n_eval > {N_EVAL_MAX}")
    ev = _frame(sh / "t0_shuffle_eval.csv")                                   # (4) shuffled labels
    n_folds = ev.groupby("method").size()
    means = ev.groupby("method").auroc.mean()
    info["shuffle_mean_outer_auroc"] = {m: float(v) for m, v in means.items()}
    info["shuffle_mean_inner_fitness"] = {m: float(v) for m, v in ev.groupby("method").output_fitness.mean().items()}
    info["waived"] = []
    for method, value in means.items():
        if n_folds[method] != M.N_OUTER_FOLDS:
            problems.append(f"(4) {method}: {n_folds[method]} folds, expected {M.N_OUTER_FOLDS}")
        if not T0_SHUFFLE_RANGE[0] <= value <= T0_SHUFFLE_RANGE[1]:
            if n_folds[method] == M.N_OUTER_FOLDS and waiver_covers(waiver, method, ev, *T0_SHUFFLE_RANGE):
                info["waived"].append({"item": "(4)", "method": method, "mean_auroc": float(value),
                                       "decision": waiver.get("decision")})
            else:
                problems.append(f"(4) shuffled labels: mean outer AUROC of {method} = {value:.4f}, "
                                f"outside {list(T0_SHUFFLE_RANGE)}")
    hist = _frame(a / "history.csv")
    overlap = float(hist[(hist.method == main) & (hist.t == 1)].leader_overlap.mean())
    info["leader_overlap_t1_gse10245"] = overlap
    info["leader_overlap_report_now"] = bool(overlap < T0_MIN_LEADER_OVERLAP)
    return problems, info


def cmd_check_t0(v2: bool = False, waiver_path: Path | None = None) -> int:
    prefix, main = ("T0v2", "mpqda_v2") if v2 else ("T0", "mpqda_de")
    waiver = None
    if waiver_path is not None:
        path = ROOT / waiver_path if not Path(waiver_path).is_absolute() else Path(waiver_path)
        waiver = json.loads(path.read_text(encoding="utf-8"))
    problems, info = check_t0(prefix=prefix, main=main, waiver=waiver)
    if waiver is not None:
        info["waiver_file"] = str(waiver_path)
        info["waiver_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    info["pip_freeze_md5"] = pip_freeze_md5()
    print(f"PIP_FREEZE_MD5 {info['pip_freeze_md5']}")
    out = results_root() / M.RESULT_SUBDIR / f"{prefix}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": not problems, "problems": problems, **info}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info.get("leader_overlap_report_now"):
        print(f"REPORT NOW: mean leader_overlap at t = 1 below {T0_MIN_LEADER_OVERLAP} (band re-encoding has no effect on real data)")
    waived = "; ".join(f"{w['item']} {w['method']} {w['mean_auroc']:.4f} ({w['decision']})" for w in info.get("waived", []))
    print((f"{prefix} PASSED" + (f" — waived under PREREG-v2, Section 12: {waived}" if waived else "")) if not problems
          else f"{prefix} FAILED:\n  " + "\n  ".join(problems))
    return 0 if not problems else 3


# ------------------------------------------------------------------------------------------------- DEV (v2)
def cmd_build_dev() -> int:
    """Build the three DEV subsets in the (writable) data root if absent; check the SHA-256 of the splits against mpqda.config. The build is deterministic
    in content (same values_sha256 and same splits); the SHA-256 of expression.h5 changes with each write and is not compared."""
    from survey_v2.data import sha256
    root = data_root()
    missing = [d for d in DEV_TCGA if not (root / C.dataset_dir(d) / "repeated_splits.csv").is_file()]
    if missing:
        if len(missing) != len(DEV_TCGA):
            print(f"STOP: only part of the DEV subsets exists ({missing} missing); delete the old DEV folders and rebuild")
            return 2
        code = subprocess.run([sys.executable, "scripts/build_dev_datasets.py", "--data-root", str(root)], cwd=ROOT).returncode
        if code != 0:
            return code
    problems = []
    for d in M.DEV_DATASETS:
        path = root / C.dataset_dir(d) / "repeated_splits.csv"
        got = sha256(path) if path.is_file() else None
        print(f"{d}: repeated_splits.csv {got}")
        if got != M.EXPECTED_SPLIT_SHA256[d]:
            # kept verbatim: this message is recorded in the archived session logs
            problems.append(f"{d}: {got} khác giá trị chờ {M.EXPECTED_SPLIT_SHA256[d]}")
    print("DEV_OK" if not problems else "DEV FAILED:\n  " + "\n  ".join(problems))
    return 0 if not problems else 3


# ------------------------------------------------------------------------------------------------- commands
def cmd_check_registered(git=None, require: str | None = None) -> int:
    """0 when HEAD carries a registration tag (original or amendment; with require: exactly that tag) and the working tree has no uncommitted change in
    tracked files; 1 otherwise."""
    from mpqda import registration as R
    git = git or R.git
    problems = []
    tag = R.registration_tag(git)
    if tag is None:
        problems.append(f"HEAD carries no registration tag ({', '.join(R.REGISTRATION_TAGS)})")
    if require is not None:
        if require not in R.REGISTRATION_TAGS:
            problems.append(f"{require} is not a registration tag")
        elif require not in set(git("tag", "--points-at", "HEAD").split()):
            problems.append(f"HEAD does not carry the tag {require}")
        else:
            tag = require
    if git("status", "--porcelain", "--untracked-files=no").strip():
        problems.append("the working tree has uncommitted changes")
    print(f"REGISTERED_OK {tag}" if not problems else "DỪNG: " + "; ".join(problems), flush=True)
    return 0 if not problems else 1


def cmd_plan():
    for session, names in SESSIONS.items():
        for name in names:
            j = JOBS[name]
            print(f"{session}  {name:11s} {j.hours:>5s} h  {j.what}")


def cmd_show(job: Job, workers: int):
    print(f"{job.name} — {job.what}\nsession {job.session}; estimate {job.hours} hours")
    for s in job.steps:
        print("  python", " ".join(render(s, workers)))
    print("checked outputs:", ", ".join(f"{rel} ({rows} rows)" if rows is not None else rel for rel, rows in job.outputs) or "—")


def cmd_run(job: Job, workers: int, log_dir: Path) -> int:
    if not job.steps:
        print(f"{job.name}: no command (no B7 dataset in mpqda.config.B7_INCLUDED)")
        return 0
    commit = head_commit()
    print(f"JOB {job.name} | commit {commit} | data {data_root()} | workers {workers}", flush=True)
    write_pip_freeze(job, log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"commands_{job.name}.log", "a", encoding="utf-8") as cmd_log:
        for i, step in enumerate(job.steps):
            argv = render(step, workers)
            t0 = time.time()
            print(f"[start {job.name}#{i}] python {' '.join(argv)}", flush=True)
            cmd_log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} commit {commit} python {' '.join(argv)}\n")
            cmd_log.flush()
            with open(log_dir / f"{job.name}_{i}.log", "w", encoding="utf-8") as log:
                code = subprocess.run([sys.executable, *argv], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"[done  {job.name}#{i}] exit {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
            cmd_log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} exit {code}\n")
            if code != 0:
                tail = (log_dir / f"{job.name}_{i}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                print("    " + "\n    ".join(tail), flush=True)
                print(f"ERROR: step {i} exited with code {code}. The written checkpoints are kept; rerunning the same job resumes.")
                return 1
    problems = verify(job, commit)
    if problems:
        print("VERIFY FAILED:\n  " + "\n  ".join(problems))
        return 3
    print(f"JOB_OK {job.name}: " + ", ".join(rel for rel, _ in job.outputs))
    return 0


def cmd_status():
    root = results_root()
    for session, names in SESSIONS.items():
        for name in names:
            j = JOBS[name]
            have = [rel for rel, _ in j.outputs if (root / rel / "config.json").exists()]
            mark = "xong" if j.outputs and len(have) == len(j.outputs) else (f"{len(have)}/{len(j.outputs)}" if j.outputs else "—")
            print(f"{session}  {name:11s} {mark}")


def main(argv=None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "list", "show", "run", "verify", "status", "check-t0", "build-dev",
                                        "check-registered"])
    ap.add_argument("--waiver", type=Path, default=None, help="check-t0: approved waiver for item (4) (PREREG-v2, Section 12)")
    ap.add_argument("job", nargs="?")
    ap.add_argument("--workers", type=int, default=4, help="number of phase A units run in parallel (Kaggle: 4 CPUs)")
    ap.add_argument("--log-dir", type=Path, default=Path(os.environ.get("KLTN_LOG_DIR", "logs")))
    ap.add_argument("--v2", action="store_true", help="check-t0: check T0-v2 (T0v2_det_*, T0v2_shuffle, mpqda_v2)")
    ap.add_argument("--require-tag", default=None, help="check-registered: HEAD must carry exactly this tag (job c4: amendment A2)")
    args = ap.parse_args(argv)
    if args.command in ("plan", "list"):
        cmd_plan()
        return 0
    if args.command == "status":
        cmd_status()
        return 0
    if args.command == "build-dev":
        return cmd_build_dev()
    if args.command == "check-t0":
        return cmd_check_t0(v2=args.v2, waiver_path=args.waiver)
    if args.command == "check-registered":
        return cmd_check_registered(require=args.require_tag)
    if args.job not in JOBS:
        ap.error(f"the job must be one of: {', '.join(JOBS)}")
    job = JOBS[args.job]
    if args.command == "show":
        cmd_show(job, args.workers)
        return 0
    if args.command == "verify":
        problems = verify(job, head_commit())
        print("VERIFY_OK" if not problems else "VERIFY FAILED:\n  " + "\n  ".join(problems))
        return 0 if not problems else 3
    return cmd_run(job, args.workers, args.log_dir)


if __name__ == "__main__":
    sys.exit(main())
