"""Việc chạy trên Kaggle của bài TAIMMA (Mục 4.4), viết theo mẫu scripts/kaggle_jobs.py của survey.

    python scripts/taimma_jobs.py plan                 # danh sách việc theo phiên S0–S3
    python scripts/taimma_jobs.py show <việc>          # các lệnh của một việc
    python scripts/taimma_jobs.py run <việc> [--workers 4] [--log-dir logs]
    python scripts/taimma_jobs.py verify <việc>        # traceable, commit, SHA-256 bản chia, số dòng chờ
    python scripts/taimma_jobs.py status
    python scripts/taimma_jobs.py check-t0             # bốn điều kiện của T0 (Mục 6) trên kết quả S0
    python scripts/taimma_jobs.py build-dev            # dựng ba tập DEV của v2 nếu chưa có, rồi kiểm SHA-256 (TQĐ23)
    python scripts/taimma_jobs.py check-registered     # HEAD mang tag taimma-2027-prereg-v2 và cây sạch (việc v1b)

Mỗi việc là một chuỗi lệnh chạy lần lượt; lệnh pha A tự chạy song song theo --workers. Mỗi `run` ghi `pip freeze`.
S1, S2, S3 chỉ mở sau khi mọi việc của S0 đạt. Thứ tự ưu tiên khi chậm: T0 → T1 → T1b → T3 → T2.
"""
from __future__ import annotations

import argparse
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
GPL570 = list(C.GPL570_COHORTS)
T0_SHUFFLE_RANGE = (0.40, 0.60)
T0_MIN_LEADER_OVERLAP = 0.2
N_EVAL_MAX = M.P * (M.T + 1) + 1


@dataclass
class Job:
    name: str
    session: str
    what: str
    steps: list                                   # mỗi phần tử: list đối số (sau "python"), chạy lần lượt
    outputs: list = field(default_factory=list)   # (thư mục tương đối results/survey_v2, số dòng chờ của selection_per_fold.csv hoặc None)
    hours: str = ""


def _run(scenario, metric, datasets, methods, k, *extra, out=None, repeats=("0", "1", "2")):
    return [RUN, "--scenario", scenario, "--metric", metric, "--datasets", *datasets, "--methods", *methods,
            "--k", *map(str, k), "--repeats", *repeats, *(["--out", out] if out else []), *extra]


def _rows(methods, k, n_folds=M.N_OUTER_FOLDS) -> int:
    return n_folds * M.n_outputs(methods) * len(k)


def _phase_a_outputs(sub, datasets, methods, k, n_folds=M.N_OUTER_FOLDS):
    return [(f"{M.RESULT_SUBDIR}/{sub}/{d}", _rows(methods, k, n_folds)) for d in datasets]


def _eval(scenario, *extra):
    """Pha B của một kịch bản: B-LL trước (chính, TQĐ11), rồi pha B của survey (B-AUC, độ nhạy)."""
    return [[EVAL, "--scenario", scenario, "--phase-b", "ll", *extra], [EVAL, "--scenario", scenario, "--phase-b", "auc", *extra]]


def _b_outputs(sub, datasets):
    return [(f"{M.RESULT_SUBDIR}/{pb}/{sub}/{d}", None) for pb in ("T5LL", "T5") for d in datasets]


def build_jobs() -> dict[str, Job]:
    W = ["--workers", "{WORKERS}"]
    t1m, b7 = list(M.t1_methods()), list(M.B7_INCLUDED)
    variants = list(M.VARIANTS)
    det = ["--repeats", "0", "--folds", "0", "1", "2"]
    jobs = [
        Job("t0_det", "S0", "T0 tất định và n_jobs: GSE10245, lần lặp 0, fold 0–2, k = 5; MPQDA-DE và GA chạy hai lần; "
            "MPQDA-DE với n_jobs_eval 1 và 4",
            [_run("T0", "logloss", ["gse10245"], ["mpqda_de", "ga"], [5], "--folds", "0", "1", "2", "--workers", "1",
                  out=f"{M.RESULT_SUBDIR}/T0_det_{tag}", repeats=("0",)) for tag in ("a", "b")]
            + [_run("T0", "logloss", ["gse10245"], ["mpqda_de"], [5], "--folds", "0", "1", "2", "--workers", "1",
                    "--n-jobs-eval", "4", out=f"{M.RESULT_SUBDIR}/T0_det_j4", repeats=("0",))],
            [(f"{M.RESULT_SUBDIR}/T0_det_{tag}/gse10245", _rows(["mpqda_de", "ga"], [5], 3)) for tag in ("a", "b")]
            + [(f"{M.RESULT_SUBDIR}/T0_det_j4/gse10245", _rows(["mpqda_de"], [5], 3))], "0,1"),
        Job("t0_shuffle", "S0", "T0 nhãn xáo trộn (Đ3): GSE37745, 15 fold, k = 5, MPQDA-DE và GA; rồi kiểm bốn điều kiện của T0",
            [_run("T0", "logloss", ["gse37745"], ["mpqda_de", "ga"], [5], "--shuffle-train-labels", *W,
                  out=f"{M.RESULT_SUBDIR}/T0_shuffle"), [SELF, "check-t0"]],
            _phase_a_outputs("T0_shuffle", ["gse37745"], ["mpqda_de", "ga"], [5]), "0,15"),
        Job("gt1", "S0", "G-T1 (phiên S0): GA, BPSO chế độ AUROC trên GSE10245, lần lặp 0, k ∈ {5, 10, 50}; rồi check_gate_gt1",
            [_run("T1b", "auroc", ["gse10245"], ["ga", "bpso"], [5, 10, 50], *W, out=f"{M.RESULT_SUBDIR}/GT1_s0",
                  repeats=("0",)),
             [GATE, "--run-root", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/GT1_s0", "--datasets", "gse10245",
              "--report", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/GT1_s0/gate_gt1.csv"]],
            _phase_a_outputs("GT1_s0", ["gse10245"], ["ga", "bpso"], [5, 10, 50], 5), "0,15"),
        Job("t1", "S1", "T1 so sánh chính (log-loss) trên ba cohort GPL570, rồi pha B-LL và B-AUC",
            [_run("T1", "logloss", GPL570, t1m, [2, 5, 10, 50], *W), *_eval("T1")],
            _phase_a_outputs("T1", GPL570, t1m, [2, 5, 10, 50]) + _b_outputs("T1", GPL570), "3,5"),
        Job("t1_b7", "S1", "T1 trên các bộ B7 đã qua cổng G1, G2 (k ∈ {5, 10}), rồi pha B-LL và B-AUC; bỏ qua nếu không bộ nào qua",
            [_run("T1", "logloss", b7, t1m, [5, 10], *W), *_eval("T1_b7")] if b7 else [],
            _phase_a_outputs("T1", b7, t1m, [5, 10]) + _b_outputs("T1", b7), "1,9"),
        Job("t1b", "S2", "T1b bão hoà (AUROC) chạy đủ, check_gate_gt1 trên toàn bộ, rồi pha B-LL và B-AUC",
            [_run("T1b", "auroc", GPL570, t1m, [5, 10, 50], *W),
             [GATE, "--run-root", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/T1b", "--report", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/T1b/gate_gt1.csv"],
             *_eval("T1b")],
            _phase_a_outputs("T1b", GPL570, t1m, [5, 10, 50]) + _b_outputs("T1b", GPL570), "3,0"),
        Job("t4", "S2", "T4 tham chiếu không wrapper (sáu selector) một lần, rồi pha B-LL và B-AUC kèm B0, random-k",
            [[EVAL, "--scenario", "T4", "--phase-b", "ll"], [EVAL, "--scenario", "T4", "--phase-b", "auc", "--no-phase-a"]],
            [(f"{M.RESULT_SUBDIR}/T4/{d}", None) for d in [*GPL570, *b7]] + _b_outputs("T4", [*GPL570, *b7]), "1,5"),
        Job("t6", "S2", "T6 chi phí: GSE37745, lần lặp 0, fold 0–4, k = 10, MPQDA-DE với n_jobs_eval 1 và 4; chạy một mình, cuối phiên",
            [_run("T6", "logloss", ["gse37745"], ["mpqda_de"], [10], "--workers", "1", "--n-jobs-eval", n,
                  out=f"{M.RESULT_SUBDIR}/T6_j{n}", repeats=("0",)) for n in ("1", "4")],
            [(f"{M.RESULT_SUBDIR}/T6_j{n}/gse37745", _rows(["mpqda_de"], [10], 5)) for n in ("1", "4")], "0,2"),
        Job("t2", "S3", "T2 ablation: bảy biến thể, k ∈ {2, 5, 10}, ba cohort GPL570, rồi pha B-LL và B-AUC",
            [_run("T2", "logloss", GPL570, variants, [2, 5, 10], *W), *_eval("T2")],
            _phase_a_outputs("T2", GPL570, variants, [2, 5, 10]) + _b_outputs("T2", GPL570), "1,6"),
        Job("t3", "S3", "T3 ổn định theo seed: GSE37745, s = 1..4, MPQDA-DE và năm baseline mức 1, k ∈ {5, 10}, rồi pha B-LL "
            "và B-AUC",
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
    """Pha phát triển của v2 (Review-2026-11-16, Mục 6.2): chỉ dữ liệu phát triển; không chạm GPL570 hay B7."""
    W = ["--workers", "{WORKERS}"]
    return [Job(
        "dev", "DEV", "DEV: dựng ba tập TCGA, pha A (ứng viên M1–M3, v1, DE-G, DE/rand/1, GA; k ∈ {5, 10}), B-LL mô tả, "
        "dev_select (quy tắc chọn và cổng G-DEV)",
        [[SELF, "build-dev"],
         _run("DEV", "logloss", DEV_TCGA, DEV_METHODS, [5, 10], *W, repeats=("0",)),
         _run("DEV", "logloss", ["gse42127"], DEV_METHODS, [5, 10], *W, repeats=("0", "1")),
         [EVAL, "--scenario", "DEV", "--phase-b", "ll"],
         ["scripts/dev_select.py", "--results", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/DEV", "--eval",
          "{RESULTS}/" + f"{M.RESULT_SUBDIR}/T5LL/DEV", "--out", "{RESULTS}/" + f"{M.RESULT_SUBDIR}/DEV/dev_selection"]],
        _phase_a_outputs("DEV", DEV_TCGA, DEV_METHODS, [5, 10], 5) + _phase_a_outputs("DEV", ["gse42127"], DEV_METHODS, [5, 10], 10)
        + [(f"{M.RESULT_SUBDIR}/T5LL/DEV/{d}", None) for d in M.DEV_DATASETS], "2,5")]


V2_PAIR = ["mpqda_v2", "de_g"]
V1_METHODS = ["mpqda_de", *M.TIER1, *M.TIER2]                     # tám phương pháp của v1 dùng lại (G-R)
PHASE_B_V2 = ("ll", "auc", "c1", "llw")                           # B-LL chính; B-AUC, B-C1, B-LL-wide độ nhạy (TQĐ27)
PHASE_B_DIRS4 = ("T5LL", "T5", "T5C1", "T5LLW")


def _eval4(scenario):
    """Pha B của v2: B-LL (chính), rồi B-AUC, B-C1 và B-LL-wide (độ nhạy), theo thứ tự đó (PREREG-v2 Mục 5)."""
    return [[EVAL, "--scenario", scenario, "--phase-b", pb] for pb in PHASE_B_V2]


def _b_outputs4(sub, datasets):
    return [(f"{M.RESULT_SUBDIR}/{pb}/{sub}/{d}", None) for pb in PHASE_B_DIRS4 for d in datasets]


def v2_ablations() -> list[str]:
    """Ablation của C2: bỏ khe, bỏ gen mới, bỏ tournament; thêm bỏ thích nghi chỉ khi bản chốt là M2."""
    return [m for m in M.V2_ABLATIONS if m != "mpqda_v2_no_adapt" or M.V2_CHOSEN == "mpqda_v2_m2"]


def build_v2_jobs() -> list[Job]:
    """Pha xác nhận của v2 (PREREG-v2): chỉ chạy được sau tag taimma-2027-prereg-v2 (cổng đăng ký trong run_taimma.py)."""
    W = ["--workers", "{WORKERS}"]
    b7, abl = list(M.B7_INCLUDED), v2_ablations()
    det = [_run("T0v2", "logloss", ["gse10245"], V2_PAIR, [5], "--folds", "0", "1", "2", "--workers", "1",
                out=f"{M.RESULT_SUBDIR}/T0v2_det_{tag}", repeats=("0",)) for tag in ("a", "b")]
    det += [_run("T0v2", "logloss", ["gse10245"], ["mpqda_v2"], [5], "--folds", "0", "1", "2", "--workers", "1",
                 "--n-jobs-eval", "4", out=f"{M.RESULT_SUBDIR}/T0v2_det_j4", repeats=("0",))]
    gr_out = f"{M.RESULT_SUBDIR}/GR"
    v1_outputs = [n for m in V1_METHODS for n in M.output_names(m)]
    return [
        Job("t0v2_det", "S0v2", "T0-v2 tất định và n_jobs: GSE10245, lần lặp 0, fold 0–2, k = 5; mpqda_v2 và de_g hai lần; "
            "mpqda_v2 với n_jobs_eval 4", det,
            [(f"{M.RESULT_SUBDIR}/T0v2_det_{tag}/gse10245", _rows(V2_PAIR, [5], 3)) for tag in ("a", "b")]
            + [(f"{M.RESULT_SUBDIR}/T0v2_det_j4/gse10245", _rows(["mpqda_v2"], [5], 3))], "0,05"),
        Job("t0v2_shuffle", "S0v2", "T0-v2 nhãn xáo trộn: GSE37745, 15 fold, k = 5, mpqda_v2 và de_g; rồi check-t0 --v2",
            [_run("T0v2", "logloss", ["gse37745"], V2_PAIR, [5], "--shuffle-train-labels", *W,
                  out=f"{M.RESULT_SUBDIR}/T0v2_shuffle"), [SELF, "check-t0", "--v2"]],
            _phase_a_outputs("T0v2_shuffle", ["gse37745"], V2_PAIR, [5]), "0,1"),
        Job("gr", "S0v2", "G-R: chạy lại 16 đơn vị của v1 và bảy baseline (gse10245 lần lặp 0 fold 0 k = 5; gse37745 lần lặp 2 "
            "fold 4 k = 10) và so với T1 của S1 (tập gen, N_eval trùng; best_fitness lệch ≤ 1e-12)",
            [_run("GR", "logloss", ["gse10245"], V1_METHODS, [5], "--folds", "0", *W, out=gr_out, repeats=("0",)),
             _run("GR", "logloss", ["gse37745"], V1_METHODS, [10], "--folds", "4", *W, out=gr_out, repeats=("2",)),
             [GATE, "--run-root", "{RESULTS}/" + gr_out, "--reference", "{V1_T1}", "--datasets", "gse10245", "gse37745",
              "--methods", *v1_outputs, "--report", "{RESULTS}/" + gr_out + "/gate_gr.csv"]],
            [(f"{gr_out}/gse10245", _rows(V1_METHODS, [5], 1)), (f"{gr_out}/gse37745", _rows(V1_METHODS, [10], 1))], "0,15"),
        Job("v1b", "S0v2", "B-C1 và B-LL-wide cho T1 của v1 và bảy baseline (panel của S1, đọc từ {V1_T1}) trên ba cohort GPL570 "
            "và B7, để HV2 có tham chiếu dưới mỗi tiêu chí (TQĐ30); chỉ chạy ở commit mang tag đăng ký",
            [[SELF, "check-registered"]]
            + [[EVAL, "--scenario", sc, "--phase-b", pb, "--selection-root", "{V1_T1}"]
               for sc in ("T1", *(("T1_b7",) if b7 else ())) for pb in ("c1", "llw")],
            [(f"{M.RESULT_SUBDIR}/{pb}/T1/{d}", None) for pb in ("T5C1", "T5LLW") for d in [*GPL570, *b7]], "0,3"),
        Job("c1", "C1", "C1: mpqda_v2 và de_g trên ba cohort GPL570 (k ∈ {2, 5, 10, 50}) và hai bộ B7 (k ∈ {5, 10}); pha B bốn "
            "tiêu chí",
            [_run("T1v2", "logloss", GPL570, V2_PAIR, [2, 5, 10, 50], *W),
             *([_run("T1v2", "logloss", b7, V2_PAIR, [5, 10], *W)] if b7 else []),
             *_eval4("T1v2"), *(_eval4("T1v2_b7") if b7 else [])],
            _phase_a_outputs("T1v2", GPL570, V2_PAIR, [2, 5, 10, 50]) + _phase_a_outputs("T1v2", b7, V2_PAIR, [5, 10])
            + _b_outputs4("T1v2", [*GPL570, *b7]), "2,0"),
        Job("c2", "C2", "C2: ablation của v2 (" + ", ".join(abl) + "), k ∈ {2, 5, 10}, ba cohort GPL570; pha B bốn tiêu chí",
            [_run("T2v2", "logloss", GPL570, abl, [2, 5, 10], *W), *_eval4("T2v2")],
            _phase_a_outputs("T2v2", GPL570, abl, [2, 5, 10]) + _b_outputs4("T2v2", GPL570), "1,5"),
        Job("c3", "C3", "C3: mpqda_v2 với seed 1–4 trên GSE37745, k ∈ {5, 10}; pha B bốn tiêu chí",
            [step for s_ in (1, 2, 3, 4) for step in (
                _run(f"T3v2_s{s_}", "logloss", ["gse37745"], ["mpqda_v2"], [5, 10], "--seed-index", str(s_), *W),
                *_eval4(f"T3v2_s{s_}"))],
            [out for s_ in (1, 2, 3, 4) for out in (
                (f"{M.RESULT_SUBDIR}/T3v2_s{s_}/gse37745", _rows(["mpqda_v2"], [5, 10])),
                *_b_outputs4(f"T3v2_s{s_}", ["gse37745"]))], "0,6"),
    ]


JOBS = {**build_jobs(), **{j.name: j for j in build_dev_jobs()}, **{j.name: j for j in build_v2_jobs()}}
SESSIONS = {"S0": ["t0_det", "t0_shuffle", "gt1"], "S1": ["t1", "t1_b7"], "S2": ["t1b", "t4", "t6"], "S3": ["t2", "t3"],
            "DEV": ["dev"], "S0v2": ["t0v2_det", "t0v2_shuffle", "gr", "v1b"], "C1": ["c1"], "C2": ["c2"], "C3": ["c3"]}


# ------------------------------------------------------------------------------------------------- helpers
def data_root() -> Path:
    root = os.environ.get("KLTN_DATA_ROOT")
    if not root:
        raise SystemExit("đặt KLTN_DATA_ROOT")
    return Path(root)


def results_root() -> Path:
    return data_root().joinpath(*C.RESULT_DIR)


def render(argv, workers: int = 4) -> list[str]:
    res = results_root().as_posix() if os.environ.get("KLTN_DATA_ROOT") else "$KLTN_DATA_ROOT/results/survey_v2"
    v1_t1 = os.environ.get("KLTN_V1_T1", "$KLTN_V1_T1")          # T1 của v1 (S1_results.zip) cho cổng G-R
    return [a.replace("{RESULTS}", res).replace("{WORKERS}", str(workers)).replace("{V1_T1}", v1_t1) for a in argv]


def pip_freeze_md5() -> str | None:
    """MD5 của `python -m pip freeze`, cùng cách tính với GATE2 của notebook (TQĐ16)."""
    import hashlib
    try:
        txt = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=180,
                             check=True).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"[pip freeze] không tính được MD5: {exc}", flush=True)
        return None
    return hashlib.md5(txt.encode("utf-8")).hexdigest()


def write_pip_freeze(job: Job, log_dir: Path) -> Path | None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / f"pip_freeze_{job.name}.txt"
        txt = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=180).stdout
        out.write_text(f"# python {sys.version.split()[0]}\n" + txt, encoding="utf-8")
        return out
    except Exception as exc:  # noqa: BLE001 — không được làm hỏng phiên vì bước ghi phụ
        print(f"[pip freeze] bỏ qua: {exc}", flush=True)
        return None


def verify(job: Job, commit: str | None) -> list[str]:
    """traceable = true, commit_hash = HEAD, SHA-256 bản chia khớp giá trị chờ, đủ số dòng chờ (Mục 4.1)."""
    problems, root = [], results_root()
    for rel, expected_rows in job.outputs:
        cfg_path = root / rel / "config.json"
        if not cfg_path.exists():
            problems.append(f"{rel}: thiếu config.json")
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("traceable") is not True:
            problems.append(f"{rel}: traceable = {cfg.get('traceable')} {cfg.get('traceable_errors', '')}")
        if commit and cfg.get("commit_hash") != commit:
            problems.append(f"{rel}: commit_hash {cfg.get('commit_hash')} khác HEAD {commit}")
        if cfg.get("resumed_from"):
            problems.append(f"{rel}: có checkpoint từ commit khác ({[r.get('commit_hash') for r in cfg['resumed_from']]})")
        if "split_sha256_match" in cfg and cfg["split_sha256_match"] is not True:
            problems.append(f"{rel}: split_sha256 {cfg.get('split_sha256')} khác giá trị chờ {cfg.get('split_sha256_expected')}")
        if expected_rows is not None:
            path = root / rel / "selection_per_fold.csv"
            n = len(pd.read_csv(path)) if path.exists() else 0
            if n != expected_rows:
                problems.append(f"{rel}: selection_per_fold.csv có {n} dòng, chờ {expected_rows}")
    return problems


# ------------------------------------------------------------------------------------------------- T0
def _frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def same_values(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Hai bảng trùng từng giá trị (không dung sai), không phụ thuộc kiểu số: một cột int trong tệp chỉ có MPQDA-DE thành float
    khi tệp có thêm dòng GA để trống cột đó (0 so với 0.0). NaN trùng NaN."""
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


def check_t0(root: Path | None = None, prefix: str = "T0", main: str = "mpqda_de") -> tuple[list[str], dict]:
    """Bốn điều kiện của T0 (Mục 6) trên kết quả của t0_det và t0_shuffle (prefix T0, phương pháp chính mpqda_de), hoặc
    của T0-v2 (prefix T0v2, mpqda_v2; PREREG-v2 Mục 9). Trả về (danh sách không đạt, số liệu ghi lại)."""
    root = results_root() / M.RESULT_SUBDIR if root is None else Path(root)
    a, b, j4 = (root / f"{prefix}_det_{t}" / "gse10245" for t in ("a", "b", "j4"))
    sh = root / f"{prefix}_shuffle" / "gse37745"
    problems, info = [], {}
    need = [d / n for d in (a, b, j4) for n in ("selected_genes.csv", "history.csv", "selection_per_fold.csv")]
    need += [sh / "t0_shuffle_eval.csv", sh / "selection_per_fold.csv"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        return [f"thiếu tệp: {m}" for m in missing], info
    for name in ("selected_genes.csv", "history.csv"):                       # (1) tất định
        same = (a / name).read_bytes() == (b / name).read_bytes()
        info[f"det_identical_{name}"] = same
        if not same:
            problems.append(f"(1) tất định: {name} của hai lần chạy cùng seed khác nhau")
    for name in ("selected_genes.csv", "history.csv"):                       # (2) n_jobs_eval 1 và 4
        one = _frame(a / name)
        one = one[one.method.isin(M.output_names(main))].reset_index(drop=True)
        same = same_values(one, _frame(j4 / name))
        info[f"njobs_identical_{name}"] = same
        if not same:
            problems.append(f"(2) song song: {name} của MPQDA-DE khác nhau giữa n_jobs_eval 1 và 4")
    sel = pd.concat([_frame(d / "selection_per_fold.csv") for d in (a, b, j4, sh)], ignore_index=True)   # (3)
    bad = sel[(sel.n_selected != sel.k) | (~sel.exact_k.astype(bool)) | (sel.n_eval > N_EVAL_MAX)]
    info["n_rows_checked"], info["n_eval_max"] = int(len(sel)), int(sel.n_eval.max())
    if len(bad):
        problems.append(f"(3) {len(bad)} dòng có |S| ≠ k, exact_k = False hoặc n_eval > {N_EVAL_MAX}")
    ev = _frame(sh / "t0_shuffle_eval.csv")                                   # (4) nhãn xáo trộn
    n_folds = ev.groupby("method").size()
    means = ev.groupby("method").auroc.mean()
    info["shuffle_mean_outer_auroc"] = {m: float(v) for m, v in means.items()}
    info["shuffle_mean_inner_fitness"] = {m: float(v) for m, v in ev.groupby("method").output_fitness.mean().items()}
    for method, value in means.items():
        if n_folds[method] != M.N_OUTER_FOLDS:
            problems.append(f"(4) {method}: {n_folds[method]} fold, chờ {M.N_OUTER_FOLDS}")
        if not T0_SHUFFLE_RANGE[0] <= value <= T0_SHUFFLE_RANGE[1]:
            problems.append(f"(4) nhãn xáo trộn: AUROC ngoài trung bình của {method} = {value:.4f}, ngoài {list(T0_SHUFFLE_RANGE)}")
    hist = _frame(a / "history.csv")
    overlap = float(hist[(hist.method == main) & (hist.t == 1)].leader_overlap.mean())
    info["leader_overlap_t1_gse10245"] = overlap
    info["leader_overlap_report_now"] = bool(overlap < T0_MIN_LEADER_OVERLAP)
    return problems, info


def cmd_check_t0(v2: bool = False) -> int:
    prefix, main = ("T0v2", "mpqda_v2") if v2 else ("T0", "mpqda_de")
    problems, info = check_t0(prefix=prefix, main=main)
    info["pip_freeze_md5"] = pip_freeze_md5()
    print(f"PIP_FREEZE_MD5 {info['pip_freeze_md5']}")
    out = results_root() / M.RESULT_SUBDIR / f"{prefix}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": not problems, "problems": problems, **info}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info.get("leader_overlap_report_now"):
        print(f"BÁO NGAY: leader_overlap trung bình ở t = 1 dưới {T0_MIN_LEADER_OVERLAP} (Đ2 không có tác dụng trên dữ liệu thật)")
    print(f"{prefix} ĐẠT" if not problems else f"{prefix} KHÔNG ĐẠT:\n  " + "\n  ".join(problems))
    return 0 if not problems else 3


# ------------------------------------------------------------------------------------------------- DEV (v2)
def cmd_build_dev() -> int:
    """Dựng ba tập DEV vào gốc dữ liệu (ghi được) nếu chưa có; kiểm SHA-256 bản chia với mpqda.config. Bản dựng tất định
    về nội dung (cùng values_sha256 và cùng bản chia); SHA-256 của expression.h5 thay đổi theo lần ghi nên không so."""
    from survey_v2.data import sha256
    root = data_root()
    missing = [d for d in DEV_TCGA if not (root / C.dataset_dir(d) / "repeated_splits.csv").is_file()]
    if missing:
        if len(missing) != len(DEV_TCGA):
            print(f"DỪNG: chỉ có một phần tập DEV ({missing} thiếu); xoá thư mục DEV cũ rồi dựng lại")
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
            problems.append(f"{d}: {got} khác giá trị chờ {M.EXPECTED_SPLIT_SHA256[d]}")
    print("DEV_OK" if not problems else "DEV KHÔNG ĐẠT:\n  " + "\n  ".join(problems))
    return 0 if not problems else 3


# ------------------------------------------------------------------------------------------------- lệnh
def cmd_check_registered(git=None) -> int:
    """0 khi HEAD mang tag taimma-2027-prereg-v2 và cây làm việc không có thay đổi chưa commit ở tệp được theo dõi; 1 nếu không."""
    from mpqda import registration as R
    git = git or R.git
    problems = []
    if R.PREREG_V2_TAG not in git("tag", "--points-at", "HEAD").split():
        problems.append(f"HEAD chưa mang tag {R.PREREG_V2_TAG}")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        problems.append("cây làm việc có thay đổi chưa commit")
    print("REGISTERED_OK" if not problems else "DỪNG: " + "; ".join(problems), flush=True)
    return 0 if not problems else 1


def cmd_plan():
    for session, names in SESSIONS.items():
        for name in names:
            j = JOBS[name]
            print(f"{session}  {name:11s} {j.hours:>5s} h  {j.what}")


def cmd_show(job: Job, workers: int):
    print(f"{job.name} — {job.what}\nphiên {job.session}; ước lượng {job.hours} giờ")
    for s in job.steps:
        print("  python", " ".join(render(s, workers)))
    print("đầu ra kiểm tra:", ", ".join(f"{rel} ({rows} dòng)" if rows is not None else rel for rel, rows in job.outputs) or "—")


def cmd_run(job: Job, workers: int, log_dir: Path) -> int:
    if not job.steps:
        print(f"{job.name}: không có lệnh nào (không bộ B7 nào trong mpqda.config.B7_INCLUDED)")
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
                print(f"LỖI: bước {i} thoát mã {code}. Checkpoint đã ghi được giữ; chạy lại cùng việc sẽ tiếp tục.")
                return 1
    problems = verify(job, commit)
    if problems:
        print("VERIFY KHÔNG ĐẠT:\n  " + "\n  ".join(problems))
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
    ap.add_argument("job", nargs="?")
    ap.add_argument("--workers", type=int, default=4, help="số đơn vị pha A chạy song song (Kaggle: 4 CPU)")
    ap.add_argument("--log-dir", type=Path, default=Path(os.environ.get("KLTN_LOG_DIR", "logs")))
    ap.add_argument("--v2", action="store_true", help="check-t0: kiểm T0-v2 (T0v2_det_*, T0v2_shuffle, mpqda_v2)")
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
        return cmd_check_t0(v2=args.v2)
    if args.command == "check-registered":
        return cmd_check_registered()
    if args.job not in JOBS:
        ap.error(f"việc phải là một trong: {', '.join(JOBS)}")
    job = JOBS[args.job]
    if args.command == "show":
        cmd_show(job, args.workers)
        return 0
    if args.command == "verify":
        problems = verify(job, head_commit())
        print("VERIFY_OK" if not problems else "VERIFY KHÔNG ĐẠT:\n  " + "\n  ".join(problems))
        return 0 if not problems else 3
    return cmd_run(job, args.workers, args.log_dir)


if __name__ == "__main__":
    sys.exit(main())
