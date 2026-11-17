"""Pha B và T4 của bài TAIMMA (Mục 4.2): chỉ dựng và chạy các lệnh của script survey trong vendor/, không có mã đánh giá riêng.

    python scripts/run_taimma_eval.py --scenario T4          # T4 pha A (sáu selector) rồi pha B kèm B0 và random-k
    python scripts/run_taimma_eval.py --scenario T1          # pha B của T1 (ba cohort GPL570), --no-references --save-scores
    python scripts/run_taimma_eval.py --scenario T1_b7       # pha B của T1 trên các bộ B7 đã qua cổng
    python scripts/run_taimma_eval.py --scenario T1b | T2 | T3_s1 .. T3_s4   [--phase-b ll|auc] [--dry-run]

--phase-b ll (mặc định, TQĐ11): scripts/run_taimma_phase_b.py --criterion logloss, ghi vào taimma/T5LL/<kịch bản>.
--phase-b auc: script pha B của survey nguyên trạng, ghi vào taimma/T5/<kịch bản> (phân tích độ nhạy).
--phase-b c1: C = 1 cố định, ghi vào taimma/T5C1/<kịch bản>; --phase-b llw: log-loss trên lưới rộng 1e-4..1e3, ghi vào
taimma/T5LLW/<kịch bản> (TQĐ27). Cả hai chỉ để phân tích độ nhạy.
--no-phase-a (chỉ T4): bỏ bước pha A, dùng khi pha A của T4 đã chạy cùng pha B kia.
--selection-root DIR: đọc panel pha A từ DIR thay cho taimma/<thư mục pha A>; dùng cho B-C1, B-LL-wide của T1 v1 (panel S1).

Sau khi chạy, kiểm config.json của từng thư mục kết quả: traceable = true và commit_hash = HEAD. Trả 0 khi đạt, 1 khi một
lệnh lỗi, 3 khi config không đạt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from survey_v2 import config as C  # noqa: E402

from mpqda import config as M  # noqa: E402
from mpqda.runlog import DATA_ROOT, head_commit, utf8_console  # noqa: E402

FS = "vendor/scripts/run_e1_feature_selection.py"
CLF = "vendor/scripts/run_e1_classification.py"
CLF_LL = "scripts/run_taimma_phase_b.py"
PHASE_B = {"ll": "T5LL", "auc": "T5", "c1": "T5C1", "llw": "T5LLW"}   # pha B -> thư mục dưới taimma/ (c1: B-C1; llw: B-LL-wide)
CRITERION = {"ll": "logloss", "c1": "c1", "llw": "llwide"}           # c_selection.criterion chờ trong config.json; auc: không có
GPL570 = list(C.GPL570_COHORTS)
REPEATS = ["0", "1", "2"]
T4_METHODS = ["l1_lr", "stability_selection", "f_test", "mrmr", "svm_rfe", "abess"]
# kịch bản pha B -> (thư mục pha A dưới taimma/, bộ dữ liệu, k, có --save-scores)
SCENARIOS = {
    "T1": ("T1", GPL570, (2, 5, 10, 50), True),
    "T1_b7": ("T1", list(M.B7_INCLUDED), (5, 10), True),
    "T1b": ("T1b", GPL570, (5, 10, 50), False),
    "T2": ("T2", GPL570, (2, 5, 10), False),
    **{f"T3_s{s}": (f"T3_s{s}", ["gse37745"], (5, 10), False) for s in (1, 2, 3, 4)},
    "DEV": ("DEV", list(M.DEV_DATASETS), (5, 10), False),            # pha phát triển của v2 (TQĐ23); chỉ để mô tả
    # pha xác nhận của v2 (PREREG-v2 Mục 5): C1 trên GPL570 và B7, C2 ablation, C3 seed
    "T1v2": ("T1v2", GPL570, (2, 5, 10, 50), True),
    "T1v2_b7": ("T1v2", list(M.B7_INCLUDED), (5, 10), True),
    "T2v2": ("T2v2", GPL570, (2, 5, 10), False),
    **{f"T3v2_s{s}": (f"T3v2_s{s}", ["gse37745"], (5, 10), False) for s in (1, 2, 3, 4)},
}


def results_root() -> Path:
    return DATA_ROOT.joinpath(*C.RESULT_DIR)


def classify_script(phase_b: str) -> list[str]:
    if phase_b not in PHASE_B:
        raise ValueError(phase_b)
    return [CLF] if phase_b == "auc" else [CLF_LL, "--criterion", CRITERION[phase_b]]


def phase_b_command(scenario: str, datasets=None, phase_b: str = "ll", selection_root=None) -> tuple[list[str], str]:
    """Lệnh pha B của một kịch bản wrapper (không tính lại B0, random-k) và thư mục kết quả tương đối."""
    sel, default_ds, k, save = SCENARIOS[scenario]
    datasets = list(datasets or default_ds)
    out = f"{M.RESULT_SUBDIR}/{PHASE_B[phase_b]}/{sel}"
    root = str(selection_root) if selection_root is not None else str(results_root() / M.RESULT_SUBDIR / sel)
    cmd = [*classify_script(phase_b), "--datasets", *datasets, "--selection-roots", root, "--out", out,
           "--repeats", *REPEATS, "--k", *map(str, k), "--no-references"]
    return cmd + (["--save-scores"] if save else []), out


def t4_commands(datasets=None, phase_b: str = "ll", phase_a: bool = True) -> list[tuple[list[str], str]]:
    """T4 pha A (sáu selector không wrapper) rồi pha B kèm B0 và random-k: lần duy nhất tính hai mốc này (mỗi pha B một lần)."""
    datasets = list(datasets or [*GPL570, *M.B7_INCLUDED])
    k = ["2", "5", "10", "50"]
    sel_out, clf_out = f"{M.RESULT_SUBDIR}/T4", f"{M.RESULT_SUBDIR}/{PHASE_B[phase_b]}/T4"
    a = [([FS, "--datasets", *datasets, "--methods", *T4_METHODS, "--k", *k, "--repeats", *REPEATS, "--out", sel_out], sel_out)]
    return (a if phase_a else []) + [
            ([*classify_script(phase_b), "--datasets", *datasets, "--selection-roots", str(results_root() / sel_out), "--out", clf_out,
              "--repeats", *REPEATS, "--k", *k], clf_out)]


def commands(scenario: str, datasets=None, phase_b: str = "ll", phase_a: bool = True,
             selection_root=None) -> list[tuple[list[str], str, list[str]]]:
    """Danh sách (lệnh, thư mục kết quả tương đối, bộ dữ liệu) theo thứ tự chạy."""
    if scenario == "T4":
        ds = list(datasets or [*GPL570, *M.B7_INCLUDED])
        return [(cmd, out, ds) for cmd, out in t4_commands(ds, phase_b, phase_a)]
    ds = list(datasets or SCENARIOS[scenario][1])
    if not ds:
        raise SystemExit(f"{scenario}: không có bộ dữ liệu nào (mpqda.config.B7_INCLUDED đang rỗng)")
    cmd, out = phase_b_command(scenario, ds, phase_b, selection_root)
    return [(cmd, out, ds)]


def check_configs(out: str, datasets, commit: str | None) -> list[str]:
    problems = []
    for name in datasets:
        path = results_root() / out / name / "config.json"
        if not path.is_file():
            problems.append(f"{out}/{name}: thiếu config.json")
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("traceable") is not True:
            problems.append(f"{out}/{name}: traceable = {cfg.get('traceable')} {cfg.get('traceable_errors', '')}")
        if cfg.get("commit_hash") != commit:
            problems.append(f"{out}/{name}: commit_hash {cfg.get('commit_hash')} khác HEAD {commit}")
        criterion = (cfg.get("c_selection") or {}).get("criterion")
        for pb, want in CRITERION.items():
            if f"/{PHASE_B[pb]}/" in f"/{out}/" and criterion != want:
                problems.append(f"{out}/{name}: c_selection.criterion = {criterion}, chờ {want}")
        if f"/{PHASE_B['auc']}/" in f"/{out}/" and criterion is not None:
            problems.append(f"{out}/{name}: có c_selection = {criterion} trong thư mục của pha B survey")
    return problems


def main(argv=None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=["T4", *SCENARIOS])
    ap.add_argument("--datasets", nargs="+", default=None, help="mặc định theo kịch bản")
    ap.add_argument("--phase-b", choices=sorted(PHASE_B), default="ll", help="ll: B-LL (chính); auc: pha B của survey; c1: C = 1; llw: B-LL-wide")
    ap.add_argument("--no-phase-a", dest="phase_a", action="store_false", help="T4: bỏ bước pha A đã chạy")
    ap.add_argument("--selection-root", type=Path, default=None, help="thư mục panel pha A thay cho taimma/<kịch bản>")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in lệnh")
    args = ap.parse_args(argv)
    if not args.phase_a and args.scenario != "T4":
        ap.error("--no-phase-a chỉ dùng với --scenario T4")
    if args.selection_root is not None and args.scenario == "T4":
        ap.error("--selection-root không dùng với T4")
    if args.selection_root is not None and not args.selection_root.is_dir():
        ap.error(f"--selection-root {args.selection_root} không phải thư mục")
    plan = commands(args.scenario, args.datasets, args.phase_b, args.phase_a, args.selection_root)
    for cmd, _, _ in plan:
        print("python " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    if not os.environ.get("KLTN_DATA_ROOT", "").strip():
        print("DỪNG: đặt KLTN_DATA_ROOT (thư mục dữ liệu và kết quả).", flush=True)
        return 2
    for cmd, out, _ in plan:
        code = subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode
        if code != 0:
            print(f"LỖI: {cmd[0]} thoát mã {code} ({out})", flush=True)
            return 1
    problems = [p for _, out, ds in plan for p in check_configs(out, ds, head_commit())]
    if problems:
        print("VERIFY KHÔNG ĐẠT:\n  " + "\n  ".join(problems), flush=True)
        return 3
    print(f"EVAL_OK {args.scenario} ({args.phase_b}): " + ", ".join(out for _, out, _ in plan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
