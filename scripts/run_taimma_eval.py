"""Pha B và T4 của bài TAIMMA (Mục 4.2): chỉ dựng và chạy các lệnh của script survey trong vendor/, không có mã đánh giá riêng.

    python scripts/run_taimma_eval.py --scenario T4          # T4 pha A (sáu selector) rồi pha B kèm B0 và random-k
    python scripts/run_taimma_eval.py --scenario T1          # pha B của T1 (ba cohort GPL570), --no-references --save-scores
    python scripts/run_taimma_eval.py --scenario T1_b7       # pha B của T1 trên các bộ B7 đã qua cổng
    python scripts/run_taimma_eval.py --scenario T1b | T2 | T3_s1 .. T3_s4   [--dry-run]

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
}


def results_root() -> Path:
    return DATA_ROOT.joinpath(*C.RESULT_DIR)


def phase_b_command(scenario: str, datasets=None) -> tuple[list[str], str]:
    """Lệnh pha B của một kịch bản wrapper (không tính lại B0, random-k) và thư mục kết quả tương đối."""
    sel, default_ds, k, save = SCENARIOS[scenario]
    datasets = list(datasets or default_ds)
    out = f"{M.RESULT_SUBDIR}/T5/{sel}"
    cmd = [CLF, "--datasets", *datasets, "--selection-roots", str(results_root() / M.RESULT_SUBDIR / sel), "--out", out,
           "--repeats", *REPEATS, "--k", *map(str, k), "--no-references"]
    return cmd + (["--save-scores"] if save else []), out


def t4_commands(datasets=None) -> list[tuple[list[str], str]]:
    """T4 pha A (sáu selector không wrapper) rồi pha B kèm B0 và random-k: lần duy nhất tính hai mốc này."""
    datasets = list(datasets or [*GPL570, *M.B7_INCLUDED])
    k = ["2", "5", "10", "50"]
    sel_out, clf_out = f"{M.RESULT_SUBDIR}/T4", f"{M.RESULT_SUBDIR}/T5/T4"
    return [([FS, "--datasets", *datasets, "--methods", *T4_METHODS, "--k", *k, "--repeats", *REPEATS, "--out", sel_out], sel_out),
            ([CLF, "--datasets", *datasets, "--selection-roots", str(results_root() / sel_out), "--out", clf_out,
              "--repeats", *REPEATS, "--k", *k], clf_out)]


def commands(scenario: str, datasets=None) -> list[tuple[list[str], str, list[str]]]:
    """Danh sách (lệnh, thư mục kết quả tương đối, bộ dữ liệu) theo thứ tự chạy."""
    if scenario == "T4":
        ds = list(datasets or [*GPL570, *M.B7_INCLUDED])
        return [(cmd, out, ds) for cmd, out in t4_commands(ds)]
    ds = list(datasets or SCENARIOS[scenario][1])
    if not ds:
        raise SystemExit(f"{scenario}: không có bộ dữ liệu nào (mpqda.config.B7_INCLUDED đang rỗng)")
    cmd, out = phase_b_command(scenario, ds)
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
    return problems


def main(argv=None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=["T4", *SCENARIOS])
    ap.add_argument("--datasets", nargs="+", default=None, help="mặc định theo kịch bản")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in lệnh")
    args = ap.parse_args(argv)
    plan = commands(args.scenario, args.datasets)
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
    print(f"EVAL_OK {args.scenario}: " + ", ".join(out for _, out, _ in plan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
