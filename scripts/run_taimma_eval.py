"""Phase B and T4 of the TAIMMA study: only builds and runs the commands of the vendored scripts in vendor/, with no evaluation code of its own.

    python scripts/run_taimma_eval.py --scenario T4          # T4 phase A (six selectors) then phase B with B0 and random-k
    python scripts/run_taimma_eval.py --scenario T1          # phase B of T1 (three GPL570 cohorts), --no-references --save-scores
    python scripts/run_taimma_eval.py --scenario T1_b7       # phase B of T1 on the B7 datasets that passed the gates
    python scripts/run_taimma_eval.py --scenario T1b | T2 | T3_s1 .. T3_s4   [--phase-b ll|auc] [--dry-run]

--phase-b ll (default): scripts/run_taimma_phase_b.py --criterion logloss, written to taimma/T5LL/<scenario>.
--phase-b auc: the phase B script of the vendored code unchanged, written to taimma/T5/<scenario> (sensitivity analysis).
--phase-b c1: C = 1 fixed, written to taimma/T5C1/<scenario>; --phase-b llw: log-loss on the wide grid 1e-4..1e3, written to
taimma/T5LLW/<scenario>. Both are only for sensitivity analysis.
--no-phase-a (T4 only): skip phase A, used when phase A of T4 already ran with the other phase B.
--selection-root DIR: read the phase A panels from DIR instead of taimma/<phase A folder>; used for B-C1, B-LL-wide of T1 of v1 (S1 panels).

After running, check config.json of each result folder: traceable = true and commit_hash = HEAD. Returns 0 on success, 1 when a
command fails, 3 when a config fails the check.
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
PHASE_B = {"ll": "T5LL", "auc": "T5", "c1": "T5C1", "llw": "T5LLW"}   # phase B -> folder under taimma/ (c1: B-C1; llw: B-LL-wide)
CRITERION = {"ll": "logloss", "c1": "c1", "llw": "llwide"}           # expected c_selection.criterion in config.json; auc: none
GPL570 = list(C.GPL570_COHORTS)
REPEATS = ["0", "1", "2"]
T4_METHODS = ["l1_lr", "stability_selection", "f_test", "mrmr", "svm_rfe", "abess"]
# phase B scenario -> (phase A folder under taimma/, datasets, k, with --save-scores)
SCENARIOS = {
    "T1": ("T1", GPL570, (2, 5, 10, 50), True),
    "T1_b7": ("T1", list(M.B7_INCLUDED), (5, 10), True),
    "T1b": ("T1b", GPL570, (5, 10, 50), False),
    "T2": ("T2", GPL570, (2, 5, 10), False),
    **{f"T3_s{s}": (f"T3_s{s}", ["gse37745"], (5, 10), False) for s in (1, 2, 3, 4)},
    "DEV": ("DEV", list(M.DEV_DATASETS), (5, 10), False),            # development phase of v2; descriptive only
    # confirmatory phase of v2 (PREREG-v2, Section 5): C1 on GPL570 and B7, C2 ablation, C3 seeds
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
    """Phase B command of one wrapper scenario (B0, random-k not recomputed) and the relative result folder."""
    sel, default_ds, k, save = SCENARIOS[scenario]
    datasets = list(datasets or default_ds)
    out = f"{M.RESULT_SUBDIR}/{PHASE_B[phase_b]}/{sel}"
    root = str(selection_root) if selection_root is not None else str(results_root() / M.RESULT_SUBDIR / sel)
    cmd = [*classify_script(phase_b), "--datasets", *datasets, "--selection-roots", root, "--out", out,
           "--repeats", *REPEATS, "--k", *map(str, k), "--no-references"]
    return cmd + (["--save-scores"] if save else []), out


def t4_commands(datasets=None, phase_b: str = "ll", phase_a: bool = True) -> list[tuple[list[str], str]]:
    """T4 phase A (six non-wrapper selectors) then phase B with B0 and random-k: the only time these two references are computed (once per phase B)."""
    datasets = list(datasets or [*GPL570, *M.B7_INCLUDED])
    k = ["2", "5", "10", "50"]
    sel_out, clf_out = f"{M.RESULT_SUBDIR}/T4", f"{M.RESULT_SUBDIR}/{PHASE_B[phase_b]}/T4"
    a = [([FS, "--datasets", *datasets, "--methods", *T4_METHODS, "--k", *k, "--repeats", *REPEATS, "--out", sel_out], sel_out)]
    return (a if phase_a else []) + [
            ([*classify_script(phase_b), "--datasets", *datasets, "--selection-roots", str(results_root() / sel_out), "--out", clf_out,
              "--repeats", *REPEATS, "--k", *k], clf_out)]


def commands(scenario: str, datasets=None, phase_b: str = "ll", phase_a: bool = True,
             selection_root=None) -> list[tuple[list[str], str, list[str]]]:
    """List of (command, relative result folder, datasets) in run order."""
    if scenario == "T4":
        ds = list(datasets or [*GPL570, *M.B7_INCLUDED])
        return [(cmd, out, ds) for cmd, out in t4_commands(ds, phase_b, phase_a)]
    ds = list(datasets or SCENARIOS[scenario][1])
    if not ds:
        raise SystemExit(f"{scenario}: no dataset (mpqda.config.B7_INCLUDED is empty)")
    cmd, out = phase_b_command(scenario, ds, phase_b, selection_root)
    return [(cmd, out, ds)]


def check_configs(out: str, datasets, commit: str | None) -> list[str]:
    problems = []
    for name in datasets:
        path = results_root() / out / name / "config.json"
        if not path.is_file():
            problems.append(f"{out}/{name}: missing config.json")
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("traceable") is not True:
            problems.append(f"{out}/{name}: traceable = {cfg.get('traceable')} {cfg.get('traceable_errors', '')}")
        if cfg.get("commit_hash") != commit:
            problems.append(f"{out}/{name}: commit_hash {cfg.get('commit_hash')} differs from HEAD {commit}")
        criterion = (cfg.get("c_selection") or {}).get("criterion")
        for pb, want in CRITERION.items():
            if f"/{PHASE_B[pb]}/" in f"/{out}/" and criterion != want:
                problems.append(f"{out}/{name}: c_selection.criterion = {criterion}, expected {want}")
        if f"/{PHASE_B['auc']}/" in f"/{out}/" and criterion is not None:
            problems.append(f"{out}/{name}: c_selection = {criterion} present in a folder of the vendored phase B")
    return problems


def main(argv=None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=["T4", *SCENARIOS])
    ap.add_argument("--datasets", nargs="+", default=None, help="default from the scenario")
    ap.add_argument("--phase-b", choices=sorted(PHASE_B), default="ll", help="ll: B-LL (main); auc: phase B of the vendored code; c1: C = 1; llw: B-LL-wide")
    ap.add_argument("--no-phase-a", dest="phase_a", action="store_false", help="T4: skip the phase A that already ran")
    ap.add_argument("--selection-root", type=Path, default=None, help="folder of phase A panels instead of taimma/<scenario>")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)
    if not args.phase_a and args.scenario != "T4":
        ap.error("--no-phase-a is only used with --scenario T4")
    if args.selection_root is not None and args.scenario == "T4":
        ap.error("--selection-root is not used with T4")
    if args.selection_root is not None and not args.selection_root.is_dir():
        ap.error(f"--selection-root {args.selection_root} is not a folder")
    plan = commands(args.scenario, args.datasets, args.phase_b, args.phase_a, args.selection_root)
    for cmd, _, _ in plan:
        print("python " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    if not os.environ.get("KLTN_DATA_ROOT", "").strip():
        print("STOP: set KLTN_DATA_ROOT (data and results folder).", flush=True)
        return 2
    for cmd, out, _ in plan:
        code = subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode
        if code != 0:
            print(f"ERROR: {cmd[0]} exited with code {code} ({out})", flush=True)
            return 1
    problems = [p for _, out, ds in plan for p in check_configs(out, ds, head_commit())]
    if problems:
        print("VERIFY FAILED:\n  " + "\n  ".join(problems), flush=True)
        return 3
    print(f"EVAL_OK {args.scenario} ({args.phase_b}): " + ", ".join(out for _, out, _ in plan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
