"""Gate G-T1: GA and BPSO in AUROC mode must reproduce the panels of the vendored code exactly.

    python scripts/check_gate_gt1.py [--run-root $KLTN_DATA_ROOT/results/survey_v2/taimma/T1b] [--datasets gse10245 ...]

Compares `selected_genes.csv` and `selection_per_fold.csv` of GA and BPSO of the run with `vendor/reference/`: same gene set in every
(repeat, fold, method, k) that was run; `n_eval` exactly equal; `best_fitness` differing by at most FITNESS_TOL = 1e-12
(PREREG, Section 9, amended before registration on 2026-11-15). Prints the differences and returns code 2 if any differ or no unit can be compared; returns 0
when every unit matches. Units whose `best_fitness` differs within the tolerance are printed and written to
`<report>_within_tol.csv`, and do not fail the gate.

Reason for the tolerance: on Kaggle on 2026-11-15 (commit 35331ed), 268/270 units of T1b matched exactly; the other two had the same
gene set and the same n_eval, and `best_fitness` differed only in the last bit (0.9993141289437585 against 0.9993141289437584), because floating-point
addition differs between the two machines. 1e-12 is the saturation threshold of PREREG (mpqda.config.SATURATION_EPS).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

REFERENCE = ROOT / "vendor" / "reference" / "E1_wrapper_feature_selection"
METHODS = ("ga", "bpso")
FITNESS_TOL = 1e-12                    # = mpqda.config.SATURATION_EPS; gene set and n_eval must still match exactly
UNIT = ["repeat", "fold", "method", "k"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _panels(genes: pd.DataFrame) -> dict:
    return {key: frozenset(g.gene.astype(str)) for key, g in genes.groupby(UNIT)}


def compare(run_dir: Path, ref_dir: Path, cohort: str = "", methods=METHODS, within_tol: list | None = None
            ) -> tuple[pd.DataFrame, int]:
    """Table of differences (empty = match) and number of units (repeat, fold, method, k) compared. If `within_tol` is given, rows with
    different best_fitness but |difference| <= FITNESS_TOL are appended to it."""
    run_sel, ref_sel = _read(Path(run_dir) / "selection_per_fold.csv"), _read(Path(ref_dir) / "selection_per_fold.csv")
    run_sel = run_sel[run_sel.method.isin(methods)]
    run_p, ref_p = _panels(_read(Path(run_dir) / "selected_genes.csv")), _panels(_read(Path(ref_dir) / "selected_genes.csv"))
    ref_rows = {tuple(r[c] for c in UNIT): r for r in ref_sel.to_dict("records")}
    rows = []
    for r in run_sel.sort_values(UNIT, kind="stable").to_dict("records"):
        key = tuple(r[c] for c in UNIT)
        base = dict(zip(["cohort", *UNIT], [cohort, *key]))
        ref = ref_rows.get(key)
        if ref is None or key not in ref_p:
            rows.append({**base, "field": "reference", "run": "present", "survey": "unit absent"})
            continue
        mine, theirs = run_p.get(key, frozenset()), ref_p[key]
        if mine != theirs:
            rows.append({**base, "field": "genes", "run": ";".join(sorted(mine - theirs)) or "(missing genes)",
                         "survey": ";".join(sorted(theirs - mine)) or "(extra genes)"})
        if not r["n_eval"] == ref["n_eval"]:                          # exactly equal; NaN also counts as a difference
            rows.append({**base, "field": "n_eval", "run": repr(r["n_eval"]), "survey": repr(ref["n_eval"])})
        a, b = r["best_fitness"], ref["best_fitness"]
        if not a == b:
            row = {**base, "field": "best_fitness", "run": repr(a), "survey": repr(b)}
            if pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) <= FITNESS_TOL:
                if within_tol is not None:
                    within_tol.append({**row, "abs_diff": abs(float(a) - float(b))})
            else:
                rows.append(row)
    return pd.DataFrame(rows, columns=["cohort", *UNIT, "field", "run", "survey"]), len(run_sel)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, default=None,
                    help="folder holding <cohort>/selected_genes.csv of the T1b run (default: results taimma/T1b)")
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--datasets", nargs="+", default=["gse10245", "gse30219", "gse37745"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--report", type=Path, default=None, help="write the table of differences to CSV (even when empty)")
    args = ap.parse_args(argv)
    if args.run_root is None:
        from mpqda import config as M
        from mpqda.runlog import DATA_ROOT
        from survey_v2 import config as C
        args.run_root = DATA_ROOT.joinpath(*C.RESULT_DIR, M.RESULT_SUBDIR, "T1b")
    tables, n_units, missing, near = [], 0, [], []
    for name in args.datasets:
        run_dir = args.run_root / name
        if not (run_dir / "selection_per_fold.csv").is_file():
            missing.append(name)
            continue
        table, n = compare(run_dir, args.reference / name, cohort=name, methods=args.methods, within_tol=near)
        tables.append(table)
        n_units += n
        print(f"[G-T1] {name}: {n} units compared, {table[UNIT].drop_duplicates().shape[0]} units differ", flush=True)
    diff = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    near_df = pd.DataFrame(near, columns=["cohort", *UNIT, "field", "run", "survey", "abs_diff"])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        diff.to_csv(args.report, index=False)
        near_df.to_csv(args.report.with_name(args.report.stem + "_within_tol.csv"), index=False)
    if len(near_df):
        print(f"[G-T1] {len(near_df)} units have best_fitness differing at rounding level (|difference| <= {FITNESS_TOL:g}; gene set and n_eval "
              "match), not counted as differences:", flush=True)
        print(near_df.to_string(index=False), flush=True)
    if missing:
        print(f"[G-T1] no results for: {missing}", flush=True)
    if n_units == 0:
        print("[G-T1] FAILED: no GA or BPSO unit to compare.", flush=True)
        return 2
    if len(diff):
        print(diff.to_string(index=False), flush=True)
        print(f"[G-T1] FAILED: {len(diff)} differing rows over {n_units} units. Stop, do not open S1–S3.", flush=True)
        return 2
    print(f"[G-T1] PASSED: {n_units} units match the vendored code (gene set, n_eval; best_fitness within {FITNESS_TOL:g}).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
