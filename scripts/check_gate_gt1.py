"""Cổng G-T1 (Mục 4.3): GA, BPSO ở chế độ AUROC phải tái tạo đúng panel của survey.

    python scripts/check_gate_gt1.py [--run-root $KLTN_DATA_ROOT/results/survey_v2/taimma/T1b] [--datasets gse10245 ...]

So `selected_genes.csv` và `selection_per_fold.csv` của GA, BPSO trong lần chạy với `vendor/reference/`: cùng tập gen ở mọi
(lần lặp, fold, phương pháp, k) đã chạy; `n_eval` bằng nhau tuyệt đối; `best_fitness` lệch không quá FITNESS_TOL = 1e-12
(PREREG Mục 9, sửa trước đăng ký ngày 15/11/2026). In bảng lệch và trả mã 2 nếu có lệch hoặc không có đơn vị nào để so; trả 0
khi mọi đơn vị trùng. Các đơn vị có `best_fitness` khác nhau nhưng trong dung sai được in ra và ghi vào
`<report>_within_tol.csv`, không làm cổng trượt.

Lý do của dung sai: trên Kaggle ngày 15/11/2026 (commit 35331ed), 268/270 đơn vị của T1b trùng hệt; hai đơn vị còn lại có cùng
tập gen và cùng n_eval, `best_fitness` chỉ khác ở bit cuối (0.9993141289437585 so với 0.9993141289437584), do phép cộng số thực
khác nhau giữa hai máy. 1e-12 là ngưỡng bão hoà của PREREG (mpqda.config.SATURATION_EPS).
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
FITNESS_TOL = 1e-12                    # = mpqda.config.SATURATION_EPS; tập gen và n_eval vẫn phải trùng tuyệt đối
UNIT = ["repeat", "fold", "method", "k"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _panels(genes: pd.DataFrame) -> dict:
    return {key: frozenset(g.gene.astype(str)) for key, g in genes.groupby(UNIT)}


def compare(run_dir: Path, ref_dir: Path, cohort: str = "", methods=METHODS, within_tol: list | None = None
            ) -> tuple[pd.DataFrame, int]:
    """Bảng lệch (rỗng = trùng) và số đơn vị (lần lặp, fold, phương pháp, k) đã so. Nếu truyền `within_tol`, các dòng có
    best_fitness khác nhau nhưng |hiệu| <= FITNESS_TOL được thêm vào đó."""
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
            rows.append({**base, "field": "reference", "run": "có", "survey": "không có đơn vị này"})
            continue
        mine, theirs = run_p.get(key, frozenset()), ref_p[key]
        if mine != theirs:
            rows.append({**base, "field": "genes", "run": ";".join(sorted(mine - theirs)) or "(thiếu gen)",
                         "survey": ";".join(sorted(theirs - mine)) or "(thừa gen)"})
        if not r["n_eval"] == ref["n_eval"]:                          # bằng nhau tuyệt đối; NaN cũng tính là lệch
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
                    help="thư mục chứa <cohort>/selected_genes.csv của lần chạy T1b (mặc định: kết quả taimma/T1b)")
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--datasets", nargs="+", default=["gse10245", "gse30219", "gse37745"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--report", type=Path, default=None, help="ghi bảng lệch ra CSV (kể cả khi rỗng)")
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
        print(f"[G-T1] {name}: {n} đơn vị đã so, {table[UNIT].drop_duplicates().shape[0]} đơn vị lệch", flush=True)
    diff = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    near_df = pd.DataFrame(near, columns=["cohort", *UNIT, "field", "run", "survey", "abs_diff"])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        diff.to_csv(args.report, index=False)
        near_df.to_csv(args.report.with_name(args.report.stem + "_within_tol.csv"), index=False)
    if len(near_df):
        print(f"[G-T1] {len(near_df)} đơn vị có best_fitness khác ở mức làm tròn (|hiệu| <= {FITNESS_TOL:g}; tập gen và n_eval "
              "trùng), không tính là lệch:", flush=True)
        print(near_df.to_string(index=False), flush=True)
    if missing:
        print(f"[G-T1] không có kết quả cho: {missing}", flush=True)
    if n_units == 0:
        print("[G-T1] KHÔNG ĐẠT: không có đơn vị GA, BPSO nào để so.", flush=True)
        return 2
    if len(diff):
        print(diff.to_string(index=False), flush=True)
        print(f"[G-T1] KHÔNG ĐẠT: {len(diff)} dòng lệch trên {n_units} đơn vị. Dừng, không mở S1–S3.", flush=True)
        return 2
    print(f"[G-T1] ĐẠT: {n_units} đơn vị trùng survey (tập gen, n_eval; best_fitness trong {FITNESS_TOL:g}).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
