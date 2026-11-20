"""Sinh thân hai bảng kết quả của bài TAIMMA 2027 (Table 3–4, source 20/11, Review-2026-11-20) từ thư mục tables_v2.

    python scripts/make_taimma_paper_tables.py --tables-v2 <tables_v2> --out <source LaTeX>/tables [--hv6 <tables_v2/corrected>]

tables_v2 do scripts/make_taimma_v2_tables.py sinh ở commit 95333d0 (tag taimma-2027-v2-results). --hv6 là thư mục ra của
scripts/make_taimma_corrected.py sau phiên C4 (hv6.csv); chỉ dùng khi cổng R50 đạt.
  table3.tex  hàng là k, cột là HV1 (v2 − v1, v2 − GA) và HV4 (v2 trừ biến thể bỏ bước gen mới, khe, tournament); mỗi ô là
              hiệu trung bình của fitness nội bộ tốt nhất và p Holm trong ngoặc. Với --hv6: thêm hàng k = 20 và k = 50, chỉ
              cột khe có số (HV6a, Holm trên 2), các cột khác là "--".
  table4.tex  log-loss và AUROC ngoài theo B-LL, trung bình đều ba cohort GPL570; khối HV2: hiệu v2 − tham chiếu và khoảng
              tin cậy Nadeau–Bengio gộp 95 % cho log-loss và AUROC. Khoảng của AUROC = auroc_diff ± q · auroc_se, q lấy từ
              cột log-loss của cùng dòng (q = (logloss_ci_high − logloss_diff) / logloss_se, tức t 0,975 với df = 14).
Bản 20/11 bỏ Table 5 của bản 19b: số của HV3 và HV5 nằm trong văn bản mục 5.3–5.4 (stats_hv3_stability.csv, stats_hv5_b7.csv).
Chú thích, nhãn và cỡ chữ nằm ở môi trường table trong sections/05_results.tex. Không chép tay số vào bảng; muốn đổi số thì
sửa nguồn rồi chạy lại. Chạy hai lần cho tệp trùng từng byte.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scipy.stats as st

NAMES = {"mpqda_de": "v1 (consensus)", "mpqda_de_best": "v1 (best)", "ga": "GA", "bpso": "BPSO",
         "de_rand1": "DE/rand/1/bin", "shade": "SHADE", "qlde": "qlDE", "jde": "jDE", "jade": "JADE",
         "mpqda_v2": "v2 (consensus)", "mpqda_v2_best": "v2 (best)", "de_g": "DE-G"}
ABL = {"mpqda_v2_no_new": "new-gene step", "mpqda_v2_no_gap": "gap", "mpqda_v2_no_tour": "tournament"}
ORDER = ["mpqda_de", "ga", "bpso", "de_rand1", "jde", "jade", "shade", "qlde", "mpqda_v2", "mpqda_v2_best", "de_g"]
HV2_REF = (("bpso", "BPSO"), ("mpqda_de", "v1"))
T3_K = (2, 5, 10)
HV6_K = (20, 50)
DF_NB = 14                                                     # 15 fold mỗi cohort, công thức gộp của HV2


def f(x: float, d: int = 3) -> str:
    s = f"{x:.{d}f}"
    return "0." + "0" * d if s in (f"-0.{'0' * d}",) else s.replace("-", "$-$")


def p(x: float) -> str:
    return "$<$0.001" if x < 0.001 else f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def cell(r) -> str:
    return f"{f(r.mean_diff, 4)} ({p(r.p_holm)})"


def first(df: pd.DataFrame, mask) -> str:
    part = df[mask]
    return cell(part.iloc[0]) if len(part) else "--"


def table3(t: Path, hv6_dir: Path | None = None) -> str:
    h1 = pd.read_csv(t / "stats_hv1_search.csv")
    ab = pd.read_csv(t / "table_v2_ablation.csv")
    rows = []
    for k in T3_K:
        cells = [first(h1, (h1.k == k) & (h1.reference == ref)) for ref in ("mpqda_de", "ga")]
        cells += [first(ab, (ab.k == k) & (ab.ablation == a)) for a in ABL]
        rows.append(f"{k} & " + " & ".join(cells) + " \\\\")
    if hv6_dir is not None:
        h6 = pd.read_csv(hv6_dir / "hv6.csv")
        for k in HV6_K:
            part = h6[h6.k == k]
            if len(part):
                rows.append(f"{k} & -- & -- & -- & {cell(part.iloc[0])} & -- \\\\")
    return "\n".join([
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{HV1} & \\multicolumn{3}{c}{HV4, v2 minus v2 without} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
        "$k$ & v2 $-$ v1 & v2 $-$ GA & new-gene step & gap & tournament \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", ""])


def ci_cell(diff: float, lo: float, hi: float) -> str:
    return f"\\multicolumn{{2}}{{r}}{{{f(diff)} [{f(lo)}, {f(hi)}]}}"


def table4(t: Path) -> str:
    m = pd.read_csv(t / "table_v2_main.csv")
    m = m[m.k.isin([5, 10])]
    pooled = m.groupby(["method", "k"])[["logloss_mean", "auroc_mean"]].mean()   # trung bình đều của ba cohort
    h2 = pd.read_csv(t / "stats_hv2_generalization.csv")
    rows = []
    for meth in ORDER:
        if (meth, 5) not in pooled.index:
            continue
        a, b = pooled.loc[(meth, 5)], pooled.loc[(meth, 10)]
        rows.append(f"{NAMES[meth]} & {f(a.logloss_mean)} & {f(a.auroc_mean)} & {f(b.logloss_mean)} & {f(b.auroc_mean)} \\\\")
        if meth == "qlde":
            rows.append("\\midrule")
    hv2 = []
    for ref, name in HV2_REF:
        ll, au = [], []
        for k in (5, 10):
            r = h2[(h2.k == k) & (h2.reference == ref)].iloc[0]
            q = (r.logloss_ci_high - r.logloss_diff) / r.logloss_se if r.logloss_se > 0 else st.t.ppf(0.975, DF_NB)
            ll.append(ci_cell(r.logloss_diff, r.logloss_ci_low, r.logloss_ci_high))
            au.append(ci_cell(r.auroc_diff, r.auroc_diff - q * r.auroc_se, r.auroc_diff + q * r.auroc_se))
        hv2.append(f"{name}, log-loss & " + " & ".join(ll) + " \\\\")
        hv2.append(f"{name}, AUROC & " + " & ".join(au) + " \\\\")
    return "\n".join([
        "\\begin{tabular}{lrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{$k=5$} & \\multicolumn{2}{c}{$k=10$} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Method & Log-loss & AUROC & Log-loss & AUROC \\\\", "\\midrule", *rows, "\\midrule",
        "HV2, v2 $-$ reference & \\multicolumn{2}{c}{$k=5$, $\\Delta$ [95\\,\\% CI]} & "
        "\\multicolumn{2}{c}{$k=10$, $\\Delta$ [95\\,\\% CI]} \\\\",
        "\\midrule", *hv2, "\\bottomrule", "\\end{tabular}", ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-v2", type=Path, required=True, help="thư mục tables_v2")
    ap.add_argument("--out", type=Path, required=True, help="thư mục tables/ của source LaTeX")
    ap.add_argument("--hv6", type=Path, default=None, help="thư mục ra của make_taimma_corrected.py (hv6.csv), sau C4")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, text in (("table3.tex", table3(a.tables_v2, a.hv6)), ("table4.tex", table4(a.tables_v2))):
        (a.out / name).write_text(text, encoding="utf-8", newline="\n")
        print("đã ghi", a.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
