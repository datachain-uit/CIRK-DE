"""Sinh thân ba bảng kết quả của bài TAIMMA 2027 (Table 3–5) từ thư mục tables_v2.

    python scripts/make_taimma_paper_tables.py --tables-v2 <tables_v2> --out <source LaTeX>/tables

tables_v2 do scripts/make_taimma_v2_tables.py sinh ở commit 95333d0 (tag taimma-2027-v2-results).
Tệp sinh ra chỉ chứa môi trường tabular, theo cấu trúc của source mẫu:
  table3.tex  HV1 (v2 so với v1 và GA) và HV4 (ablation), fitness nội bộ
  table4.tex  log-loss và AUROC ngoài theo B-LL, trung bình đều ba cohort GPL570; khối HV2
  table5.tex  HV3 (Φ của Nogueira, GSE37745) và HV5 (AUROC trên gse31210_egfr)
Chú thích, nhãn và cỡ chữ nằm ở môi trường table trong sections/05_results.tex, kèm dòng "% source:".
Không chép tay số vào bảng; muốn đổi số thì sửa nguồn rồi chạy lại. Chạy hai lần cho tệp trùng từng byte.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NAMES = {"mpqda_de": "v1 (consensus)", "mpqda_de_best": "v1 (best)", "ga": "GA", "bpso": "BPSO",
         "de_rand1": "DE/rand/1/bin", "shade": "SHADE", "qlde": "qlDE", "jde": "jDE", "jade": "JADE",
         "mpqda_v2": "v2 (consensus)", "mpqda_v2_best": "v2 (best)", "de_g": "DE-G"}
ABL = {"mpqda_v2_no_new": "new-gene step", "mpqda_v2_no_gap": "gapped re-encoding", "mpqda_v2_no_tour": "archive tournament"}
ORDER = ["mpqda_de", "ga", "bpso", "de_rand1", "jde", "jade", "shade", "qlde", "mpqda_v2", "mpqda_v2_best", "de_g"]


def f(x: float, d: int = 3) -> str:
    s = f"{x:.{d}f}"
    return "0." + "0" * d if s in (f"-0.{'0' * d}",) else s.replace("-", "$-$")


def p(x: float) -> str:
    return "$<$0.001" if x < 0.001 else f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def table3(t: Path) -> str:
    h1 = pd.read_csv(t / "stats_hv1_search.csv")
    ab = pd.read_csv(t / "table_v2_ablation.csv")
    rows = []
    for _, r in h1.iterrows():
        ref = "v1" if r.reference == "mpqda_de" else "GA"
        rows.append(f"HV1 & v2 $-$ {ref} & {int(r.k)} & {f(r.mean_diff, 4)} & [{f(r.ci_low, 4)}, {f(r.ci_high, 4)}] & "
                    f"{r.share_v2_better:.2f} & {p(r.p_holm)} \\\\")
    rows.append("\\midrule")
    for a in ABL:
        for _, r in ab[ab.ablation == a].iterrows():
            rows.append(f"HV4 & without {ABL[a]} & {int(r.k)} & {f(r.mean_diff, 4)} & -- & {r.share_v2_better:.2f} & {p(r.p_holm)} \\\\")
    return "\n".join([
        "\\begin{tabular}{llrrcrr}", "\\toprule",
        "Test & Comparison & $k$ & Mean diff. & 95\\,\\% CI & Share v2 better & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", ""])


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
    for ref in ("bpso", "mpqda_de"):
        cells = []
        for k in (5, 10):
            r = h2[(h2.k == k) & (h2.reference == ref)].iloc[0]
            cells.append(f"{f(r.logloss_diff)} [{f(r.logloss_ci_low)}, {f(r.logloss_ci_high)}] & {f(r.auroc_diff)}")
        name = "BPSO" if ref == "bpso" else "v1"
        hv2.append(f"v2 $-$ {name} & " + " & ".join(cells) + " \\\\")
    return "\n".join([
        "\\begin{tabular}{lrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{$k=5$} & \\multicolumn{2}{c}{$k=10$} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Method & Log-loss & AUROC & Log-loss & AUROC \\\\", "\\midrule", *rows, "\\midrule",
        "HV2 & $\\Delta$ log-loss [95\\,\\% CI] & $\\Delta$ AUROC & $\\Delta$ log-loss [95\\,\\% CI] & $\\Delta$ AUROC \\\\",
        "\\midrule", *hv2, "\\bottomrule", "\\end{tabular}", ""])


def table5(t: Path) -> str:
    h3 = pd.read_csv(t / "stats_hv3_stability.csv")
    h5 = pd.read_csv(t / "stats_hv5_b7.csv")
    rows = []
    for k in (5, 10):
        v1 = h3[(h3.k == k) & (h3.reference == "mpqda_de")].iloc[0]
        ga = h3[(h3.k == k) & (h3.reference == "ga")].iloc[0]
        rows.append(f"HV3, $\\Phi$ & {k} & {f(v1.phi_v2)} & {f(v1.phi_reference)} & {f(ga.phi_reference)} & "
                    f"{p(v1.p_holm)}; {p(ga.p_holm)} \\\\")
    for k in (5, 10):
        r = h5[h5.k == k].iloc[0]
        rows.append(f"HV5, AUROC & {k} & {f(r.auroc_v2)} & {f(r.auroc_v1)} & -- & {p(r.p_holm)} \\\\")
    return "\n".join([
        "\\begin{tabular}{lrrrrl}", "\\toprule",
        "Quantity & $k$ & v2 & v1 & GA & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-v2", type=Path, required=True, help="thư mục tables_v2")
    ap.add_argument("--out", type=Path, required=True, help="thư mục tables/ của source LaTeX")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("table3.tex", table3), ("table4.tex", table4), ("table5.tex", table5)):
        (a.out / name).write_text(fn(a.tables_v2), encoding="utf-8", newline="\n")
        print("đã ghi", a.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
