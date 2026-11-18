"""Mô tả, không đăng ký (Review-2026-11-18, Mục 6.2): số gen khác nhau trong các panel của năm seed trên gse37745.

Với mỗi (method, k, repeat, fold), đếm số gen khác nhau trong hợp của panel đồng thuận của seed 0 (T1, T1v2) và seed 1–4
(T3_s1..4, T3v2_s1..4), rồi lấy trung bình, min, max qua 15 (repeat, fold).

    python scripts/genes_across_seeds.py <thư mục chứa T1, T1v2, T3_s*, T3v2_s*> tables_v2/genes_across_seeds_gse37745.csv
"""
import sys
import pandas as pd
from pathlib import Path
R = Path(sys.argv[1])
SC = (("T1", "T3_s", ("mpqda_de", "ga")),
      ("T1v2", "T3v2_s", ("mpqda_v2",)))
out = []
for s0, pre, methods in SC:
    f = "gse37745/selected_genes.csv"
    parts = [pd.read_csv(R / s0 / f)]
    parts += [pd.read_csv(R / f"{pre}{s}" / f) for s in (1, 2, 3, 4)]
    g = pd.concat(parts)
    g = g[g.method.isin(methods) & g.k.isin([5, 10])]
    n = g.groupby(["method", "k", "repeat", "fold"]).gene.nunique()
    out.append(n.groupby(["method", "k"]).agg(["mean", "min", "max", "count"]))
t = pd.concat(out)
print(t)
t.to_csv(sys.argv[2])
