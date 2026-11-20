"""Bảng của bài (source 20/11, Review-2026-11-20): Table 3 dạng hàng k, Table 4 có khoảng tin cậy AUROC của HV2, hàng HV6."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import make_taimma_paper_tables as P  # noqa: E402

TABLE3_PAPER = r"""\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{2}{c}{HV1} & \multicolumn{3}{c}{HV4, v2 minus v2 without} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}
$k$ & v2 $-$ v1 & v2 $-$ GA & new-gene step & gap & tournament \\
\midrule
2 & -- & -- & 0.0698 ($<$0.001) & $-$0.0017 (1.00) & $-$0.0005 (1.00) \\
5 & 0.0058 (0.012) & 0.0019 (0.46) & 0.0372 ($<$0.001) & 0.0036 (0.33) & 0.0004 (1.00) \\
10 & 0.0096 ($<$0.001) & $-$0.0005 (0.46) & 0.0175 ($<$0.001) & 0.0045 ($<$0.001) & 0.0043 (0.009) \\
\bottomrule
\end{tabular}
"""


def _write(t: Path) -> None:
    pd.DataFrame([
        {"k": 5, "reference": "mpqda_de", "mean_diff": 0.0058, "p_holm": 0.012},
        {"k": 5, "reference": "ga", "mean_diff": 0.0019, "p_holm": 0.46},
        {"k": 10, "reference": "mpqda_de", "mean_diff": 0.0096, "p_holm": 0.0004},
        {"k": 10, "reference": "ga", "mean_diff": -0.0005, "p_holm": 0.46},
    ]).to_csv(t / "stats_hv1_search.csv", index=False)
    abl = {"mpqda_v2_no_new": [(2, 0.0698, 1e-4), (5, 0.0372, 1e-4), (10, 0.0175, 1e-4)],
           "mpqda_v2_no_gap": [(2, -0.0017, 1.0), (5, 0.0036, 0.33), (10, 0.0045, 5e-4)],
           "mpqda_v2_no_tour": [(2, -0.0005, 1.0), (5, 0.0004, 1.0), (10, 0.0043, 0.009)]}
    pd.DataFrame([{"ablation": a, "k": k, "mean_diff": d, "p_holm": q} for a, rows in abl.items() for k, d, q in rows]
                 ).to_csv(t / "table_v2_ablation.csv", index=False)
    main = []
    for i, meth in enumerate(P.ORDER):
        for k in (2, 5, 10, 50):
            for c, shift in (("gse10245", -0.01), ("gse30219", 0.0), ("gse37745", 0.01)):
                main.append({"cohort": c, "method": meth, "k": k, "logloss_mean": 0.2 + i / 100 + shift,
                             "auroc_mean": 0.97 - i / 1000 + shift / 10})
    pd.DataFrame(main).to_csv(t / "table_v2_main.csv", index=False)
    q = st.t.ppf(0.975, 14)
    hv2 = []
    for k in (5, 10):
        for ref in ("bpso", "mpqda_de"):
            hv2.append({"k": k, "reference": ref, "logloss_diff": 0.02, "logloss_se": 0.05, "logloss_ci_low": 0.02 - q * 0.05,
                        "logloss_ci_high": 0.02 + q * 0.05, "auroc_diff": 0.004, "auroc_se": 0.01})
    pd.DataFrame(hv2).to_csv(t / "stats_hv2_generalization.csv", index=False)


def test_table3_reproduces_the_paper_layout(tmp_path):
    _write(tmp_path)
    assert P.table3(tmp_path) == TABLE3_PAPER


def test_table3_adds_hv6_rows_only_in_the_gap_column(tmp_path):
    _write(tmp_path)
    corr = tmp_path / "corrected"
    corr.mkdir()
    pd.DataFrame([{"k": 20, "mean_diff": 0.0081, "p_holm": 0.0004}, {"k": 50, "mean_diff": 0.0123, "p_holm": 0.02}]
                 ).to_csv(corr / "hv6.csv", index=False)
    lines = P.table3(tmp_path, corr).splitlines()
    assert lines[-4] == "20 & -- & -- & -- & 0.0081 ($<$0.001) & -- \\\\"
    assert lines[-3] == "50 & -- & -- & -- & 0.0123 (0.020) & -- \\\\"
    assert "\n".join(lines[:-4]) == "\n".join(TABLE3_PAPER.splitlines()[:-2])


def test_table4_auroc_interval_uses_the_log_loss_quantile(tmp_path):
    _write(tmp_path)
    text = P.table4(tmp_path)
    assert "BPSO, log-loss & \\multicolumn{2}{r}{0.020 [$-$0.087, 0.127]} & \\multicolumn{2}{r}{0.020 [$-$0.087, 0.127]} \\\\" in text
    assert "v1, AUROC & \\multicolumn{2}{r}{0.004 [$-$0.017, 0.025]} & \\multicolumn{2}{r}{0.004 [$-$0.017, 0.025]} \\\\" in text
    rows = [r for r in text.splitlines() if r.startswith(("v1 (consensus)", "qlDE", "DE-G"))]
    assert rows[0] == "v1 (consensus) & 0.200 & 0.970 & 0.200 & 0.970 \\\\"
    assert text.count("\\midrule") == 4 and text.endswith("\\end{tabular}\n")


def test_main_writes_two_tables_byte_stable(tmp_path):
    _write(tmp_path)
    out = tmp_path / "tables"
    assert P.main(["--tables-v2", str(tmp_path), "--out", str(out)]) == 0
    first = {n: (out / n).read_bytes() for n in ("table3.tex", "table4.tex")}
    assert not (out / "table5.tex").exists()
    P.main(["--tables-v2", str(tmp_path), "--out", str(out)])
    assert first == {n: (out / n).read_bytes() for n in ("table3.tex", "table4.tex")}
