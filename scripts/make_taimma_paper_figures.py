"""Sinh hai hình của bài TAIMMA 2027 từ tệp kết quả, dạng PDF véc-tơ, vừa bề rộng chữ của LNCS (12,2 cm).

    python scripts/make_taimma_paper_figures.py --disruption docs/taimma/disruption.csv \
        --optimism <tables_v2>/optimism_v2.csv --out <source LaTeX>/figures

fig_disruption.pdf (Fig. 1): tỉ lệ gen ngẫu nhiên trong panel thử và tỉ lệ gen của leader được giữ, theo F (CR = 0,9), hai
độ rộng dải dưới ε = 0,5 (v1) và ε = 0,01 (v2), hai mức chung gen của quần thể (share 0 và 0,5). Nguồn: disruption.csv.
fig_optimism.pdf (Fig. 2): log-loss nội bộ và log-loss ngoài (B-C1) theo phương pháp, k = 5 và 10; khoảng cách tới đường
chéo là optimism. Nguồn: optimism_v2.csv (scripts/make_taimma_optimism.py, TQĐ38). Chưa có tệp thì không vẽ; khi đó PDF
của bài hiện một khung TODO ở chỗ của hình.
Hình vẽ từ số liệu bằng matplotlib, không dùng công cụ sinh ảnh. Chạy hai lần cho cùng hình (metadata ngày tháng tắt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})
W = 12.2 / 2.54                      # bề rộng chữ LNCS, inch
META = {"CreationDate": None, "ModDate": None, "Producer": None}
NAMES = {"mpqda_de": "v1", "mpqda_de_best": "v1 best", "ga": "GA", "bpso": "BPSO", "de_rand1": "DE/rand/1", "shade": "SHADE",
         "qlde": "qlDE", "jde": "jDE", "jade": "JADE", "mpqda_v2": "v2", "mpqda_v2_best": "v2 best", "de_g": "DE-G"}


def disruption(csv: Path, out: Path) -> None:
    d = pd.read_csv(csv)
    d = d[d.CR == 0.9]
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.40), constrained_layout=True)
    styles = {(0.5, 0.0): ("k-o", "v1 band, $\\varepsilon=0.5$, disjoint panels"),
              (0.5, 0.5): ("k--s", "v1 band, $\\varepsilon=0.5$, shared genes"),
              (0.01, 0.0): ("-^", "v2 gap, $\\varepsilon=0.01$, disjoint panels"),
              (0.01, 0.5): ("--v", "v2 gap, $\\varepsilon=0.01$, shared genes")}
    for ax, col, lab in ((axes[0], "random_genes", "random genes (%)"),
                         (axes[1], "leader_kept", "leader kept (%)")):
        for (eps, share), (st, name) in styles.items():
            g = d[(d.eps == eps) & (d.share == share)].sort_values("F")
            color = "0.0" if eps == 0.5 else "0.55"
            ax.plot(g.F, 100 * g[col], st.replace("k", ""), color=color, ms=3, lw=0.9, label=name)
        ax.set_xlabel("scale factor $F$")
        ax.set_ylabel(lab)
        ax.set_ylim(-3, 103)
        ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ax.grid(True, lw=0.3, color="0.85")
    h, lab_ = axes[0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="outside lower center", ncol=2, frameon=False, handlelength=2.2)
    fig.savefig(out / "fig_disruption.pdf", metadata=META)
    plt.close(fig)


def optimism(csv: Path, out: Path) -> None:
    d = pd.read_csv(csv)
    d = d[(d.phase_b == "B-C1") & d.k.isin([5, 10])]
    # Dumbbell: một hàng cho mỗi đầu ra, nối log-loss nội bộ (tròn rỗng) với log-loss ngoài (tròn đặc);
    # độ dài đoạn nối là optimism. Thứ tự hàng cố định theo NAMES để hai ô so được với nhau.
    order = [m for m in NAMES if m in set(d.method)]
    ypos = {m: len(order) - 1 - i for i, m in enumerate(order)}
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.36), constrained_layout=True, sharex=True, sharey=True)
    for ax, k in zip(axes, (5, 10)):
        g = d[d.k == k].set_index("method")
        for m in order:
            r, y = g.loc[m], ypos[m]
            ours = m.startswith("mpqda_v2") or m == "de_g"
            c = "0.0" if ours else "0.5"
            ax.plot([r.inner_logloss, r.outer_logloss], [y, y], color=c, lw=0.9)
            ax.plot(r.inner_logloss, y, "o", mfc="white", mec=c, ms=3.2, mew=0.8)
            ax.plot(r.outer_logloss, y, "o", color=c, ms=3.2)
        ax.set_title(f"$k={k}$", fontsize=8)
        ax.set_xlabel("log-loss")
        ax.set_xlim(0, 0.21)
        ax.grid(True, axis="x", lw=0.3, color="0.85")
    axes[0].set_yticks([ypos[m] for m in order])
    axes[0].set_yticklabels([NAMES[m] for m in order], fontsize=6)
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    h = [plt.Line2D([], [], ls="", marker="o", mfc="white", mec="0.0", ms=3.2, mew=0.8),
         plt.Line2D([], [], ls="", marker="o", color="0.0", ms=3.2)]
    fig.legend(h, ["inner log-loss (selection criterion)", "outer log-loss (B-C1)"], loc="outside lower center",
               ncol=2, frameon=False)
    fig.savefig(out / "fig_optimism.pdf", metadata=META)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disruption", type=Path, required=True, help="docs/taimma/disruption.csv")
    ap.add_argument("--optimism", type=Path, default=None, help="tables_v2/optimism_v2.csv (TQĐ38)")
    ap.add_argument("--out", type=Path, required=True, help="thư mục figures/ của source LaTeX")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    disruption(a.disruption, a.out)
    print("đã ghi", a.out / "fig_disruption.pdf")
    if a.optimism is not None and a.optimism.is_file():
        optimism(a.optimism, a.out)
        print("đã ghi", a.out / "fig_optimism.pdf")
    else:
        print("chưa có optimism_v2.csv: không vẽ fig_optimism.pdf (PDF của bài hiện khung TODO ở Fig. 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
