"""Bảng, hình, kiểm định và sổ số liệu của bài TAIMMA (Mục 4.5; PREREG Mục 7).

    python scripts/make_taimma_tables.py --results <thư mục chứa T1, T1b, T2, T3_s1..4, T5LL/..., T5/...> --out tables

Pha B (TQĐ11): mọi bảng chính, H3, H4 (qua H3), H5 (Q2), phép thử phụ và Friedman đọc B-LL (`T5LL/`). Bảng độ nhạy
`*_bauc` đọc pha B của survey (`T5/`), nhãn trong claims.csv có hậu tố `_BAUC`. `table_s_c_choice` ghi tỉ lệ panel nhận
từng giá trị C dưới hai pha B. Cột `phase_b` của claims.csv ghi pha của từng số (B-LL, B-AUC hoặc phase A).
Đầu vào là các thư mục kết quả đã giải nén (cấu trúc của results/survey_v2/taimma/). SHA-256 của mọi tệp đầu vào ghi vào
`inputs_sha256.csv`. Mọi phép thống kê dùng `survey_v2.stats` (nadeau_bengio_ttest, holm, bayesian_correlated_ttest),
`survey_v2.stability.nogueira`, `scipy.stats.wilcoxon`, `scipy.stats.friedmanchisquare`; công thức gộp ba cohort là
`pooled_nb`. Chạy hai lần cho kết quả trùng từng byte: sắp dòng cố định, định dạng số cố định, không ghi thời điểm.
Kịch bản chưa có kết quả thì bảng tương ứng bị bỏ qua và được ghi trong `missing_inputs.txt`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from survey_v2 import config as C  # noqa: E402
from survey_v2.stability import nogueira  # noqa: E402
from survey_v2.stats import bayesian_correlated_ttest, holm, nadeau_bengio_ttest  # noqa: E402

from mpqda import config as M  # noqa: E402

COHORTS = list(C.GPL570_COHORTS)
FOLD = ["repeat", "fold"]
MARGIN_LOGLOSS, ROPE_AUROC, MARGIN_NOGUEIRA, ALPHA = 0.02, C.ROPE_AUROC, 0.2, C.ALPHA
PRIMARY_K = (5, 10)
BASELINE_ORDER = ("ga", "bpso", "de_rand1", "shade", "qlde", "jde", "jade")
NEMENYI_Q05 = {6: 2.850, 8: 3.031}             # Demšar (2006), alpha = 0,05
N_GENES_DEFAULT = 16479
H5_VARIANTS = ("mpqda_a0_noband", "mpqda_a1_uniform", "mpqda_a2a_bestleader", "mpqda_a2b_randleader", "mpqda_a3_fixedF",
               "mpqda_a5_A1")                  # TQĐ14: A0 vào họ Holm chính
COMPONENT = {"mpqda_a0_noband": "band re-encoding", "mpqda_a1_uniform": "chaotic initialisation", "mpqda_a2a_bestleader": "archive-based leader",
             "mpqda_a2b_randleader": "archive-based leader", "mpqda_a5_A1": "archive-based leader",
             "mpqda_a3_fixedF": "Q-learning control", "A4": "consensus output"}
FMT = "%.10g"
PHASE_B_DIRS = {"B-LL": "T5LL", "B-AUC": "T5"}  # pha B -> thư mục kết quả
PHASE_A = "phase A"
B7_K = (5, 10)


# ------------------------------------------------------------------------------------------------- thống kê
def pooled_nb(d_by_cohort, ratio: float = C.NB_TEST_TRAIN_RATIO) -> dict:
    """d̄ = trung bình (trọng số bằng nhau) của d̄_c; SE² = (1/C²)·Σ_c (1/J_c + ρ)·s_c²; df = min J_c − 1.
    Trả về khoảng tin cậy hai phía 95 % (ci_high dùng cho biên không kém)."""
    d_by_cohort = [np.asarray(d, dtype=float) for d in d_by_cohort]
    n = len(d_by_cohort)
    mean = float(np.mean([d.mean() for d in d_by_cohort]))
    se = float(np.sqrt(sum((1.0 / len(d) + ratio) * d.var(ddof=1) for d in d_by_cohort)) / n)
    df = min(len(d) for d in d_by_cohort) - 1
    q = st.t.ppf(0.975, df)
    return {"mean": mean, "se": se, "df": df, "ci_low": mean - q * se, "ci_high": mean + q * se}


def pooled_p_greater(pool: dict) -> float:
    """p một phía P(t(df) > d̄/SE)."""
    if pool["se"] == 0:
        return 0.0 if pool["mean"] > 0 else 1.0
    return float(st.t.sf(pool["mean"] / pool["se"], pool["df"]))


def wilcoxon_p(d, alternative: str) -> float:
    """scipy.stats.wilcoxon(zero_method="wilcox", method="auto"); mọi hiệu bằng 0 thì p = 1 (không có bằng chứng)."""
    d = np.asarray(d, dtype=float)
    if not np.any(d != 0):
        return 1.0
    return float(st.wilcoxon(d, zero_method="wilcox", alternative=alternative, method="auto").pvalue)


# ------------------------------------------------------------------------------------------------- đọc đầu vào
class Inputs:
    def __init__(self, root: Path):
        self.root, self.used, self.missing = Path(root), {}, []

    def read(self, rel: str, **kw) -> pd.DataFrame | None:
        path = self.root / rel
        if not path.is_file():
            self.missing.append(rel)
            return None
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        self.used[Path(rel).as_posix()] = h
        return pd.read_csv(path, float_precision="round_trip", **kw)

    def scenario(self, scenario: str, name: str, cohorts=COHORTS, **kw) -> pd.DataFrame | None:
        frames = [self.read(f"{scenario}/{c}/{name}", **kw) for c in cohorts]
        frames = [f for f in frames if f is not None]
        return pd.concat(frames, ignore_index=True) if len(frames) == len(cohorts) else None

    def evaluation(self, scenario: str, cohorts=COHORTS, phase_b: str = "B-LL") -> pd.DataFrame | None:
        ev = self.scenario(f"{PHASE_B_DIRS[phase_b]}/{scenario}", "classification_per_fold.csv", cohorts, encoding="utf-8-sig")
        return None if ev is None else ev[ev.model == "lr_l2"].reset_index(drop=True)


def paired(frame, a, b, value, k, cohort) -> np.ndarray:
    """Hiệu a − b của cột `value` theo fold (ghép theo lần lặp, fold) trong một cohort và một k."""
    part = frame[(frame.cohort == cohort) & (frame.k == k)]
    pa = part[part.method == a].set_index(FOLD)[value]
    pb = part[part.method == b].set_index(FOLD)[value]
    idx = pa.index.intersection(pb.index).sort_values()
    return (pa.loc[idx] - pb.loc[idx]).to_numpy(dtype=float)


def _decision(n_sig: int, n: int, need: int) -> str:
    return "supported" if n_sig >= need else ("rejected" if n_sig == 0 else "partially supported")


# ------------------------------------------------------------------------------------------------- H1–H5
def h1(sel_b: pd.DataFrame) -> pd.DataFrame:
    """Tỉ lệ bão hoà của T1b theo cohort: một lần chạy cho mỗi (phương pháp, k, fold), xét best_fitness."""
    runs = sel_b[~sel_b.method.str.endswith(M.BEST_SUFFIX)]
    sat = runs.best_fitness >= 1 - M.SATURATION_EPS
    rows = [{"cohort": c, "n_runs": int((runs.cohort == c).sum()), "n_saturated": int(sat[runs.cohort == c].sum())} for c in COHORTS]
    two = runs.cohort.isin(COHORTS[:2])
    rows.append({"cohort": "pooled_gse10245_gse30219", "n_runs": int(two.sum()), "n_saturated": int(sat[two].sum())})
    out = pd.DataFrame(rows)
    out["rate"] = out.n_saturated / out.n_runs
    r = out.set_index("cohort").rate
    out["decision"] = ("supported" if r[COHORTS[0]] >= 0.90 and r[COHORTS[1]] >= 0.90
                       else "rejected" if r["pooled_gse10245_gse30219"] < 0.50 else "partially supported")
    return out


def h2(sel: pd.DataFrame) -> pd.DataFrame:
    """best_fitness(MPQDA-DE) − best_fitness(DE/rand/1/bin): Wilcoxon một phía theo cohort và k ∈ {5, 10}; một họ Holm 6."""
    rows = []
    for k in PRIMARY_K:
        for c in COHORTS:
            d = paired(sel, "mpqda_de", "de_rand1", "best_fitness", k, c)
            rows.append({"k": k, "cohort": c, "n": len(d), "mean_diff": float(d.mean()), "p": wilcoxon_p(d, "greater")})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["significant"] = out.p_holm < ALPHA
    n_sig = out.groupby("k").significant.sum()
    out["decision_k"] = out.k.map(lambda k: _decision(int(n_sig[k]), len(COHORTS), 2))
    out["decision"] = "supported" if (out.decision_k == "supported").all() else "not supported at both k"
    return out


def comparator(ev: pd.DataFrame, k: int) -> str:
    """m*: baseline wrapper có log-loss ngoài trung bình thấp nhất trên 45 fold; hoà theo BASELINE_ORDER."""
    part = ev[(ev.k == k) & ev.cohort.isin(COHORTS)]
    means = {m: float(part[part.method == m].logloss.mean()) for m in BASELINE_ORDER if (part.method == m).any()}
    return min(means, key=lambda m: (means[m], BASELINE_ORDER.index(m)))


def h3(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in PRIMARY_K:
        m = comparator(ev, k)
        ll = pooled_nb([paired(ev, "mpqda_de", m, "logloss", k, c) for c in COHORTS])
        au = pooled_nb([paired(ev, "mpqda_de", m, "auroc", k, c) for c in COHORTS])
        p_worse = (float(au["mean"] < -ROPE_AUROC) if au["se"] == 0
                   else float(st.t.cdf((-ROPE_AUROC - au["mean"]) / au["se"], au["df"])))
        a, b = ll["ci_high"] < MARGIN_LOGLOSS, p_worse < ALPHA
        rows.append({"k": k, "comparator": m, "logloss_diff": ll["mean"], "logloss_se": ll["se"], "logloss_ci_low": ll["ci_low"],
                     "logloss_ci_high": ll["ci_high"], "noninferior_logloss": a, "auroc_diff": au["mean"], "auroc_se": au["se"],
                     "p_auroc_below_minus_rope": p_worse, "auroc_ok": b, "h3_holds": bool(a and b)})
    return pd.DataFrame(rows)


def secondary(ev: pd.DataFrame, ks) -> pd.DataFrame:
    """Trong mỗi (cohort, k), MPQDA-DE (đồng thuận) so với từng baseline: Nadeau–Bengio và Wilcoxon hai phía, Holm trong các
    baseline; Bayesian correlated t-test cho AUROC với ROPE 0,01."""
    rows = []
    for c in sorted(ev.cohort.unique()):
        for k in ks:
            block = []
            for m in BASELINE_ORDER:
                d = paired(ev, "mpqda_de", m, "logloss", k, c)
                if len(d) < 2:
                    continue
                nb = nadeau_bengio_ttest(d, alternative="two-sided")
                bay = bayesian_correlated_ttest(paired(ev, "mpqda_de", m, "auroc", k, c))
                block.append({"cohort": c, "k": k, "baseline": m, "n": len(d), "logloss_diff": nb["mean"], "ci_low": nb["ci_low"],
                              "ci_high": nb["ci_high"], "p_nb": nb["p"], "p_wilcoxon": wilcoxon_p(d, "two-sided"),
                              "auroc_p_left": bay["p_left"], "auroc_p_rope": bay["p_rope"], "auroc_p_right": bay["p_right"]})
            if block:
                b = pd.DataFrame(block)
                b["p_nb_holm"], b["p_wilcoxon_holm"] = holm(b.p_nb.to_numpy()), holm(b.p_wilcoxon.to_numpy())
                rows.append(b)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def friedman(ev: pd.DataFrame, ks) -> tuple[pd.DataFrame, dict]:
    """Friedman trên log-loss ngoài trung bình qua các khối (cohort × k); hạng trung bình và critical difference của Nemenyi."""
    methods = ["mpqda_de", *[m for m in BASELINE_ORDER if (ev.method == m).any()]]
    part = ev[ev.method.isin(methods) & ev.cohort.isin(COHORTS) & ev.k.isin(ks)]
    table = part.groupby(["cohort", "k", "method"]).logloss.mean().unstack("method")[methods].dropna()
    ranks = table.rank(axis=1, method="average")
    n, m = table.shape
    stat = st.friedmanchisquare(*[table[c].to_numpy() for c in methods]) if n >= 2 and m >= 3 else None
    q = NEMENYI_Q05.get(m, np.nan)
    info = {"n_blocks": n, "n_methods": m, "chi2": float(stat.statistic) if stat else np.nan,
            "p": float(stat.pvalue) if stat else np.nan, "cd": float(q * np.sqrt(m * (m + 1) / (6.0 * n))) if n else np.nan}
    return ranks.mean().rename("mean_rank").reset_index(), info


def seed_stability(inputs: Inputs, n_genes: int = N_GENES_DEFAULT) -> pd.DataFrame | None:
    """Φ theo fold của GSE37745: Nogueira của năm panel (seed index 0 từ T1, 1–4 từ T3_s1..4) cho từng đầu ra và k ∈ {5, 10}."""
    frames = [inputs.read("T1/gse37745/selected_genes.csv")]
    frames += [inputs.read(f"T3_s{s}/gse37745/selected_genes.csv") for s in (1, 2, 3, 4)]
    if any(f is None for f in frames):
        return None
    genes = pd.concat([f.assign(seed_index=s) for s, f in enumerate(frames)], ignore_index=True)
    genes = genes[genes.k.isin(PRIMARY_K)]
    rows = []
    for (method, k, repeat, fold), g in genes.groupby(["method", "k", "repeat", "fold"]):
        if g.seed_index.nunique() < 5:
            continue
        universe = sorted(g.gene.astype(str).unique())
        Z = np.zeros((5, n_genes), dtype=np.int8)
        for s, panel in g.groupby("seed_index"):
            Z[int(s), [universe.index(x) for x in panel.gene.astype(str)]] = 1
        rows.append({"method": method, "k": int(k), "repeat": int(repeat), "fold": int(fold), "phi": nogueira(Z)["stability"]})
    return pd.DataFrame(rows)


def h4(phi: pd.DataFrame, h3_table: pd.DataFrame | None) -> pd.DataFrame:
    def diff(k, a, b, shift=0.0):
        pa = phi[(phi.k == k) & (phi.method == a)].set_index(FOLD).phi
        pb = phi[(phi.k == k) & (phi.method == b)].set_index(FOLD).phi
        idx = pa.index.intersection(pb.index).sort_values()
        return (pa.loc[idx] - pb.loc[idx]).to_numpy(dtype=float) - shift
    rows = []
    for k in PRIMARY_K:
        for test, other, shift in (("i_consensus_vs_best", "mpqda_de" + M.BEST_SUFFIX, 0.0), ("ii_vs_ga", "ga", MARGIN_NOGUEIRA),
                                   ("iii_vs_bpso", "bpso", MARGIN_NOGUEIRA)):
            d = diff(k, "mpqda_de", other, shift)
            rows.append({"k": k, "test": test, "n": len(d), "mean_diff_minus_margin": float(d.mean()), "p": wilcoxon_p(d, "greater")})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["significant"] = out.p_holm < ALPHA
    h3_ok = {} if h3_table is None else dict(zip(h3_table.k, h3_table.h3_holds))
    all_sig = out.groupby("k").significant.all()
    out["h4_holds"] = out.k.map(lambda k: bool(all_sig[k] and h3_ok.get(k, False)))
    return out


def h5(sel1, sel2, ev1, ev2) -> pd.DataFrame:
    """Mỗi biến thể ghép với lần chạy MPQDA-DE đầy đủ của T1. Q1: best_fitness(đầy đủ) − best_fitness(biến thể), Wilcoxon
    một phía trên 45 fold; Q2: log-loss(đầu ra chính của biến thể) − log-loss(đồng thuận đầy đủ), Nadeau–Bengio gộp một phía.
    Một họ Holm 39 phép thử (A0, A1, A2a, A2b, A3, A5_A1 × {Q1, Q2} × 3 k, cộng A4 × Q2 × 3 k; TQĐ14); A5_A30 chỉ mô tả."""
    sel, ev = pd.concat([sel1, sel2], ignore_index=True), pd.concat([ev1, ev2], ignore_index=True)
    rows = []
    for variant in (*H5_VARIANTS, "A4", "mpqda_a5_A30"):
        other = "mpqda_de" + M.BEST_SUFFIX if variant == "A4" else variant
        family = "main" if variant in (*H5_VARIANTS, "A4") else "descriptive"
        for k in (2, 5, 10):
            if variant != "A4":
                d = np.concatenate([paired(sel, "mpqda_de", other, "best_fitness", k, c) for c in COHORTS])
                rows.append({"variant": variant, "k": k, "question": "Q1_inner_fitness", "family": family, "n": len(d),
                             "mean_diff": float(d.mean()), "ci_low": np.nan, "ci_high": np.nan, "p": wilcoxon_p(d, "greater")})
            diffs = [paired(ev, other, "mpqda_de", "logloss", k, c) for c in COHORTS]
            pool = pooled_nb(diffs)
            rows.append({"variant": variant, "k": k, "question": "Q2_outer_logloss", "family": family, "n": sum(map(len, diffs)),
                         "mean_diff": pool["mean"], "ci_low": pool["ci_low"], "ci_high": pool["ci_high"], "p": pooled_p_greater(pool)})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    mask = (out.family == "main").to_numpy()
    out.loc[mask, "p_holm"] = holm(out.p.to_numpy()[mask])
    out["significant"] = out.p_holm < ALPHA
    out["component"] = out.variant.map(COMPONENT).fillna("")
    out["phase_b"] = np.where(out.question == "Q2_outer_logloss", "B-LL", PHASE_A)
    return out


def auprc_per_fold(scores: pd.DataFrame) -> pd.DataFrame:
    """AUPRC (average precision, lớp 1 là dương) trên test ngoài của từng (cohort, lần lặp, fold, phương pháp, k), từ điểm từng
    mẫu mà pha B lưu (--save-scores)."""
    rows = [{"cohort": c, "repeat": int(r), "fold": int(f), "method": m, "k": int(k),
             "auprc": float(average_precision_score(g.label.to_numpy(dtype=int), g.score.to_numpy(dtype=float)))}
            for (c, r, f, m, k), g in scores.groupby(["cohort", "repeat", "fold", "method", "k"], sort=True)]
    return pd.DataFrame(rows, columns=["cohort", "repeat", "fold", "method", "k", "auprc"])


def b7_summary(ev: pd.DataFrame, scores: pd.DataFrame | None, methods) -> pd.DataFrame:
    """Bảng B7 (PREREG Mục 8): log-loss, AUROC và AUPRC ngoài (trung bình, độ lệch chuẩn) theo bộ, k, phương pháp."""
    out = summary_table(ev, methods, B7_K)
    if scores is not None and len(scores):
        ap = auprc_per_fold(scores[scores.method.isin(methods) & scores.k.isin(B7_K)])
        agg = ap.groupby(["cohort", "k", "method"]).agg(auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std")).reset_index()
        out = out.merge(agg, on=["cohort", "k", "method"], how="left")
        out["positive_rate"] = out.cohort.map(scores.groupby("cohort").label.mean())
    return out.sort_values(["cohort", "k", "method"], kind="stable").reset_index(drop=True)


def c_choice(evs: dict) -> pd.DataFrame:
    """Tỉ lệ panel nhận từng giá trị C của LR-L2 theo pha B, kịch bản, phương pháp và k (B0, random-k không tính)."""
    rows = []
    for (phase_b, scenario), ev in sorted(evs.items()):
        part = ev[~ev.method.isin(["b0", "random_k"]) & ev["C"].notna()]
        for (method, k), g in part.groupby(["method", "k"], sort=True):
            n = len(g)
            for c in C.C_GRID:
                hit = int(np.isclose(g["C"].to_numpy(dtype=float), c).sum())
                rows.append({"phase_b": phase_b, "scenario": scenario, "method": method, "k": int(k), "C": float(c),
                             "n_panels": n, "n_with_C": hit, "rate": hit / n})
    return pd.DataFrame(rows, columns=["phase_b", "scenario", "method", "k", "C", "n_panels", "n_with_C", "rate"])


# ------------------------------------------------------------------------------------------------- ghi tệp
def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        frame.to_csv(f, index=False, float_format=FMT, lineterminator="\n")


def write_tex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cols = list(frame.columns)
    def cell(v):
        return (FMT % v if isinstance(v, (float, np.floating)) else str(v)).replace("_", r"\_")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}\label{{{label}}}", r"\begin{tabular}{" + "l" * len(cols) + "}",
             r"\hline", " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\hline"]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in frame.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def summary_table(ev: pd.DataFrame, methods, ks) -> pd.DataFrame:
    """Trung bình ± độ lệch chuẩn (kèm số fold) của log-loss và AUROC ngoài theo cohort, k, phương pháp."""
    part = ev[ev.method.isin(methods) & ev.k.isin(ks)]
    g = part.groupby(["cohort", "k", "method"])
    out = g.agg(n=("logloss", "size"), logloss_mean=("logloss", "mean"), logloss_sd=("logloss", "std"),
                auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std")).reset_index()
    return out.sort_values(["cohort", "k", "method"], kind="stable").reset_index(drop=True)


def _savefig(fig, path: Path) -> None:
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})


def fig_anytime(hist: pd.DataFrame, path: Path, methods) -> None:
    """Fig. 2: fitness tốt nhất theo N_eval, trung vị và IQR qua các lần chạy (cohort × fold) của mỗi k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(hist.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 2.8), squeeze=False)
    for ax, k in zip(axes[0], ks):
        for m in methods:
            part = hist[(hist.k == k) & (hist.method == m)]
            if part.empty:
                continue
            q = part.groupby("t").agg(x=("n_eval", "median"), lo=("best_fitness", lambda v: v.quantile(0.25)),
                                      mid=("best_fitness", "median"), hi=("best_fitness", lambda v: v.quantile(0.75)))
            ax.plot(q.x, q.mid, label=m, linewidth=1.0)
            ax.fill_between(q.x, q.lo, q.hi, alpha=0.15)
        n_runs = int(hist[hist.k == k].groupby(["cohort", "repeat", "fold"]).ngroups)
        ax.set_title(f"k = {k} (median and IQR, {n_runs} runs)", fontsize=8)
        ax.set_xlabel("fitness evaluations", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("best inner fitness", fontsize=8)
    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def fig_cd(ranks: pd.DataFrame, info: dict, path: Path) -> None:
    """Fig. 3: sơ đồ critical difference (Demšar, 2006) trên hạng log-loss ngoài trung bình."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = ranks.sort_values(["mean_rank", "method"], kind="stable")
    fig, ax = plt.subplots(figsize=(5.0, 0.35 * len(r) + 1.0))
    ax.hlines(0, 1, len(r), color="black", linewidth=0.8)
    for i, row in enumerate(r.itertuples(index=False)):
        ax.plot([row.mean_rank, row.mean_rank], [0, -0.3 - 0.25 * i], color="black", linewidth=0.6)
        ax.text(row.mean_rank, -0.35 - 0.25 * i, f"{row.method} ({row.mean_rank:.2f})", fontsize=7, ha="center", va="top")
    if np.isfinite(info["cd"]):
        ax.hlines(0.25, 1, 1 + info["cd"], color="black", linewidth=1.5)
        ax.text(1 + info["cd"] / 2, 0.3, f"CD = {info['cd']:.2f}", fontsize=7, ha="center", va="bottom")
    ax.set_xlim(0.8, len(r) + 0.2)
    ax.set_ylim(-0.6 - 0.25 * len(r), 0.6)
    ax.set_yticks([])
    ax.set_xlabel(f"mean rank of outer log-loss ({info['n_blocks']} blocks)", fontsize=8)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


CLAIM_COLUMNS = ["claim_id", "value", "file", "column", "filter", "phase_b"]


def claims_rows(tag: str, frame: pd.DataFrame, file: str, columns, keys, phase_b: str) -> list[dict]:
    """Sổ số liệu: mỗi giá trị một dòng (mã số liệu, giá trị, tệp, cột, bộ lọc, pha B). Bảng có cột phase_b thì lấy theo dòng."""
    rows = []
    for rec in frame.to_dict("records"):
        flt = ";".join(f"{c}={rec[c]}" for c in keys)
        for col in columns:
            v = rec[col]
            rows.append({"claim_id": f"{tag}:{flt}:{col}", "value": FMT % v if isinstance(v, (float, np.floating)) else str(v),
                         "file": file, "column": col, "filter": flt, "phase_b": rec.get("phase_b", phase_b)})
    return rows


# ------------------------------------------------------------------------------------------------- main
def build(results: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    inp, claims, done = Inputs(results), [], []

    def emit(name, frame, caption=None, tag=None, columns=(), keys=(), phase_b=PHASE_A):
        write_csv(frame, out / f"{name}.csv")
        if caption:
            write_tex(frame, out / f"{name}.tex", caption, f"tab:{name}")
        if tag:
            claims.extend(claims_rows(tag, frame, f"{name}.csv", columns, keys, phase_b))
        done.append(name)

    sel1, sel1b, sel2 = (inp.scenario(s, "selection_per_fold.csv") for s in ("T1", "T1b", "T2"))
    ev1, ev1b, ev2 = (inp.evaluation(s) for s in ("T1", "T1b", "T2"))
    ev1_bauc = inp.evaluation("T1", phase_b="B-AUC")
    hist1 = inp.scenario("T1", "history.csv")
    h3_table = None
    if sel1b is not None:
        emit("stats_h1_saturation", h1(sel1b), tag="H1", columns=["n_runs", "n_saturated", "rate", "decision"], keys=["cohort"])
    if sel1 is not None and (sel1.method == "de_rand1").any():
        emit("stats_h2_inner_fitness", h2(sel1), tag="H2", columns=["mean_diff", "p", "p_holm", "decision_k"], keys=["k", "cohort"])
    methods = ["mpqda_de", "mpqda_de" + M.BEST_SUFFIX, *BASELINE_ORDER]
    for ev, suffix, phase_b in ((ev1, "", "B-LL"), (ev1_bauc, "_bauc", "B-AUC")):
        if ev is None:
            continue
        tag_sfx = suffix.upper()
        caption_sfx = "" if not suffix else " Sensitivity analysis: phase B of the survey (C chosen by the inner AUROC)."
        emit(f"table3_main{suffix}", summary_table(ev, methods, (2, 5, 10, 50)),
             "Outer log-loss and AUROC (mean and SD over the outer folds; n = number of folds)." + caption_sfx, f"T3{tag_sfx}",
             ["n", "logloss_mean", "logloss_sd", "auroc_mean", "auroc_sd"], ["cohort", "k", "method"], phase_b)
        table = h3(ev)
        h3_table = table if not suffix else h3_table
        emit(f"stats_h3_noninferiority{suffix}", table, tag=f"H3{tag_sfx}", keys=["k"], phase_b=phase_b,
             columns=["comparator", "logloss_diff", "logloss_ci_high", "auroc_diff", "p_auroc_below_minus_rope", "h3_holds"])
        emit(f"stats_secondary_pairwise{suffix}", secondary(ev, (2, 5, 10, 50)))
        if not suffix:
            ranks, info = friedman(ev, (2, 5, 10, 50))
            emit("stats_friedman_ranks", ranks.assign(**info), tag="FRIEDMAN", columns=["mean_rank", "chi2", "p", "cd"],
                 keys=["method"], phase_b=phase_b)
            fig_cd(ranks, info, out / "fig3_critical_difference.pdf")
            done.append("fig3_critical_difference.pdf")
    evs = {("B-LL", "T1"): ev1, ("B-AUC", "T1"): ev1_bauc}
    for phase_b, sub in PHASE_B_DIRS.items():                 # T4 (selector không wrapper) nếu đã có
        if (results / sub / "T4").is_dir():
            evs[(phase_b, "T4")] = inp.evaluation("T4", phase_b=phase_b)
    evs = {key: frame for key, frame in evs.items() if frame is not None}
    if evs:
        emit("table_s_c_choice", c_choice(evs), "Share of panels receiving each value of C (LR-L2) under the two phase B rules.",
             "CCHOICE", ["n_panels", "n_with_C", "rate"], ["phase_b", "scenario", "method", "k", "C"])
    if hist1 is not None:
        fig_anytime(hist1, out / "fig2_anytime.pdf", ["mpqda_de", *BASELINE_ORDER])
        done.append("fig2_anytime.pdf")
    if all(x is not None for x in (sel1, sel2, ev1, ev2)):
        t = h5(sel1, sel2, ev1, ev2)
        emit("table4_ablation", t, "Ablation: paired differences against the full MPQDA-DE run (one Holm family of 39 tests).",
             "H5", ["mean_diff", "ci_low", "ci_high", "p", "p_holm", "significant"], ["variant", "k", "question"])
    b7 = [d for d in M.B7_INCLUDED if (results / PHASE_B_DIRS["B-LL"] / "T1" / d).is_dir()]
    if b7:                                                    # B7: bảng riêng, không gộp với GPL570 (PREREG Mục 8)
        ev_b7 = inp.evaluation("T1", cohorts=b7)
        sc_b7 = inp.scenario(f"{PHASE_B_DIRS['B-LL']}/T1", "e1_scores.csv", b7, encoding="utf-8")
        if ev_b7 is not None:
            emit("table_b7_main", b7_summary(ev_b7, sc_b7, methods),
                 "B7 (EGFR mutation in LUAD): outer log-loss, AUROC and AUPRC (mean and SD over the outer folds; n = number of "
                 "folds); separate from the GPL570 cohorts.", "B7",
                 ["n", "logloss_mean", "logloss_sd", "auroc_mean", "auroc_sd", "auprc_mean", "auprc_sd"],
                 ["cohort", "k", "method"], "B-LL")
            emit("stats_b7_secondary", secondary(ev_b7, B7_K), tag="B7_SEC", phase_b="B-LL",
                 columns=["logloss_diff", "ci_low", "ci_high", "p_nb_holm", "p_wilcoxon_holm", "auroc_p_left", "auroc_p_rope",
                          "auroc_p_right"], keys=["cohort", "k", "baseline"])
    phi = seed_stability(inp)
    if phi is not None and len(phi):
        table5 = phi.groupby(["k", "method"]).phi.agg(n="size", phi_mean="mean", phi_sd="std").reset_index()
        emit("table5_seed_stability", table5, "Seed stability on GSE37745 (Nogueira over five seeds; mean and SD over the outer folds).",
             "T5", ["n", "phi_mean", "phi_sd"], ["k", "method"])
        if {"mpqda_de", "mpqda_de" + M.BEST_SUFFIX, "ga", "bpso"} <= set(phi.method):
            emit("stats_h4_stability", h4(phi, h3_table), tag="H4", columns=["mean_diff_minus_margin", "p", "p_holm", "h4_holds"],
                 keys=["k", "test"], phase_b="phase A; h4_holds uses H3 under B-LL")
    write_csv(pd.DataFrame(sorted(inp.used.items()), columns=["file", "sha256"]), out / "inputs_sha256.csv")
    write_csv(pd.DataFrame(claims, columns=CLAIM_COLUMNS), out / "claims.csv")
    (out / "missing_inputs.txt").write_text("".join(m + "\n" for m in sorted(set(inp.missing))), encoding="utf-8", newline="\n")
    return {"written": done, "missing": sorted(set(inp.missing))}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="thư mục chứa T1, T1b, T2, T3_s1..4, T5LL, T5 (đã giải nén)")
    ap.add_argument("--out", type=Path, default=Path("tables"))
    args = ap.parse_args(argv)
    res = build(args.results, args.out)
    print("đã ghi: " + ", ".join(res["written"]))
    if res["missing"]:
        print(f"thiếu {len(res['missing'])} tệp đầu vào (xem {args.out / 'missing_inputs.txt'}); bảng tương ứng bị bỏ qua")
    return 0


if __name__ == "__main__":
    sys.exit(main())
