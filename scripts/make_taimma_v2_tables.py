"""Bảng và kiểm định của pha xác nhận MPQDA-DE v2 (PREREG-v2 Mục 6–7). Không sửa make_taimma_tables.py (TQĐ19): chỉ dùng
lại các hàm thống kê và đọc tệp của nó, nên bảng của v1 không đổi.

    python scripts/make_taimma_v2_tables.py --results <thư mục chứa T1, T1v2, T2v2, T3_s*, T3v2_s*, T5LL, T5, T5C1> --out tables_v2

Đầu vào: kết quả v1 (S1–S3, tag taimma-2027-v1-results) và kết quả C1–C3 của v2, giải nén cùng một thư mục.
  table_v2_main         log-loss, AUROC ngoài (B-LL) theo cohort, k, phương pháp: tám phương pháp của v1, v2 (hai đầu ra), DE-G
  stats_hv1_search      HV1: best_fitness(v2) − best_fitness(m), m ∈ {v1, GA}, k ∈ {5, 10}; Wilcoxon một phía, Holm trên 4;
                        CI bootstrap 95 % phân tầng theo cohort; tỉ lệ fold v2 tốt hơn
  stats_hv2_generalization[_bauc|_bc1]  HV2: log-loss, AUROC ngoài của v2 so với BPSO và v1; Nadeau–Bengio gộp, Holm trên 4
  stats_hv3_stability   HV3: Φ(v2 đồng thuận) − Φ(m), m ∈ {v1 đồng thuận, GA}, GSE37745, k ∈ {5, 10}; Wilcoxon một phía, Holm 4
  table_v2_ablation     HV4: best_fitness(v2) − best_fitness(ablation), k ∈ {2, 5, 10}; Wilcoxon một phía, Holm trên mọi phép thử;
                        ghi nhận khi đạt ở ít nhất hai trong ba k; log-loss ngoài của ablation chỉ mô tả
  table_b7_v2           B7: log-loss, AUROC, AUPRC ngoài (B-LL) của mọi phương pháp, kèm v2 và DE-G
  diagnostics_v2_behaviour  leader_overlap, success_rate, n_new_genes, diversity theo thế hệ, v1 so với v2 (C1)
  stats_v2_friedman_ranks   Friedman, critical difference trên mười phương pháp (q = 3,164), chỉ mô tả
  claims_v2.csv, inputs_sha256_v2.csv
Chạy hai lần cho tệp trùng từng byte (seed bootstrap cố định).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from survey_v2.stability import nogueira  # noqa: E402
from survey_v2.stats import holm  # noqa: E402

from mpqda import config as M  # noqa: E402
from scripts import make_taimma_tables as T  # noqa: E402

COHORTS = T.COHORTS
V1, V2, DEG = "mpqda_de", "mpqda_v2", "de_g"
V1_METHODS = [V1, *M.TIER1, *M.TIER2]
HV1_REF = (V1, "ga")                     # GA: best_fitness trung bình cao nhất trong T1 của v1 ở k = 5 và 10 (PREREG-v2 Mục 7)
HV2_REF = ("bpso", V1)
ABLATIONS = ("mpqda_v2_no_gap", "mpqda_v2_no_new", "mpqda_v2_no_tour", "mpqda_v2_no_adapt")
NEMENYI_Q05_10 = 3.164
BOOT_REPS, BOOT_SEED = 2000, 20261118
PHASE_B = {"B-LL": "T5LL", "B-AUC": "T5", "B-C1": "T5C1"}


def bootstrap_ci(diffs_by_cohort, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> tuple[float, float]:
    """CI 95 % của hiệu trung bình (trọng số theo số fold) bằng bootstrap lấy lại fold trong từng cohort."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(d, dtype=float) for d in diffs_by_cohort if len(d)]
    n = sum(len(d) for d in arrays)
    means = np.empty(reps)
    for b in range(reps):
        means[b] = sum(rng.choice(d, size=len(d), replace=True).sum() for d in arrays) / n
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def hv1(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in T.PRIMARY_K:
        for ref in HV1_REF:
            by_c = [T.paired(sel, V2, ref, "best_fitness", k, c) for c in COHORTS]
            d = np.concatenate(by_c)
            lo, hi = bootstrap_ci(by_c)
            rows.append({"k": k, "reference": ref, "n": len(d), "mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi,
                         "share_v2_better": float(np.mean(d > 0)), "p": T.wilcoxon_p(d, "greater")})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["significant"] = out.p_holm < T.ALPHA
    return out


def hv2(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in T.PRIMARY_K:
        for ref in HV2_REF:
            ll = T.pooled_nb([T.paired(ev, V2, ref, "logloss", k, c) for c in COHORTS])
            au = T.pooled_nb([T.paired(ev, V2, ref, "auroc", k, c) for c in COHORTS])
            p_two = 1.0 if ll["se"] == 0 else float(2 * st.t.sf(abs(ll["mean"]) / ll["se"], ll["df"]))
            p_worse = (float(au["mean"] < -T.ROPE_AUROC) if au["se"] == 0
                       else float(st.t.cdf((-T.ROPE_AUROC - au["mean"]) / au["se"], au["df"])))
            rows.append({"k": k, "reference": ref, "logloss_diff": ll["mean"], "logloss_se": ll["se"], "logloss_ci_low": ll["ci_low"],
                         "logloss_ci_high": ll["ci_high"], "logloss_half_width": (ll["ci_high"] - ll["ci_low"]) / 2,
                         "p_two_sided": p_two, "auroc_diff": au["mean"], "auroc_se": au["se"], "p_auroc_below_minus_rope": p_worse})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_two_sided.to_numpy())
    out["different"] = out.p_holm < T.ALPHA
    return out


def stability(inputs: T.Inputs, n_genes: int = T.N_GENES_DEFAULT) -> pd.DataFrame | None:
    """Φ theo fold của GSE37745 cho v1 (T1, T3_s1..4) và v2 (T1v2, T3v2_s1..4), k ∈ {5, 10}."""
    frames = []
    for seed0, seeds in (("T1", "T3_s"), ("T1v2", "T3v2_s")):
        parts = [inputs.read(f"{seed0}/gse37745/selected_genes.csv")] + [inputs.read(f"{seeds}{s}/gse37745/selected_genes.csv")
                                                                          for s in (1, 2, 3, 4)]
        if any(f is None for f in parts):
            return None
        frames.append(pd.concat([f.assign(seed_index=s) for s, f in enumerate(parts)], ignore_index=True))
    genes = pd.concat(frames, ignore_index=True)
    genes = genes[genes.k.isin(T.PRIMARY_K) & genes.method.isin([V1, "ga", V2])]
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


def hv3(phi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in T.PRIMARY_K:
        for ref in (V1, "ga"):
            pa = phi[(phi.k == k) & (phi.method == V2)].set_index(T.FOLD).phi
            pb = phi[(phi.k == k) & (phi.method == ref)].set_index(T.FOLD).phi
            idx = pa.index.intersection(pb.index).sort_values()
            d = (pa.loc[idx] - pb.loc[idx]).to_numpy(dtype=float)
            rows.append({"k": k, "reference": ref, "n": len(d), "phi_v2": float(pa.loc[idx].mean()), "phi_reference": float(pb.loc[idx].mean()),
                         "mean_diff": float(d.mean()), "p": T.wilcoxon_p(d, "greater")})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["significant"] = out.p_holm < T.ALPHA
    return out


def hv4(sel: pd.DataFrame, ev: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    present = [a for a in ABLATIONS if (sel.method == a).any()]
    for a in present:
        for k in (2, 5, 10):
            d = np.concatenate([T.paired(sel, V2, a, "best_fitness", k, c) for c in COHORTS])
            row = {"ablation": a, "k": k, "n": len(d), "mean_diff": float(d.mean()), "share_v2_better": float(np.mean(d > 0)),
                   "p": T.wilcoxon_p(d, "greater")}
            if ev is not None:
                ll = T.pooled_nb([T.paired(ev, a, V2, "logloss", k, c) for c in COHORTS])
                row.update({"outer_logloss_ablation_minus_v2": ll["mean"], "outer_ci_low": ll["ci_low"], "outer_ci_high": ll["ci_high"]})
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["significant"] = out.p_holm < T.ALPHA
    n_sig = out.groupby("ablation").significant.sum()
    out["component_credited"] = out.ablation.map(lambda a: bool(n_sig[a] >= 2))
    return out


def behaviour(h1: pd.DataFrame, h2: pd.DataFrame) -> pd.DataFrame:
    cols = ["leader_overlap", "success_rate", "n_new_genes", "diversity", "exploration_rate"]
    h = pd.concat([h1[h1.method == V1], h2[h2.method == V2]], ignore_index=True)
    out = h.groupby(["method", "k", "t"])[cols].mean().reset_index()
    return out.sort_values(["method", "k", "t"], kind="stable").reset_index(drop=True)


def friedman10(ev: pd.DataFrame, ks) -> tuple[pd.DataFrame, dict]:
    methods = [m for m in (*V1_METHODS, V2, DEG) if (ev.method == m).any()]
    part = ev[ev.method.isin(methods) & ev.cohort.isin(COHORTS) & ev.k.isin(ks)]
    table = part.groupby(["cohort", "k", "method"]).logloss.mean().unstack("method")[methods].dropna()
    ranks = table.rank(axis=1, method="average")
    n, m = table.shape
    stat = st.friedmanchisquare(*[table[c].to_numpy() for c in methods]) if n >= 2 and m >= 3 else None
    q = NEMENYI_Q05_10 if m == 10 else T.NEMENYI_Q05.get(m, np.nan)     # q(0,05) = 3,164 cho mười phương pháp (PREREG-v2)
    info = {"n_blocks": n, "n_methods": m, "chi2": float(stat.statistic) if stat else np.nan,
            "p": float(stat.pvalue) if stat else np.nan, "cd": float(q * np.sqrt(m * (m + 1) / (6.0 * n))) if n else np.nan}
    return ranks.mean().rename("mean_rank").reset_index(), info


def build(results: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    inp, claims, done = T.Inputs(results), [], []

    def emit(name, frame, caption=None, tag=None, columns=(), keys=(), phase_b=T.PHASE_A):
        T.write_csv(frame, out / f"{name}.csv")
        if caption:
            T.write_tex(frame, out / f"{name}.tex", caption, f"tab:{name}")
        if tag:
            claims.extend(T.claims_rows(tag, frame, f"{name}.csv", columns, keys, phase_b))
        done.append(name)

    sel1, sel2 = inp.scenario("T1", "selection_per_fold.csv"), inp.scenario("T1v2", "selection_per_fold.csv")
    if sel1 is not None and sel2 is not None:
        sel = pd.concat([sel1, sel2], ignore_index=True)
        emit("stats_hv1_search", hv1(sel), tag="HV1", keys=["k", "reference"],
             columns=["mean_diff", "ci_low", "ci_high", "share_v2_better", "p", "p_holm", "significant"])
    evs: dict[str, pd.DataFrame] = {}
    for pb, sub in PHASE_B.items():
        v1_ev = inp.scenario(f"{sub}/T1", "classification_per_fold.csv", encoding="utf-8-sig") if sub != "T5C1" else None
        v2_ev = inp.scenario(f"{sub}/T1v2", "classification_per_fold.csv", encoding="utf-8-sig")
        if v2_ev is None or (pb != "B-C1" and v1_ev is None):
            continue
        ev = pd.concat([f for f in (v1_ev, v2_ev) if f is not None], ignore_index=True)
        ev = ev[ev.model == "lr_l2"].reset_index(drop=True)
        evs[pb] = ev
        if pb == "B-C1":                          # v1 và baseline không có B-C1 (TQĐ24: chỉ thêm cho v2); chỉ mô tả v2, DE-G
            emit("table_v2_main_bc1", T.summary_table(ev, [V2, V2 + M.BEST_SUFFIX, DEG], (2, 5, 10, 50)), tag="V2T_BC1",
                 columns=["n", "logloss_mean", "auroc_mean"], keys=["cohort", "k", "method"], phase_b=pb)
            continue
        sfx = "" if pb == "B-LL" else "_bauc"
        if pb == "B-LL":
            methods = [*[n for m in V1_METHODS for n in M.output_names(m)], V2, V2 + M.BEST_SUFFIX, DEG]
            emit("table_v2_main", T.summary_table(ev, methods, (2, 5, 10, 50)),
                 "Outer log-loss and AUROC (B-LL), v1, the seven baselines, v2 and DE-G (mean and SD over the outer folds).", "V2T",
                 ["n", "logloss_mean", "logloss_sd", "auroc_mean", "auroc_sd"], ["cohort", "k", "method"], pb)
            ranks, info = friedman10(ev, (2, 5, 10, 50))
            emit("stats_v2_friedman_ranks", ranks.assign(**info), tag="V2FRIEDMAN", columns=["mean_rank", "chi2", "p", "cd"],
                 keys=["method"], phase_b=pb)
        emit(f"stats_hv2_generalization{sfx}", hv2(ev), tag=f"HV2{sfx.upper()}", keys=["k", "reference"], phase_b=pb,
             columns=["logloss_diff", "logloss_ci_low", "logloss_ci_high", "logloss_half_width", "p_holm", "different", "auroc_diff",
                      "p_auroc_below_minus_rope"])
    phi = stability(inp)
    if phi is not None and len(phi):
        emit("stats_hv3_stability", hv3(phi), tag="HV3", keys=["k", "reference"],
             columns=["phi_v2", "phi_reference", "mean_diff", "p", "p_holm", "significant"])
    sel_abl = inp.scenario("T2v2", "selection_per_fold.csv")
    if sel2 is not None and sel_abl is not None:
        ev_abl = inp.scenario("T5LL/T2v2", "classification_per_fold.csv", encoding="utf-8-sig")
        ev_all = None
        if ev_abl is not None and "B-LL" in evs:
            ev_all = pd.concat([evs["B-LL"], ev_abl[ev_abl.model == "lr_l2"]], ignore_index=True)
        emit("table_v2_ablation", hv4(pd.concat([sel2, sel_abl], ignore_index=True), ev_all),
             "Ablation of v2: paired differences in best inner fitness (one Holm family).", "HV4",
             ["mean_diff", "share_v2_better", "p", "p_holm", "significant", "component_credited"], ["ablation", "k"])
    h1, h2 = inp.scenario("T1", "history.csv"), inp.scenario("T1v2", "history.csv")
    if h1 is not None and h2 is not None:
        emit("diagnostics_v2_behaviour", behaviour(h1, h2))
    b7 = [d for d in M.B7_INCLUDED if (results / "T5LL" / "T1v2" / d).is_dir() and (results / "T5LL" / "T1" / d).is_dir()]
    if b7:
        ev_b7 = pd.concat([f for f in (inp.evaluation("T1", cohorts=b7), inp.scenario("T5LL/T1v2", "classification_per_fold.csv", b7,
                                                                                        encoding="utf-8-sig")) if f is not None])
        sc = [inp.scenario(f"T5LL/{s}", "e1_scores.csv", b7, encoding="utf-8") for s in ("T1", "T1v2")]
        sc = pd.concat([f for f in sc if f is not None], ignore_index=True) if any(f is not None for f in sc) else None
        methods = [*[n for m in V1_METHODS for n in M.output_names(m)], V2, V2 + M.BEST_SUFFIX, DEG]
        emit("table_b7_v2", T.b7_summary(ev_b7[ev_b7.model == "lr_l2"], sc, methods),
             "B7 with v2: outer log-loss, AUROC and AUPRC (B-LL); separate from the GPL570 cohorts.", "B7V2",
             ["n", "logloss_mean", "auroc_mean", "auprc_mean"], ["cohort", "k", "method"], "B-LL")
    T.write_csv(pd.DataFrame(sorted(inp.used.items()), columns=["file", "sha256"]), out / "inputs_sha256_v2.csv")
    T.write_csv(pd.DataFrame(claims, columns=T.CLAIM_COLUMNS), out / "claims_v2.csv")
    (out / "missing_inputs_v2.txt").write_text("".join(m + "\n" for m in sorted(set(inp.missing))), encoding="utf-8", newline="\n")
    return {"written": done, "missing": sorted(set(inp.missing))}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tables_v2"))
    args = ap.parse_args(argv)
    res = build(args.results, args.out)
    print("đã ghi: " + ", ".join(res["written"]))
    if res["missing"]:
        print(f"thiếu {len(res['missing'])} tệp đầu vào (xem {args.out / 'missing_inputs_v2.txt'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
