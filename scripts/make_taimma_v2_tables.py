"""Tables and tests of the confirmatory phase of CIRK-DE (PREREG-v2, Sections 6–7). make_taimma_tables.py is not edited: only its
statistical functions and file readers are reused, so the tables of v1 do not change.

    python scripts/make_taimma_v2_tables.py --results <folder holding T1, T1v2, T2v2, T3_s*, T3v2_s*, T5LL, T5, T5C1, T5LLW> --out tables_v2

Inputs: results of v1 (S1–S3, tag taimma-2027-v1-results), results of S0v2 (job v1b: T5C1/T1, T5LLW/T1) and C1–C3 of v2,
extracted into one folder.
  table_v2_main         outer log-loss and AUROC (B-LL) by cohort, k, method: the eight methods of v1, v2 (two outputs), DE-G
  stats_hv1_search      HV1: best_fitness(v2) − best_fitness(m), m ∈ {v1, GA}, k ∈ {5, 10}; one-sided Wilcoxon, Holm over 4;
                        95 % bootstrap CI stratified by cohort; share of folds where v2 is better
  stats_hv2_generalization[_bauc|_bc1|_bllw]  HV2: outer log-loss and AUROC of v2 against BPSO and v1; pooled Nadeau–Bengio, Holm
                        over 4 within each phase B criterion; _bc1, _bllw need job v1b (without it only table_v2_main_bc1|_bllw is written)
  stats_hv5_b7          HV5: outer AUROC (B-LL) of v1 − v2 on gse31210_egfr, k ∈ {5, 10}; one-sided Wilcoxon over 15 folds,
                        Holm over 2; with best_fitness v2 − v1 on this dataset, log-loss v2 − v1 (single-dataset Nadeau–Bengio CI) and
                        AUROC GA − v1, descriptive only
  diagnostics_v2_c_choice  share of panels with C at the edge of the grid by phase B criterion, method, k
  Every HV table has the columns holm_family and family_size: Holm runs over the whole family, not per cohort or k.
  stats_hv3_stability   HV3: Φ(v2 consensus) − Φ(m), m ∈ {v1 consensus, GA}, GSE37745, k ∈ {5, 10}; one-sided Wilcoxon, Holm 4
  table_v2_ablation     HV4: best_fitness(v2) − best_fitness(ablation), k ∈ {2, 5, 10}; one-sided Wilcoxon, Holm over all tests;
                        supported when significant for at least two of the three k; outer log-loss of the ablations descriptive only
  table_b7_v2           B7: outer log-loss, AUROC, AUPRC (B-LL) of every method, with v2 and DE-G
  diagnostics_v2_behaviour  leader_overlap, success_rate, n_new_genes, diversity per generation, v1 against v2 (C1)
  stats_v2_friedman_ranks   Friedman, critical difference over ten methods (q = 3.164), descriptive only
  claims_v2.csv, inputs_sha256_v2.csv
Two runs give byte-identical files (fixed bootstrap seed).
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
HV1_REF = (V1, "ga")                     # GA: highest mean best_fitness in T1 of v1 at k = 5 and 10 (PREREG-v2, Section 7)
HV2_REF = ("bpso", V1)
ABLATIONS = ("mpqda_v2_no_gap", "mpqda_v2_no_new", "mpqda_v2_no_tour", "mpqda_v2_no_adapt")
NEMENYI_Q05_10 = 3.164
BOOT_REPS, BOOT_SEED = 2000, 20261118
PHASE_B = {"B-LL": "T5LL", "B-AUC": "T5", "B-C1": "T5C1", "B-LL-wide": "T5LLW"}
SFX = {"B-LL": "", "B-AUC": "_bauc", "B-C1": "_bc1", "B-LL-wide": "_bllw"}
GRID_EDGES = {"B-LL": (0.01, 10.0), "B-AUC": (0.01, 10.0), "B-LL-wide": (1e-4, 1000.0)}   # B-C1: C = 1, no grid
HV5_COHORT = "gse31210_egfr"             # B7 dataset independent of the development data; tcga_luad_egfr shares its source with TCGA DEV
FAMILY_SIZE = {"HV1": 4, "HV2": 4, "HV3": 4, "HV4": 9, "HV5": 2}   # PREREG-v2; HV4 is 12 if the chosen configuration is M2


def bootstrap_ci(diffs_by_cohort, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95 % CI of the mean difference (weighted by number of folds) by bootstrap resampling of folds within each cohort."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(d, dtype=float) for d in diffs_by_cohort if len(d)]
    n = sum(len(d) for d in arrays)
    means = np.empty(reps)
    for b in range(reps):
        means[b] = sum(rng.choice(d, size=len(d), replace=True).sum() for d in arrays) / n
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def with_holm(out: pd.DataFrame, p_col: str, family: str) -> pd.DataFrame:
    """Holm over every row of the table (one family), writing the family name and size next to p_holm."""
    out["p_holm"] = holm(out[p_col].to_numpy())
    out["holm_family"] = family
    out["family_size"] = len(out)
    return out


def hv1(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in T.PRIMARY_K:
        for ref in HV1_REF:
            by_c = [T.paired(sel, V2, ref, "best_fitness", k, c) for c in COHORTS]
            d = np.concatenate(by_c)
            lo, hi = bootstrap_ci(by_c)
            rows.append({"k": k, "reference": ref, "n": len(d), "mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi,
                         "share_v2_better": float(np.mean(d > 0)), "p": T.wilcoxon_p(d, "greater")})
    out = with_holm(pd.DataFrame(rows), "p", "HV1")
    out["significant"] = out.p_holm < T.ALPHA
    return out


def hv2(ev: pd.DataFrame, family: str = "HV2") -> pd.DataFrame:
    rows = []
    for k in T.PRIMARY_K:
        for ref in HV2_REF:
            d_ll = [T.paired(ev, V2, ref, "logloss", k, c) for c in COHORTS]
            ll = T.pooled_nb(d_ll)
            au = T.pooled_nb([T.paired(ev, V2, ref, "auroc", k, c) for c in COHORTS])
            p_two = 1.0 if ll["se"] == 0 else float(2 * st.t.sf(abs(ll["mean"]) / ll["se"], ll["df"]))
            p_worse = (float(au["mean"] < -T.ROPE_AUROC) if au["se"] == 0
                       else float(st.t.cdf((-T.ROPE_AUROC - au["mean"]) / au["se"], au["df"])))
            rows.append({"k": k, "reference": ref, "n": int(sum(len(d) for d in d_ll)), "logloss_diff": ll["mean"],
                         "logloss_se": ll["se"], "logloss_ci_low": ll["ci_low"],
                         "logloss_ci_high": ll["ci_high"], "logloss_half_width": (ll["ci_high"] - ll["ci_low"]) / 2,
                         "p_two_sided": p_two, "auroc_diff": au["mean"], "auroc_se": au["se"], "p_auroc_below_minus_rope": p_worse})
    out = with_holm(pd.DataFrame(rows), "p_two_sided", family)
    out["different"] = out.p_holm < T.ALPHA
    return out


def stability(inputs: T.Inputs, n_genes: int = T.N_GENES_DEFAULT) -> pd.DataFrame | None:
    """Φ per fold of GSE37745 for v1 (T1, T3_s1..4) and v2 (T1v2, T3v2_s1..4), k ∈ {5, 10}."""
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
    out = with_holm(pd.DataFrame(rows), "p", "HV3")
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
    out = with_holm(pd.DataFrame(rows), "p", "HV4")
    out["significant"] = out.p_holm < T.ALPHA
    n_sig = out.groupby("ablation").significant.sum()
    out["component_credited"] = out.ablation.map(lambda a: bool(n_sig[a] >= 2))
    return out


def hv5(ev: pd.DataFrame, sel: pd.DataFrame | None = None, cohort: str = HV5_COHORT) -> pd.DataFrame:
    """HV5: outer AUROC (B-LL) of v2 lower than v1 on one B7 dataset. d = AUROC(v1) − AUROC(v2) per (repeat, fold);
    one-sided Wilcoxon (d > 0), Holm over k ∈ {5, 10}. best_fitness v2 − v1 on the same dataset (when sel is given), log-loss v2 − v1
    (single-dataset Nadeau–Bengio CI) and AUROC GA − v1 are descriptive only, outside the Holm family."""
    rows = []
    for k in T.PRIMARY_K:
        d = T.paired(ev, V1, V2, "auroc", k, cohort)
        if not len(d):
            raise ValueError(f"HV5: no v1, v2 pair on {cohort}, k = {k}")
        ll = T.pooled_nb([T.paired(ev, V2, V1, "logloss", k, cohort)])
        ga = T.paired(ev, "ga", V1, "auroc", k, cohort)
        rows.append({"cohort": cohort, "k": k, "n": len(d), "auroc_v1": float(ev[(ev.cohort == cohort) & (ev.k == k) & (ev.method == V1)].auroc.mean()),
                     "auroc_v2": float(ev[(ev.cohort == cohort) & (ev.k == k) & (ev.method == V2)].auroc.mean()),
                     "auroc_v1_minus_v2": float(d.mean()), "share_v2_lower": float(np.mean(d > 0)), "p": T.wilcoxon_p(d, "greater"),
                     "logloss_v2_minus_v1": ll["mean"], "logloss_ci_low": ll["ci_low"], "logloss_ci_high": ll["ci_high"],
                     "auroc_ga_minus_v1": float(ga.mean()) if len(ga) else np.nan})
        inner = T.paired(sel, V2, V1, "best_fitness", k, cohort) if sel is not None else np.array([])
        rows[-1].update({"inner_fitness_v2_minus_v1": float(inner.mean()) if len(inner) else np.nan,
                         "share_v2_inner_better": float(np.mean(inner > 0)) if len(inner) else np.nan})
    out = with_holm(pd.DataFrame(rows), "p", "HV5")
    out["significant"] = out.p_holm < T.ALPHA
    return out


def c_choice(evs: dict) -> pd.DataFrame:
    """Distribution of the chosen C (column C of classification_per_fold) by phase B criterion, method, k: share at the lower and upper edge
    of the grid, median. B-C1 has no grid and is not in the table."""
    rows = []
    for pb, ev in evs.items():
        if pb not in GRID_EDGES:
            continue
        lo, hi = GRID_EDGES[pb]
        part = ev[ev.cohort.isin(COHORTS) & ev.C.notna()]
        for (method, k), g in part.groupby(["method", "k"], sort=True):
            c = g.C.astype(float)
            rows.append({"phase_b": pb, "method": method, "k": int(k), "n": len(c), "share_at_lower": float(np.isclose(c, lo).mean()),
                         "share_at_upper": float(np.isclose(c, hi).mean()), "median_C": float(c.median())})
    return pd.DataFrame(rows, columns=["phase_b", "method", "k", "n", "share_at_lower", "share_at_upper", "median_C"])


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
    q = NEMENYI_Q05_10 if m == 10 else T.NEMENYI_Q05.get(m, np.nan)     # q(0.05) = 3.164 for ten methods (PREREG-v2)
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
    all_methods = [*[n for m in V1_METHODS for n in M.output_names(m)], V2, V2 + M.BEST_SUFFIX, DEG]
    for pb, sub in PHASE_B.items():
        v1_ev = inp.scenario(f"{sub}/T1", "classification_per_fold.csv", encoding="utf-8-sig")
        v2_ev = inp.scenario(f"{sub}/T1v2", "classification_per_fold.csv", encoding="utf-8-sig")
        if v2_ev is None or (pb in ("B-LL", "B-AUC") and v1_ev is None):
            continue
        ev = pd.concat([f for f in (v1_ev, v2_ev) if f is not None], ignore_index=True)
        ev = ev[ev.model == "lr_l2"].reset_index(drop=True)
        evs[pb] = ev
        sfx = SFX[pb]
        if v1_ev is None:                         # B-C1, B-LL-wide before job v1b has run: describe only v2 and DE-G
            emit(f"table_v2_main{sfx}", T.summary_table(ev, [V2, V2 + M.BEST_SUFFIX, DEG], (2, 5, 10, 50)), tag=f"V2T{sfx.upper()}",
                 columns=["n", "logloss_mean", "auroc_mean"], keys=["cohort", "k", "method"], phase_b=pb)
            continue
        if pb in ("B-C1", "B-LL-wide"):
            emit(f"table_v2_main{sfx}", T.summary_table(ev, all_methods, (2, 5, 10, 50)), tag=f"V2T{sfx.upper()}",
                 columns=["n", "logloss_mean", "auroc_mean"], keys=["cohort", "k", "method"], phase_b=pb)
        if pb == "B-LL":
            methods = all_methods
            emit("table_v2_main", T.summary_table(ev, methods, (2, 5, 10, 50)),
                 "Outer log-loss and AUROC (B-LL), v1, the seven baselines, v2 and DE-G (mean and SD over the outer folds).", "V2T",
                 ["n", "logloss_mean", "logloss_sd", "auroc_mean", "auroc_sd"], ["cohort", "k", "method"], pb)
            ranks, info = friedman10(ev, (2, 5, 10, 50))
            emit("stats_v2_friedman_ranks", ranks.assign(**info), tag="V2FRIEDMAN", columns=["mean_rank", "chi2", "p", "cd"],
                 keys=["method"], phase_b=pb)
        emit(f"stats_hv2_generalization{sfx}", hv2(ev, f"HV2{sfx.upper()}"), tag=f"HV2{sfx.upper()}", keys=["k", "reference"],
             phase_b=pb, columns=["logloss_diff", "logloss_ci_low", "logloss_ci_high", "logloss_half_width", "p_holm", "different",
                                  "auroc_diff", "p_auroc_below_minus_rope"])
    if evs:
        emit("diagnostics_v2_c_choice", c_choice(evs))
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
        if HV5_COHORT in b7:
            sel_b7 = [inp.scenario(s, "selection_per_fold.csv", [HV5_COHORT]) for s in ("T1", "T1v2")]
            sel_b7 = pd.concat(sel_b7, ignore_index=True) if all(f is not None for f in sel_b7) else None
            emit("stats_hv5_b7", hv5(ev_b7[ev_b7.model == "lr_l2"].reset_index(drop=True), sel_b7),
                 "HV5: outer AUROC (B-LL) of v1 minus v2 on GSE31210 (EGFR), one-sided Wilcoxon over 15 folds, Holm over k.",
                 "HV5", ["auroc_v1", "auroc_v2", "auroc_v1_minus_v2", "share_v2_lower", "p", "p_holm", "significant",
                         "inner_fitness_v2_minus_v1", "share_v2_inner_better", "logloss_v2_minus_v1", "logloss_ci_low",
                         "logloss_ci_high", "auroc_ga_minus_v1"], ["cohort", "k"], "B-LL")
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
    print("written: " + ", ".join(res["written"]))
    if res["missing"]:
        print(f"missing {len(res['missing'])} input files (see {args.out / 'missing_inputs_v2.txt'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
