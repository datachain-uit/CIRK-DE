"""Re-analysis of existing results and analysis of amendment A2 (PREREG-v2, Section 12; docs/taimma/PREREG-v2_A2.md).

No new search is run. Reads the result folder of the paper (same layout as make_taimma_v2_tables.py: T1, T1v2, T2v2, T5LL/...,
and T4v2 once session C4 has run) and writes to --out:

* corrected_inner.csv (not registered, sensitivity): every difference of best inner fitness stated in the paper (HV1, HV4, E2 and the decomposition
  v2 − v1 = (v2 − DE-G) + (DE-G − DE/rand/1/bin) + (DE/rand/1/bin − v1)); each row has the mean, the mean difference per cohort,
  the number of cohorts with a positive difference, a stratified bootstrap CI (as in HV1, same seed), a Nadeau–Bengio CI pooled over the three cohorts (formula of HV2:
  ρ = 1/4, J = 15, df = 14), the one-sided p of the corrected t, and the one-sided Wilcoxon p (as registered) for comparison.
* corrected_outer.csv: HV2 under B-LL, differences of outer log-loss and AUROC (v2 minus BPSO, v2 minus v1) with the pooled Nadeau–Bengio CI.
* corrected_hv5.csv: HV5 (gse31210_egfr), single-dataset Nadeau–Bengio CI for AUROC v1 − v2 (descriptive).
* runtime.csv: mean gene-selection time per run (column selection_time_s) by method and k, three GPL570 cohorts.
* gate_r50.csv and hv6.csv, hv6_page.csv (only when T4v2 exists): reproduction gate R50 and HV6 of amendment A2.
* claims_corrected.csv and inputs_sha256_corrected.csv: register of numbers and SHA-256 of the inputs.

    python scripts/make_taimma_corrected.py --results <$KLTN_DATA_ROOT/results/survey_v2/taimma> --out tables_v2/corrected

Reading rules written before computing: docs/taimma/PREREG-v2_A2.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from survey_v2.stats import holm  # noqa: E402

from scripts import make_taimma_tables as T  # noqa: E402
from scripts.make_taimma_v2_tables import HV5_COHORT, bootstrap_ci  # noqa: E402

COHORTS = T.COHORTS
V1, V2, DEG, DR, NOGAP = "mpqda_de", "mpqda_v2", "de_g", "de_rand1", "mpqda_v2_no_gap"
ALL_K = (2, 5, 10, 50)
# (group, label, method a, method b, the k values, registered?)  — difference a − b of best_fitness per fold
INNER = (
    ("HV1", "v2 - v1", V2, V1, (5, 10), True),
    ("HV1", "v2 - GA", V2, "ga", (5, 10), True),
    ("HV4", "v2 - no gap", V2, NOGAP, (2, 5, 10), True),
    ("HV4", "v2 - no new-gene step", V2, "mpqda_v2_no_new", (2, 5, 10), True),
    ("HV4", "v2 - no archive tournament", V2, "mpqda_v2_no_tour", (2, 5, 10), True),
    ("E2", "DE-G - DE/rand/1/bin", DEG, DR, ALL_K, False),
    ("D", "v2 - v1", V2, V1, ALL_K, False),
    ("D", "v2 - DE-G", V2, DEG, ALL_K, False),
    ("D", "DE-G - DE/rand/1/bin", DEG, DR, ALL_K, False),
    ("D", "DE/rand/1/bin - v1", DR, V1, ALL_K, False),
)
HV6_K = (20, 50)
HV6_TREND_K = (2, 5, 10, 20, 50)
R50_TOL = 1e-12


# ------------------------------------------------------------------------------------------------- statistics
def describe(by_c, alternative: str = "greater") -> dict:
    """Describe one paired difference per fold, split by the three cohorts (list of arrays, same order as COHORTS)."""
    by_c = [np.asarray(d, dtype=float) for d in by_c]
    d = np.concatenate(by_c)
    lo, hi = bootstrap_ci(by_c)
    nb = T.pooled_nb(by_c)
    p_nb = T.pooled_p_greater(nb) if alternative == "greater" else float(st.t.cdf(nb["mean"] / nb["se"], nb["df"]))
    out = {"n": int(len(d)), "mean_diff": float(d.mean()), "share_positive": float(np.mean(d > 0)),
           "boot_ci_low": lo, "boot_ci_high": hi, "nb_se": nb["se"], "nb_ci_low": nb["ci_low"], "nb_ci_high": nb["ci_high"],
           "nb_excludes_zero": bool(nb["ci_low"] > 0 or nb["ci_high"] < 0), "p_nb_one_sided": p_nb,
           "p_wilcoxon_one_sided": T.wilcoxon_p(d, alternative)}
    for c, dc in zip(COHORTS, by_c):
        out[f"mean_{c}"] = float(dc.mean()) if len(dc) else np.nan
    out["n_cohorts_positive"] = int(sum(float(dc.mean()) > 0 for dc in by_c if len(dc)))
    return out


def inner_table(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, label, a, b, ks, registered in INNER:
        if not ((sel.method == a).any() and (sel.method == b).any()):
            continue
        for k in ks:
            by_c = [T.paired(sel, a, b, "best_fitness", k, c) for c in COHORTS]
            if not all(len(d) for d in by_c):
                continue
            rows.append({"group": group, "comparison": label, "a": a, "b": b, "k": k, "registered_test": registered,
                         **describe(by_c)})
    return pd.DataFrame(rows)


def decomposition_check(inner: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """v2 − v1 must equal the sum of the three terms (same fold, same k); returns the discrepancy per k (≈ 0)."""
    part = inner[inner.group == "D"].set_index(["comparison", "k"]).mean_diff
    rows = []
    for k in sorted(inner[inner.group == "D"].k.unique()):
        total = part.get(("v2 - v1", k), np.nan)
        terms = sum(part.get((t, k), np.nan) for t in ("v2 - DE-G", "DE-G - DE/rand/1/bin", "DE/rand/1/bin - v1"))
        rows.append({"k": int(k), "v2_minus_v1": total, "sum_of_terms": terms, "gap": total - terms,
                     "ok": bool(abs(total - terms) <= tol)})
    return pd.DataFrame(rows)


def outer_table(ev: pd.DataFrame, refs=("bpso", V1), ks=(5, 10)) -> pd.DataFrame:
    rows = []
    for k in ks:
        for ref in refs:
            for metric in ("logloss", "auroc"):
                by_c = [T.paired(ev, V2, ref, metric, k, c) for c in COHORTS]
                if not all(len(d) for d in by_c):
                    continue
                nb = T.pooled_nb(by_c)
                rows.append({"k": k, "reference": ref, "metric": metric, "n": int(sum(len(d) for d in by_c)),
                             "diff": nb["mean"], "nb_se": nb["se"], "nb_ci_low": nb["ci_low"], "nb_ci_high": nb["ci_high"],
                             "half_width": (nb["ci_high"] - nb["ci_low"]) / 2})
    return pd.DataFrame(rows)


def hv5_table(ev_b7: pd.DataFrame, cohort: str = HV5_COHORT, ks=(5, 10)) -> pd.DataFrame:
    rows = []
    for k in ks:
        d = T.paired(ev_b7, V1, V2, "auroc", k, cohort)
        if not len(d):
            continue
        nb = T.pooled_nb([d])
        rows.append({"cohort": cohort, "k": k, "n": int(len(d)), "auroc_v1_minus_v2": nb["mean"], "nb_ci_low": nb["ci_low"],
                     "nb_ci_high": nb["ci_high"], "half_width": (nb["ci_high"] - nb["ci_low"]) / 2})
    return pd.DataFrame(rows)


def runtime_table(sel: pd.DataFrame, ks=(5, 10)) -> pd.DataFrame:
    part = sel[sel.cohort.isin(COHORTS) & sel.k.isin(ks) & ~sel.method.str.endswith("_best")]
    if "selection_time_s" not in part:
        return pd.DataFrame()
    g = part.groupby(["method", "k"]).selection_time_s
    return g.agg(n="size", mean_s="mean", median_s="median").reset_index()


# ------------------------------------------------------------------------------------------------- amendment A2
def gate_r50(sel_c1: pd.DataFrame, sel_c4: pd.DataFrame, genes_c1: pd.DataFrame | None = None,
             genes_c4: pd.DataFrame | None = None, k: int = 50, tol: float = R50_TOL) -> pd.DataFrame:
    """R50: mpqda_v2 at k = 50 of C4 must match C1 (T1v2): same best_fitness (difference ≤ tol), same N_eval, same panel."""
    key = ["cohort", "repeat", "fold", "method"]
    names = (V2, V2 + "_best")
    a = sel_c1[(sel_c1.k == k) & sel_c1.method.isin(names)].set_index(key).sort_index()
    b = sel_c4[(sel_c4.k == k) & sel_c4.method.isin(names)].set_index(key).sort_index()
    idx = a.index.union(b.index)
    rows = []
    for i in idx:
        ok_a, ok_b = i in a.index, i in b.index
        row = dict(zip(key, i))
        row["in_c1"], row["in_c4"] = ok_a, ok_b
        if ok_a and ok_b:
            row["fitness_abs_diff"] = abs(float(a.loc[i, "best_fitness"]) - float(b.loc[i, "best_fitness"]))
            row["same_n_eval"] = int(a.loc[i, "n_eval"]) == int(b.loc[i, "n_eval"])
            if genes_c1 is not None and genes_c4 is not None:
                def panel(g):
                    m = (g.cohort == i[0]) & (g.repeat == i[1]) & (g.fold == i[2]) & (g.method == i[3]) & (g.k == k)
                    return tuple(sorted(g[m].gene.astype(str)))
                row["same_panel"] = panel(genes_c1) == panel(genes_c4)
            else:
                row["same_panel"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        same_panel = out.same_panel.fillna(True).astype(bool) if "same_panel" in out else True
        out["pass"] = out.in_c1 & out.in_c4 & (out.fitness_abs_diff <= tol) & out.same_n_eval & same_panel
    return out


def hv6_diffs(sel_small: pd.DataFrame, sel_c4: pd.DataFrame) -> pd.DataFrame:
    """d_k = best_fitness(v2) − best_fitness(v2 without gap) per fold: k ≤ 10 from C1 + C2, k ∈ {20, 50} from C4."""
    key = ["cohort", "repeat", "fold"]
    frames = []
    for k in HV6_TREND_K:
        src = sel_c4 if k in HV6_K else sel_small
        part = src[src.k == k]
        a = part[part.method == V2].set_index(key).best_fitness
        b = part[part.method == NOGAP].set_index(key).best_fitness
        idx = a.index.intersection(b.index)
        frames.append(pd.DataFrame({"k": k, "d": (a.loc[idx] - b.loc[idx])}).reset_index())
    return pd.concat(frames, ignore_index=True)


def hv6(diffs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HV6a: one-sided Wilcoxon (d > 0) at k ∈ {20, 50}, Holm over 2; HV6: Page's L, increasing trend over k ∈ {2, 5, 10, 20, 50}."""
    rows = []
    for k in HV6_K:
        part = diffs[diffs.k == k]
        by_c = [part[part.cohort == c].sort_values(["repeat", "fold"]).d.to_numpy() for c in COHORTS]
        if not all(len(d) for d in by_c):
            continue
        rows.append({"k": k, **describe(by_c)})
    a = pd.DataFrame(rows)
    if len(a):
        a["p_holm"] = holm(a.p_wilcoxon_one_sided.to_numpy())
        a["holm_family"], a["family_size"] = "HV6a", len(a)
        a["significant"] = a.p_holm < T.ALPHA
    wide = diffs.pivot_table(index=["cohort", "repeat", "fold"], columns="k", values="d")
    wide = wide[[k for k in HV6_TREND_K if k in wide.columns]].dropna()
    page = pd.DataFrame()
    if wide.shape[1] == len(HV6_TREND_K) and len(wide) >= 2:
        res = st.page_trend_test(wide.to_numpy())          # columns in increasing k = predicted ranks 1..5
        page = pd.DataFrame([{"n_blocks": int(len(wide)), "ks": " ".join(map(str, HV6_TREND_K)), "L": float(res.statistic),
                              "p": float(res.pvalue), "method": res.method, "significant": bool(res.pvalue < T.ALPHA),
                              **{f"mean_d_k{k}": float(wide[k].mean()) for k in HV6_TREND_K}}])
    return a, page


def hv6_reading(a: pd.DataFrame, page: pd.DataFrame) -> str:
    """Reading rule of amendment A2 (written before the run): supported / partially supported / not supported."""
    trend = bool(len(page) and page.significant.iloc[0])
    at50 = bool(len(a) and a[a.k == 50].significant.any())
    if trend and at50:
        return "supported"
    if trend or at50:
        return "partially supported"
    return "not supported"


# ------------------------------------------------------------------------------------------------- main
def build(results: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    inp, claims, done = T.Inputs(results), [], []

    def emit(name, frame, tag=None, columns=(), keys=(), phase_b=T.PHASE_A):
        T.write_csv(frame, out / f"{name}.csv")
        if tag and len(frame):
            claims.extend(T.claims_rows(tag, frame, f"{name}.csv", columns, keys, phase_b))
        done.append(name)

    frames = [inp.scenario(s, "selection_per_fold.csv") for s in ("T1", "T1v2", "T2v2")]
    if all(f is not None for f in frames):
        sel = pd.concat(frames, ignore_index=True)
        inner = inner_table(sel)
        emit("corrected_inner", inner, "CORR", ["mean_diff", "boot_ci_low", "boot_ci_high", "nb_ci_low", "nb_ci_high",
                                                 "n_cohorts_positive", "p_nb_one_sided", "share_positive"],
             ["group", "comparison", "k"])
        emit("decomposition_check", decomposition_check(inner))
        emit("runtime", runtime_table(sel), "RUNTIME", ["mean_s", "median_s"], ["method", "k"])
    ev = [inp.scenario(f"T5LL/{s}", "classification_per_fold.csv", encoding="utf-8-sig") for s in ("T1", "T1v2")]
    if all(f is not None for f in ev):
        ev = pd.concat(ev, ignore_index=True)
        emit("corrected_outer", outer_table(ev[ev.model == "lr_l2"].reset_index(drop=True)), "CORROUT",
             ["diff", "nb_ci_low", "nb_ci_high", "half_width"], ["k", "reference", "metric"], "B-LL")
    b7 = [inp.scenario(f"T5LL/{s}", "classification_per_fold.csv", [HV5_COHORT], encoding="utf-8-sig") for s in ("T1", "T1v2")]
    if all(f is not None for f in b7):
        b7 = pd.concat(b7, ignore_index=True)
        emit("corrected_hv5", hv5_table(b7[b7.model == "lr_l2"].reset_index(drop=True)), "CORRHV5",
             ["auroc_v1_minus_v2", "nb_ci_low", "nb_ci_high"], ["cohort", "k"], "B-LL")
    sel_c4 = inp.scenario("T4v2", "selection_per_fold.csv")
    status = "C4 not available (T4v2): R50 and HV6 skipped"
    if sel_c4 is not None and frames[1] is not None and frames[2] is not None:
        g1, g4 = inp.scenario("T1v2", "selected_genes.csv"), inp.scenario("T4v2", "selected_genes.csv")
        gate = gate_r50(frames[1], sel_c4, g1, g4)
        emit("gate_r50", gate)
        if not len(gate) or not gate["pass"].all():
            status = "R50 FAILED: stop, HV6 not computed, report it"
        else:
            a, page = hv6(hv6_diffs(pd.concat([frames[1], frames[2]], ignore_index=True), sel_c4))
            reading = hv6_reading(a, page)
            emit("hv6", a.assign(reading=reading), "HV6",
                 ["mean_diff", "boot_ci_low", "boot_ci_high", "nb_ci_low", "nb_ci_high", "share_positive", "p_wilcoxon_one_sided",
                  "p_holm", "significant"], ["k"])
            emit("hv6_page", page.assign(reading=reading), "HV6PAGE", ["L", "p", "significant"], ["ks"])
            status = f"R50 passed; HV6: {reading}"
    T.write_csv(pd.DataFrame(sorted(inp.used.items()), columns=["file", "sha256"]), out / "inputs_sha256_corrected.csv")
    T.write_csv(pd.DataFrame(claims, columns=T.CLAIM_COLUMNS), out / "claims_corrected.csv")
    return {"written": done, "missing": sorted(set(inp.missing)), "status": status}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tables_v2/corrected"))
    args = ap.parse_args(argv)
    res = build(args.results, args.out)
    print("written: " + ", ".join(res["written"]))
    print(res["status"])
    if res["missing"]:
        print(f"missing {len(res['missing'])} files: " + ", ".join(res["missing"][:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
