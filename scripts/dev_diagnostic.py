"""Diagnosis of the reversal on the development data.

    python scripts/dev_diagnostic.py --taimma <KLTN_DATA_ROOT>/results/survey_v2/taimma --out dev_diagnostic

Reads only DEV results (phase A in taimma/DEV, phase B in taimma/T5LL|T5C1|T5LLW/DEV); never touches the confirmatory data. Every table is
DESCRIPTIVE for the development phase: two-sided Wilcoxon p values, uncorrected, for orientation only, not used for any claim of the paper.

  dev_diag_by_dataset.csv   by dataset, k, output, phase B criterion: inner log-loss (−output_fitness), outer log-loss and AUROC,
                            optimism = outer log-loss − inner log-loss (B-C1 is the matched comparison: same LR-L2, C = 1 as in the fitness)
  dev_diag_pairs.csv        pairs (M1, v1), (M1, GA), (M1, DE-G), (M1, DE/rand/1), (v1, GA), (consensus, best individual):
                            difference a − b per unit (dataset, repeat, fold), by k and pooled over the two k
  dev_diag_inner_outer.csv  Spearman between inner and outer log-loss over the seven main outputs: within each unit (then summarised),
                            and on the per-method means; Spearman between Δinner and Δouter of M1 − v1 over the units
  dev_diag_c_choice.csv     share of C at the lower and upper edge of the grid (B-LL, B-LL-wide), median C
  dev_diag_overlap.csv      Jaccard between the panels of two methods on the same unit
  dev_diag_inputs_sha256.csv
Two runs give byte-identical files.
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

from mpqda import config as M  # noqa: E402
from scripts import make_taimma_tables as T  # noqa: E402

DATASETS = tuple(M.DEV_DATASETS)
KS = (5, 10)
M1, V1, GA, DEG, DER = "mpqda_v2_m1", "mpqda_de", "ga", "de_g", "de_rand1"
MAIN = ("mpqda_v2_m1", "mpqda_v2_m2", "mpqda_v2_m3", V1, DEG, DER, GA)        # the seven main outputs (consensus for v1, v2)
PAIRS = ((M1, V1), (M1, GA), (M1, DEG), (M1, DER), (V1, GA), (M1, M1 + M.BEST_SUFFIX), (V1, V1 + M.BEST_SUFFIX))
PHASE_B = {"B-LL": "T5LL", "B-C1": "T5C1", "B-LL-wide": "T5LLW"}
GRID_EDGES = {"B-LL": (0.01, 10.0), "B-LL-wide": (1e-4, 1000.0)}
UNIT = ["cohort", "repeat", "fold"]
HIGHER_IS_BETTER = {"inner_fitness": True, "outer_auroc": True, "outer_logloss": False, "optimism": False}


def load(taimma: Path):
    inp = T.Inputs(taimma)
    sel = inp.scenario("DEV", "selection_per_fold.csv", DATASETS)
    genes = inp.scenario("DEV", "selected_genes.csv", DATASETS)
    if sel is None or genes is None:
        raise SystemExit(f"missing DEV phase A: {sorted(set(inp.missing))}")
    evs = {}
    for pb, sub in PHASE_B.items():
        ev = inp.scenario(f"{sub}/DEV", "classification_per_fold.csv", DATASETS, encoding="utf-8-sig")
        if ev is not None:
            evs[pb] = ev[ev.model == "lr_l2"].reset_index(drop=True)
    if "B-LL" not in evs:
        raise SystemExit("missing T5LL/DEV (B-LL of DEV)")
    return inp, sel, genes, evs


def merged(sel: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (unit, output, k): inner fitness of exactly the evaluated panel and its outer metrics."""
    a = sel[[*UNIT, "method", "k", "output_fitness"]].rename(columns={"output_fitness": "inner_fitness"})
    b = ev[[*UNIT, "method", "k", "logloss", "auroc", "C"]].rename(columns={"logloss": "outer_logloss", "auroc": "outer_auroc"})
    m = a.merge(b, on=[*UNIT, "method", "k"], how="inner", validate="one_to_one")
    m["inner_logloss"] = -m.inner_fitness
    m["optimism"] = m.outer_logloss - m.inner_logloss
    return m


def by_dataset(frames: dict) -> pd.DataFrame:
    rows = []
    for pb, m in frames.items():
        for (cohort, k, method), g in m.groupby(["cohort", "k", "method"], sort=True):
            rows.append({"phase_b": pb, "cohort": cohort, "k": int(k), "method": method, "n": len(g),
                         "inner_logloss": g.inner_logloss.mean(), "outer_logloss": g.outer_logloss.mean(),
                         "outer_auroc": g.outer_auroc.mean(), "optimism": g.optimism.mean(), "optimism_sd": g.optimism.std(ddof=1)})
        for (k, method), g in m.groupby(["k", "method"], sort=True):
            rows.append({"phase_b": pb, "cohort": "all", "k": int(k), "method": method, "n": len(g),
                         "inner_logloss": g.inner_logloss.mean(), "outer_logloss": g.outer_logloss.mean(),
                         "outer_auroc": g.outer_auroc.mean(), "optimism": g.optimism.mean(), "optimism_sd": g.optimism.std(ddof=1)})
    return pd.DataFrame(rows)


def _diff(m: pd.DataFrame, a: str, b: str, col: str, ks) -> np.ndarray:
    part = m[m.k.isin(ks)]
    pa = part[part.method == a].set_index([*UNIT, "k"])[col]
    pb = part[part.method == b].set_index([*UNIT, "k"])[col]
    idx = pa.index.intersection(pb.index).sort_values()
    return (pa.loc[idx] - pb.loc[idx]).to_numpy(dtype=float)


def pairs(frames: dict) -> pd.DataFrame:
    rows = []
    for pb, m in frames.items():
        for a, b in PAIRS:
            for k_label, ks in (("5", (5,)), ("10", (10,)), ("5+10", KS)):
                for col, higher in HIGHER_IS_BETTER.items():
                    d = _diff(m, a, b, col, ks)
                    if not len(d):
                        continue
                    better = d > 0 if higher else d < 0
                    rows.append({"phase_b": pb, "a": a, "b": b, "k": k_label, "quantity": col, "n": len(d),
                                 "mean_diff": float(d.mean()), "share_a_better": float(better.mean()),
                                 "p_two_sided_exploratory": T.wilcoxon_p(d, "two-sided")})
    return pd.DataFrame(rows)


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(st.spearmanr(x, y).statistic)


def inner_outer(frames: dict) -> pd.DataFrame:
    rows = []
    for pb, m in frames.items():
        main = m[m.method.isin(MAIN)]
        per_unit = []
        for (cohort, repeat, fold, k), g in main.groupby([*UNIT, "k"], sort=True):
            per_unit.append({"cohort": cohort, "k": int(k), "rho": _spearman(g.inner_logloss, g.outer_logloss)})
        pu = pd.DataFrame(per_unit)
        for cohort, g in [*pu.groupby("cohort", sort=True), ("all", pu)]:
            r = g.rho.dropna()
            rows.append({"phase_b": pb, "analysis": "within_unit_across_methods", "cohort": cohort, "n": len(r),
                         "rho_mean": float(r.mean()), "rho_median": float(r.median()), "share_rho_negative": float((r < 0).mean())})
        for cohort in (*DATASETS, "all"):
            g = main if cohort == "all" else main[main.cohort == cohort]
            means = g.groupby("method")[["inner_logloss", "outer_logloss"]].mean()
            rows.append({"phase_b": pb, "analysis": "method_means", "cohort": cohort, "n": len(means),
                         "rho_mean": _spearman(means.inner_logloss, means.outer_logloss), "rho_median": np.nan,
                         "share_rho_negative": np.nan})
        di, do = _diff(m, M1, V1, "inner_logloss", KS), _diff(m, M1, V1, "outer_logloss", KS)
        rows.append({"phase_b": pb, "analysis": "delta_m1_minus_v1_across_units", "cohort": "all", "n": len(di),
                     "rho_mean": _spearman(di, do), "rho_median": np.nan, "share_rho_negative": np.nan})
    return pd.DataFrame(rows)


def c_choice(evs: dict) -> pd.DataFrame:
    rows = []
    for pb, (lo, hi) in GRID_EDGES.items():
        if pb not in evs:
            continue
        ev = evs[pb][evs[pb].C.notna()]
        for (method, k), g in ev.groupby(["method", "k"], sort=True):
            c = g.C.astype(float)
            rows.append({"phase_b": pb, "method": method, "k": int(k), "n": len(c), "share_at_lower": float(np.isclose(c, lo).mean()),
                         "share_at_upper": float(np.isclose(c, hi).mean()), "median_C": float(c.median())})
    return pd.DataFrame(rows)


def overlap(genes: pd.DataFrame) -> pd.DataFrame:
    panels = {key: set(g.gene.astype(str)) for key, g in genes.groupby([*UNIT, "method", "k"], sort=True)}
    rows = []
    for a, b in ((M1, V1), (M1, GA), (V1, GA), (M1, DEG), (M1, M1 + M.BEST_SUFFIX), (V1, V1 + M.BEST_SUFFIX)):
        for k in KS:
            vals = []
            for (cohort, repeat, fold, method, kk), pa in panels.items():
                if method != a or kk != k or (cohort, repeat, fold, b, kk) not in panels:
                    continue
                pb = panels[(cohort, repeat, fold, b, kk)]
                vals.append(len(pa & pb) / len(pa | pb))
            if vals:
                rows.append({"a": a, "b": b, "k": k, "n": len(vals), "jaccard_mean": float(np.mean(vals)),
                             "jaccard_median": float(np.median(vals))})
    return pd.DataFrame(rows)


def build(taimma: Path, out: Path) -> dict:
    inp, sel, genes, evs = load(taimma)
    frames = {pb: merged(sel, ev) for pb, ev in evs.items()}
    tables = {"dev_diag_by_dataset": by_dataset(frames), "dev_diag_pairs": pairs(frames), "dev_diag_inner_outer": inner_outer(frames),
              "dev_diag_c_choice": c_choice(evs), "dev_diag_overlap": overlap(genes)}
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        T.write_csv(frame, out / f"{name}.csv")
    T.write_csv(pd.DataFrame(sorted(inp.used.items()), columns=["file", "sha256"]), out / "dev_diag_inputs_sha256.csv")
    return {"written": sorted(tables), "phase_b": sorted(evs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taimma", type=Path, required=True, help="results/survey_v2/taimma folder holding DEV and T5*/DEV")
    ap.add_argument("--out", type=Path, default=Path("dev_diagnostic"))
    args = ap.parse_args(argv)
    info = build(args.taimma, args.out)
    print("written: " + ", ".join(info["written"]) + " | phase B: " + ", ".join(info["phase_b"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
