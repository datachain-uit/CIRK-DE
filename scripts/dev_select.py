"""Selection rule of the CIRK-DE configuration in the development phase and gate G-DEV (PREREG-v2, Section 2.2). Reads only results of the
development data (tcga_dev_a, tcga_dev_b, tcga_dev_c, gse42127); never reads GPL570 or B7.


    python scripts/dev_select.py --results <...>/taimma/DEV --out dev_selection \
        [--eval <...>/taimma/T5LL/DEV]

Comparison unit = (dataset, repeat, fold, k) with k ∈ {5, 10}; score = best_fitness (inner fitness of the best individual
at T = 50, same budget).
  1. Within each unit, rank the three candidates (1 = best; ties get the mean rank). Choose the candidate with the smallest mean
     rank. When two candidates differ by less than 0.10 rank, choose by the priority order M1, M3, M2 (fewer components first).
  2. Gate G-DEV: the chosen candidate beats v1 (mpqda_de) on best_fitness, one-sided Wilcoxon (greater) over all units,
     p < 0.05. If it fails, there is no confirmatory phase for v2.
  3. Descriptive, not used for the choice: best_fitness differences against GA, DE-G, DE/rand/1 (mean, two-sided Wilcoxon) and outer
     B-LL log-loss of the main output if --eval is given.
Writes <out>_ranks.csv, <out>_pairs.csv, <out>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CANDIDATES = ("mpqda_v2_m1", "mpqda_v2_m2", "mpqda_v2_m3")
PREFERENCE = ("mpqda_v2_m1", "mpqda_v2_m3", "mpqda_v2_m2")
REFERENCES = ("mpqda_de", "ga", "de_g", "de_rand1")
DEV_DATASETS = ("tcga_dev_a", "tcga_dev_b", "tcga_dev_c", "gse42127")
UNIT = ["cohort", "repeat", "fold", "k"]


def load(results: Path, datasets, k=(5, 10)) -> pd.DataFrame:
    frames = []
    for d in datasets:
        path = results / d / "selection_per_fold.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.k.isin(k) & df.method.isin([*CANDIDATES, *REFERENCES])]
    return df.drop_duplicates(subset=[*UNIT, "method"], keep="last")


def choose(df: pd.DataFrame) -> tuple[str, pd.DataFrame, int]:
    wide = df[df.method.isin(CANDIDATES)].pivot_table(index=UNIT, columns="method", values="best_fitness")
    wide = wide.dropna()
    ranks = wide.rank(axis=1, ascending=False, method="average")
    mean_rank = ranks.mean().reindex(CANDIDATES)
    best = mean_rank.min()
    near = [m for m in PREFERENCE if mean_rank[m] - best < 0.10]
    table = pd.DataFrame({"method": mean_rank.index, "mean_rank": mean_rank.values,
                          "mean_best_fitness": wide.mean().reindex(CANDIDATES).values, "n_units": len(wide)})
    return near[0], table, len(wide)


def paired(df: pd.DataFrame, a: str, b: str, alternative: str) -> dict:
    wide = df[df.method.isin([a, b])].pivot_table(index=UNIT, columns="method", values="best_fitness").dropna()
    d = (wide[a] - wide[b]).to_numpy()
    p = float(wilcoxon(d, alternative=alternative, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
    return {"a": a, "b": b, "n": int(len(d)), "mean_diff": float(d.mean()), "share_a_better": float(np.mean(d > 0)),
            "alternative": alternative, "p": p}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DEV_DATASETS))
    ap.add_argument("--eval", type=Path, default=None, help="B-LL folder of DEV (classification_per_fold.csv)")
    ap.add_argument("--out", default="dev_selection")
    a = ap.parse_args(argv)
    if set(a.datasets) - set(DEV_DATASETS):
        raise SystemExit(f"STOP: only the development data may be used: {DEV_DATASETS}")
    df = load(a.results, a.datasets)
    chosen, ranks, n_units = choose(df)
    gate = paired(df, chosen, "mpqda_de", "greater")
    pairs = [gate] + [paired(df, chosen, m, "two-sided") for m in ("ga", "de_g", "de_rand1")]
    out = {"chosen": chosen, "n_units": n_units, "g_dev_passed": bool(gate["p"] < 0.05), "g_dev": gate}
    if a.eval is not None:
        ev = pd.concat([pd.read_csv(a.eval / d / "classification_per_fold.csv", encoding="utf-8-sig") for d in a.datasets])
        ev = ev[ev.k.isin((5, 10)) & (ev.model == "lr_l2")]
        out["dev_outer_logloss"] = ev.groupby("method").logloss.mean().round(6).to_dict()
    ranks.to_csv(f"{a.out}_ranks.csv", index=False)
    pd.DataFrame(pairs).to_csv(f"{a.out}_pairs.csv", index=False)
    Path(f"{a.out}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["g_dev_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
