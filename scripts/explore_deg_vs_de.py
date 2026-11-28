"""Exploratory, not registered: best inner fitness of DE-G minus DE/rand/1/bin on the three
GPL570 cohorts, paired by (repeat, fold); 95 % bootstrap CI stratified by cohort and one-sided Wilcoxon (DE-G better) as in HV1,
no Holm, in no family.

    python scripts/explore_deg_vs_de.py --results <folder holding T1 and T1v2> --out tables_v2/explore_deg_vs_de.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from scripts import make_taimma_tables as T  # noqa: E402
from scripts.make_taimma_v2_tables import bootstrap_ci  # noqa: E402

A, B = "de_g", "de_rand1"


def compare(sel: pd.DataFrame, ks=(2, 5, 10, 50)) -> pd.DataFrame:
    rows = []
    for k in ks:
        by_c = [T.paired(sel, A, B, "best_fitness", k, c) for c in T.COHORTS]
        d = np.concatenate(by_c)
        lo, hi = bootstrap_ci(by_c)
        rows.append({"k": k, "n": len(d), "mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi,
                     "share_deg_better": float((d > 0).mean()), "p_one_sided": T.wilcoxon_p(d, "greater")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    inp = T.Inputs(a.results)
    s1, s2 = inp.scenario("T1", "selection_per_fold.csv"), inp.scenario("T1v2", "selection_per_fold.csv")
    if s1 is None or s2 is None:
        sys.exit(f"missing files: {inp.missing}")
    out = compare(pd.concat([s1, s2], ignore_index=True))
    out.to_csv(a.out, index=False, lineterminator="\n")
    print(out.round(4).to_string(index=False))
    for rel, h in sorted(inp.used.items()):
        print(h, rel)


if __name__ == "__main__":
    main()
