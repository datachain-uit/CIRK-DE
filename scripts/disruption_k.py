"""Exploratory, not registered (k = 20 added by amendment A2 of PREREG-v2): number of foreign genes that the
mutation + crossover operator brings into a trial panel as a function of k, with the band ε = 1/2 (v1 and the DE baselines) and the gap ε = 0.01 (v2,
DE-G), F = 0.5, CR = 0.9. Uses population() and keys() of scripts/disruption_sim.py; its own seed (3) and its own output file, so
disruption.csv and disruption_plain.csv do not change.

    python scripts/disruption_k.py [--reps 200] [--ks 2 5 10 50 20] [--out docs/taimma/disruption_k.csv]

One random generator goes through the k values in the order of --ks. The default order 2, 5, 10, 50 reproduces exactly the
first 16 rows of docs/taimma/disruption_k.csv; putting k = 20 LAST (--ks 2 5 10 50 20) leaves the first 16 rows unchanged and only adds
the four rows of k = 20 (amendment A2 of PREREG-v2 cites these rows).

random_genes_per_trial = random_share · k: number of genes that belong to neither the leader, the target nor the added vectors in a trial panel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mpqda  # noqa: E402,F401  (puts vendor/ and the repository root on sys.path)
from mpqda.encoding import decode_topk, repair_midpoint  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disruption_sim import keys, population  # noqa: E402

DEFAULT_KS = (2, 5, 10, 50)


def simulate_k(reps=200, P=30, p0=5000, ks=DEFAULT_KS, F=0.5, CR=0.9, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        for eps in (0.5, 0.01):
            for share in (0.0, 0.5):
                kept_l, rand = [], []
                for _ in range(reps):
                    pans = population(rng, P, p0, k, share)
                    X = np.array([keys(rng, S, p0, k, eps) for S in pans])
                    r1, r2 = rng.choice(np.arange(2, P), size=2, replace=False)
                    v = repair_midpoint(X[0] + F * (X[r1] - X[r2]), X[1])
                    m = rng.random(p0) <= CR
                    m[rng.integers(p0)] = True
                    S = set(decode_topk(np.where(m, v, X[1]), k).tolist())
                    SL, ST, S1 = (set(pans[j].tolist()) for j in (0, 1, r1))
                    kept_l.append(len(S & SL) / k)
                    rand.append(len(S - SL - ST - S1) / k)
                rows.append({"k": k, "eps": eps, "share": share, "F": F, "CR": CR,
                             "leader_kept": np.mean(kept_l), "random_share": np.mean(rand),
                             "random_genes_per_trial": np.mean(rand) * k, "reps": reps})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS),
                    help="run order; add new k values at the end so that the existing rows do not change")
    ap.add_argument("--out", default="docs/taimma/disruption_k.csv")
    a = ap.parse_args(argv)
    if len(set(a.ks)) != len(a.ks):
        ap.error("--ks has repeated values")
    df = simulate_k(a.reps, ks=tuple(a.ks))
    df.to_csv(a.out, index=False, lineterminator="\n")
    print(df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
