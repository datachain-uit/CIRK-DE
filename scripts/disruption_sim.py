"""Mechanism analysis (no data): how many random genes the mutation + crossover operator of random-key DE with band re-encoding
brings into a trial panel, as a function of F, CR, the width of the lower band eps and the gene overlap of the population.

    python scripts/disruption_sim.py [--reps 200] [--out docs/taimma/disruption.csv]

Each repetition: build a population of P panels (each panel takes share·k genes from a common core of 2k genes, the rest at random), encode,
take leader L = 0, target i = 1, random r1, r2, build u = crossover(repair(x_L + F(x_r1 − x_r2)), x_i), then count the share of genes
of S(u) that belong to the leader panel, the target panel, and to neither the leader, the target nor r1 (random genes). No new-gene mutation yet.
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
from mpqda.de_v2 import gapped_band_encode  # noqa: E402


def population(rng, P, p0, k, share):
    core = rng.choice(p0, size=2 * k, replace=False)
    out = []
    for _ in range(P):
        m = int(round(share * k))
        a = rng.choice(core, size=m, replace=False)
        rest = rng.choice(np.setdiff1d(np.arange(p0), a), size=k - m, replace=False)
        out.append(np.sort(np.concatenate([a, rest])))
    return out


def keys(rng, S, p0, k, eps):
    x = rng.random(p0) * 0.5
    x[S] = 0.5 + rng.random(len(S)) * 0.5
    return gapped_band_encode(x, k, eps)


def simulate(reps=200, P=30, p0=5000, k=10, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for share in (0.0, 0.5):
        for eps in (0.5, 0.01):
            for F in (0.3, 0.5, 0.7, 0.9):
                for CR in (0.9, 0.5):
                    kept_l, kept_t, rand = [], [], []
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
                        kept_t.append(len(S & ST) / k)
                        rand.append(len(S - SL - ST - S1) / k)
                    rows.append({"share": share, "eps": eps, "F": F, "CR": CR, "leader_kept": np.mean(kept_l),
                                 "target_kept": np.mean(kept_t), "random_genes": np.mean(rand), "reps": reps})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="docs/taimma/disruption.csv")
    a = ap.parse_args()
    df = simulate(a.reps)
    df.to_csv(a.out, index=False)
    print(df.round(3).to_string(index=False))
