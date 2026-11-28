"""Exploratory, not registered: the mutation + crossover operator of random-key DE WITHOUT
key re-encoding (plain keys), same setting as scripts/disruption_sim.py. Uses the population generator of disruption_sim.py,
but its own seed (2) and its own output file, so docs/taimma/disruption.csv and its random stream do not change.

    python scripts/disruption_plain.py [--reps 200] [--out docs/taimma/disruption_plain.csv]

Plain keys of a panel S: the k largest values of a uniform vector U(0, 1)^p0 are put (in random order) on the genes of S,
the rest on the genes outside S; this is the state of a random-key DE population at initialisation without re-encoding.
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
from disruption_sim import population  # noqa: E402


def plain_keys(rng, S, p0, k):
    """Uniform keys whose k largest values lie on the genes of S."""
    u = np.sort(rng.random(p0))
    x = np.empty(p0)
    others = np.setdiff1d(np.arange(p0), S)
    x[rng.permutation(S)] = u[p0 - k:]
    x[rng.permutation(others)] = u[:p0 - k]
    return x


def simulate_plain(reps=200, P=30, p0=5000, k=10, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for share in (0.0, 0.5):
        for F in (0.3, 0.5, 0.7, 0.9):
            for CR in (0.9, 0.5):
                kept_l, kept_t, rand = [], [], []
                for _ in range(reps):
                    pans = population(rng, P, p0, k, share)
                    X = np.array([plain_keys(rng, S, p0, k) for S in pans])
                    r1, r2 = rng.choice(np.arange(2, P), size=2, replace=False)
                    v = repair_midpoint(X[0] + F * (X[r1] - X[r2]), X[1])
                    m = rng.random(p0) <= CR
                    m[rng.integers(p0)] = True
                    S = set(decode_topk(np.where(m, v, X[1]), k).tolist())
                    SL, ST, S1 = (set(pans[j].tolist()) for j in (0, 1, r1))
                    kept_l.append(len(S & SL) / k)
                    kept_t.append(len(S & ST) / k)
                    rand.append(len(S - SL - ST - S1) / k)
                rows.append({"share": share, "encoding": "plain", "F": F, "CR": CR,
                             "leader_kept": np.mean(kept_l), "target_kept": np.mean(kept_t),
                             "random_genes": np.mean(rand), "reps": reps})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="docs/taimma/disruption_plain.csv")
    a = ap.parse_args()
    df = simulate_plain(a.reps)
    df.to_csv(a.out, index=False, lineterminator="\n")
    print(df.round(3).to_string(index=False))
