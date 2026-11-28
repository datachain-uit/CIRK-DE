"""Random-key representation and decoding to exactly k genes (PREREG, Section 3.1).

An individual is a key vector x ∈ (0, 1)^p0. Panel S(x) is the k indices with the largest keys; equal keys put the smaller index
first, so every individual is feasible and |S(x)| = k.
"""
from __future__ import annotations

import numpy as np

from . import config as M


def decode_topk(x, k: int) -> np.ndarray:
    """The k indices with the largest keys; equal keys put the smaller index first. Returns an ascending array."""
    order = np.argsort(-np.asarray(x, dtype=float), kind="stable")
    return np.sort(order[:k])


def jaccard_distance(a, b) -> float:
    """d(a, b) = 1 − |a ∩ b| / |a ∪ b|; two empty sets give 0."""
    a, b = set(map(int, a)), set(map(int, b))
    u = len(a | b)
    return 0.0 if u == 0 else 1.0 - len(a & b) / u


def init_chaotic(P: int, p: int, rng: np.random.Generator, mu: float = M.MU, z0_range=M.Z0_RANGE) -> np.ndarray:
    """Logistic map z(t + 1) = mu·z(t)·(1 − z(t)); z_i(0) ~ U(0.001; 0.999), one value per individual; the key of gene j
    is z_i(j + 1). Uses exactly one call rng.uniform(size=P)."""
    z = rng.uniform(z0_range[0], z0_range[1], size=P)
    X = np.empty((P, p))
    for j in range(p):
        z = mu * z * (1.0 - z)
        X[:, j] = z
    return X


def init_uniform(P: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform keys U(0, 1): variant A1 and every DE baseline. Uses exactly one call rng.random((P, p))."""
    return rng.random((P, p))


def repair_midpoint(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Midpoint bound repair (as in JADE, SHADE): v > 1 becomes (1 + x)/2, v < 0 becomes x/2, where x is the TARGET individual (same
    shape as v), not the leader."""
    v = np.where(v > 1.0, (1.0 + x) / 2.0, v)
    return np.where(v < 0.0, x / 2.0, v)


def band_encode(x: np.ndarray, k: int) -> np.ndarray:
    """Band re-encoding: genes of S(x) get 0.5 + 0.5·(r + 0.5)/k, other genes get 0.5·(r + 0.5)/(p0 − k), where r is
    the stable rank of the key within its group. The panel and the key order within each group are unchanged."""
    x = np.asarray(x, dtype=float)
    S = decode_topk(x, k)
    in_s = np.zeros(len(x), dtype=bool)
    in_s[S] = True
    out = np.empty(len(x))
    for group, low in ((in_s, 0.5), (~in_s, 0.0)):
        idx = np.flatnonzero(group)
        if len(idx) == 0:
            continue
        rank = np.argsort(np.argsort(x[idx], kind="stable"), kind="stable")
        out[idx] = low + 0.5 * (rank + 0.5) / len(idx)
    assert np.array_equal(decode_topk(out, k), S)
    return out
