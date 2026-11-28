"""CIRK-DE (v2; PREREG-v2, Section 3). Function bodies and the order of random draws are fixed by the registration.

v2 is a random-key DE with exactly k genes; it fixes the source of disruption in v1 and then adds selection pressure:
  (1) two-band re-encoding with a gap: genes in the panel get keys in (1/2, 1) by rank as in v1; genes outside the panel are compressed into
      (0, eps). In v1 (eps = 1/2), the difference vector pushes many random genes of the lower band into the upper band; with a small eps,
      mutation and crossover only exchange genes between the existing panels;
  (2) GA-style new-gene mutation: each gene of the trial panel is replaced, with probability rate_new (default 1/k), by a gene outside the
      panel drawn uniformly from p0, implemented by swapping the keys of the two genes. This is the only source of new genes;
  (3) the leader and the vector for r1 come from the archive by tournament on fitness rank (tour = 3);
  (4) F, CR fixed (default), or adapted from the success history as in SHADE (adapt = True, candidate M2);
  (5) the main output is the consensus panel of the archive as in v1, together with the best individual.
eps = 1/2, rate_new = 0, tour = 1, r1_from = "population" give back the operator of v1 with a uniformly drawn leader (A2b) and fixed F.

Uses decode_topk, repair_midpoint, Archive, history_row, exploration_rate of the mpqda package (branch taimma-2027).
Order of random draws in each generation (part of the specification): (1) if adapt: memory cell index, F (Cauchy, redrawn when ≤ 0), CR
for i = 0..P−1; (2) leader: tour integers for each i; (3) r1 ≠ r2 ≠ i for each i; (4) if r1_from = "archive": tour
integers for each i; (5) U = rng.random((P, p0)), j_rand; (6) new-gene mutation for i = 0..P−1: k uniform numbers, then choice if
a gene is replaced. Initialisation: rng.random((P, p0)).
"""
from __future__ import annotations

import numpy as np

from mpqda.archive import Archive
from mpqda.de import exploration_rate, history_row
from mpqda.encoding import decode_topk, repair_midpoint

EPS_LOW = 0.01                                     # width of the lower band
F_FIXED, CR_FIXED = 0.5, 0.9                       # as DE/rand/1 of the paper
H_MEM, M_INIT, F_SCALE, CR_SD, F_MAX = 6, 0.5, 0.1, 0.1, 1.0   # used only when adapt = True (as in SHADE)

# Candidates of the development phase (PREREG-v2, Section 2.2) and ablation variants of the confirmatory phase
CANDIDATES = {
    "mpqda_v2_m1": dict(),                                    # core: gap, new genes, tournament 3 for the leader and r1, fixed F, CR
    "mpqda_v2_m2": dict(adapt=True),                          # M1 + SHADE-style adaptation
    "mpqda_v2_m3": dict(tour=2),                              # M1 with tournament 2
}
ABLATIONS = {                                                 # remove one component of the chosen configuration (overrides the chosen configuration)
    "v2_no_gap": dict(eps=0.5),
    "v2_no_new": dict(rate_new=0.0),
    "v2_no_tour": dict(tour=1, r1_from="population"),
}


def gapped_band_encode(x: np.ndarray, k: int, eps: float = EPS_LOW) -> np.ndarray:
    """Genes of S(x) get 0.5 + 0.5·(r + 0.5)/k; other genes get eps·(r + 0.5)/(p0 − k); r is the stable rank within the group.
    eps = 0.5 gives exactly band_encode of v1. The panel and the key order within each group are unchanged."""
    x = np.asarray(x, dtype=float)
    if not 0.0 < eps <= 0.5:
        raise ValueError(eps)
    S = decode_topk(x, k)
    in_s = np.zeros(len(x), dtype=bool)
    in_s[S] = True
    out = np.empty(len(x))
    for group, low, width in ((in_s, 0.5, 0.5), (~in_s, 0.0, eps)):
        idx = np.flatnonzero(group)
        if len(idx) == 0:
            continue
        rank = np.argsort(np.argsort(x[idx], kind="stable"), kind="stable")
        out[idx] = low + width * (rank + 0.5) / len(idx)
    assert np.array_equal(decode_topk(out, k), S)
    return out


def innovate(u: np.ndarray, k: int, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Each gene of panel S(u) is replaced with probability rate by a gene outside the panel drawn uniformly; the keys of the old and the new
    gene are swapped, so the new S(u) is exactly the modified panel. Always draws k uniform numbers; one more choice call when a gene is replaced."""
    S = decode_topk(u, k)
    flip = rng.random(k) < rate
    n = int(flip.sum())
    if n == 0:
        return u, 0
    outside = np.setdiff1d(np.arange(len(u)), S)
    new = rng.choice(outside, size=n, replace=False)
    u = u.copy()
    for old, nw in zip(S[flip], new):
        u[old], u[nw] = u[nw], u[old]
    return u, n


def _lehmer(values, weights) -> float:
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    v = np.asarray(values, dtype=float)
    return float(np.sum(w * v ** 2) / np.sum(w * v))


def _tournament(rng, n_arch: int, tour: int, rank_of: dict) -> int:
    cand = rng.integers(0, n_arch, size=tour)
    return int(min(cand, key=lambda m: rank_of[int(m)]))


def run_mpqda_v2(fitness, p0: int, k: int, rng: np.random.Generator, P: int = 30, T: int = 50, eps: float = EPS_LOW,
                 rate_new: float | None = None, tour: int = 3, r1_from: str = "archive", adapt: bool = False,
                 F_fixed: float = F_FIXED, CR_fixed: float = CR_FIXED, A_max: int = 15, delta: float = 0.2,
                 n_jobs: int = 1) -> dict:
    """One run of CIRK-DE (v2) for one outer fold and one k. Returns the same keys as run_mpqda of v1."""
    if r1_from not in ("archive", "population") or tour < 1:
        raise ValueError((r1_from, tour))
    rate = (1.0 / k) if rate_new is None else float(rate_new)
    X = rng.random((P, p0))
    panels = [decode_topk(X[i], k) for i in range(P)]
    X = np.array([gapped_band_encode(X[i], k, eps) for i in range(P)])
    f = np.array(fitness.evaluate_many(panels, n_jobs))
    arch = Archive(A_max=A_max, delta=delta)
    for i in range(P):
        arch.offer(panels[i], f[i], X[i])
    MF, MCR, pos = np.full(H_MEM, M_INIT), np.full(H_MEM, CR_FIXED), 0
    hist = [history_row(0, fitness.n_eval, panels, f, n_ties_accepted=0, mean_F=F_fixed, sd_F=0.0,
                        archive_size=len(arch.members), mean_CR=CR_fixed, n_new_genes=0.0)]
    for t in range(1, T + 1):
        if adapt:
            r = rng.integers(0, H_MEM, size=P)
            F = np.empty(P)
            for i in range(P):
                Fi = 0.0
                while Fi <= 0.0:
                    Fi = MF[r[i]] + F_SCALE * np.tan(np.pi * (rng.random() - 0.5))
                F[i] = min(Fi, F_MAX)
            CR = np.clip(MCR[r] + CR_SD * rng.standard_normal(P), 0.0, 1.0)
        else:
            F, CR = np.full(P, F_fixed), np.full(P, CR_fixed)
        n_arch = len(arch.members)
        rank_of = {m: r_ for r_, m in enumerate(arch.ranked())}
        XL, lead_panels, from_best = np.empty((P, p0)), [], 0
        for i in range(P):
            j = _tournament(rng, n_arch, tour, rank_of)
            XL[i] = arch.members[j]["x"]
            lead_panels.append(arch.members[j]["key"])
            from_best += int(rank_of[j] == 0)
        R = np.empty((P, 2), dtype=int)
        for i in range(P):
            R[i] = rng.choice(np.delete(np.arange(P), i), size=2, replace=False)
        X1 = X[R[:, 0]].copy()
        if r1_from == "archive":
            for i in range(P):
                X1[i] = arch.members[_tournament(rng, n_arch, tour, rank_of)]["x"]
        U = rng.random((P, p0))
        jr = rng.integers(0, p0, size=P)
        V = repair_midpoint(XL + F[:, None] * (X1 - X[R[:, 1]]), X)
        mask = U <= CR[:, None]
        mask[np.arange(P), jr] = True
        Ut = np.where(mask, V, X)
        n_new = 0
        for i in range(P):
            Ut[i], m = innovate(Ut[i], k, rate, rng)
            n_new += m
        trial = [decode_topk(Ut[i], k) for i in range(P)]
        fu = np.array(fitness.evaluate_many(trial, n_jobs))
        improved, accepted = fu > f, fu >= f
        explore = exploration_rate(trial, panels)
        overlap = float(np.mean([len(set(trial[i].tolist()) & set(lead_panels[i])) / k for i in range(P)]))
        if adapt and improved.any():
            w = fu[improved] - f[improved]
            MF[pos] = _lehmer(F[improved], w)
            MCR[pos] = float(np.sum(w / w.sum() * CR[improved]))
            pos = (pos + 1) % H_MEM
        for i in range(P):                                        # selection then archive, in index order
            if accepted[i]:
                X[i] = gapped_band_encode(Ut[i], k, eps)
                f[i], panels[i] = fu[i], trial[i]
                arch.offer(panels[i], f[i], X[i])
        hist.append(history_row(
            t, fitness.n_eval, panels, f, success_rate=float(improved.mean()), exploration_rate=explore,
            n_ties_accepted=int((accepted & ~improved).sum()), mean_F=float(F.mean()), sd_F=float(F.std()),
            archive_size=len(arch.members), leader_is_archive_best=from_best / P, leader_overlap=overlap,
            mean_CR=float(CR.mean()), n_new_genes=n_new / P))
    ibest = int(np.argmax(f))
    S_c, pi = arch.consensus(k)
    f_c = fitness.evaluate_many([S_c])[0]                          # counted in n_eval if the panel is new
    return {"consensus": S_c, "consensus_fitness": float(f_c), "best": np.asarray(panels[ibest]),
            "best_fitness": float(f[ibest]), "pi": pi,
            "archive": [(arch.members[i]["key"], arch.members[i]["f"]) for i in arch.ranked()], "history": hist,
            "n_eval": fitness.n_eval, "Q": None, "MF": MF.copy() if adapt else None, "MCR": MCR.copy() if adapt else None}


def run_de_gapped(fitness, p0: int, k: int, rng: np.random.Generator, P: int = 30, T: int = 50, F: float = F_FIXED,
                  CR: float = CR_FIXED, eps: float = EPS_LOW, rate_new: float | None = None, n_jobs: int = 1) -> dict:
    """Baseline DE-G: DE/rand/1/bin with gap re-encoding and new-gene mutation; no archive, no tournament, fixed F,
    CR. Separates the contribution of the representation from the other components of v2. Order of random draws as in
    run_de_rand1 of the paper, with the new-gene mutation step added after crossover."""
    rate = (1.0 / k) if rate_new is None else float(rate_new)
    X = rng.random((P, p0))
    panels = [decode_topk(X[i], k) for i in range(P)]
    X = np.array([gapped_band_encode(X[i], k, eps) for i in range(P)])
    f = np.array(fitness.evaluate_many(panels, n_jobs))
    hist = [history_row(0, fitness.n_eval, panels, f, mean_F=F, sd_F=0.0)]
    for t in range(1, T + 1):
        R = np.empty((P, 3), dtype=int)
        for i in range(P):
            R[i] = rng.choice(np.delete(np.arange(P), i), size=3, replace=False)
        U = rng.random((P, p0))
        jr = rng.integers(0, p0, size=P)
        V = repair_midpoint(X[R[:, 0]] + F * (X[R[:, 1]] - X[R[:, 2]]), X)
        mask = U <= CR
        mask[np.arange(P), jr] = True
        Ut = np.where(mask, V, X)
        for i in range(P):
            Ut[i], _ = innovate(Ut[i], k, rate, rng)
        trial = [decode_topk(Ut[i], k) for i in range(P)]
        fu = np.array(fitness.evaluate_many(trial, n_jobs))
        improved, accepted = fu > f, fu >= f
        explore = exploration_rate(trial, panels)
        for i in range(P):
            if accepted[i]:
                X[i] = gapped_band_encode(Ut[i], k, eps)
                f[i], panels[i] = fu[i], trial[i]
        hist.append(history_row(t, fitness.n_eval, panels, f, success_rate=float(improved.mean()), exploration_rate=explore,
                                n_ties_accepted=int((accepted & ~improved).sum()), mean_F=F, sd_F=0.0))
    ibest = int(np.argmax(f))
    return {"best": np.asarray(panels[ibest]), "best_fitness": float(f[ibest]), "history": hist, "n_eval": fitness.n_eval}
