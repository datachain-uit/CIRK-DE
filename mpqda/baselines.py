"""Wrapper baselines under the same budget and the same fitness (PREREG, Section 4).

All DE baselines share: uniform keys `rng.random((P, p0))`, band re-encoding, midpoint bound repair against the target
individual, binomial crossover with j_rand, selection u_i replaces x_i when f(u_i) ≥ f(x_i), output is the best individual at the stop
(ties: smallest index). GA and BPSO call the vendored code in vendor/ directly with the new fitness object.
The order of random draws of each baseline is given in the docstring of its function.
"""
from __future__ import annotations

import numpy as np
from survey_v2 import wrappers as W

from . import config as M
from .de import exploration_rate, history_from_log, history_row
from .encoding import band_encode, decode_topk, init_uniform, repair_midpoint


class _Population:
    """Common part of the DE baselines: initialisation, binomial crossover, evaluation, ≥ selection and history."""

    def __init__(self, fitness, p0, k, rng, P, n_jobs):
        self.fitness, self.p0, self.k, self.P, self.n_jobs = fitness, p0, k, P, n_jobs
        X = init_uniform(P, p0, rng)
        self.panels = [decode_topk(X[i], k) for i in range(P)]
        self.X = np.array([band_encode(X[i], k) for i in range(P)])
        self.f = np.array(fitness.evaluate_many(self.panels, n_jobs))
        self.t = 0
        self.history = []

    def record(self, **extra):
        self.history.append(history_row(self.t, self.fitness.n_eval, self.panels, self.f, **extra))

    def step(self, V, CR, U, jr):
        """Binomial crossover with CR (scalar or per-individual vector), evaluation, ≥ selection. Returns (improved, accepted, old X, old f, fu)."""
        P = self.P
        V = repair_midpoint(V, self.X)
        mask = U <= (np.asarray(CR, dtype=float).reshape(-1, 1) if np.ndim(CR) else CR)
        mask[np.arange(P), jr] = True
        Ut = np.where(mask, V, self.X)
        trial = [decode_topk(Ut[i], self.k) for i in range(P)]
        fu = np.array(self.fitness.evaluate_many(trial, self.n_jobs))
        improved, accepted = fu > self.f, fu >= self.f
        self._explore = exploration_rate(trial, self.panels)
        X_old, f_old = self.X.copy(), self.f.copy()
        for i in np.flatnonzero(accepted):
            self.X[i], self.f[i], self.panels[i] = band_encode(Ut[i], self.k), fu[i], trial[i]
        self.t += 1
        self._stats = {"success_rate": float(improved.mean()), "exploration_rate": self._explore,
                       "n_ties_accepted": int((accepted & ~improved).sum())}
        return improved, accepted, X_old, f_old, fu

    def result(self, **extra) -> dict:
        ibest = int(np.argmax(self.f))
        return {"best": np.asarray(self.panels[ibest]), "best_fitness": float(self.f[ibest]), "history": self.history,
                "n_eval": self.fitness.n_eval, **extra}


def _others(rng, P, i, size):
    return rng.choice(np.delete(np.arange(P), i), size=size, replace=False)


def _crossover_numbers(rng, P, p0):
    U = rng.random((P, p0))
    return U, rng.integers(0, p0, size=P)


def _cauchy_F(rng, loc, scale=M.SHADE_SIGMA):
    """F ~ Cauchy(loc; scale): redrawn when ≤ 0, set to 1 when larger than 1 (JADE, SHADE)."""
    while True:
        F = loc + scale * rng.standard_cauchy()
        if F > 0:
            return min(F, 1.0)


def _F_stats(F):
    F = np.asarray(F, dtype=float)
    return {"mean_F": float(F.mean()), "sd_F": float(F.std())}


# ---------------------------------------------------------------------------------------------------
# DE/rand/1/bin (Storn and Price, 1997)
# ---------------------------------------------------------------------------------------------------
def run_de_rand1(fitness, p0: int, k: int, rng: np.random.Generator, P: int = M.P, T: int = M.T,
                 F: float = M.DE_RAND1_F, CR: float = M.CR, n_jobs: int = 1) -> dict:
    """v_i = x_r1 + F·(x_r2 − x_r3), r1, r2, r3 distinct and different from i.
    Draw order: uniform keys rng.random((P, p0)); each generation: (r1, r2, r3) for i = 0..P − 1 by
    rng.choice(np.delete(np.arange(P), i), size=3, replace=False); U = rng.random((P, p0)); j_rand = rng.integers(0, p0, size=P)."""
    pop = _Population(fitness, p0, k, rng, P, n_jobs)
    pop.record(n_ties_accepted=0, **_F_stats(np.full(P, F)))
    for _ in range(T):
        R = np.array([_others(rng, P, i, 3) for i in range(P)])
        U, jr = _crossover_numbers(rng, P, p0)
        pop.step(pop.X[R[:, 0]] + F * (pop.X[R[:, 1]] - pop.X[R[:, 2]]), CR, U, jr)
        pop.record(**pop._stats, **_F_stats(np.full(P, F)))
    return pop.result()


# ---------------------------------------------------------------------------------------------------
# jDE (Brest et al., 2006) — tier 2
# ---------------------------------------------------------------------------------------------------
def run_jde(fitness, p0: int, k: int, rng: np.random.Generator, P: int = M.P, T: int = M.T, tau=M.JDE_TAU,
            F_range=M.JDE_F_RANGE, init=M.JDE_INIT, n_jobs: int = 1) -> dict:
    """F_i = 0.5, CR_i = 0.9 initially. Each i draws four uniform numbers (rand1, rand2, rand3, rand4): F_i' = F_l + F_u·rand1 when
    rand2 < tau1, otherwise F_i; CR_i' = rand3 when rand4 < tau2, otherwise CR_i. DE/rand/1/bin with F_i', CR_i'; the new parameters
    are kept when u_i is accepted (f(u_i) ≥ f(x_i)).
    Draw order: uniform keys; each generation: rng.random(4) for i = 0..P − 1; then (r1, r2, r3) for i = 0..P − 1; U; j_rand."""
    pop = _Population(fitness, p0, k, rng, P, n_jobs)
    F, CRv = np.full(P, init[0]), np.full(P, init[1])
    pop.record(n_ties_accepted=0, **_F_stats(F))
    for _ in range(T):
        F_new, CR_new = F.copy(), CRv.copy()
        for i in range(P):
            r = rng.random(4)
            if r[1] < tau[0]:
                F_new[i] = F_range[0] + F_range[1] * r[0]
            if r[3] < tau[1]:
                CR_new[i] = r[2]
        R = np.array([_others(rng, P, i, 3) for i in range(P)])
        U, jr = _crossover_numbers(rng, P, p0)
        V = pop.X[R[:, 0]] + F_new[:, None] * (pop.X[R[:, 1]] - pop.X[R[:, 2]])
        _, accepted, *_ = pop.step(V, CR_new, U, jr)
        F[accepted], CRv[accepted] = F_new[accepted], CR_new[accepted]
        pop.record(**pop._stats, **_F_stats(F))
    return pop.result(F=F.copy(), CR=CRv.copy())


# ---------------------------------------------------------------------------------------------------
# current-to-pbest/1 with an external archive: JADE (Zhang and Sanderson, 2009) and SHADE (Tanabe and Fukunaga, 2013)
# ---------------------------------------------------------------------------------------------------
def _pbest_mutation(pop, rng, F, n_pbest, ext):
    """v_i = x_i + F_i·(x_pbest − x_i) + F_i·(x_r1 − x̃_r2). For i = 0..P − 1 in order: pbest = rng.integers(0, n_pbest_i)
    among the best individuals (stable sort by decreasing fitness); r1 = rng.choice over the population except i; r2 = rng.integers over
    population ∪ external archive, except i and r1 (order: population indices ascending, then the archive in entry order)."""
    P, X = pop.P, pop.X
    order = np.argsort(-pop.f, kind="stable")
    V = np.empty_like(X)
    for i in range(P):
        pb = order[int(rng.integers(0, n_pbest[i]))]
        r1 = int(rng.choice(np.delete(np.arange(P), i)))
        cand = [j for j in range(P) if j not in (i, r1)]
        j2 = int(rng.integers(0, len(cand) + len(ext)))
        x2 = X[cand[j2]] if j2 < len(cand) else ext[j2 - len(cand)]
        V[i] = X[i] + F[i] * (X[pb] - X[i]) + F[i] * (X[r1] - x2)
    return V


def _archive_successes(rng, ext, X_old, improved, size):
    """Old x_i of successful trials enter the external archive in order of i; when oversized, one element drawn uniformly is removed (rng.integers)."""
    for i in np.flatnonzero(improved):
        ext.append(X_old[i].copy())
        if len(ext) > size:
            ext.pop(int(rng.integers(0, len(ext))))


def run_jade(fitness, p0: int, k: int, rng: np.random.Generator, P: int = M.P, T: int = M.T, p_best: float = M.JADE_P,
             p_min: int = M.JADE_P_MIN, c: float = M.JADE_C, archive_size: int | None = None, n_jobs: int = 1) -> dict:
    """mu_CR = mu_F = 0.5; CR_i ~ N(mu_CR; 0.1) clipped to [0, 1]; F_i ~ Cauchy(mu_F; 0.1) (redrawn when ≤ 0, set to 1 when larger than 1);
    current-to-pbest/1 with the max(2, ⌈0.05·P⌉) best individuals and an external archive of size P. When there are successful trials (f(u) > f(x)):
    mu_CR ← 0,9·mu_CR + 0,1·mean(S_CR); mu_F ← 0,9·mu_F + 0,1·mean_Lehmer(S_F).
    Draw order: uniform keys; each generation: for i = 0..P − 1: CR_i (rng.normal), F_i (rng.standard_cauchy, repeated until > 0);
    then for i = 0..P − 1: pbest, r1, r2; U; j_rand; after evaluation: the element removed from the external archive (if full)."""
    archive_size = P if archive_size is None else archive_size       # default from the actual P (M.DE_ARCHIVE_SIZE = M.P)
    pop = _Population(fitness, p0, k, rng, P, n_jobs)
    mu_cr = mu_f = 0.5
    n_pb = np.full(P, max(p_min, int(np.ceil(p_best * P))))
    ext: list[np.ndarray] = []
    pop.record(n_ties_accepted=0, archive_size=0, mean_F=mu_f, sd_F=0.0)
    for _ in range(T):
        CRv, F = np.empty(P), np.empty(P)
        for i in range(P):
            CRv[i] = np.clip(rng.normal(mu_cr, M.SHADE_SIGMA), 0.0, 1.0)
            F[i] = _cauchy_F(rng, mu_f)
        V = _pbest_mutation(pop, rng, F, n_pb, ext)
        U, jr = _crossover_numbers(rng, P, p0)
        improved, _, X_old, _, _ = pop.step(V, CRv, U, jr)
        _archive_successes(rng, ext, X_old, improved, archive_size)
        if improved.any():
            s_f = F[improved]
            mu_cr = (1 - c) * mu_cr + c * float(CRv[improved].mean())
            mu_f = (1 - c) * mu_f + c * float((s_f ** 2).sum() / s_f.sum())
        pop.record(**pop._stats, archive_size=len(ext), **_F_stats(F))
    return pop.result(mu_CR=mu_cr, mu_F=mu_f, ext_archive_size=len(ext))


def run_shade(fitness, p0: int, k: int, rng: np.random.Generator, P: int = M.P, T: int = M.T, H: int | None = None,
              p_range=None, archive_size: int | None = None, n_jobs: int = 1) -> dict:
    """Memory of H cells M_CR, M_F initialised to 0.5. For each i: cell r_i drawn uniformly; CR_i ~ N(M_CR[r_i]; 0.1) clipped to [0, 1];
    F_i ~ Cauchy(M_F[r_i]; 0.1) (redrawn when ≤ 0, set to 1 when larger than 1); p_i ~ U(2/P; 0.2); current-to-pbest/1 with the
    max(2, round(p_i·P)) best individuals and an external archive of size P. Successful trial (f(u) > f(x)): old x_i enters the external archive,
    CR_i, F_i and Δf are recorded. At the end of a generation with successes: M_CR[pos] = weighted mean, M_F[pos] = weighted Lehmer
    mean (weights Δf/ΣΔf); pos advances cyclically.
    Draw order: uniform keys; each generation: for i = 0..P − 1: r_i (rng.integers(0, H)), CR_i (rng.normal), F_i
    (rng.standard_cauchy, repeated until > 0), p_i (rng.random); then for i = 0..P − 1: pbest, r1, r2; U; j_rand; after evaluation:
    the element removed from the external archive (if full)."""
    H = P if H is None else H                                       # default from the actual P (M.SHADE_H = M.P)
    p_range = (2 / P, M.SHADE_P_RANGE[1]) if p_range is None else p_range
    archive_size = P if archive_size is None else archive_size
    pop = _Population(fitness, p0, k, rng, P, n_jobs)
    M_cr, M_f, pos = np.full(H, 0.5), np.full(H, 0.5), 0
    ext: list[np.ndarray] = []
    pop.record(n_ties_accepted=0, archive_size=0, mean_F=0.5, sd_F=0.0)
    for _ in range(T):
        CRv, F, n_pb = np.empty(P), np.empty(P), np.empty(P, dtype=int)
        for i in range(P):
            r = int(rng.integers(0, H))
            CRv[i] = np.clip(rng.normal(M_cr[r], M.SHADE_SIGMA), 0.0, 1.0)
            F[i] = _cauchy_F(rng, M_f[r])
            p_i = p_range[0] + (p_range[1] - p_range[0]) * rng.random()    # U(2/P; 0.2); P < 10 (smoke test) swaps the two bounds
            n_pb[i] = min(P, max(2, int(round(p_i * P))))
        V = _pbest_mutation(pop, rng, F, n_pb, ext)
        U, jr = _crossover_numbers(rng, P, p0)
        improved, _, X_old, f_old, fu = pop.step(V, CRv, U, jr)
        _archive_successes(rng, ext, X_old, improved, archive_size)
        if improved.any():
            w = (fu - f_old)[improved]
            w = w / w.sum()
            s_f = F[improved]
            M_cr[pos] = float((w * CRv[improved]).sum())
            M_f[pos] = float((w * s_f ** 2).sum() / (w * s_f).sum())
            pos = (pos + 1) % H
        pop.record(**pop._stats, archive_size=len(ext), **_F_stats(F))
    return pop.result(M_CR=M_cr.copy(), M_F=M_f.copy(), ext_archive_size=len(ext))


# ---------------------------------------------------------------------------------------------------
# qlDE (Huynh, Do and Lee, 2021) — configuration taken from the full text, docs/taimma/baselines/qlde.md
# ---------------------------------------------------------------------------------------------------
def qlde_alpha(t: int, T: int, start: float, drop: float) -> float:
    """Learning rate of Eq. (7): alpha = start − drop·(eval / MaxEval), with eval = P·(t + 1) after generation t and MaxEval = P·(T + 1)
    (nominal number of evaluations as in Algorithm 2 of the original paper, cache hits not subtracted)."""
    return start - drop * (t + 1) / (T + 1)


def run_qlde(fitness, p0: int, k: int, rng: np.random.Generator, P: int = M.P, T: int = M.T, actions=None,
             gamma: float | None = None, n_jobs: int = 1) -> dict:
    """qlDE following docs/taimma/baselines/qlde.md: DE/rand/1/bin, each generation uses one (F, Cr) pair for the whole population.
    Start of a generation: the action with the largest Q (greedy; ties: smallest index); the Q table starts at 0 and has one state.
    End of a generation: r = +1 if the best fitness of the population strictly increases, otherwise −1 (Eq. 8);
    Q[a] += alpha·(r + gamma·max Q − Q[a]) (Eq. 6), alpha from Eq. (7), gamma = 0.8.
    Differs from the original paper only in the part shared by all DE baselines: uniform keys and top-k decoding, band re-encoding, midpoint bound repair
    (original: reflection), fixed budget P, T (original: an additional convergence criterion), no constraint penalty function.
    Draw order: as DE/rand/1/bin — uniform keys rng.random((P, p0)); each generation: (r1, r2, r3) for i = 0..P − 1 by
    rng.choice(np.delete(np.arange(P), i), size=3, replace=False); U = rng.random((P, p0)); j_rand = rng.integers(0, p0, size=P).
    Action selection uses no random numbers."""
    cfg = M.QLDE
    actions = tuple(cfg["actions"] if actions is None else actions)
    gamma = cfg["gamma"] if gamma is None else gamma
    pop = _Population(fitness, p0, k, rng, P, n_jobs)
    Q = np.full(len(actions), float(cfg["q_init"]))
    chosen, rewards = [], []
    pop.record(n_ties_accepted=0)
    for t in range(1, T + 1):
        a = int(np.argmax(Q))                                      # greedy; ties: smallest index
        F, CR = actions[a]
        best_before = float(pop.f.max())
        R = np.array([_others(rng, P, i, 3) for i in range(P)])
        U, jr = _crossover_numbers(rng, P, p0)
        pop.step(pop.X[R[:, 0]] + F * (pop.X[R[:, 1]] - pop.X[R[:, 2]]), CR, U, jr)
        r = cfg["reward"][0] if float(pop.f.max()) > best_before else cfg["reward"][1]
        alpha = qlde_alpha(t, T, cfg["alpha_start"], cfg["alpha_drop"])
        Q[a] += alpha * (r + gamma * Q.max() - Q[a])
        chosen.append(a)
        rewards.append(r)
        pop.record(**pop._stats, **_F_stats(np.full(P, F)))
    return pop.result(Q=Q.copy(), actions=chosen, rewards=rewards)


# ---------------------------------------------------------------------------------------------------
# GA and BPSO of the vendored code with the new fitness
# ---------------------------------------------------------------------------------------------------
def run_survey_wrapper(method: str, fitness, p0: int, k: int, rng: np.random.Generator, P: int = M.P, T: int = M.T) -> dict:
    """Calls `genetic_algorithm` / `binary_pso` of vendor/survey_v2/wrappers.py unchanged. rng must be
    default_rng([seed_run, k, 0]) for GA and default_rng([seed_run, k, 1]) for BPSO, as in `run_wrapper` of the vendored code.
    The history is built from fitness.log in blocks of P calls."""
    if method == "ga":
        best, best_fit, _ = W.genetic_algorithm(fitness, p0, k, rng, pop=P, generations=T)
    elif method == "bpso":
        best, best_fit, _ = W.binary_pso(fitness, p0, k, rng, pop=P, iterations=T)
    else:
        raise ValueError(method)
    return {"best": np.sort(np.asarray(best, dtype=int)), "best_fitness": float(best_fit),
            "history": history_from_log(fitness.log, P), "n_eval": fitness.n_eval}


RUNNERS = {"de_rand1": run_de_rand1, "shade": run_shade, "qlde": run_qlde, "jde": run_jde, "jade": run_jade}
SURVEY_RNG_TAG = {"ga": 0, "bpso": 1}


def make_rng(method: str, seed_run: int, k: int) -> np.random.Generator:
    """Random generators per k: DE family default_rng([seed_run, k]); GA [seed_run, k, 0]; BPSO [seed_run, k, 1]."""
    if method in SURVEY_RNG_TAG:
        return np.random.default_rng([seed_run, k, SURVEY_RNG_TAG[method]])
    return np.random.default_rng([seed_run, k])


def run_method(method: str, fitness, p0: int, k: int, rng: np.random.Generator, P: int = M.P, T: int = M.T,
               n_jobs: int = 1) -> dict:
    """Run one method (including the T2 variants) with budget P, T. The MPQDA family also returns consensus, pi, archive."""
    from .de import run_mpqda
    from .de_v2 import run_de_gapped, run_mpqda_v2
    if method in M.V2_FAMILY:                                       # v2: candidates, chosen configuration and ablations
        return run_mpqda_v2(fitness, p0, k, rng, P=P, T=T, n_jobs=n_jobs, **M.v2_config(method))
    if method == "de_g":
        return run_de_gapped(fitness, p0, k, rng, P=P, T=T, n_jobs=n_jobs)
    if method in M.MPQDA_FAMILY:
        return run_mpqda(fitness, p0, k, rng, P=P, T=T, n_jobs=n_jobs, **M.VARIANTS.get(method, {}))
    if method in SURVEY_RNG_TAG:
        return run_survey_wrapper(method, fitness, p0, k, rng, P=P, T=T)
    if method in RUNNERS:
        return RUNNERS[method](fitness, p0, k, rng, P=P, T=T, n_jobs=n_jobs)
    raise ValueError(f"unknown method: {method}")
