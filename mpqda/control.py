"""Q-learning controller of F with a per-individual state (PREREG, Section 3.3, steps 4a, 4g, 4h).

State s_i = 0 if the previous trial of individual i improved, 1 otherwise (initially 1). One shared 2 × 3 Q table.
Draw order: `__init__` draws the initial F_i with one call rng.uniform(size=P) (right after the initial keys); `propose` draws
exactly one number rng.random() for each i in index order.
"""
from __future__ import annotations

import numpy as np

from . import config as M


class QController:
    ACTIONS = np.array(M.ACTIONS)

    def __init__(self, P: int, rng: np.random.Generator, eta: float = M.ETA, gamma: float = M.GAMMA, tau: float = M.TAU,
                 F_bounds=M.F_BOUNDS, F_init=M.F_INIT):
        self.P, self.eta, self.gamma, self.tau, self.F_bounds = P, eta, gamma, tau, F_bounds
        self.Q = np.zeros((2, len(self.ACTIONS)))
        self.s = np.ones(P, dtype=int)
        self.F = rng.uniform(F_init[0], F_init[1], size=P)

    def probs(self, s: int) -> np.ndarray:
        """softmax(Q[s, ·] / tau)."""
        z = self.Q[s] / self.tau
        e = np.exp(z - z.max())
        return e / e.sum()

    def propose(self, rng: np.random.Generator):
        """a_i ~ softmax(Q[s_i, ·]/tau) by the cumulative distribution function; F_i' = clip(F_i + a_i). Returns (a, F')."""
        a = np.empty(self.P, dtype=int)
        last = len(self.ACTIONS) - 1
        for i in range(self.P):
            cdf = np.cumsum(self.probs(self.s[i]))
            a[i] = min(int(np.searchsorted(cdf, rng.random(), side="right")), last)
        return a, np.clip(self.F + self.ACTIONS[a], *self.F_bounds)

    def update(self, a, improved, accepted, F_trial) -> None:
        """After evaluating the whole generation, in order of i: r_i = 1 if f(u_i) > f(x_i); s_i' = 0 if r_i = 1, otherwise 1;
        Q[s_i, a_i] += eta·(r_i + gamma·max Q[s_i', ·] − Q[s_i, a_i]); s_i ← s_i'; F_i ← F_i' only when u_i is accepted."""
        for i in range(self.P):
            r = 1.0 if improved[i] else 0.0
            s_next = 0 if improved[i] else 1
            q = self.Q[self.s[i], a[i]]
            self.Q[self.s[i], a[i]] = q + self.eta * (r + self.gamma * self.Q[s_next].max() - q)
            self.s[i] = s_next
            if accepted[i]:
                self.F[i] = F_trial[i]
