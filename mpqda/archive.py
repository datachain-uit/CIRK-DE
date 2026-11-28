"""Archive, leader and consensus panel (PREREG, Sections 3.4 and 3.5).

Rules (numbers in parentheses are the rule numbers of the specification):
(1) panel identical to a member: ignore; (2) a member lies closer than delta to the candidate: take the nearest member (ties:
higher fitness, then earlier entry), replace it in place when the candidate's fitness is strictly higher, otherwise reject the candidate; (3) no
near member: add, and if A_max is exceeded remove the member with the smallest 0.5·novelty + 0.5·fitness (min–max normalised,
candidate included), never removing the best member; equal scores: remove the lower fitness, then the newer member;
(4) novelty = mean Jaccard distance to the 3 nearest members; (5) leader: one number u per selection.
"""
from __future__ import annotations

import numpy as np

from . import config as M
from .encoding import jaccard_distance


def _minmax(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)


class Archive:
    def __init__(self, A_max: int = M.A_MAX, delta: float = M.DELTA, n_neighbors: int = M.N_NEIGHBORS,
                 w_leader=M.W_LEADER, w_remove=M.W_REMOVE):
        self.A_max, self.delta, self.n_neighbors = A_max, delta, n_neighbors
        self.w_leader, self.w_remove = w_leader, w_remove
        self.members: list[dict] = []           # dict(key, set, f, x, order)
        self._stamp = 0

    def _new(self, key, f, x):
        self._stamp += 1
        return {"key": key, "set": set(key), "f": float(f), "x": np.array(x, dtype=float), "order": self._stamp}

    def novelty(self, members=None) -> np.ndarray:
        members = self.members if members is None else members
        n = len(members)
        if n < 2:
            return np.zeros(n)
        D = np.array([[jaccard_distance(a["set"], b["set"]) for b in members] for a in members])
        out = np.empty(n)
        for i in range(n):
            d = np.sort(np.delete(D[i], i))
            out[i] = d[: min(self.n_neighbors, n - 1)].mean()
        return out

    def best_index(self, members=None) -> int:
        """Best member: highest fitness, ties broken by earlier entry into the archive."""
        members = self.members if members is None else members
        return min(range(len(members)), key=lambda i: (-members[i]["f"], members[i]["order"]))

    def ranked(self) -> list[int]:
        """Member indices by fitness rank (1 = best; ties: earlier entry first)."""
        return sorted(range(len(self.members)), key=lambda i: (-self.members[i]["f"], self.members[i]["order"]))

    def offer(self, panel, f: float, x) -> str:
        """Put (panel, fitness, re-encoded keys) into the archive. Returns duplicate, replaced, rejected_near, added or
        rejected_full."""
        key = tuple(sorted(int(g) for g in panel))
        if any(m["key"] == key for m in self.members):
            return "duplicate"
        if self.members:
            d = [jaccard_distance(key, m["set"]) for m in self.members]
            near = [i for i, di in enumerate(d) if di < self.delta]
            if near:
                j = min(near, key=lambda i: (d[i], -self.members[i]["f"], self.members[i]["order"]))
                if f > self.members[j]["f"]:
                    self.members[j] = self._new(key, f, x)       # replace in place, takes a new time stamp
                    return "replaced"
                return "rejected_near"
        cand = self._new(key, f, x)
        self.members.append(cand)
        if len(self.members) <= self.A_max:
            return "added"
        nov = _minmax(self.novelty())
        fit = _minmax([m["f"] for m in self.members])
        score = self.w_remove[0] * nov + self.w_remove[1] * fit
        best = self.best_index()
        r = min((i for i in range(len(self.members)) if i != best),
                key=lambda i: (score[i], self.members[i]["f"], -self.members[i]["order"]))
        removed = self.members.pop(r)
        return "rejected_full" if removed is cand else "added"

    def leader_index(self, u: float, mode: str = "roulette") -> int:
        """One number u ∈ [0, 1) per selection. roulette: weight 0.7·novelty + 0.3·fitness (min–max normalised within the
        archive; uniform choice when the weights sum to 0). uniform (A2b): int(u·|A|)."""
        n = len(self.members)
        if mode == "uniform" or n == 1:
            return min(int(u * n), n - 1)
        w = self.w_leader[0] * _minmax(self.novelty()) + self.w_leader[1] * _minmax([m["f"] for m in self.members])
        if w.sum() <= 0:
            w = np.ones(n)
        cdf = np.cumsum(w) / w.sum()
        return min(int(np.searchsorted(cdf, u, side="right")), n - 1)

    def consensus(self, k: int):
        """Consensus panel: w_a = |A| − fitness rank + 1; pi_j = Σ_a w_a·1[j ∈ S_a] / Σ_a w_a; take the k genes with the largest pi_j.
        Ties: genes of the best member first, then larger mean key (over the members), then smaller index.
        Returns (sorted panel, {gene: pi})."""
        n = len(self.members)
        order = self.ranked()
        w = np.empty(n)
        for rank, i in enumerate(order, start=1):
            w[i] = n - rank + 1
        genes = sorted(set().union(*(m["set"] for m in self.members)))
        pi = {j: sum(w[a] for a in range(n) if j in self.members[a]["set"]) / w.sum() for j in genes}
        best_set = self.members[order[0]]["set"]
        mean_key = np.mean([m["x"] for m in self.members], axis=0)
        ranked = sorted(genes, key=lambda j: (-pi[j], -(j in best_set), -mean_key[j], j))
        return np.sort(np.array(ranked[:k], dtype=int)), pi
