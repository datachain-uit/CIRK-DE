"""Archive, leader và panel đồng thuận (Mục 3.5; PREREG Mục 3.4, 3.5).

Quy tắc (số trong ngoặc là số quy tắc của đặc tả):
(1) panel trùng hệt một thành viên: bỏ qua; (2) có thành viên cách ứng viên dưới delta: lấy thành viên gần nhất (hoà:
fitness cao hơn, rồi vào sớm hơn), thay tại chỗ khi fitness của ứng viên cao hơn hẳn, không thì loại ứng viên; (3) không
có thành viên gần: thêm, và nếu vượt A_max thì bỏ thành viên có 0,5·novelty + 0,5·fitness nhỏ nhất (chuẩn hoá min–max,
kể cả ứng viên), không bao giờ bỏ thành viên tốt nhất; điểm bằng nhau: bỏ fitness thấp hơn, rồi thành viên mới hơn;
(4) novelty = trung bình khoảng cách Jaccard tới 3 thành viên gần nhất; (5) leader: một số u cho mỗi lần chọn.
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
        """Thành viên tốt nhất: fitness cao nhất, hoà thì vào archive sớm hơn."""
        members = self.members if members is None else members
        return min(range(len(members)), key=lambda i: (-members[i]["f"], members[i]["order"]))

    def ranked(self) -> list[int]:
        """Chỉ số thành viên theo hạng fitness (1 = tốt nhất; hoà thì vào sớm hơn đứng trước)."""
        return sorted(range(len(self.members)), key=lambda i: (-self.members[i]["f"], self.members[i]["order"]))

    def offer(self, panel, f: float, x) -> str:
        """Đưa (panel, fitness, khoá đã mã hoá lại) vào archive. Trả về duplicate, replaced, rejected_near, added hoặc
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
                    self.members[j] = self._new(key, f, x)       # thay tại chỗ, nhận dấu thời gian mới
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
        """Một số u ∈ [0, 1) cho mỗi lần chọn. roulette: trọng số 0,7·novelty + 0,3·fitness (chuẩn hoá min–max trong
        archive; tổng bằng 0 thì chọn đều). uniform (A2b): int(u·|A|)."""
        n = len(self.members)
        if mode == "uniform" or n == 1:
            return min(int(u * n), n - 1)
        w = self.w_leader[0] * _minmax(self.novelty()) + self.w_leader[1] * _minmax([m["f"] for m in self.members])
        if w.sum() <= 0:
            w = np.ones(n)
        cdf = np.cumsum(w) / w.sum()
        return min(int(np.searchsorted(cdf, u, side="right")), n - 1)

    def consensus(self, k: int):
        """Panel đồng thuận: w_a = |A| − hạng fitness + 1; pi_j = Σ_a w_a·1[j ∈ S_a] / Σ_a w_a; lấy k gen có pi_j lớn nhất.
        Hoà: gen của thành viên tốt nhất trước, rồi khoá trung bình (trên các thành viên) lớn hơn, rồi chỉ số nhỏ hơn.
        Trả về (panel đã sắp, {gen: pi})."""
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
