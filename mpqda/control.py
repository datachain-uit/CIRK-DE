"""Bộ điều khiển Q-learning của F theo trạng thái từng cá thể (Mục 3.6; PREREG Mục 3.3 bước 4a, 4g, 4h).

Trạng thái s_i = 0 nếu lần thử trước của cá thể i cải thiện, 1 nếu không (ban đầu 1). Một bảng Q 2 × 3 dùng chung.
Thứ tự rút số: `__init__` rút F_i ban đầu bằng một lời gọi rng.uniform(size=P) (ngay sau khoá khởi tạo); `propose` rút
đúng một số rng.random() cho mỗi i theo thứ tự chỉ số.
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
        """a_i ~ softmax(Q[s_i, ·]/tau) bằng hàm phân phối tích luỹ; F_i' = clip(F_i + a_i). Trả về (a, F')."""
        a = np.empty(self.P, dtype=int)
        last = len(self.ACTIONS) - 1
        for i in range(self.P):
            cdf = np.cumsum(self.probs(self.s[i]))
            a[i] = min(int(np.searchsorted(cdf, rng.random(), side="right")), last)
        return a, np.clip(self.F + self.ACTIONS[a], *self.F_bounds)

    def update(self, a, improved, accepted, F_trial) -> None:
        """Sau khi chấm cả thế hệ, theo thứ tự i: r_i = 1 nếu f(u_i) > f(x_i); s_i' = 0 nếu r_i = 1, ngược lại 1;
        Q[s_i, a_i] += eta·(r_i + gamma·max Q[s_i', ·] − Q[s_i, a_i]); s_i ← s_i'; F_i ← F_i' chỉ khi u_i được nhận."""
        for i in range(self.P):
            r = 1.0 if improved[i] else 0.0
            s_next = 0 if improved[i] else 1
            q = self.Q[self.s[i], a[i]]
            self.Q[self.s[i], a[i]] = q + self.eta * (r + self.gamma * self.Q[s_next].max() - q)
            self.s[i] = s_next
            if accepted[i]:
                self.F[i] = F_trial[i]
