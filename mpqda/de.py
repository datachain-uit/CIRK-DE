"""Vòng lặp MPQDA-DE và các cờ ablation (Mục 3.7; PREREG Mục 3.3, Algorithm 1).

Thứ tự rút số ngẫu nhiên là một phần của đặc tả. Khởi tạo: khoá (init_chaotic: rng.uniform(size=P); init_uniform:
rng.random((P, p0))), rồi F_i ban đầu (rng.uniform(size=P); bỏ ở A3). Mỗi thế hệ:
  1. bộ điều khiển rút a_i cho i = 0..P − 1, mỗi i một rng.random() (bỏ ở A3);
  2. leader cho i = 0..P − 1 từ archive ở đầu thế hệ, mỗi i một rng.random() (A2a vẫn rút để giữ thứ tự số);
  3. r1 ≠ r2 ≠ i cho i = 0..P − 1: rng.choice(np.delete(np.arange(P), i), size=2, replace=False);
  4. U = rng.random((P, p0)) rồi j_rand = rng.integers(0, p0, size=P);
  5. v_i = repair_midpoint(x_L + F_i'·(x_r1 − x_r2), x_i); u_ij = v_ij nếu U_ij ≤ CR hoặc j = j_rand, ngược lại x_ij;
  6. chấm f(topk(u_i)) cho cả thế hệ bằng evaluate_many (mọi số ngẫu nhiên đã rút xong ở tiến trình chính);
  7. cập nhật bộ điều khiển; với i có f(u_i) ≥ f(x_i): x_i ← band_encode(u_i), cập nhật f_i, panel, đưa vào archive;
  8. ghi lịch sử của thế hệ.
"""
from __future__ import annotations

import numpy as np

from . import config as M
from .archive import Archive
from .control import QController
from .encoding import band_encode, decode_topk, init_chaotic, init_uniform, jaccard_distance, repair_midpoint

Q_COLUMNS = tuple(f"q_{s}{a}" for s in range(2) for a in range(len(M.ACTIONS)))
HISTORY_COLUMNS = ("t", "n_eval", "best_fitness", "mean_fitness", "diversity", "success_rate", "exploration_rate",
                   "n_ties_accepted", "mean_F", "sd_F", "archive_size", "leader_is_archive_best", "leader_overlap",
                   "n_best_ties", *Q_COLUMNS, "mean_CR", "n_new_genes")   # hai cột cuối: v2 (Review-2026-11-16)


def diversity(panels) -> float:
    """Độ đa dạng của quần thể: trung bình khoảng cách Jaccard giữa mọi cặp panel."""
    n = len(panels)
    return float(np.mean([jaccard_distance(panels[i], panels[j]) for i in range(n) for j in range(i + 1, n)]))


def n_best_ties(panels, f) -> int:
    """Số panel KHÁC NHAU trong quần thể có fitness bằng đúng giá trị tốt nhất (đo vùng phẳng, RQ1)."""
    f = np.asarray(f, dtype=float)
    return len({tuple(int(g) for g in panels[i]) for i in np.flatnonzero(f == f.max())})


def exploration_rate(trial, prev_panels, threshold: float = M.EXPLORE_THRESHOLD) -> float:
    """Tỉ lệ panel thử có khoảng cách Jaccard tới panel gần nhất của quần thể trước lớn hơn ngưỡng."""
    return float(np.mean([min(jaccard_distance(q, r) for r in prev_panels) > threshold for q in trial]))


def history_row(t, n_eval, panels, f, **extra) -> dict:
    """Một dòng lịch sử với đủ cột của HISTORY_COLUMNS; đại lượng không xác định là NaN."""
    row = {c: np.nan for c in HISTORY_COLUMNS}
    row.update({"t": int(t), "n_eval": int(n_eval), "best_fitness": float(np.max(f)), "mean_fitness": float(np.mean(f)),
                "diversity": diversity(panels), "n_best_ties": n_best_ties(panels, f)})
    row.update(extra)
    return row


def history_from_log(log, P: int, explore_threshold: float = M.EXPLORE_THRESHOLD) -> list[dict]:
    """Lịch sử theo thế hệ cho GA, BPSO của survey (gọi fitness từng panel, P lời gọi mỗi thế hệ): khối t gồm các lời
    gọi [t·P, (t + 1)·P). best_fitness là tốt nhất tính tới hết khối; success_rate, mean_F không xác định (NaN)."""
    out, best, prev = [], -np.inf, None
    for t in range(len(log) // P):
        block = log[t * P:(t + 1) * P]
        panels = [b[0] for b in block]
        vals = np.array([b[1] for b in block])
        best = max(best, float(vals.max()))
        row = history_row(t, block[-1][2], panels, vals)
        row["best_fitness"] = best
        if prev is not None:
            row["exploration_rate"] = exploration_rate(panels, prev, explore_threshold)
        out.append(row)
        prev = panels
    return out


def _q_entries(ctrl) -> dict:
    if ctrl is None:
        return {}
    return {f"q_{s}{a}": float(ctrl.Q[s, a]) for s in range(ctrl.Q.shape[0]) for a in range(ctrl.Q.shape[1])}


def run_mpqda(fitness, p0: int, k: int, rng: np.random.Generator, P: int = M.P, T: int = M.T, CR: float = M.CR,
              init: str = "chaotic", leader: str = "archive", control: str = "q", F_fixed: float = M.FIXED_F,
              A_max: int = M.A_MAX, delta: float = M.DELTA, explore_threshold: float = M.EXPLORE_THRESHOLD,
              reencode: bool = True, n_jobs: int = 1) -> dict:
    """leader ∈ {archive (đầy đủ), best (A2a), random (A2b)}; init ∈ {chaotic, uniform (A1)}; control ∈ {q, fixed (A3)};
    A_max = 1 hoặc 30 cho A5; reencode = False cho A0. Đầu ra chính: panel đồng thuận; A4 = cá thể tốt nhất của cùng lần
    chạy (hoà thì chỉ số nhỏ nhất)."""
    if init not in ("chaotic", "uniform") or leader not in ("archive", "best", "random") or control not in ("q", "fixed"):
        raise ValueError((init, leader, control))
    X = init_chaotic(P, p0, rng) if init == "chaotic" else init_uniform(P, p0, rng)
    ctrl = QController(P, rng) if control == "q" else None
    panels = [decode_topk(X[i], k) for i in range(P)]
    if reencode:
        X = np.array([band_encode(X[i], k) for i in range(P)])
    f = np.array(fitness.evaluate_many(panels, n_jobs))
    arch = Archive(A_max=A_max, delta=delta)
    for i in range(P):
        arch.offer(panels[i], f[i], X[i])
    F_now = ctrl.F.copy() if ctrl else np.full(P, F_fixed)
    hist = [history_row(0, fitness.n_eval, panels, f, n_ties_accepted=0, mean_F=float(F_now.mean()),
                        sd_F=float(F_now.std()), archive_size=len(arch.members), **_q_entries(ctrl))]
    for t in range(1, T + 1):
        if ctrl:
            a, F_trial = ctrl.propose(rng)
        else:
            a, F_trial = None, np.full(P, F_fixed)
        best_arch = arch.best_index()
        XL, from_best = np.empty((P, p0)), 0
        ibest = int(np.argmax(f))                                  # cá thể tốt nhất; hoà thì chỉ số nhỏ nhất
        for i in range(P):
            u = rng.random()                                       # luôn đúng một số cho mỗi i, kể cả A2a
            if leader == "best":
                XL[i] = X[ibest]
            else:
                j = arch.leader_index(u, "roulette" if leader == "archive" else "uniform")
                XL[i] = arch.members[j]["x"]
                from_best += int(j == best_arch)
        R = np.empty((P, 2), dtype=int)
        for i in range(P):
            R[i] = rng.choice(np.delete(np.arange(P), i), size=2, replace=False)
        U = rng.random((P, p0))
        jr = rng.integers(0, p0, size=P)
        V = repair_midpoint(XL + F_trial[:, None] * (X[R[:, 0]] - X[R[:, 1]]), X)
        mask = U <= CR
        mask[np.arange(P), jr] = True
        Ut = np.where(mask, V, X)
        trial = [decode_topk(Ut[i], k) for i in range(P)]
        fu = np.array(fitness.evaluate_many(trial, n_jobs))
        improved, accepted = fu > f, fu >= f
        explore = exploration_rate(trial, panels, explore_threshold)
        if ctrl:
            ctrl.update(a, improved, accepted, F_trial)
            F_now = ctrl.F.copy()
        leader_overlap = float(np.mean([len(set(trial[i].tolist()) & set(decode_topk(XL[i], k).tolist())) / k
                                        for i in range(P)]))
        for i in range(P):                                         # chọn lọc rồi archive, theo thứ tự chỉ số
            if accepted[i]:
                X[i] = band_encode(Ut[i], k) if reencode else Ut[i]
                f[i], panels[i] = fu[i], trial[i]
                arch.offer(panels[i], f[i], X[i])
        hist.append(history_row(
            t, fitness.n_eval, panels, f, success_rate=float(improved.mean()), exploration_rate=explore,
            n_ties_accepted=int((accepted & ~improved).sum()), mean_F=float(F_now.mean()), sd_F=float(F_now.std()),
            archive_size=len(arch.members), leader_is_archive_best=np.nan if leader == "best" else from_best / P,
            leader_overlap=leader_overlap, **_q_entries(ctrl)))
    ibest = int(np.argmax(f))
    S_c, pi = arch.consensus(k)
    f_c = fitness.evaluate_many([S_c])[0]                          # tính vào n_eval nếu là panel mới
    return {"consensus": S_c, "consensus_fitness": float(f_c), "best": np.asarray(panels[ibest]),
            "best_fitness": float(f[ibest]), "pi": pi,
            "archive": [(arch.members[i]["key"], arch.members[i]["f"]) for i in arch.ranked()], "history": hist,
            "n_eval": fitness.n_eval, "Q": None if ctrl is None else ctrl.Q.copy()}
