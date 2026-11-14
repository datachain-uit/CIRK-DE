"""Biểu diễn random key và giải mã đúng k gen (Mục 3.3; PREREG Mục 3.1).

Cá thể là vector khoá x ∈ (0, 1)^p0. Panel S(x) là k chỉ số có khoá lớn nhất; khoá bằng nhau thì chỉ số nhỏ hơn đứng
trước, nên mọi cá thể đều khả thi và |S(x)| = k.
"""
from __future__ import annotations

import numpy as np

from . import config as M


def decode_topk(x, k: int) -> np.ndarray:
    """k chỉ số có khoá lớn nhất; khoá bằng nhau thì chỉ số nhỏ hơn đứng trước. Trả về mảng tăng dần."""
    order = np.argsort(-np.asarray(x, dtype=float), kind="stable")
    return np.sort(order[:k])


def jaccard_distance(a, b) -> float:
    """d(a, b) = 1 − |a ∩ b| / |a ∪ b|; hai tập rỗng cho 0."""
    a, b = set(map(int, a)), set(map(int, b))
    u = len(a | b)
    return 0.0 if u == 0 else 1.0 - len(a & b) / u


def init_chaotic(P: int, p: int, rng: np.random.Generator, mu: float = M.MU, z0_range=M.Z0_RANGE) -> np.ndarray:
    """Bản đồ logistic z(t + 1) = mu·z(t)·(1 − z(t)); z_i(0) ~ U(0,001; 0,999), một giá trị cho mỗi cá thể; khoá của gen j
    là z_i(j + 1). Dùng đúng một lời gọi rng.uniform(size=P)."""
    z = rng.uniform(z0_range[0], z0_range[1], size=P)
    X = np.empty((P, p))
    for j in range(p):
        z = mu * z * (1.0 - z)
        X[:, j] = z
    return X


def init_uniform(P: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Khoá đều U(0, 1): biến thể A1 và mọi baseline DE. Dùng đúng một lời gọi rng.random((P, p))."""
    return rng.random((P, p))


def repair_midpoint(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sửa biên trung điểm (Đ1, như JADE, SHADE): v > 1 thành (1 + x)/2, v < 0 thành x/2, với x là cá thể ĐÍCH (cùng
    hình dạng với v), không phải leader."""
    v = np.where(v > 1.0, (1.0 + x) / 2.0, v)
    return np.where(v < 0.0, x / 2.0, v)


def band_encode(x: np.ndarray, k: int) -> np.ndarray:
    """Mã hoá lại theo dải (Đ2): gen của S(x) nhận 0,5 + 0,5·(r + 0,5)/k, gen ngoài nhận 0,5·(r + 0,5)/(p0 − k), với r là
    hạng ổn định của khoá trong nhóm. Panel và thứ tự khoá trong từng nhóm không đổi."""
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
