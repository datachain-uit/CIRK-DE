"""Khám phá, không đăng ký (Review-2026-11-18c, E3; mở rộng k = 20 theo Review-2026-11-20, TQĐ51): số gen lạ mà toán tử
đột biến + lai ghép đưa vào một panel thử theo ngân sách k, với dải ε = 1/2 (v1 và các baseline DE) và khe ε = 0,01 (v2,
DE-G), F = 0,5, CR = 0,9. Dùng population() và keys() của scripts/disruption_sim.py; seed riêng (3) và tệp ra riêng, nên
disruption.csv và disruption_plain.csv không đổi.

    python scripts/disruption_k.py [--reps 200] [--ks 2 5 10 50 20] [--out docs/taimma/disruption_k.csv]

Một bộ sinh số ngẫu nhiên đi qua các k theo đúng thứ tự của --ks. Thứ tự mặc định 2, 5, 10, 50 cho lại đúng
expected/disruption_k.csv của taimma_ref_v7; đặt k = 20 ở CUỐI (--ks 2 5 10 50 20) thì 16 dòng đầu không đổi và chỉ thêm
bốn dòng của k = 20 (sửa đổi A2 của PREREG-v2 trích các dòng này).

random_genes_per_trial = random_share · k: số gen không thuộc leader, đích hay vector được cộng trong một panel thử.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mpqda  # noqa: E402,F401  (đặt vendor/ và gốc kho vào sys.path)
from mpqda.encoding import decode_topk, repair_midpoint  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disruption_sim import keys, population  # noqa: E402

DEFAULT_KS = (2, 5, 10, 50)


def simulate_k(reps=200, P=30, p0=5000, ks=DEFAULT_KS, F=0.5, CR=0.9, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        for eps in (0.5, 0.01):
            for share in (0.0, 0.5):
                kept_l, rand = [], []
                for _ in range(reps):
                    pans = population(rng, P, p0, k, share)
                    X = np.array([keys(rng, S, p0, k, eps) for S in pans])
                    r1, r2 = rng.choice(np.arange(2, P), size=2, replace=False)
                    v = repair_midpoint(X[0] + F * (X[r1] - X[r2]), X[1])
                    m = rng.random(p0) <= CR
                    m[rng.integers(p0)] = True
                    S = set(decode_topk(np.where(m, v, X[1]), k).tolist())
                    SL, ST, S1 = (set(pans[j].tolist()) for j in (0, 1, r1))
                    kept_l.append(len(S & SL) / k)
                    rand.append(len(S - SL - ST - S1) / k)
                rows.append({"k": k, "eps": eps, "share": share, "F": F, "CR": CR,
                             "leader_kept": np.mean(kept_l), "random_share": np.mean(rand),
                             "random_genes_per_trial": np.mean(rand) * k, "reps": reps})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS),
                    help="thứ tự chạy; thêm k mới ở cuối để các dòng cũ không đổi")
    ap.add_argument("--out", default="docs/taimma/disruption_k.csv")
    a = ap.parse_args(argv)
    if len(set(a.ks)) != len(a.ks):
        ap.error("--ks có giá trị lặp")
    df = simulate_k(a.reps, ks=tuple(a.ks))
    df.to_csv(a.out, index=False, lineterminator="\n")
    print(df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
