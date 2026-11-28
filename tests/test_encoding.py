import numpy as np

from mpqda import encoding as E


def test_decode_topk_size_and_ties():
    rng = np.random.default_rng(1)
    for k in (1, 2, 5, 10, 50):
        for _ in range(20):
            S = E.decode_topk(rng.random(500), k)
            assert len(S) == k == len(set(S.tolist())) and (np.diff(S) > 0).all()
    assert E.decode_topk(np.zeros(10), 3).tolist() == [0, 1, 2]                      # ties: smaller index first
    assert E.decode_topk(np.array([0.2, 0.9, 0.9, 0.1, 0.9]), 2).tolist() == [1, 2]


def test_jaccard():
    rng = np.random.default_rng(2)
    assert E.jaccard_distance({1, 2}, {1, 2}) == 0.0 and E.jaccard_distance({1, 2}, {3, 4}) == 1.0
    a, b = set(range(10)), set(range(1, 11))                                         # 9/10 genes in common
    assert abs(E.jaccard_distance(a, b) - (1 - 9 / 11)) < 1e-12 and E.jaccard_distance(a, b) < 0.2
    for _ in range(50):
        x, y = rng.choice(40, 7, replace=False), rng.choice(40, 7, replace=False)
        d = E.jaccard_distance(x, y)
        assert d == E.jaccard_distance(y, x) and 0.0 <= d <= 1.0


def test_chaotic_keys_in_open_interval_and_reproducible():
    X = E.init_chaotic(30, 500, np.random.default_rng(3))
    assert X.shape == (30, 500) and X.min() > 0 and X.max() < 1
    assert np.array_equal(X, E.init_chaotic(30, 500, np.random.default_rng(3)))
    assert not np.array_equal(X, E.init_chaotic(30, 500, np.random.default_rng(4)))
    U = E.init_uniform(30, 500, np.random.default_rng(3))
    assert np.array_equal(U, np.random.default_rng(3).random((30, 500)))


def test_repair_midpoint_uses_target_and_never_hits_one():
    rng = np.random.default_rng(4)
    x = rng.random((30, 500))
    v = x + 0.9 * (rng.random((30, 500)) - rng.random((30, 500))) + 0.3
    r = E.repair_midpoint(v, x)
    assert r.min() >= 0 and r.max() <= 1 and not (r == 1.0).any()
    assert (v > 1).any() and np.array_equal(r[v > 1], (1 + x[v > 1]) / 2)
    low = E.repair_midpoint(np.array([-0.2, 0.4]), np.array([0.6, 0.9]))
    assert low.tolist() == [0.3, 0.4]


def test_band_encode_keeps_panel_bands_and_order():
    rng = np.random.default_rng(13)
    for k in (1, 2, 5, 10, 50):
        x = rng.random(800)
        b = E.band_encode(x, k)
        S = E.decode_topk(x, k)
        assert E.decode_topk(b, k).tolist() == S.tolist()
        assert (b[S] > 0.5).all() and (np.delete(b, S) < 0.5).all() and b.min() > 0 and b.max() < 1
        out = np.delete(np.arange(800), S)
        assert (np.argsort(b[out], kind="stable") == np.argsort(x[out], kind="stable")).all()
        assert (np.argsort(b[S], kind="stable") == np.argsort(x[S], kind="stable")).all()
        r = np.argsort(np.argsort(x[S], kind="stable"), kind="stable")
        assert np.allclose(b[S], 0.5 + 0.5 * (r + 0.5) / k)
    assert E.decode_topk(E.band_encode(np.zeros(20), 4), 4).tolist() == [0, 1, 2, 3]   # tied keys still keep the panel


def test_no_low_index_bias_after_repair():
    """Regression for the midpoint repair: with clipping to [0, 1], hundreds of keys equal 1.0 and the panel collapses to small indices."""
    rng = np.random.default_rng(0)
    P, p, k = 30, 5000, 10
    X = E.init_chaotic(P, p, rng)
    share_mid, share_clip = [], []
    for _ in range(100):
        i, L, r1, r2 = rng.choice(P, 4, replace=False)
        raw = X[L] + 0.5 * (X[r1] - X[r2])
        mask = rng.random(p) <= 0.9
        for repaired, acc in ((E.repair_midpoint(raw, X[i]), share_mid), (np.clip(raw, 0, 1), share_clip)):
            acc.append(np.mean(E.decode_topk(np.where(mask, repaired, X[i]), k) < p // 10))
    assert abs(np.mean(share_mid) - 0.1) < 0.05
    assert np.mean(share_clip) > 0.9


def test_leader_information_transmitted_only_with_band_encoding():
    """Band re-encoding: with re-encoding, the trial panel keeps most of the leader panel at F = 0.5; without re-encoding almost all of it is lost."""
    rng = np.random.default_rng(14)
    P, p, k = 30, 5000, 10
    X = E.init_uniform(P, p, rng)
    B = np.array([E.band_encode(X[i], k) for i in range(P)])
    for keys, lo, hi in ((B, 0.35, 0.65), (X, 0.0, 0.05)):
        ov = []
        for _ in range(200):
            i, L, r1, r2 = rng.choice(P, 4, replace=False)
            v = E.repair_midpoint(keys[L] + 0.5 * (keys[r1] - keys[r2]), keys[i])
            mask = rng.random(p) <= 0.9
            mask[rng.integers(p)] = True
            u = np.where(mask, v, keys[i])
            ov.append(len(set(E.decode_topk(u, k)) & set(E.decode_topk(keys[L], k))) / k)
        assert lo <= np.mean(ov) <= hi
