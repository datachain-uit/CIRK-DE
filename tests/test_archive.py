import numpy as np

from mpqda.archive import Archive


def test_size_and_best_member_invariants_over_1000_offers():
    rng = np.random.default_rng(11)
    arch = Archive(A_max=15, delta=0.2)
    best_seen = -np.inf
    for _ in range(1000):
        row = rng.choice(60, size=5, replace=False)
        f = float(rng.normal())
        best_seen = max(best_seen, f)
        arch.offer(row, f, rng.random(60))
        assert len(arch.members) <= 15
        assert max(m["f"] for m in arch.members) == best_seen                        # thành viên tốt nhất không bao giờ bị bỏ
        assert len({m["key"] for m in arch.members}) == len(arch.members)
    S, pi = arch.consensus(5)
    assert len(S) == 5 and abs(sum(pi.values()) - 5.0) < 1e-9


def test_delta_rule_and_duplicates():
    arch = Archive(A_max=15, delta=0.2)
    a, near = list(range(10)), list(range(9)) + [99]                                 # chung 9 gen: d = 0,18 < delta
    assert arch.offer(a, 0.5, np.zeros(100)) == "added"
    assert arch.offer(list(reversed(a)), 0.9, np.zeros(100)) == "duplicate"
    assert arch.offer(near, 0.4, np.zeros(100)) == "rejected_near"
    assert arch.offer(near, 0.5, np.zeros(100)) == "rejected_near"                   # phải cao hơn hẳn
    assert arch.offer(near, 0.6, np.zeros(100)) == "replaced"
    assert [m["key"] for m in arch.members] == [tuple(sorted(near))]
    assert arch.offer(list(range(50, 60)), 0.1, np.zeros(100)) == "added" and len(arch.members) == 2


def test_size_one_keeps_best_panel():
    one = Archive(A_max=1)
    assert one.offer([1, 2], 0.1, np.zeros(5)) == "added"
    assert one.offer([3, 4], 0.3, np.zeros(5)) == "added"
    assert one.offer([0, 4], 0.2, np.zeros(5)) == "rejected_full"
    assert [m["key"] for m in one.members] == [(3, 4)] and one.consensus(2)[0].tolist() == [3, 4]


def test_removal_score_by_hand():
    """A_max = 3, các panel rời nhau (novelty bằng nhau): bỏ thành viên có fitness nhỏ nhất, không phải thành viên tốt nhất."""
    arch = Archive(A_max=3)
    for genes, f in (([0, 1], 0.9), ([2, 3], 0.2), ([4, 5], 0.5)):
        arch.offer(genes, f, np.zeros(10))
    assert arch.offer([6, 7], 0.4, np.zeros(10)) == "added"
    assert sorted(m["key"] for m in arch.members) == [(0, 1), (4, 5), (6, 7)]
    assert arch.offer([8, 9], 0.1, np.zeros(10)) == "rejected_full"


def test_novelty_and_leader():
    arch = Archive(A_max=15)
    arch.offer([0, 1, 2, 3], 0.5, np.zeros(20))
    assert arch.novelty().tolist() == [0.0] and arch.leader_index(0.99) == 0
    arch.offer([0, 1, 2, 9], 0.6, np.zeros(20))                                       # d = 1 − 3/5 = 0,4
    arch.offer([10, 11, 12, 13], 0.7, np.zeros(20))                                   # d = 1 tới hai panel kia
    assert np.allclose(arch.novelty(), [0.7, 0.7, 1.0])
    # trọng số roulette: 0,7·novelty_mm + 0,3·fitness_mm = [0; 0,15; 1,0] -> cdf = [0; 0,1304; 1]
    assert [arch.leader_index(u) for u in (0.0, 0.1, 0.2, 0.999)] == [1, 1, 2, 2]
    assert [arch.leader_index(u, "uniform") for u in (0.0, 0.34, 0.67, 0.999)] == [0, 1, 2, 2]
    assert arch.best_index() == 2 and arch.ranked() == [2, 1, 0]


def test_consensus_weights_and_ties_by_hand():
    arch = Archive(A_max=15)
    x = np.zeros(10)
    x[7] = 1.0                                         # khoá trung bình của gen 7 lớn hơn
    arch.offer([0, 1], 0.9, x)                         # tốt nhất: w = 3
    arch.offer([2, 3], 0.5, x)                         # w = 1
    arch.offer([2, 7], 0.7, x)                         # w = 2
    S, pi = arch.consensus(2)
    # pi: 0, 1 = 3/6; 2 = 3/6; 3 = 1/6; 7 = 2/6 -> hoà 0, 1, 2; ưu tiên gen của thành viên tốt nhất -> {0, 1}
    assert S.tolist() == [0, 1]
    assert abs(pi[2] - 0.5) < 1e-12 and abs(pi[7] - 2 / 6) < 1e-12 and abs(pi[3] - 1 / 6) < 1e-12
    assert arch.consensus(3)[0].tolist() == [0, 1, 2] and arch.consensus(4)[0].tolist() == [0, 1, 2, 7]
