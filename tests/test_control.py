import numpy as np

from mpqda.control import QController


def test_q_update_three_steps_by_hand():
    q = QController(P=1, rng=np.random.default_rng(0))
    q.F[:] = 0.5
    # bước 1: s = 1, a = 2 (+0,05), cải thiện -> r = 1, s' = 0
    q.update(np.array([2]), np.array([True]), np.array([True]), np.array([0.55]))
    assert np.isclose(q.Q[1, 2], 0.1) and q.s[0] == 0 and np.isclose(q.F[0], 0.55)
    # bước 2: s = 0, a = 1 (0), không cải thiện nhưng được nhận (hoà) -> r = 0, s' = 1; max Q[1] = 0,1
    q.update(np.array([1]), np.array([False]), np.array([True]), np.array([0.55]))
    assert np.isclose(q.Q[0, 1], 0.1 * (0 + 0.9 * 0.1)) and q.s[0] == 1
    # bước 3: s = 1, a = 0 (−0,05), không cải thiện, bị loại -> F giữ 0,55
    q.update(np.array([0]), np.array([False]), np.array([False]), np.array([0.50]))
    assert np.isclose(q.Q[1, 0], 0.1 * (0 + 0.9 * 0.1)) and np.isclose(q.F[0], 0.55)
    assert np.count_nonzero(q.Q) == 3


def test_zero_q_gives_uniform_policy_and_initial_state():
    q = QController(P=3, rng=np.random.default_rng(1))
    assert np.allclose(q.probs(0), 1 / 3) and np.allclose(q.probs(1), 1 / 3)
    assert q.s.tolist() == [1, 1, 1] and q.Q.shape == (2, 3) and not q.Q.any()
    assert ((q.F >= 0.3) & (q.F <= 0.8)).all()
    assert np.array_equal(q.F, np.random.default_rng(1).uniform(0.3, 0.8, size=3))   # F ban đầu: một lời gọi uniform(size=P)


def test_propose_uses_one_number_per_individual_and_keeps_F_in_bounds():
    rng = np.random.default_rng(2)
    q = QController(P=30, rng=rng)
    check = np.random.default_rng(2)
    check.uniform(0.3, 0.8, size=30)
    a, F_trial = q.propose(rng)
    u = check.random(30)
    assert a.tolist() == np.searchsorted(np.cumsum([1 / 3] * 3), u, side="right").clip(max=2).tolist()   # Q = 0: cdf đều
    assert rng.random() == check.random()                                              # đúng 30 số đã dùng
    for _ in range(300):
        a, F_trial = q.propose(rng)
        assert ((F_trial >= 0.1) & (F_trial <= 1.0)).all()
        improved = rng.random(30) < 0.3
        q.update(a, improved, improved | (rng.random(30) < 0.2), F_trial)
        assert ((q.F >= 0.1) & (q.F <= 1.0)).all()
    assert np.array_equal(q.s, np.where(improved, 0, 1))


def test_trial_F_kept_only_when_accepted():
    q = QController(P=4, rng=np.random.default_rng(3))
    before = q.F.copy()
    F_trial = before + 0.05
    q.update(np.array([2, 2, 2, 2]), np.array([True, False, False, False]), np.array([True, True, False, False]), F_trial)
    assert np.array_equal(q.F[:2], F_trial[:2]) and np.array_equal(q.F[2:], before[2:])
