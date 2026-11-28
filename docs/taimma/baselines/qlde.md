# qlDE — configuration taken from the original paper

Source: Huynh TN, Do DTT, Lee J (2021) Q-learning-based parameter control in differential evolution for structural
optimization. Applied Soft Computing 107:107464. doi:10.1016/j.asoc.2021.107464

**Status: filled in from the full text on 2026-11-14** (publisher's PDF, 14 pages). The configuration is in
`mpqda.config.QLDE`; the implementation is `mpqda.baselines.run_qlde`. The four points that the original paper leaves open or contradicts (Section 3 below) were recorded
in `CHOICES_TO_CONFIRM` and **all four were approved** before the main runs: one state; α from Eq. (7) and
γ = 0.8 from Section 3.1; ties broken by the smallest index; one (F, CR) pair for the whole population per generation. Section 4 of the paper states the two points the original
paper leaves open (state, tie-breaking).

## 1. Configuration of the original paper

| Item | Value in the original paper | Section, equation, table |
|---|---|---|
| Mutation | DE/rand/1: v_i = x_R1 + F·(x_R2 − x_R3), i ≠ R1 ≠ R2 ≠ R3 | Section 2, Eq. (2); Algorithm 2 lines 15–17 |
| Bound repair | reflection (reflecting wall): v < x_min becomes 2·x_min − v; v > x_max becomes 2·x_max − v | Section 2, Eq. (3) |
| Crossover | binomial: u_ij = v_ij if j = K_i or rand ≤ Cr; K_i drawn uniformly from {1, …, D} | Section 2, Eq. (4); Algorithm 2 lines 18–23 |
| Selection | u_i replaces x_i when f(u_i) ≤ f(x_i) (minimisation problem) | Section 2, Eq. (5); Algorithm 2 lines 26–28 |
| Actions | 7 pairs [F, Cr]: [0.4; 0.7], [0.6; 0.7], [0.8; 0.7], [0.4; 0.9], [0.6; 0.9], [0.8; 0.9], [0.9; 0.9] | Section 3.2, Eq. (9); Table 1 |
| Scope of an action | one pair for the whole population in one iteration: "at the beginning of each DE iteration, a combination of control parameters is chosen" | Section 3.2; Fig. 2 |
| Reward | r_t = +1 if best_{t+1} < best_t (the best solution of the population improves), otherwise −1; computed at the end of the iteration | Section 3.2, Eq. (8) |
| Q update | Q(s_t, a_t) ← Q(s_t, a_t) + α·[r_t + γ·max_a Q(s_{t+1}, a) − Q(s_t, a_t)], once per iteration | Section 3.1, Eq. (6); Algorithm 2 line 30 |
| Learning rate | α(t) = 1 − 0.9·(eval / MaxEval), decreasing linearly | Section 3.1, Eq. (7) |
| Discount factor | γ = 0.8 | Section 3.1 |
| Policy | choose the action with the largest Q in the current state (greedy, no exploration) | Algorithm 2 lines 1–2, 14 |
| Initial values | Q table all 0 | Algorithm 2 line 10 |
| State | **not defined.** Section 3.1 only gives S = {s_1, …, s_m} in general form; Section 3.2, Table 1 and Algorithm 2 do not say what the state is or how many there are | – |
| Budget in the original paper | NP = 50 (and 20), MaxEval = 150,000, stop when δ = \|f_mean / f_best − 1\| < 1e−6 or MaxEval is reached; 30 runs | Section 4 |
| DE compared in the original paper | F = 0.8, Cr = 0.9 | Table 1 |

## 2. Differences from the original paper in the TAIMMA study

Only the parts shared by every DE baseline are changed.

| Item | Original paper | TAIMMA study | Reason |
|---|---|---|---|
| Representation | real vector of design variables | uniform keys `rng.random((P, p0))`, top-k decoding, band re-encoding | same encoding for every wrapper |
| Bound repair | reflection, Eq. (3) | midpoint against the target individual | applied to every DE variant |
| Selection | f(u) ≤ f(x), minimisation | f(u) ≥ f(x), maximisation | same rule, reversed because the fitness is maximised |
| Reward | best_{t+1} < best_t | the best fitness of the population after selection is strictly larger than before selection | as above |
| Constraint handling | penalty function, Eq. (11), ε_1 = 1.5, ε_2 = 3 | none | every individual is feasible (exactly k genes) |
| Budget and stopping rule | NP = 50, MaxEval = 150,000, or δ < 1e−6 | P = 30, T = 50, no convergence criterion | every wrapper has the same budget |
| eval / MaxEval in Eq. (7) | number of objective function calls | P·(t + 1) / (P·(T + 1)) after generation t, so α goes from 0.965 (t = 1) to 0.1 (t = 50) | N_eval of this study counts distinct panels (with a cache); the nominal count keeps α independent of the cache |

## 3. Points left open or contradictory in the original paper

| # | Issue | Current implementation | Alternative |
|---|---|---|---|
| 1 | The state is not defined | one state, Q table 1 × 7; max_a Q(s_{t+1}, a) is taken over that same row | defining a state of our own (for example from the result of the previous iteration) would no longer be the setting of the original paper |
| 2 | Table 1 gives "QLearning: α = 0.8", while Section 3.1 gives α from Eq. (7) and "γ = 0.8" | following Section 3.1: α from Eq. (7), γ = 0.8 (the line of Table 1 is read as a notation error for γ) | α = 0.8 fixed |
| 3 | Tie-breaking when several actions share the largest Q is not stated; with Q all 0 and a greedy policy, this choice decides the first action | smallest index (order of Table 1), so the first generation always uses (F, Cr) = (0.4; 0.7); no random number used | uniform draw among the tied actions (one more random number per generation) |
| 4 | Algorithm 2 calls ChooseAction inside the loop `for i = 1 → NP`, while Section 3.2 and Fig. 2 say one pair per iteration | one pair per generation. The two readings give the same result because the Q table is updated only at the end of the iteration and the policy is deterministic | – |

Consequence for other documents: PREREG describes qlDE as "Q-learning choosing an (F, CR) pair for each
individual". According to the full text, qlDE chooses one pair for the whole population per iteration. Table 1 and Table 2 of the paper must state this correctly.

Behaviour to know when reading the results: a reward of −1 makes the Q of the action just used negative, so the greedy policy moves to the next
action that still has Q = 0. The controller therefore goes through the seven pairs in turn during generations without improvement and returns to a pair that
was once rewarded.

## 4. Order of random draws of the implementation

As DE/rand/1/bin: uniform keys `rng.random((P, p0))`; each generation: (r1, r2, r3) for i = 0..P − 1 by
`rng.choice(np.delete(np.arange(P), i), size=3, replace=False)`; `U = rng.random((P, p0))`; `j_rand = rng.integers(0, p0, size=P)`.
Action selection and the Q update use no random numbers. `tests/test_baselines.py` asserts: with a single action
(0.5; 0.9), `run_qlde` gives exactly the result of `run_de_rand1` with the same seed.
