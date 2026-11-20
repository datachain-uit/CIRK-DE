# Amendment A2 to PREREG-v2: does the contribution of the gap grow with the panel budget?

| Field | Value |
|---|---|
| Status | Registered. Binding from the annotated tag `taimma-2027-prereg-v2.2`; the tag message gives the commit hash and the SHA-256 of this file and of `docs/taimma/disruption_k.csv` |
| Registered on | 2026-11-20 17:09 ICT |
| Author | LuanNT (student ID 23520899), 23520899@gm.uit.edu.vn |
| Approved by | Supervisor, decisions TQĐ50–TQĐ51 of Review-2026-11-20 |
| Relation to PREREG-v2 | Adds HV6 and session C4. Sections 1–11 of PREREG-v2 are unchanged. HV1–HV5, their families, their reading rules and the credit rule of HV4 are unchanged; HV6 does not credit any component retroactively |

## 1. Rationale

Under the band re-encoding (ε = 1/2) the DE mutation and crossover insert genes that belong to no parent into the trial panel
(PREREG-v2, Section 1). An operator simulation (`scripts/disruption_k.py`, exploratory, seed 3, 200 repetitions, F = 0.5,
CR = 0.9, p0 = 5,000) gives, on average per trial panel, 0.8 such genes at k = 2, 1.9–2.1 at k = 5, 3.4–3.7 at k = 10, 5.9–6.5 at
k = 20 and 11.7–13.7 at k = 50 (the two values are the two levels of gene sharing), and at most 0.005 under the gap (ε = 0.01). v2 adds
one controlled new gene per trial. The rows for k ∈ {2, 5, 10, 50} were computed on 18 November 2026 (Review-2026-11-18c,
`expected/disruption_k.csv` of taimma_ref_v7, SHA-256 b4fcb0e5…9e92); the rows for k = 20 were appended before this registration
with the same seed and order (`--ks 2 5 10 50 20`), so that the first 16 rows are unchanged.

Prediction: the contribution of the gap to the best inner fitness of v2, d_k = f(v2) − f(v2 without the gap), increases with k,
and is positive at the budgets where the band inserts several foreign genes per trial.

## 2. What had been seen before this registration

- HV4 (C2, k ∈ {2, 5, 10}): mean d = −0.0017, 0.0036 and 0.0045; significant only at k = 10.
- E2 (exploratory, C1 and S1): DE-G minus DE/rand/1/bin, k ∈ {2, 5, 10, 50}: 0.0023, −0.0002, 0.0068 and 0.0055.
- The simulation of Section 1.
- No run of `mpqda_v2_no_gap` at k ∈ {20, 50} and no run of `mpqda_v2` at k = 20 existed. `mpqda_v2` at k = 50 had been run in C1.

## 3. Design (session C4)

- Commit: the commit carrying `taimma-2027-prereg-v2.2`; the job `c4` starts with
  `python scripts/taimma_jobs.py check-registered --require-tag taimma-2027-prereg-v2.2`.
- Environment: the Kaggle image of S0-v2 and C1–C3; the notebook stops if the `pip freeze` MD5 differs from that of S0-v2.
- Runs: `mpqda_v2` and `mpqda_v2_no_gap` (ε = 0.5, every other setting of v2), three GPL570 cohorts, 15 outer folds each,
  k ∈ {20, 50}, seed index 0 (METHOD_ID 19 for both, common random numbers as in C2), the fitness, inner folds, prefilter,
  P = 30 and T = 50 of PREREG-v2. 180 runs. No phase B.
- Gate R50: for every outer fold, `mpqda_v2` at k = 50 in C4 must reproduce C1 (`T1v2`): identical panels (consensus and best
  output), identical N_eval, best fitness within 1e−12 (`scripts/make_taimma_corrected.py`, `gate_r50.csv`). A failure stops the
  analysis of HV6 and is reported to the supervisor; no run is repeated or replaced.

## 4. Hypotheses and tests

- HV6 (primary of this amendment). For each outer fold, d_k for k ∈ {2, 5, 10, 20, 50}; k ∈ {2, 5, 10} from C1 (`mpqda_v2`) and
  C2 (`mpqda_v2_no_gap`), k ∈ {20, 50} from C4. Page's L test for an increasing trend over the five ordered budgets, 45 blocks
  (outer folds), one-sided, α = 0.05 (`scipy.stats.page_trend_test`, columns in increasing k).
- HV6a (secondary). For k ∈ {20, 50}: one-sided Wilcoxon signed-rank test (d > 0) over the 45 folds, `zero_method="wilcox"`,
  `method="auto"`, Holm over the 2 tests. Reported with the mean of d, the 95 % bootstrap CI stratified by cohort (as HV1), the
  share of folds with d > 0 and, as a sensitivity analysis, the pooled Nadeau–Bengio 95 % CI (as HV2).

## 5. Reading rules

- Supported: HV6 holds and HV6a holds at k = 50. The paper states that the contribution of the gap grew with the panel budget, as
  the simulation predicted, and gives the estimates at k = 20 and 50.
- Partially supported: exactly one of HV6 and HV6a at k = 50 holds. The paper reports both results and makes no statement on a
  trend.
- Not supported: neither holds. The paper states that the predicted growth was not confirmed.
- HV6a at k = 20 is reported in every case and does not change the reading.
- The outer performance of the C4 panels is not computed; no statement on generalisation is based on C4.

## 6. Compute

About 25 to 40 s per run on the Kaggle CPU image (PREREG-v2, Section 11; larger k is slower), about 0.5 h with 4 workers.

## 7. What will not be done

No other value of k or ε, no change to v2, no phase B for C4, no new dataset, no change to HV1–HV5 or to the credit rule of HV4,
no re-run of C1–C3.
