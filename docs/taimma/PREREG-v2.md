# Pre-registration v2: MPQDA-DE v2 for exactly-k gene panels (TAIMMA 2027, round 3)

| Field | Value |
|---|---|
| Status | DRAFT for supervisor review. Binding from the annotated tag `taimma-2027-prereg-v2` |
| Repository | `datachain-uit/MPQDA-DE`, branch `taimma-2027` |
| Registration commit | [TO FILL at registration: 7-character hash; the tag message repeats it with the SHA-256 of this file and of `DEV_results.zip`] |
| Registered on | [TO FILL: date and time, ICT] |
| Author | LuanNT (student ID 23520899), 23520899@gm.uit.edu.vn |
| Reviewed by | [TO FILL: supervisor], on [TO FILL: date of approval] |
| Vendored survey code | `RNA-Seq-FS-Survey`, branch `dmkd-revision`, commit `c1828d2`, unchanged (`vendor/VENDORED.md`) |
| Basis | Review-2026-11-16 (decisions TQĐ19–TQĐ26); results of v1 under tag `taimma-2027-v1-results` |

Sections 1–11 do not change after registration. Any later change is logged in Section 12 with its date, the reason, and
whether any output of the confirmatory phase (C1–C3) had been seen.

## 1. Purpose, scope and relation to v1

v1 of MPQDA-DE was evaluated on three GPL570 cohorts and two EGFR datasets (B7); its protocol is `docs/taimma/PREREG.md`
(executed before supervisor approval, not a pre-registration; tags `taimma-2027-v1-protocol` and `taimma-2027-v1-results`).
v1 did not support H2–H4. A diagnosis of the search operator showed that band re-encoding with a lower band of width 1/2 lets
the DE difference vector lift random out-of-panel genes into the panel band (about a third of a trial panel at F = 0.5).
MPQDA-DE v2 replaces that encoding with a gapped one, adds a GA-style source of new genes, and draws the leader and the added
vector from the archive by tournament.

Three statements hold for every claim about v2 (TQĐ26):

1. v2 was designed after the results of v1 on the three GPL570 cohorts had been seen.
2. The configuration of v2 was chosen on separate development data (Section 2.2) by a rule written before those runs.
3. The GPL570 results of v2 confirm the chosen configuration; they are not an independent confirmation of the redesign idea.
   `gse31210_egfr` is the only dataset that took part in neither the diagnosis, the design nor the development phase.

Research questions:

- RQV1 (search). With the same budget and fitness, does v2 reach a better inner fitness than v1 and than the strongest
  baseline of v1?
- RQV2 (generalisation). How large is the difference in outer log-loss and AUROC between v2 and BPSO (best baseline of H3)
  and between v2 and v1, and with what precision is it estimated?
- RQV3 (stability). Is the consensus panel of v2 more stable across seeds than that of v1 and than GA?
- RQV4 (components). Which of the gap, the new-gene step and the tournament contribute to the inner fitness of v2?

## 2. Data

### 2.1 Confirmatory data (as in PREREG v1, Section 2)

| Dataset | Role | n (class 0 / 1) | Genes | `repeated_splits.csv` SHA-256 |
|---|---|---|---|---|
| gse10245 | main | 55 (37 / 18) | 16,479 | `823637702262a5fea795a52e9d2972f7857d9b75156d0667b1f7cac769496d5e` |
| gse30219 | main | 141 (81 / 60) | 16,479 | `de92722bd27a853fef77abe4fda22bcd64886ef2724c298ac386b0bfd9b979bc` |
| gse37745 | main; C3 | 169 (104 / 65) | 16,479 | `18dc9577a10b86f9dec236cefad6beac48fb43db4c9d1383fed547b80712fd20` |
| tcga_luad_egfr | B7 | 493 (435 / 58) | 19,938 | `6302d7f2e752fc80e7fa2b6675ed5451efddf53b82ba35323fa12c0475d6a7f6` |
| gse31210_egfr | B7 | 224 (98 / 126) | 16,479 | `91ff5e92e75eff26159bcba1310abb787b4b2c536e460c0faac4913ddc5dfa05` |

Splits, repeats (0, 1, 2; 15 outer folds per dataset), prefilter (p0 = 5,000 genes by variance on the outer-training part)
and the other input files are those of PREREG v1, Section 2. `tcga_luad_egfr` uses the expression profiles of the TCGA
cohort from which the development subsets are drawn (same profiles, different labels), so v2 results on it are not
independent of the development data; the paper says so and relies on `gse31210_egfr` when discussing B7 for v2.

### 2.2 Development data and results (TQĐ23)

| Dataset | Source | n (LUAD / LUSC) | Outer folds | `repeated_splits.csv` SHA-256 |
|---|---|---|---|---|
| tcga_dev_a | TCGA LUAD/LUSC, patient-disjoint subset | 170 (86 / 84) | 5 (one repeat, split seed 1171) | `4b551e87837ef251172594f719534adc6c7c3f46632ad1d8d410f8e5485cc4bd` |
| tcga_dev_b | idem | 170 (86 / 84) | 5 (split seed 1172) | `7c3c4db3830de08be96536f3589f6d040fb2b645acb13a631a22044850046494` |
| tcga_dev_c | idem | 170 (86 / 84) | 5 (split seed 1173) | `34f6f2146f6692eab183e2d6a61e8239fed93af6432c3a5fea1e34da7eb7e2b4` |
| gse42127 | survey cohort (GPL6884) | survey | 10 (repeats 0, 1) | `6a0f194aa586b94f3836019b273d12ac2bc8abbdfe0151e612c876be073fd959` |

The three TCGA subsets are built by `scripts/build_dev_datasets.py` (one sample per patient, class-stratified draw with seed
20261117, 170 samples each); the card with value hashes is `docs/taimma/dev_datasets_card.json`. The build is deterministic in
content (matrix values, `metadata.csv`, `repeated_splits.csv`).

Development run: session DEV on Kaggle, commit `20f838c`, 16 November 2026; `DEV_results.zip` SHA-256
`be24f71d4c485db527040edc303b9f775195da1ef41ca0e30dabb21f87988c4a`; 50 units (4 datasets, k ∈ {5, 10}); methods: the three
candidates, v1, DE-G, DE/rand/1/bin, GA. Selection rule and gate (`scripts/dev_select.py`, written before the run):

| Candidate | Mean rank of best fitness (50 units) | Mean best fitness |
|---|---|---|
| `mpqda_v2_m1` (fixed F, CR) | 1.92 | −0.1348 |
| `mpqda_v2_m3` (tournament 2) | 1.98 | −0.1369 |
| `mpqda_v2_m2` (SHADE-type adaptation) | 2.10 | −0.1340 |

Chosen configuration: **`mpqda_v2_m1`** (smallest mean rank). Gate G-DEV passed: best fitness of M1 minus v1, mean +0.0162,
M1 better in 72 % of the units, one-sided Wilcoxon p = 3.6·10⁻⁴. Descriptive only, not used for the choice: M1 − GA −0.0035
(p = 0.92, two-sided); M1 − DE-G +0.0035 (p = 0.15); M1 − DE/rand/1/bin +0.0121 (p = 0.006). Outer log-loss under B-LL on the
development data (descriptive): v1 0.352, M1 0.425, DE-G 0.416, DE/rand/1/bin 0.382, GA 0.476. No change was made to v2 after
these results other than the choice of the candidate.

## 3. Method (Algorithm 2)

`mpqda/de_v2.py` (`run_mpqda_v2`) with `V2_CHOSEN = mpqda_v2_m1`: eps = 0.01, rate_new = 1/k, tour = 3, r1_from = archive,
adapt = False, F = 0.5, CR = 0.9; P = 30, T = 50, A_max = 15, δ = 0.2. The order of random numbers per generation is part of the
specification (docstring of `de_v2.py`). Fitness, inner folds, archive rules, consensus panel and bound handling are those of
PREREG v1, Section 3.

```
Input:  X_tr, y_tr (outer-train part after the prefilter), k, seed
gband(x): genes of topk(x) -> 0.5 + 0.5 (r + 0.5) / k; other genes -> eps (r + 0.5) / (p0 - k); r = stable rank within each group
1 x_i ~ U(0, 1)^p0; S_i = topk(x_i); x_i = gband(x_i); f_i = f(S_i); offer (S_i, f_i, x_i) to the archive in the order of i
2 for t = 1..T:
  b leader L_i: best (by fitness rank) of tour archive members drawn uniformly with replacement
  c r1 != r2 != i uniform in the population; a_i: best of tour archive members (replaces x_r1)
  d v_i = repair(x_Li + F (a_i - x_r2), x_i); u_ij = v_ij if U_ij <= CR or j = j_rand,i, else x_ij
  e innovation: each gene of topk(u_i) is replaced with probability 1/k by a gene drawn uniformly outside the panel (keys swapped)
  f evaluate f(topk(u_i)) for all i (one batch)
  g for each i in order: if f(u_i) >= f(x_i): x_i = gband(u_i); f_i = f(u_i); offer (topk(u_i), f_i, x_i) to the archive
3 S_c = consensus(archive, k); evaluate f(S_c); return S_c (primary output), S_best, history
```

N_eval ≤ P(T + 1) + 1 = 1,531. Ablations (C2): `mpqda_v2_no_gap` (eps = 0.5), `mpqda_v2_no_new` (rate_new = 0),
`mpqda_v2_no_tour` (tour = 1 and r1 drawn from the population). Seeds: METHOD_ID 19 for every name of the v2 family (common
random numbers for the ablations), 20 for DE-G; otherwise as PREREG v1, Section 3.8.

## 4. Baselines, references and gate G-R

v1 and the seven baselines (GA, BPSO, DE/rand/1/bin, SHADE, qlDE, jDE, JADE) are not re-run: their T1, T2, T3 and phase B
results from S1–S3 (tag `taimma-2027-v1-results`) are reused. DE-G (`run_de_gapped`: DE/rand/1/bin with the gap and the
new-gene step, no archive, no tournament, F = 0.5, CR = 0.9) is the internal control that separates the representation from the
other components of v2.

Gate G-R (session S0-v2, at the registration commit): 16 units are run again, the eight methods (`mpqda_de`, `ga`, `bpso`,
`de_rand1`, `shade`, `qlde`, `jde`, `jade`) × (gse10245, repeat 0, fold 0, k = 5) and (gse37745, repeat 2, fold 4, k = 10), and
compared with T1 of v1 by `scripts/check_gate_gt1.py`: identical gene sets (consensus and best output for v1) and N_eval;
best fitness within 1e−12 (TQĐ20). A failure stops the confirmatory phase and is reported to the supervisor.

## 5. Scenarios of the confirmatory phase

| Code | Data | k | Methods | Seed indices |
|---|---|---|---|---|
| T0-v2 | GSE10245 (repeat 0, folds 0–2) for determinism and n_jobs; GSE37745 (15 folds) with shuffled training labels | 5 | `mpqda_v2`, `de_g` | 0 |
| C1 | 3 GPL570 × 15 folds | 2, 5, 10, 50 | `mpqda_v2`, `de_g` | 0 |
| C1-B7 | 2 B7 × 15 folds | 5, 10 | `mpqda_v2`, `de_g` | 0 |
| C2 | 3 GPL570 × 15 folds | 2, 5, 10 | three ablations | 0 (seed of `mpqda_v2`) |
| C3 | GSE37745 × 15 folds | 5, 10 | `mpqda_v2` | 1–4 (0 from C1) |
| Phase B | every panel of C1–C3 | – | LR-L2: B-LL (primary), B-AUC and B-C1 (C = 1 fixed; sensitivity) | – |

Commands: `python scripts/taimma_jobs.py show <job>` for the jobs `t0v2_det`, `t0v2_shuffle`, `gr`, `c1`, `c2`, `c3`.

## 6. Endpoints

- Primary: best inner fitness at T = 50 (HV1).
- Secondary: outer log-loss and AUROC under B-LL, estimated with precision (HV2); seed stability of the consensus panel (HV3);
  component contributions (HV4); outer metrics under B-AUC and B-C1; B7.
- Descriptive: leader overlap, success rate and new genes per generation (v1 against v2, C1); Friedman ranks and critical
  difference over ten methods (q(0.05) = 3.164); N_eval; time.

## 7. Hypotheses, tests and decision rules

Conventions: paired unit = outer fold; Holm within each hypothesis; α = 0.05; Wilcoxon signed-rank with
`zero_method="wilcox"`; pooled Nadeau–Bengio as PREREG v1, Section 7.

Expected precision (from v1, TQĐ24): the SD of paired best-fitness differences over 45 folds is 0.015 (k = 5) and 0.010
(k = 10) (GA − v1), so the smallest difference detectable with 80 % power in HV1 at level 0.05/4 is about 0.007 (k = 5) and
0.005 (k = 10). The SD of fold-wise Φ differences (consensus v1 − GA) is 0.030, so the smallest detectable difference in HV3 is
about 0.025. For HV2 the half-width of the pooled 95 % CI is about 0.13 and the smallest detectable difference about 0.19.

- **HV1 (primary).** For k ∈ {5, 10}: d = best fitness(v2) − best fitness(m), m ∈ {v1, GA} (GA has the highest mean best
  fitness among the baselines of T1 at both values of k). One-sided Wilcoxon (greater) over the 45 folds; Holm over 4 tests.
  Report the mean difference, a 95 % bootstrap CI stratified by cohort, and the share of folds where v2 is better.
- **HV2 (estimation).** Outer log-loss and AUROC (B-LL) of the v2 consensus panel against BPSO and against v1, k ∈ {5, 10}; pooled
  Nadeau–Bengio CIs as H3 of v1, Holm over the 4 log-loss comparisons; ROPE probability for AUROC as v1. "Different" only if the
  Holm-adjusted CI excludes 0. No margin; neither "equivalent" nor "non-inferior" is used. Repeated under B-AUC and B-C1.
- **HV3 (stability).** Φ (Nogueira, seed indices 0–4, GSE37745) of the v2 consensus panel minus that of the v1 consensus panel
  and minus that of GA, k ∈ {5, 10}; one-sided Wilcoxon over 15 folds; Holm over 4 tests; no margin.
- **HV4 (components).** Best fitness of v2 minus that of each ablation, k ∈ {2, 5, 10}; one-sided Wilcoxon over 45 folds; Holm over
  9 tests. A component is credited if its test is significant at two of the three values of k. Outer log-loss of the ablations is
  descriptive.

## 8. Reading rules

- If HV1 holds and HV2 finds no difference: the paper states that v2 optimises the wrapper objective better and that, with the
  precision of this design, no difference in outer log-loss was detected (Reunanen, 2003).
- If v2 beats v1 but not the strongest baseline: the paper states that v2 corrects the defect of v1 without beating that
  baseline; "competitive" is not used without a number.
- Only credited components (HV4) are described as effective. The development results of Section 2.2, including the outer
  log-loss order on the development data, are reported.
- B7 is reported in a separate table; on `tcga_luad_egfr` the dependence on the development profiles is stated.

## 9. Gates

- Registration gate (`mpqda/registration.py`): `mpqda_v2`, its ablations and DE-G run on the confirmatory data only if
  `V2_CHOSEN` is set, HEAD carries `taimma-2027-prereg-v2` and the working tree is clean; the candidates never run there. No
  environment variable or flag bypasses it.
- T0-v2: two runs with the same seed give identical `selected_genes.csv` and `history.csv`; n_jobs 1 and 4 give identical
  values; every output has |S| = k and N_eval ≤ 1,531; with shuffled training labels the mean outer AUROC of each method lies in
  [0.40, 0.60].
- G-R as Section 4. C1–C3 start only after T0-v2 and G-R pass and the `pip freeze` MD5 equals that of S0-v2.

## 10. What will not be done

- No change to v2, its constants or the candidate after the development results, except a recorded bug fix followed by a full
  re-run of DEV.
- No run of any v2 method on the confirmatory data before the tag; no use of outer-test data for any choice.
- No re-run or re-tuning of v1 or the baselines.

## 11. Software and compute

| Item | Value |
|---|---|
| Environment | Kaggle CPU image of S0–S3 and DEV: Python 3.13.18; `pip freeze` MD5 `6af55eea01de6bb6b9b41ece2d33e480`; threadpoolctl 3.7.0 differs from the survey (3.6.0) |
| Measured cost (DEV) | about 26 s per run of v2 or DE-G, 38 s for GA (n = 170, p0 = 5,000, P = 30, T = 50); DEV took 0.75 h with 4 workers |
| Expected cost | S0-v2 about 0.3 h; C1 about 1.5 h; C2 about 1.5 h; C3 about 0.5 h; phase B with three criteria about 1 h |
| Parallel sessions | 3 |
| Tags | `taimma-2027-prereg-v2` (this registration); `taimma-2027-submission` (code and results cited in the paper) |

## 12. Change log

| Date | Section | Change | Reason | Outputs of C1–C3 seen before the change? |
|---|---|---|---|---|
| [TO FILL] | – | Registration | – | No |
