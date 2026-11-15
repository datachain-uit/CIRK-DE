# Pre-registration: MPQDA-DE for exactly-k gene panels (TAIMMA 2027)

| Field | Value |
|---|---|
| Status | DRAFT. Binding from the annotated tag `taimma-2027-prereg`, pushed before the first real run (session S0) |
| Repository | `datachain-uit/MPQDA-DE`, branch `taimma-2027` |
| Registration commit | [TO FILL: 7-character hash of the tagged commit; the tag message repeats it with the SHA-256 of this file] |
| Registered on | [TO FILL: date and time, ICT] |
| Author | LuanNT (student ID 23520899), 23520899@gm.uit.edu.vn |
| Reviewed by | [TO FILL: supervisor], on [TO FILL: date of approval] |
| Vendored survey code | `RNA-Seq-FS-Survey`, branch `dmkd-revision`, commit `c1828d2`, unchanged until submission (TQĐ16; file list and SHA-256 in `vendor/VENDORED.md`) |
| Basis | Review-2026-11-14 (decisions TQĐ1–TQĐ10); code request YEU-CAU-DOI-CODE-2026-11-14-TAIMMA (adjustments Đ1–Đ7); Review-2026-11-14b (decisions TQĐ11–TQĐ18) |

Sections 1–11 do not change after registration. Any later change is a deviation: it is logged in Section 12 with its date, the
reason, and whether any output of T1–T5 had been seen.

To fill before registration: the fields above; Section 11 (environment of S0, parallel sessions); Section 12 (registration row). Delete this paragraph at registration.

## 1. Purpose and scope

Task: binary classification of lung adenocarcinoma (LUAD, label 0) versus squamous cell carcinoma (LUSC, label 1) from microarray
expression, with a hard budget of exactly k genes per panel. Wrapper selectors search panels of exactly k genes with an inner-CV
fitness on the outer-training part of each fold, under a fixed budget of fitness evaluations. In the survey that provides the data
and the evaluation protocol, 262 of 270 GA and BPSO runs with an AUROC fitness reached the ceiling of 1.000 (GSE10245 90/90,
GSE30219 90/90, GSE37745 82/90; survey commit `9b2d6a5`), so the returned panel was one of many tied optima.

Research questions:

- RQ1 (fitness). Under an exactly-k budget, does a log-loss fitness remove the plateau of the AUROC fitness, and how does it change
  the outer performance and the stability of wrapper panels?
- RQ2 (search quality). With the same evaluation budget, does MPQDA-DE reach a better inner fitness than DE/rand/1/bin, and an
  outer log-loss and AUROC that are not worse than those of GA, BPSO, DE/rand/1/bin, SHADE, qlDE, jDE and JADE?
- RQ3 (components). What does each component contribute: band re-encoding, chaotic initialisation, the archive-based leader, the
  per-individual Q-learning control of F, and the consensus output?
- RQ4 (stability). Is the archive consensus panel more stable across seeds than the best-individual panel and the panels of other
  wrappers, without loss of outer performance?

| Contribution | RQ | Scenarios | Hypotheses | Paper section |
|---|---|---|---|---|
| C1. Exactly-k wrapper selection with a log-loss fitness, with evidence that the AUROC fitness saturates | RQ1 | T1, T1b | H1 | 5.1 |
| C2. Band re-encoding of random keys, which keeps the panel of the base vector under DE mutation with exactly-k decoding, and MPQDA-DE, which combines it with a leader drawn from a Jaccard archive and per-individual Q-learning control of F, two components adapted from earlier DE designs to exactly-k panels | RQ2, RQ3 | T1, T2 | H2, H3, H5 (A0) | 3, 5.2, 5.3 |
| C3. Consensus panel from the archive | RQ4 | T1, T3 | H3, H4, H5 (A4) | 3, 5.4 |
| C4. Leakage-free, equal-budget, pre-registered evaluation with code and result files | – | all | – | 4 |

Scope: one main task, three GPL570 cohorts, one fitness model (LR-L2). The EGFR-mutation task (B7) is added only for the datasets
that passed the survey gates (Section 2). No claim is made beyond these data and settings.

## 2. Data, splits and prefilter

| Dataset | Role | Platform | n (class 0 / 1) | Genes | Status |
|---|---|---|---|---|---|
| `gse10245` | main | GPL570 | 55 (37 / 18) | 16,479 | fixed |
| `gse30219` | main | GPL570 | 141 (81 / 60) | 16,479 | fixed |
| `gse37745` | main; T0 shuffled labels; T3 | GPL570 | 169 (104 / 65) | 16,479 | fixed |
| `tcga_luad_egfr` | B7 (EGFR mutation in LUAD) | RNA-seq, log2(TPM + 1) | 493 (435 / 58; wild type / mutant) | 19,938 | included (passed G1 and G2) |
| `gse31210_egfr` | B7 (EGFR mutation in LUAD) | GPL570 | 224 (98 / 126; negative / positive; 2 samples removed by the survey QC from 99 / 127) | 16,479 | included (passed G1 and G2) |

SHA-256 of the input files. GPL570 values come from the provenance of the survey's wrapper runs (commit `9b2d6a5`, Kaggle input
`lfreedom2750/experiment-data-v3`); B7 values come from the hand-over of survey B7 step 1 (commit `c1828d2`, Kaggle session of
13 November 2026, `b7_step1_results.zip` SHA-256 `3febfbc23d377804f96a25bef35db29da4e28b0683c6a60ded2777acf0b28ee9`, folder
`B7_data/experiment-data-b7/`). The runner stops if `repeated_splits.csv` differs.

| Dataset | File | SHA-256 |
|---|---|---|
| gse10245 | expression.h5 | `7cb29e07d6c1489dbd90ad81ebfa9675548bac729289f70cf04766f2119226ea` |
| gse10245 | metadata.csv | `571f9d0ce2fc78275f41d09aa57a71f7a49ac3c6f1814c5b6f5b0002c12c2e26` |
| gse10245 | repeated_splits.csv | `823637702262a5fea795a52e9d2972f7857d9b75156d0667b1f7cac769496d5e` |
| gse30219 | expression.h5 | `6c2f7377ad802a7df3d804473884ad64fa08beadfa44ecb5232343801230e2d9` |
| gse30219 | metadata.csv | `570923f3f20216def97dfc47ca06a2e78f30669f4b7829f59b6426d7a32501bc` |
| gse30219 | repeated_splits.csv | `de92722bd27a853fef77abe4fda22bcd64886ef2724c298ac386b0bfd9b979bc` |
| gse37745 | expression.h5 | `f5dcaf11d7f36defc37ad67cf47da30b2f85823e3360fef2f8f4d0ab96a34ec7` |
| gse37745 | metadata.csv | `568b557e43c09daee94f8959f0144db642064502d73cb6493e8f5ff77c031328` |
| gse37745 | repeated_splits.csv | `18dc9577a10b86f9dec236cefad6beac48fb43db4c9d1383fed547b80712fd20` |
| tcga_luad_egfr | expression.h5 | `f688eb7b5c6785933915f5572b1a8143780953f0e36f4b1f3048a7f53176960d` |
| tcga_luad_egfr | metadata.csv | `63ca7cade4d45b2ca64dae7e97dbc635d0b0b6101aa1dec09761ed8c13ebbdc5` |
| tcga_luad_egfr | repeated_splits.csv | `6302d7f2e752fc80e7fa2b6675ed5451efddf53b82ba35323fa12c0475d6a7f6` |
| gse31210_egfr | expression.h5 | `6d2a9fbfc65194d2acb7b3b520006a54af244eec2f1fe06255d5e03a81019d0e` |
| gse31210_egfr | metadata.csv | `2679de2dc105205b880e30de5c537b81dc8585bd24c3427e54dc4ee330514b00` |
| gse31210_egfr | repeated_splits.csv | `91ff5e92e75eff26159bcba1310abb787b4b2c536e460c0faac4913ddc5dfa05` |

B7 gates (from survey B7 step 1; the survey's TP53 fallback is not used here). A B7 dataset is included only if it passed both
survey gates before registration; otherwise it is excluded from this study, and the exclusion is not a deviation. If no gate result
is available at 08:00 ICT on 16 November 2026, both B7 datasets are excluded (TQĐ17). Gate results below are copied from
`B7_gate/b7_gate_table.csv` of the hand-over (200 permutations; 25 folds; 100 random-k draws per fold); every `config.json` of
the hand-over has `traceable = true` at survey commit `c1828d2`.

| Dataset | G1: permutation p of E0 (≤ 0.05) | G2: mean AUROC(B0) − mean AUROC(random-k, k = 10) over 25 folds (≥ 0.05) | Decision |
|---|---|---|---|
| tcga_luad_egfr | p = 0.0050 (pass) | 0.934 − 0.744 = 0.190 (pass) | included |
| gse31210_egfr | p = 0.0050 (pass) | 0.888 − 0.655 = 0.234 (pass) | included |

Splits: the survey's patient-level stratified repeated 5-fold CV (split seeds 42–46). This study uses repeats 0, 1, 2 (split seeds
42, 43, 44), that is 15 outer folds per dataset, identical for every method.

Prefilter: inside each outer fold, the 5,000 genes with the largest variance on the outer-training samples (label-free; survey
`variance_prefilter`), p0 = 5,000. Selection reads only the outer-training samples after the prefilter. Outer test samples are read
only in phase B.

## 3. Method and fixed defaults

### 3.1 Representation

- An individual is a key vector x ∈ (0, 1)^p0. Its panel S(x) is the set of indices of the k largest keys; ties go to the smaller
  index. Every individual is feasible and |S(x)| = k.
- Band re-encoding: whenever a vector enters the population (initialisation or an accepted trial), the keys of S(x) are replaced by
  0.5 + 0.5·(r + 0.5)/k and the other keys by 0.5·(r + 0.5)/(p0 − k), where r is the stable rank of the key within its group.
  S(x) and the order of keys within each group do not change.
- Bound handling for every DE variant: a mutant coordinate v_j > 1 becomes (1 + x_ij)/2 and v_j < 0 becomes x_ij/2, where x_i is
  the target vector.

### 3.2 Fitness

- Log-loss mode (main): f(S) = −(1/3)·Σ log-loss over 3 stratified inner folds of the outer-training part. Model: standardisation
  fitted on the inner-training part, then logistic regression with L2 penalty (liblinear, C = 1, class_weight = balanced,
  max_iter = 5000). Probabilities are clipped to [1e−6, 1 − 1e−6]. The inner folds are shared by all wrappers on the same outer fold:
  `StratifiedKFold(3, shuffle=True, random_state=seed_for(cid, split_seed, fold, 900))`.
- AUROC mode (T1b only): f(S) = mean inner AUROC of the same model (decision function), with inner folds seeded by the method's
  search seed. This is identical to the survey's fitness.
- Fitness values are cached by panel. N_eval is the number of distinct panels evaluated in a run.

### 3.3 MPQDA-DE (Algorithm 1)

```
Input:  X_tr, y_tr (outer-training part after the prefilter), k, seed
Params: P = 30, T = 50, CR = 0.9, F_i(0) ~ U(0.3, 0.8), F in [0.1, 1.0]
        archive A_max = 15, delta = 0.2, novelty neighbours = 3, leader weights 0.7 novelty + 0.3 fitness,
        removal weights 0.5 novelty + 0.5 fitness, actions {-0.05, 0, +0.05}, eta = 0.1, gamma = 0.9, tau = 1, mu = 3.99
1  build f (Section 3.2)
2  for i = 1..P: z_i(0) ~ U(0.001, 0.999); x_ij = z_i(j + 1) with z(t + 1) = mu z(t) (1 - z(t)); then draw F_i; s_i = 1; Q = 0 (2 x 3)
3  S_i = topk(x_i); x_i = band(x_i); evaluate f(S_i) for all i; offer (S_i, f_i, x_i) to the archive in the order of i
4  for t = 1..T:
   a  for each i: a_i ~ softmax(Q[s_i, .] / tau); F'_i = clip(F_i + a_i, 0.1, 1.0)
   b  for each i: leader L_i drawn from the archive by roulette (one uniform number per i)
   c  for each i: r1 != r2 != i drawn uniformly from the population
   d  U ~ U(0, 1)^(P x p0); j_rand,i ~ U{0, ..., p0 - 1}
   e  v_i = repair(x_Li + F'_i (x_r1 - x_r2), x_i); u_ij = v_ij if U_ij <= CR or j = j_rand,i, else x_ij
   f  evaluate f(topk(u_i)) for all i (one batch; parallel)
   g  for each i in order: r_i = 1[f(u_i) > f(x_i)]; s'_i = 0 if r_i = 1 else 1
         Q[s_i, a_i] += eta (r_i + gamma max Q[s'_i, .] - Q[s_i, a_i]); s_i = s'_i
   h  for each i in order: if f(u_i) >= f(x_i): x_i = band(u_i); F_i = F'_i; f_i = f(u_i); offer (S(u_i), f_i, x_i) to the archive
   i  record the generation (Section 3.8)
5  S_c = consensus(archive, k); evaluate f(S_c); return S_c (primary output), S_best (best individual at T), history
```

### 3.4 Archive and leader

1. A candidate identical to a member is ignored.
2. If some member is at Jaccard distance below δ = 0.2 from the candidate, take the nearest one (ties: higher fitness, then earlier
   entry). The candidate replaces it in place only if its fitness is strictly higher; otherwise the candidate is rejected.
3. Otherwise the candidate is added. If the archive then exceeds A_max, the member with the smallest 0.5·novelty + 0.5·fitness is
   removed (both min–max normalised over the current members, candidate included; a zero range gives 0). The best member (highest
   fitness, ties: earlier entry) is never removed. Ties in the score: lower fitness first, then the more recent entry.
4. Novelty of a member: mean Jaccard distance to its 3 nearest other members (all others if fewer than 3; 0 if alone).
5. Leader: one uniform number u per draw; roulette on 0.7·novelty + 0.3·fitness (min–max normalised within the archive; uniform if
   all weights are 0).
6. The archive receives the initial individuals and the accepted trials only, in the order of i, with their band-re-encoded keys.

### 3.5 Consensus panel

w_a = |A| − (fitness rank of a) + 1; π_j = Σ_a w_a·1[j ∈ S_a] / Σ_a w_a; S_c = the k genes with the largest π_j. Ties: genes of the
best member first, then the larger mean key over members, then the smaller index. S_c is evaluated once; a new panel counts in
N_eval, so N_eval ≤ P·(T + 1) + 1 = 1,531.

### 3.6 Parallel evaluation

All random numbers of a generation are drawn in the main process before evaluation. New panels are evaluated in contiguous chunks
by joblib (loky) workers with BLAS limited to one thread. Results must be bit-identical for `n_jobs` = 1 and 4. Parallel
evaluation is an implementation detail; it is neither a contribution nor a research question.

### 3.7 Ablation variants (T2)

All variants use the search seed of MPQDA-DE (common random numbers) and report both outputs (consensus and best individual).

| Variant | Change relative to MPQDA-DE | Component examined |
|---|---|---|
| A0 | no band re-encoding | band re-encoding (credited through H5) |
| A1 | uniform keys U(0, 1) instead of chaotic keys | chaotic initialisation |
| A2a | leader = best individual of the population (one uniform number still drawn per i) | archive-based leader |
| A2b | leader drawn uniformly from the archive | roulette weights of the leader |
| A3 | no Q-learning; F = 0.5 for every individual and generation | Q-learning control of F |
| A4 | output = best individual of the same MPQDA-DE run (from T1) | consensus output |
| A5_A1 | A_max = 1 | archive |
| A5_A30 | A_max = 30 | archive size (sensitivity) |

### 3.8 Seeds and recorded behaviour

cid = survey `cohort_id` (gse10245 = 101, gse30219 = 102, gse37745 = 103, tcga_luad_egfr = 105, gse31210_egfr = 106); split_seed ∈
{42, 43, 44}; METHOD_ID = survey values (GA 11, BPSO 12) followed by mpqda_de 13, de_rand1 14, shade 15, qlde 16, jde 17, jade 18.

| Quantity | Definition |
|---|---|
| search seed, seed index 0 | `seed_run = seed_for(cid, split_seed, fold, METHOD_ID[method])`; T2 variants use METHOD_ID of mpqda_de |
| search seed, seed index s = 1..4 (T3) | `seed_for(cid, split_seed, fold, METHOD_ID[method], s)` |
| random generator per k | DE family: `default_rng([seed_run, k])`; GA, BPSO: `default_rng([seed_run, k, 0 or 1])` (survey) |
| inner folds | log-loss mode: `seed_for(cid, split_seed, fold, 900)`, shared; AUROC mode: `seed_run` of the method (survey) |
| label permutation in T0 | `default_rng([cid, split_seed, fold, 999]).permutation(n_train)` |
| phase B | B-LL and B-AUC use the same seeds, survey `seed_for(cid, split_seed, fold, k, METHOD_ID_survey.get(method, 90))`; every DE-family output, absent from the survey table, uses tag 90 |

Per generation t = 0..T: N_eval; best and mean fitness; diversity (mean pairwise Jaccard distance among population panels); success
rate (share of i with f(u_i) > f(x_i)); exploration rate (share of trial panels whose Jaccard distance to the nearest panel of the
previous population exceeds 0.5); ties accepted; mean and SD of F; archive size; share of leaders that are the best archive member;
leader overlap (mean |S(u_i) ∩ S(x_Li)|/k); number of distinct panels tied at the best fitness; Q-table entries. GA and BPSO
histories are rebuilt from the fitness call log in blocks of P calls. Anytime curve: best fitness against N_eval, median and IQR
over the 45 runs (3 cohorts × 15 folds) of each k.

## 4. Baselines, references and budget

Every wrapper: P = 30, T = 50 generations, the same fitness object and the same inner folds (log-loss mode), N_eval ≤ 1,530 (+1 for
the MPQDA-DE consensus panel). The DE family shares the encoding of Section 3.1 (uniform keys, band re-encoding, midpoint bound
handling), binomial crossover with j_rand, selection when f(u_i) ≥ f(x_i), and outputs the best individual. GA and BPSO keep the
survey encodings (exactly k by construction or by repair) and the survey code unchanged; only the fitness object is replaced.

| Baseline | Tier | Settings | Reference |
|---|---|---|---|
| GA | 1 | tournament 3, elitism 2, crossover keeps the common genes then fills at random, mutation 1/k per gene | Siedlecki and Sklansky (1989); survey code |
| BPSO | 1 | inertia 0.9 → 0.4, c1 = c2 = 2, v_max = 6, sigmoid sampling, repair to exactly k | Kennedy and Eberhart (1997); survey code |
| DE/rand/1/bin | 1 | F = 0.5, CR = 0.9 | Storn and Price (1997) |
| SHADE | 1 | memory H = 30 (M_CR = M_F = 0.5 initially); CR ~ N(M_CR, 0.1) clipped to [0, 1]; F ~ Cauchy(M_F, 0.1), regenerated if ≤ 0, set to 1 if > 1; p ~ U(2/P, 0.2); current-to-pbest/1 with an external archive of size P; weighted arithmetic and weighted Lehmer means | Tanabe and Fukunaga (2013) |
| qlDE | 1 | DE/rand/1/bin with one (F, CR) pair per generation for the whole population, chosen greedily (largest Q; ties: smallest index) among seven pairs: (0.4, 0.7), (0.6, 0.7), (0.8, 0.7), (0.4, 0.9), (0.6, 0.9), (0.8, 0.9), (0.9, 0.9) (Table 1, Eq. 9); Q-table of zeros with a single state (the paper defines no state); reward +1 if the best fitness of the population improved strictly in the generation, otherwise −1 (Eq. 8); Q[a] += α·(r + γ·max Q − Q[a]) (Eq. 6) with α = 1 − 0.9·(t + 1)/(T + 1) after generation t (Eq. 7) and γ = 0.8 (Section 3.1). Open points of the paper and deviations: `docs/taimma/baselines/qlde.md` | Huynh, Do and Lee (2021) |
| jDE | 2 | τ1 = τ2 = 0.1, F ∈ [0.1, 1.0] when regenerated (0.1 + 0.9·rand), initial F = 0.5, CR = 0.9 | Brest et al. (2006) |
| JADE | 2 | p = 0.05 (at least 2 individuals), c = 0.1, external archive of size P, μ_CR = μ_F = 0.5 initially | Zhang and Sanderson (2009) |

Tier-2 decision: run (`RUN_TIER2 = True` in `mpqda/config.py`, TQĐ13). jDE and JADE join T1 and T1b, the comparator set of H3,
the secondary comparisons and the Friedman test (8 methods); T3 keeps tier 1.

Non-wrapper references (T4, survey code unchanged): L1-LR, stability selection, F-test, mRMR, SVM-RFE, abess, at
k ∈ {2, 5, 10, 50}; B0 (LR-L2 on all genes, no prefilter); random-k (100 random panels per fold inside the prefilter).

Phase B. Primary (B-LL, `scripts/run_taimma_phase_b.py`): the survey procedure of `run_e1_classification.py` with one change:
C ∈ {0.01, 0.1, 1, 10} is chosen by the smallest inner 3-fold out-of-fold log-loss (probabilities clipped to [1e−6, 1 − 1e−6];
ties keep the smaller C). Unchanged: LR-L2 with standardisation and class_weight = balanced; the inner folds and seeds of the
survey phase B; the decision threshold maximising balanced accuracy on the inner out-of-fold scores of the chosen C; refit on the
whole outer-training part; rule M5 (C = 1, threshold 0.5 when the smaller class has at most 10 training samples).
Sensitivity (B-AUC): the survey `run_e1_classification.py` unchanged (C chosen by the inner AUROC). With that rule and the
ascending grid, ties at the maximum keep C = 0.01; in the survey this held for 874 of 1,125 selector panels at k ∈ {5, 10} on the
three GPL570 cohorts, and the outer log-loss then reflects the shrinkage of LR-L2 rather than the panel. Outer-test metrics are
unchanged: AUROC, balanced accuracy, MCC, log-loss (probabilities clipped to [1e−6, 1 − 1e−6]) and Brier score.

## 5. Scenarios

| Code | Purpose | Data | k | Methods | Seed indices | Fitness |
|---|---|---|---|---|---|---|
| T0 | pre-run checks | GSE10245 (repeat 0, folds 0–2) for determinism and n_jobs; GSE37745 (15 folds) with shuffled training labels | 5 | MPQDA-DE, GA | 0 | log-loss |
| T1 | main comparison | 3 GPL570 × 15 folds; included B7 datasets × 15 folds at k ∈ {5, 10} | 2, 5, 10, 50 | MPQDA-DE, tier 1 (+ tier 2) | 0 | log-loss |
| T1b | saturation | 3 GPL570 × 15 folds | 5, 10, 50 | as T1 | 0 | AUROC |
| T2 | ablation | 3 GPL570 × 15 folds | 2, 5, 10 | A0, A1, A2a, A2b, A3, A5_A1, A5_A30 | 0 (seed of MPQDA-DE) | log-loss |
| T3 | seed stability | GSE37745 × 15 folds | 5, 10 | MPQDA-DE, tier 1 | 1–4 (0 from T1) | log-loss |
| T4 | non-wrapper references | 3 GPL570 (+ included B7) × 15 folds | 2, 5, 10, 50 | six selectors, B0, random-k | – | – |
| T5 | phase B: B-LL (primary) and B-AUC (sensitivity) on every panel of T1–T4 | every panel of T1–T4 | – | LR-L2 | – | – |
| T6 | cost | GSE37745, repeat 0, folds 0–4, one machine, nothing else running | 10 | MPQDA-DE with n_jobs ∈ {1, 4} | 0 | log-loss |

## 6. Endpoints

- Primary: outer log-loss of the MPQDA-DE consensus panel against the best wrapper baseline under B-LL, at k ∈ {5, 10}, on the
  three GPL570 cohorts, 15 outer folds each (H3).
- Secondary: outer AUROC (ROPE 0.01), balanced accuracy, MCC, Brier; k ∈ {2, 50}; best inner fitness at T = 50; anytime curves;
  seed stability (Nogueira, T3); fold stability (descriptive); Jaccard overlap between methods; N_eval; time.
- Exploratory: B7; A5_A30; behaviour metrics; T6.
- Sensitivity: H3 and the secondary comparisons under B-AUC.
- Every quantity of H1–H5 that comes from phase B (outer log-loss, AUROC and the other outer metrics), the main tables and the
  figures use B-LL.

## 7. Hypotheses, tests and decision rules

Conventions:

- The paired unit is the outer fold (15 per cohort); differences are taken fold by fold.
- Nadeau–Bengio variance with J = 15 and ρ = n_test/n_train = 1/4: (1/J + ρ)·s².
- Pooling over the three GPL570 cohorts, equal weights: d̄ = (1/3)·Σ_c d̄_c; SE = (1/3)·√(Σ_c (1/15 + 1/4)·s_c²); df = 14;
  two-sided 95 % CI d̄ ± t(0.975, 14)·SE.
- Wilcoxon signed-rank: `scipy.stats.wilcoxon(zero_method="wilcox", method="auto")`, one-sided where stated.
- Holm correction within the family stated for each hypothesis; α = 0.05.
- A run is saturated when its best inner fitness is ≥ 1 − 1e−12 (AUROC mode).

H1 (RQ1; T1b, T1). Saturation rate per cohort over all wrapper runs of T1b, one run per (method, k, fold), judged on the best inner
fitness found in the run (for MPQDA-DE, one run gives both outputs and counts once).
Supported if the rate is ≥ 0.90 on GSE10245 and on GSE30219. Rejected if the pooled rate over these two cohorts is < 0.50.
Otherwise partially supported. GSE37745 is descriptive. Also reported, descriptively: the number of distinct panels tied at the
best fitness at T = 50 in both modes, and for each wrapper the change from T1b to T1 in outer log-loss (two-sided Nadeau–Bengio per
cohort and k, Holm within the wrapper) and in fold stability.

H2 (RQ2; T1). d = best inner fitness of MPQDA-DE minus that of DE/rand/1/bin on the same outer fold (same fitness function).
One-sided Wilcoxon (greater) per cohort and per k ∈ {5, 10}: 6 tests, one Holm family. Supported at a given k if the adjusted
p < 0.05 in at least 2 of the 3 cohorts; rejected at that k if the adjusted p ≥ 0.05 in all three; otherwise partially supported.
H2 is supported if it is supported at both values of k. The same comparison against every other baseline is descriptive.

H3 (RQ2, primary; T1, T5). For each k ∈ {5, 10}, the comparator m* is the wrapper baseline with the lowest mean outer log-loss over
the 45 folds (ties: first in the order GA, BPSO, DE/rand/1/bin, SHADE, qlDE, jDE, JADE).
(a) Log-loss: d = log-loss(MPQDA-DE consensus) − log-loss(m*). Non-inferior if the upper limit of the pooled two-sided 95 % CI is
below 0.02.
(b) AUROC: Δ = AUROC(MPQDA-DE consensus) − AUROC(m*). With the pooled posterior Student t (df 14, location Δ̄, scale SE_Δ),
P(Δ < −0.01) < 0.05.
H3 holds at k if (a) and (b) hold. Test (a) is one-sided at level 2.5 %, so over the two values of k its family-wise error stays at
or below 5 %. Choosing m* after seeing the results can only make the test harder for MPQDA-DE. H3 is computed under B-LL and
repeated under B-AUC; if the two decisions differ, both are reported and the B-LL decision stands.
Secondary: in each (cohort, k), against each baseline, a two-sided Nadeau–Bengio t-test and a two-sided Wilcoxon test with Holm
within the baselines, and the Bayesian correlated t-test for AUROC with ROPE 0.01 (P(left), P(rope), P(right)). Friedman test on
the mean outer log-loss over 12 blocks (3 cohorts × 4 values of k), Nemenyi critical difference with q(0.05) = 2.850 for 6 methods
and 3.031 for 8 methods, and a critical-difference diagram (Demšar, 2006); descriptive.

H4 (RQ4; T1, T3). For each outer fold of GSE37745 and k ∈ {5, 10}, Φ_m is the Nogueira stability of the five panels of method m
over seed indices 0–4 (d = 16,479 genes; survey `nogueira`). One-sided Wilcoxon (greater) over the 15 folds for (i) Φ_c − Φ_b, with
c the consensus output and b the best-individual output of MPQDA-DE; (ii) Φ_c − Φ_GA − 0.2; (iii) Φ_c − Φ_BPSO − 0.2. Six tests (3 × 2
values of k), one Holm family. H4 holds at k if (i), (ii) and (iii) are significant and H3 holds at k. Descriptive: mean ± SD of Φ
for every method, fold stability (Nogueira over the 15 fold panels of seed index 0), pairwise Jaccard.

H5 (RQ3; T1, T2). Each variant is paired with the full MPQDA-DE run of T1 (seed index 0, same folds), for k ∈ {2, 5, 10}.
Q1 (search): d = best inner fitness(full) − best inner fitness(variant); one-sided Wilcoxon (greater) over the 45 folds of the three
cohorts. Q2 (outer): d = log-loss(primary output of the variant) − log-loss(full consensus); one-sided pooled Nadeau–Bengio test,
p = P(t(14) > d̄/SE). Family: A0, A1, A2a, A2b, A3, A5_A1 × {Q1, Q2} × 3 values of k (36 tests) plus A4 × Q2 × 3 values of k
(3 tests, d = log-loss(best individual) − log-loss(consensus) of the same run): 39 tests, one Holm family. Credit rule: band
re-encoding ← A0; chaotic initialisation ← A1; archive-based leader ← A2a, A2b or A5_A1; Q-learning control ← A3; consensus output
← A4 (Q2) or H4 (i). A component is credited only if at least one of its tests is significant after Holm; otherwise the paper states
that it had no measurable effect. Every DE baseline also uses band re-encoding and midpoint bound handling, so the effect of
band re-encoding is credited only through A0. A5_A30 is descriptive. Two-sided CIs are reported for every difference, so a removal
that improves a quantity is visible.

## 8. Reading rules and negative results

- Tables give mean ± SD with the number of runs. Bold only marks differences that are significant after the stated correction.
  Curves give the median and IQR; captions state the number of runs.
- If MPQDA-DE equals DE/rand/1/bin on every quantity, the paper reports a negative result for the archive and Q-learning
  components and keeps the saturation analysis (C1) and the stability analysis (C3) as contributions. If H3 fails, the paper says
  so and does not describe MPQDA-DE as competitive or comparable.
- The primary endpoint, margins, families and scenario definitions do not change after results are seen. No cohort, k or seed is
  chosen after results are seen.
- "Novel", "superior" and "significantly" appear only next to the number and the test that support them; "state of the art" is not
  used.
- C is chosen by log-loss because log-loss is the endpoint; this protocol decision was taken before any run, and the paper does not
  present B-LL as an improvement of the results.
- B7 results are reported in a separate table: per included dataset, the secondary comparisons of H3 (each baseline, Holm within
  baselines) and AUPRC from the saved outer-test scores. They are not pooled with the GPL570 cohorts and do not enter H1–H5.

## 9. Pre-run checks and gates

- T0 passes when: (1) two runs with the same seed give byte-identical `selected_genes.csv` and `history.csv` for MPQDA-DE and GA;
  (2) MPQDA-DE with `n_jobs` 1 and 4 gives byte-identical outputs and histories; (3) every output has |S| = k and N_eval ≤ 1,531;
  (4) with shuffled training labels on GSE37745 (15 folds, k = 5), the mean outer AUROC of each method lies in [0.40, 0.60].
  The mean leader overlap at t = 1 of MPQDA-DE is recorded; a value below 0.2 is reported before any other run.
- G-T1 passes when GA and BPSO in AUROC mode reproduce the survey panels (`vendor/reference/`) exactly, with identical N_eval and
  a best fitness that differs by at most 1e−12 (the saturation threshold of Section 7): first on GSE10245, repeat 0,
  k ∈ {5, 10, 50} (session S0), then on every GA and BPSO unit of T1b. Units whose best fitness differs within that tolerance
  are listed in `gate_gt1_within_tol.csv` and reported in the paper's reproducibility note.
- GO/NO-GO at 12:00 ICT on 18 November 2026: GO when T0 passed, G-T1 passed on all of T1b, T1 and T1b with both phase B variants
  (B-LL and B-AUC) are complete for the three GPL570 cohorts, T2 and T3 are complete, and every `config.json` has `traceable = true` at a pushed commit.
  NO-GO moves the submission to the third round (deadline 2 December 2026 AoE) with the same design. B7 is not a GO condition.
- A failed check stops the dependent runs. The fix is a new commit; outputs produced before the fix are discarded and listed in
  Section 12.

## 10. What will not be done

- No use of outer-test data for selection, early stopping, tuning or the choice of a variant.
- No tuning of MPQDA-DE or baseline parameters on these cohorts; all defaults are fixed in Sections 3 and 4.
- No choice of k, cohort, seed or fold after results are seen. A failed run is re-run from the same seed; if it fails again it is
  reported as missing with the reason. No run is dropped silently.
- No change of the primary endpoint, of the margins (0.02 log-loss, ROPE 0.01 AUROC, 0.2 Nogueira) or of the multiplicity families.
- No number from the ICAART version is reused, and GSE131907 is not used.

## 11. Software and compute

| Item | Value |
|---|---|
| Python and packages | The Kaggle image of session S0, listed in `logs/S0/pip_freeze_session.txt` with its MD5 [TO FILL: MD5]. numpy, scipy, pandas and scikit-learn of the image are not overwritten; only skrebate 0.8.4 and abess 0.4.11 are installed (`--no-deps`). Packages that differ from the survey `vendor/requirements-kaggle.txt`: [TO FILL: list printed by GATE2]. This difference is not a deviation; G-T1 in S0 is the equivalence check with the survey environment (TQĐ16) |
| Same environment in every session | S1–S3 start only if the MD5 of their `pip freeze` equals that of S0 (GATE2). If G-T1 fails because of package versions, `PIN_SURVEY_ENV = True` installs `vendor/requirements-kaggle.txt`, S0 is re-run, and every later session uses the same flag |
| Machine | Kaggle CPU sessions, 4 vCPU; BLAS limited to one thread |
| Parallel sessions | [TO FILL: number available on the account] |
| Tags | `taimma-2027-prereg` (this registration); `taimma-2027-submission` (code and results cited in the paper) |

## 12. Change log

| Date | Section | Change | Reason | Outputs of T1–T5 seen before the change? |
|---|---|---|---|---|
| 2026-11-15 | 9 | Before registration: one trial run of session S0 on Kaggle at commit `72cccb9` (jobs `t0_det`, `t0_shuffle`; `gt1` not reached). Its outputs are discarded and S0 is run again after registration. It showed that condition (2) of `check-t0` compared CSV text, so an integer column read as float (0 versus 0.0) was reported as a difference although every value was identical; the check now compares values (`taimma_jobs.same_values`). A second trial run of the whole S0 at commit `90a6f7e` (finished 00:45 ICT on 15 November 2026; `S0_results.zip` SHA-256 `fc2e2d54c64a479308e61e09b1fa6d941d4d3efd463c812237798571f6bcf82e`) passed T0 and G-T1 (30 of 30 units identical to the survey); it is also a pre-registration run and is not cited. A third trial run of S0 at commit `38bc3fe` (B7 datasets included; finished 01:41 ICT on 15 November 2026; `S0_results.zip` SHA-256 `e14b00f2e4245eb32523fcafbc1f2e6c5b4250dcb6c11212e7f804144970726a`) passed T0 and G-T1 again, with outputs identical to the second run except the timing columns; it is not cited either | bug in the check, not in the method | No (no job of T1–T5 was run) |
| 2026-11-15 | 9 | Before registration: G-T1 on all of T1b (session S2, commit `35331ed`) matched the survey in gene sets and N_eval for all 270 units and in best fitness for 268; the two others (gse37745, repeat 1, fold 3, BPSO, k = 5 and repeat 2, fold 1, GA, k = 5) differed only in the last bit (0.9993141289437585 against 0.9993141289437584), the only two units with that value in the survey reference. The exact-equality rule for best fitness was replaced by a tolerance of 1e−12 (`check_gate_gt1.FITNESS_TOL`); gene sets and N_eval must still be identical. Session S2 is re-run at the commit with this change; sessions S1 and S3 stay at `35331ed`, since the change touches only `scripts/check_gate_gt1.py`, which they do not run. All sessions S1–S3 ran before registration | floating-point rounding between machines, not a difference of panels | No (only T1b phase A was complete; its outer test was not evaluated) |
| 2026-11-15 | 5, 9 | Before registration: sessions S1 (commit `35331ed`; `S1_results.zip` SHA-256 `8d5a21d3356aaa86be1f221ed78fb3acd87fea705bcb2805892954b8f2ff424c`), S2 (commit `99e099b`; `65b2c9d7ff15679e413dfee04ec8f3d1ad8d12ff45434764659aa1ed2409dccc`) and S3 (commit `35331ed`; `3fc35a7202297f4c56e7356873690410a89c03e539f6d9e33d18ea7626028117`) completed every job with exit code 0, the same `pip freeze` MD5 as S0 and `taimma_jobs.py verify` passing for all 62 result folders; G-T1 on all of T1b passed (270 units; the two last-bit units listed in `gate_gt1_within_tol.csv`). Tables were built from these outputs on 15 November 2026, so every result of T1–T5 was seen before registration | the student ran S1–S3 before supervisor approval | Yes |
| [TO FILL] | – | Registration | – | No |
