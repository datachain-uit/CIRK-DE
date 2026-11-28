"""Constants fixed in advance for the TAIMMA study (PREREG, Sections 3 and 4).

Every value here was copied into docs/taimma/PREREG.md before the first real run and was not changed after seeing results.
`vendor/survey_v2/config.py` is not edited; new method codes are appended after METHOD_ID of the vendored code so that the GA and BPSO seeds do not change.
"""
from __future__ import annotations

from survey_v2 import config as C

METHOD_ID = {**C.METHOD_ID, "mpqda_de": 13, "de_rand1": 14, "shade": 15, "qlde": 16, "jde": 17, "jade": 18}
assert all(METHOD_ID[m] == C.METHOD_ID[m] for m in C.METHOD_ID)          # existing codes unchanged (G-T1)
assert len(set(METHOD_ID.values())) == len(METHOD_ID)

DE_FAMILY = ("mpqda_de", "de_rand1", "shade", "qlde", "jde", "jade")
SURVEY_WRAPPERS = ("ga", "bpso")
TIER1 = ("ga", "bpso", "de_rand1", "shade", "qlde")
TIER2 = ("jde", "jade")
VARIANTS = {  # T2 variant -> flags of run_mpqda; uses the seed of mpqda_de (common random numbers)
    "mpqda_a0_noband": dict(reencode=False), "mpqda_a1_uniform": dict(init="uniform"),
    "mpqda_a2a_bestleader": dict(leader="best"), "mpqda_a2b_randleader": dict(leader="random"),
    "mpqda_a3_fixedF": dict(control="fixed"), "mpqda_a5_A1": dict(A_max=1), "mpqda_a5_A30": dict(A_max=30)}
# Each run of the MPQDA family writes two outputs: <name> (consensus panel, main output) and <name>_best (best individual; A4 = mpqda_de_best)
MPQDA_FAMILY = ("mpqda_de", *VARIANTS)
BEST_SUFFIX = "_best"
ALL_METHODS = (*MPQDA_FAMILY, *TIER1, *TIER2)

P, T, CR = 30, 50, 0.9
F_INIT, F_BOUNDS, ACTIONS = (0.3, 0.8), (0.1, 1.0), (-0.05, 0.0, 0.05)
ETA, GAMMA, TAU = 0.1, 0.9, 1.0
MU, Z0_RANGE = 3.99, (0.001, 0.999)
A_MAX, DELTA, N_NEIGHBORS, W_LEADER, W_REMOVE = 15, 0.2, 3, (0.7, 0.3), (0.5, 0.5)
EXPLORE_THRESHOLD, INNER_FOLDS, FITNESS_C, PROB_CLIP = 0.5, 3, 1.0, 1e-6
INNER_FOLD_TAG, SHUFFLE_TAG = 900, 999
FIXED_F = 0.5                                   # ablation A3
DE_RAND1_F = 0.5
SHADE_H, SHADE_P_RANGE, DE_ARCHIVE_SIZE = P, (2 / P, 0.2), P
SHADE_SIGMA = 0.1                               # standard deviation of N(M_CR; 0.1) and scale of Cauchy(M_F; 0.1)
JDE_TAU, JDE_F_RANGE, JDE_INIT = (0.1, 0.1), (0.1, 0.9), (0.5, 0.9)
JADE_P, JADE_P_MIN, JADE_C = 0.05, 2, 0.1
SATURATION_EPS = 1e-12
EXPECTED_SPLIT_SHA256 = {   # the runner stops if a repeated_splits.csv differs (source: config.json of the E1 wrapper run of the vendored code, commit 9b2d6a5)
    "gse10245": "823637702262a5fea795a52e9d2972f7857d9b75156d0667b1f7cac769496d5e",
    "gse30219": "de92722bd27a853fef77abe4fda22bcd64886ef2724c298ac386b0bfd9b979bc",
    "gse37745": "18dc9577a10b86f9dec236cefad6beac48fb43db4c9d1383fed547b80712fd20",
    # B7: B7 step-1 hand-over of the vendored code, commit c1828d2, Kaggle session 2026-11-13 (b7_step1_results.zip,
    # SHA-256 3febfbc23d377804f96a25bef35db29da4e28b0683c6a60ded2777acf0b28ee9); both datasets pass G1 and G2.
    "tcga_luad_egfr": "6302d7f2e752fc80e7fa2b6675ed5451efddf53b82ba35323fa12c0475d6a7f6",
    "gse31210_egfr": "91ff5e92e75eff26159bcba1310abb787b4b2c536e460c0faac4913ddc5dfa05",
    # Development data of v2. Three TCGA subsets: scripts/build_dev_datasets.py, docs/taimma/dev_datasets_card.json
    # (built on 2026-11-16 from experiment-data-v3, CSV written with LF; SHA-256 identical to the build on Kaggle). gse42127: the split
    # of the vendored code in experiment-data-v3.
    "tcga_dev_a": "4b551e87837ef251172594f719534adc6c7c3f46632ad1d8d410f8e5485cc4bd",
    "tcga_dev_b": "7c3c4db3830de08be96536f3589f6d040fb2b645acb13a631a22044850046494",
    "tcga_dev_c": "34f6f2146f6692eab183e2d6a61e8239fed93af6432c3a5fea1e34da7eb7e2b4",
    "gse42127": "6a0f194aa586b94f3836019b273d12ac2bc8abbdfe0151e612c876be073fd959",
}

# qlDE: taken from the full text of Huynh, Do and Lee (2021), Applied Soft Computing 107:107464; section numbers, equations and the points
# left open by the original paper are recorded in docs/taimma/baselines/qlde.md. When QLDE is None, the runner refuses --methods qlde.
QLDE = {
    "actions": ((0.4, 0.7), (0.6, 0.7), (0.8, 0.7), (0.4, 0.9), (0.6, 0.9), (0.8, 0.9), (0.9, 0.9)),   # (F, Cr), Table 1
    "gamma": 0.8,                               # Section 3.1
    "alpha_start": 1.0, "alpha_drop": 0.9,      # Eq. (7): alpha = 1 − 0.9·eval/MaxEval
    "reward": (1.0, -1.0),                      # Eq. (8): best fitness of the population strictly increases / does not
    "q_init": 0.0,                              # Algorithm 2, line 10
    "n_states": 1,                              # the original paper defines no state
    "policy": "greedy", "tie_break": "lowest_index",
}

# Run tier 2 (jDE, JADE) in T1, T1b: True (PREREG, Section 5). T3 keeps tier 1.
RUN_TIER2 = True
# The B7 datasets passed both gates G1 and G2 of the vendored code: B7_gate/b7_gate_table.csv of the B7 step-1 hand-over.
B7_INCLUDED = ("tcga_luad_egfr", "gse31210_egfr")
N_OUTER_FOLDS = 15                              # 3 repeats × 5 folds

RESULT_SUBDIR = "taimma"                        # $KLTN_DATA_ROOT/results/survey_v2/taimma/<scenario>/<cohort>/
ENV_SMOKE = "MPQDA_SMOKE"                       # only for smoke tests on synthetic data; config.json records smoke = true, traceable = false

# Implementation choices not stated by the specification (general rule 5); written to config.json of every run. All 13 items were approved
# before the main runs. τ = 1 is kept; nothing is changed after seeing results.
CHOICES_TO_CONFIRM = (
    # Kept verbatim (archived in config.json): selection_time_s and cpu_time_s are measured per (method, k); main process only.
    "selection_time_s và cpu_time_s đo cho từng (phương pháp, k), không gộp mọi k của một đơn vị như survey; "
    "cpu_time_s chỉ tính tiến trình chính (không gồm tiến trình con khi n_jobs_eval > 1)",
    # Kept verbatim (archived): best_fitness is the fitness of the best individual at the stop; output_fitness of <name> is the consensus fitness.
    "best_fitness = fitness của cá thể tốt nhất khi dừng (Đ7); với họ MPQDA hai dòng <tên> và <tên>_best có cùng best_fitness, "
    "output_fitness của dòng <tên> là fitness của panel đồng thuận",
    # Kept verbatim (archived): consensus_weights.csv lists only genes with pi > 0.
    "consensus_weights.csv chỉ ghi gen có pi > 0 (gen thuộc ít nhất một thành viên archive)",
    # Kept verbatim (archived): SHADE, JADE number of pbest individuals uses Python round; r2 and external archive removal drawn uniformly.
    "SHADE, JADE: số cá thể pbest = max(2, round(p_i * P)) với round của Python (làm tròn về số chẵn khi .5); "
    "r2 rút đều trên quần thể ∪ archive ngoài, trừ i và r1; archive ngoài đầy thì bỏ một phần tử rút đều",
    # Kept verbatim (archived): SHADE, JADE memory is updated only in generations with a success; JADE uses the unweighted Lehmer mean.
    "SHADE, JADE: bộ nhớ chỉ cập nhật ở thế hệ có ít nhất một lần thử thành công (f(u) > f(x)); "
    "JADE dùng trung bình Lehmer không trọng số như bài gốc",
    # Kept verbatim (archived): jDE draws four uniform numbers in the order (rand1, rand2, rand3, rand4).
    "jDE: bốn số đều rút theo thứ tự (rand1, rand2, rand3, rand4); F' = 0,1 + 0,9*rand1 khi rand2 < tau1; CR' = rand3 khi rand4 < tau2",
    # Kept verbatim (archived): GA, BPSO history in blocks of P fitness calls per generation; success_rate, mean_F are NaN.
    "lịch sử GA, BPSO: khối P lời gọi fitness mỗi thế hệ (history_from_log); success_rate, mean_F không xác định (NaN)",
    # Kept verbatim (archived): with the smoke environment variable set, another split SHA-256 and --nan-outer-test are allowed.
    f"biến môi trường {ENV_SMOKE}=1 (chỉ smoke test): cho phép SHA-256 bản chia khác giá trị chờ và cờ --nan-outer-test; "
    "config.json ghi smoke = true và traceable = false",
    # Kept verbatim (archived): the outer LR-L2 of the shuffled-label T0 uses the phase B seed of the vendored code.
    "T0 nhãn xáo trộn: LR-L2 ngoài dùng seed pha B của survey, seed_for(cid, split_seed, fold, k, 90)",
    # Kept verbatim (archived): qlDE uses one state (Q table 1 × 7) and one (F, Cr) pair for the whole population per generation.
    "qlDE: bài gốc không định nghĩa trạng thái; cài đặt dùng một trạng thái (bảng Q 1 × 7) và một tổ hợp (F, Cr) cho cả "
    "quần thể mỗi thế hệ, như Mục 3.2 và Algorithm 2 (không phải theo từng cá thể như Bảng 6 của Review ghi)",
    # Kept verbatim (archived): qlDE follows Section 3.1 of the original paper for alpha (Eq. 7) and gamma = 0.8.
    "qlDE: Table 1 của bài gốc ghi alpha = 0,8 còn Mục 3.1 ghi alpha theo Eq. (7) và gamma = 0,8; cài đặt theo Mục 3.1",
    # Kept verbatim (archived): qlDE greedy policy without exploration; ties broken by the smallest index.
    "qlDE: chính sách tham lam không có thăm dò (Algorithm 2); bài gốc không nêu cách phá hoà, cài đặt lấy chỉ số nhỏ nhất, "
    "nên thế hệ đầu luôn dùng (F, Cr) = (0,4; 0,7)",
    # Kept verbatim (archived): qlDE eval / MaxEval uses nominal evaluation counts; the convergence criterion is dropped.
    "qlDE: eval / MaxEval của Eq. (7) tính bằng số lần đánh giá danh nghĩa P·(t + 1) / (P·(T + 1)), không trừ lần trúng cache; "
    "bỏ tiêu chí hội tụ delta < tol của bài gốc để mọi wrapper cùng T = 50",
)


# ---------------------------------------------------------------- CIRK-DE (v2; PREREG-v2)
# Written out here, not imported from mpqda.de_v2 (de.py imports config.py; a reverse import would create a cycle). A test checks that the two tables
# match CANDIDATES, ABLATIONS of mpqda/de_v2.py after renaming the keys.
METHOD_ID = {**METHOD_ID, "mpqda_v2": 19, "de_g": 20}
V2_CANDIDATES = {"mpqda_v2_m1": {}, "mpqda_v2_m2": {"adapt": True},
                 "mpqda_v2_m3": {"tour": 2}}
V2_ABLATIONS = {"mpqda_v2_no_gap": {"eps": 0.5},
                "mpqda_v2_no_new": {"rate_new": 0.0},
                "mpqda_v2_no_tour": {"tour": 1, "r1_from": "population"},
                "mpqda_v2_no_adapt": {"adapt": False}}   # only if M2 is chosen
V2_CHOSEN = "mpqda_v2_m1"   # fixed in the registration commit (PREREG-v2; DEV: smallest rank, G-DEV passed)
V2_FAMILY = ("mpqda_v2", *V2_CANDIDATES, *V2_ABLATIONS)
TWO_OUTPUTS = (*MPQDA_FAMILY, *V2_FAMILY)   # consensus and <name>_best
ALL_METHODS = (*ALL_METHODS, *V2_FAMILY, "de_g")
DEV_DATASETS = ("tcga_dev_a", "tcga_dev_b", "tcga_dev_c", "gse42127")


def base_method(method: str) -> str:
    """Method used for the seed: T2 variants use the seed of mpqda_de; every name of the v2 family uses the seed of mpqda_v2."""
    if method in VARIANTS:
        return "mpqda_de"
    return "mpqda_v2" if method in V2_FAMILY else method


def v2_config(method: str) -> dict:
    """Parameters passed to run_mpqda_v2: a candidate uses its own configuration; mpqda_v2 and the ablations need V2_CHOSEN."""
    if method in V2_CANDIDATES:
        return dict(V2_CANDIDATES[method])
    if V2_CHOSEN is None:
        raise ValueError("V2_CHOSEN is not fixed")
    return {**V2_CANDIDATES[V2_CHOSEN], **V2_ABLATIONS.get(method, {})}


def output_names(method: str) -> tuple[str, ...]:
    """Names of the output rows of one run: the MPQDA v1 and v2 families have <name> (consensus) and <name>_best."""
    return (method, method + BEST_SUFFIX) if method in TWO_OUTPUTS else (method,)


def t1_methods() -> tuple[str, ...]:
    """Methods of T1, T1b: QRK-DE, tier 1 and (if RUN_TIER2) tier 2."""
    return ("mpqda_de", *TIER1, *(TIER2 if RUN_TIER2 else ()))


def n_outputs(methods) -> int:
    return sum(len(output_names(m)) for m in methods)


def constants() -> dict:
    """Every constant (upper-case name) in a form that can be written to config.json."""
    out = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and not name.startswith("_") and name != "C":
            out[name] = list(value) if isinstance(value, tuple) else value
    return out
