"""Hằng số đặt trước của bài TAIMMA (Mục 3.1 của YEU-CAU-DOI-CODE-2026-11-14-TAIMMA; PREREG Mục 3, 4).

Mọi giá trị ở đây được chép vào docs/taimma/PREREG.md trước lần chạy thật đầu tiên và không chỉnh sau khi xem số.
Không sửa `vendor/survey_v2/config.py`; mã phương pháp mới nối sau METHOD_ID của survey để seed của GA, BPSO không đổi.
"""
from __future__ import annotations

from survey_v2 import config as C

METHOD_ID = {**C.METHOD_ID, "mpqda_de": 13, "de_rand1": 14, "shade": 15, "qlde": 16, "jde": 17, "jade": 18}
assert all(METHOD_ID[m] == C.METHOD_ID[m] for m in C.METHOD_ID)          # số cũ không đổi (G-T1)
assert len(set(METHOD_ID.values())) == len(METHOD_ID)

DE_FAMILY = ("mpqda_de", "de_rand1", "shade", "qlde", "jde", "jade")
SURVEY_WRAPPERS = ("ga", "bpso")
TIER1 = ("ga", "bpso", "de_rand1", "shade", "qlde")
TIER2 = ("jde", "jade")
VARIANTS = {  # biến thể T2 -> cờ của run_mpqda; dùng seed của mpqda_de (common random numbers)
    "mpqda_a0_noband": dict(reencode=False), "mpqda_a1_uniform": dict(init="uniform"),
    "mpqda_a2a_bestleader": dict(leader="best"), "mpqda_a2b_randleader": dict(leader="random"),
    "mpqda_a3_fixedF": dict(control="fixed"), "mpqda_a5_A1": dict(A_max=1), "mpqda_a5_A30": dict(A_max=30)}
# Mỗi lần chạy họ MPQDA ghi hai đầu ra: <tên> (panel đồng thuận, đầu ra chính) và <tên>_best (cá thể tốt nhất; A4 = mpqda_de_best)
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
SHADE_SIGMA = 0.1                               # độ lệch chuẩn của N(M_CR; 0,1) và tham số tỉ lệ của Cauchy(M_F; 0,1)
JDE_TAU, JDE_F_RANGE, JDE_INIT = (0.1, 0.1), (0.1, 0.9), (0.5, 0.9)
JADE_P, JADE_P_MIN, JADE_C = 0.05, 2, 0.1
SATURATION_EPS = 1e-12
EXPECTED_SPLIT_SHA256 = {   # runner dừng nếu repeated_splits.csv khác (nguồn: config.json E1 wrapper của survey, commit 9b2d6a5)
    "gse10245": "823637702262a5fea795a52e9d2972f7857d9b75156d0667b1f7cac769496d5e",
    "gse30219": "de92722bd27a853fef77abe4fda22bcd64886ef2724c298ac386b0bfd9b979bc",
    "gse37745": "18dc9577a10b86f9dec236cefad6beac48fb43db4c9d1383fed547b80712fd20",
    # "tcga_luad_egfr", "gse31210_egfr": điền từ bàn giao B7 bước 1
}

# qlDE (Đ6): trích từ toàn văn Huynh, Do và Lee (2021), Applied Soft Computing 107:107464; số mục, phương trình và các chỗ
# bài gốc để ngỏ ghi trong docs/taimma/baselines/qlde.md. Khi QLDE là None, runner từ chối --methods qlde.
QLDE = {
    "actions": ((0.4, 0.7), (0.6, 0.7), (0.8, 0.7), (0.4, 0.9), (0.6, 0.9), (0.8, 0.9), (0.9, 0.9)),   # (F, Cr), Table 1
    "gamma": 0.8,                               # Mục 3.1
    "alpha_start": 1.0, "alpha_drop": 0.9,      # Eq. (7): alpha = 1 − 0,9·eval/MaxEval
    "reward": (1.0, -1.0),                      # Eq. (8): fitness tốt nhất của quần thể tăng hẳn / không
    "q_init": 0.0,                              # Algorithm 2 dòng 10
    "n_states": 1,                              # bài gốc không định nghĩa trạng thái
    "policy": "greedy", "tie_break": "lowest_index",
}

# Quyết định trước khi đăng ký PREREG (Mục 3.8, PREREG Mục 4): chạy mức 2 (jDE, JADE) hay không. False = không chạy.
RUN_TIER2 = False
# Bộ B7 đã qua cả hai cổng G1, G2 của survey (TQĐ3); điền sau bàn giao B7 bước 1, kèm SHA-256 ở EXPECTED_SPLIT_SHA256.
B7_INCLUDED = ()
N_OUTER_FOLDS = 15                              # 3 lần lặp × 5 fold

RESULT_SUBDIR = "taimma"                        # $KLTN_DATA_ROOT/results/survey_v2/taimma/<kịch bản>/<cohort>/
ENV_SMOKE = "MPQDA_SMOKE"                       # chỉ cho smoke test trên dữ liệu giả; config.json ghi smoke = true, traceable = false

# Lựa chọn cài đặt mà đặc tả chưa nêu (quy tắc chung 5); ghi vào config.json của mọi lần chạy để giảng viên duyệt.
CHOICES_TO_CONFIRM = (
    "selection_time_s và cpu_time_s đo cho từng (phương pháp, k), không gộp mọi k của một đơn vị như survey; "
    "cpu_time_s chỉ tính tiến trình chính (không gồm tiến trình con khi n_jobs_eval > 1)",
    "best_fitness = fitness của cá thể tốt nhất khi dừng (Đ7); với họ MPQDA hai dòng <tên> và <tên>_best có cùng best_fitness, "
    "output_fitness của dòng <tên> là fitness của panel đồng thuận",
    "consensus_weights.csv chỉ ghi gen có pi > 0 (gen thuộc ít nhất một thành viên archive)",
    "SHADE, JADE: số cá thể pbest = max(2, round(p_i * P)) với round của Python (làm tròn về số chẵn khi .5); "
    "r2 rút đều trên quần thể ∪ archive ngoài, trừ i và r1; archive ngoài đầy thì bỏ một phần tử rút đều",
    "SHADE, JADE: bộ nhớ chỉ cập nhật ở thế hệ có ít nhất một lần thử thành công (f(u) > f(x)); "
    "JADE dùng trung bình Lehmer không trọng số như bài gốc",
    "jDE: bốn số đều rút theo thứ tự (rand1, rand2, rand3, rand4); F' = 0,1 + 0,9*rand1 khi rand2 < tau1; CR' = rand3 khi rand4 < tau2",
    "lịch sử GA, BPSO: khối P lời gọi fitness mỗi thế hệ (history_from_log); success_rate, mean_F không xác định (NaN)",
    f"biến môi trường {ENV_SMOKE}=1 (chỉ smoke test): cho phép SHA-256 bản chia khác giá trị chờ và cờ --nan-outer-test; "
    "config.json ghi smoke = true và traceable = false",
    "T0 nhãn xáo trộn: LR-L2 ngoài dùng seed pha B của survey, seed_for(cid, split_seed, fold, k, 90)",
    "qlDE: bài gốc không định nghĩa trạng thái; cài đặt dùng một trạng thái (bảng Q 1 × 7) và một tổ hợp (F, Cr) cho cả "
    "quần thể mỗi thế hệ, như Mục 3.2 và Algorithm 2 (không phải theo từng cá thể như Bảng 6 của Review ghi)",
    "qlDE: Table 1 của bài gốc ghi alpha = 0,8 còn Mục 3.1 ghi alpha theo Eq. (7) và gamma = 0,8; cài đặt theo Mục 3.1",
    "qlDE: chính sách tham lam không có thăm dò (Algorithm 2); bài gốc không nêu cách phá hoà, cài đặt lấy chỉ số nhỏ nhất, "
    "nên thế hệ đầu luôn dùng (F, Cr) = (0,4; 0,7)",
    "qlDE: eval / MaxEval của Eq. (7) tính bằng số lần đánh giá danh nghĩa P·(t + 1) / (P·(T + 1)), không trừ lần trúng cache; "
    "bỏ tiêu chí hội tụ delta < tol của bài gốc để mọi wrapper cùng T = 50",
)


def base_method(method: str) -> str:
    """Phương pháp cho seed: mọi biến thể T2 dùng seed của mpqda_de (common random numbers)."""
    return "mpqda_de" if method in VARIANTS else method


def output_names(method: str) -> tuple[str, ...]:
    """Tên các dòng đầu ra của một lần chạy: họ MPQDA có <tên> (đồng thuận) và <tên>_best."""
    return (method, method + BEST_SUFFIX) if method in MPQDA_FAMILY else (method,)


def t1_methods() -> tuple[str, ...]:
    """Phương pháp của T1, T1b: MPQDA-DE, mức 1 và (nếu RUN_TIER2) mức 2."""
    return ("mpqda_de", *TIER1, *(TIER2 if RUN_TIER2 else ()))


def n_outputs(methods) -> int:
    return sum(len(output_names(m)) for m in methods)


def constants() -> dict:
    """Mọi hằng số (tên viết hoa) ở dạng ghi được vào config.json."""
    out = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and not name.startswith("_") and name != "C":
            out[name] = list(value) if isinstance(value, tuple) else value
    return out
