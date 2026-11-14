# Baseline — nháp Table 1 (so sánh cơ chế) và Table 2 (cài đặt)

Nguồn của từng dòng: mã trong `mpqda/baselines.py` và `vendor/survey_v2/wrappers.py`, PREREG Mục 4. Mọi wrapper dùng cùng đối
tượng fitness, cùng fold nội bộ (chế độ log-loss), cùng P = 30, T = 50 và cùng bản chia ngoài.

## Table 1 (Mục 2 của bài): so sánh cơ chế

| Phương pháp | Mã hoá đúng k | Khởi tạo | Điều khiển tham số | Archive | Đầu ra |
|---|---|---|---|---|---|
| GA (Siedlecki và Sklansky, 1989) | tập chỉ số kích thước k | đều | cố định | không | cá thể tốt nhất |
| BPSO (Kennedy và Eberhart, 1997) | nhị phân, sửa về đúng k | đều | quán tính giảm tuyến tính | không | gbest |
| DE/rand/1/bin (Storn và Price, 1997) | random key, top-k | khoá đều | cố định F = 0,5, CR = 0,9 | không | cá thể tốt nhất |
| SHADE (Tanabe và Fukunaga, 2013) | random key, top-k | khoá đều | bộ nhớ lịch sử thành công, theo cá thể | archive ngoài (đa dạng cho đột biến) | cá thể tốt nhất |
| qlDE (Huynh, Do và Lee, 2021) | random key, top-k | khoá đều | Q-learning chọn một (F, Cr) cho cả quần thể mỗi thế hệ | không | cá thể tốt nhất |
| jDE (Brest et al., 2006), mức 2 | random key, top-k | khoá đều | tự thích nghi theo cá thể | không | cá thể tốt nhất |
| JADE (Zhang và Sanderson, 2009), mức 2 | random key, top-k | khoá đều | cập nhật μ_F, μ_CR | archive ngoài | cá thể tốt nhất |
| MPQDA-DE (bài này) | random key, top-k, mã hoá lại theo dải | bản đồ logistic | Q-learning điều khiển F theo trạng thái từng cá thể | archive theo Jaccard, chọn leader | panel đồng thuận |

Fitness: mọi dòng dùng log-loss CV nội bộ của LR-L2 (chính) và AUROC (T1b). Bài gốc của các baseline dùng hàm mục tiêu khác;
bài này chỉ giữ cơ chế tìm kiếm của chúng.

## Table 2 (Mục 4 của bài): cài đặt

| Phương pháp | Tham số | Ghi chú |
|---|---|---|
| GA | tournament 3, elitism 2, lai ghép giữ phần giao rồi bổ sung ngẫu nhiên, đột biến 1/k mỗi gen | mã survey nguyên trạng |
| BPSO | w 0,9 → 0,4, c1 = c2 = 2, v_max = 6, lấy mẫu sigmoid, sửa về đúng k | mã survey nguyên trạng |
| DE/rand/1/bin | F = 0,5, CR = 0,9 | |
| SHADE | H = 30, M_CR = M_F = 0,5 ban đầu, CR ~ N(M_CR; 0,1), F ~ Cauchy(M_F; 0,1), p ~ U(2/P; 0,2), archive ngoài cỡ P | trung bình có trọng số và trung bình Lehmer có trọng số |
| qlDE | 7 tổ hợp (F, Cr) của Table 1 bài gốc, phần thưởng ±1, α = 1 − 0,9·eval/MaxEval, γ = 0,8, chính sách tham lam, một trạng thái | bốn điểm bài gốc để ngỏ: `docs/taimma/baselines/qlde.md` Mục 3 |
| jDE | τ1 = τ2 = 0,1, F = 0,1 + 0,9·rand khi rút lại, F = 0,5, CR = 0,9 ban đầu | chỉ khi `RUN_TIER2 = True` |
| JADE | p = 0,05 (ít nhất 2), c = 0,1, archive ngoài cỡ P, μ_CR = μ_F = 0,5 ban đầu | chỉ khi `RUN_TIER2 = True` |
| MPQDA-DE | CR = 0,9; F ~ U(0,3; 0,8) ban đầu, F ∈ [0,1; 1,0]; A_max = 15, δ = 0,2, 3 láng giềng; trọng số leader 0,7/0,3, loại bỏ 0,5/0,5; hành động {−0,05; 0; +0,05}, η = 0,1, γ = 0,9, τ = 1; μ = 3,99 | |

Chung cho họ DE: sửa biên trung điểm theo cá thể đích, lai ghép nhị thức có j_rand, chọn lọc ≥.

## Tham chiếu không wrapper (T4)

L1-LR, stability selection, F-test, mRMR, SVM-RFE, abess (mã survey nguyên trạng), B0 (LR-L2 trên mọi gen) và random-k (100 panel
ngẫu nhiên mỗi fold trong phần lọc sơ bộ).
