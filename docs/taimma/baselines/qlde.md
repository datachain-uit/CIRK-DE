# qlDE — cấu hình trích từ bài gốc (Đ6)

Nguồn: Huynh TN, Do DTT, Lee J (2021) Q-learning-based parameter control in differential evolution for structural
optimization. Applied Soft Computing 107:107464. doi:10.1016/j.asoc.2021.107464

**Trạng thái: đã điền từ toàn văn ngày 14/11/2026** (bản PDF của nhà xuất bản, 14 trang). Cấu hình nằm ở
`mpqda.config.QLDE`; cài đặt là `mpqda.baselines.run_qlde`. Bốn chỗ bài gốc để ngỏ hoặc tự mâu thuẫn (Mục 3 dưới đây) đã ghi
vào `CHOICES_TO_CONFIRM` và **được giảng viên chấp nhận cả bốn** (TQĐ12, Review-2026-11-14b): một trạng thái; α theo Eq. (7) và
γ = 0,8 theo Mục 3.1; phá hoà bằng chỉ số nhỏ nhất; một tổ hợp (F, CR) cho cả quần thể mỗi thế hệ. Mục 4 của bài nêu hai chỗ bài
gốc để ngỏ (trạng thái, cách phá hoà).

## 1. Cấu hình của bài gốc

| Hạng mục | Giá trị trong bài gốc | Mục, phương trình, bảng |
|---|---|---|
| Đột biến | DE/rand/1: v_i = x_R1 + F·(x_R2 − x_R3), i ≠ R1 ≠ R2 ≠ R3 | Mục 2, Eq. (2); Algorithm 2 dòng 15–17 |
| Sửa biên | phản xạ (reflecting wall): v < x_min thành 2·x_min − v; v > x_max thành 2·x_max − v | Mục 2, Eq. (3) |
| Lai ghép | nhị thức: u_ij = v_ij nếu j = K_i hoặc rand ≤ Cr; K_i rút đều trong {1, …, D} | Mục 2, Eq. (4); Algorithm 2 dòng 18–23 |
| Chọn lọc | u_i thay x_i khi f(u_i) ≤ f(x_i) (bài toán cực tiểu) | Mục 2, Eq. (5); Algorithm 2 dòng 26–28 |
| Hành động | 7 tổ hợp [F, Cr]: [0,4; 0,7], [0,6; 0,7], [0,8; 0,7], [0,4; 0,9], [0,6; 0,9], [0,8; 0,9], [0,9; 0,9] | Mục 3.2, Eq. (9); Table 1 |
| Phạm vi của hành động | một tổ hợp cho cả quần thể trong một vòng lặp: "at the beginning of each DE iteration, a combination of control parameters is chosen" | Mục 3.2; Fig. 2 |
| Phần thưởng | r_t = +1 nếu best_{t+1} < best_t (nghiệm tốt nhất của quần thể tốt lên), ngược lại −1; tính ở cuối vòng lặp | Mục 3.2, Eq. (8) |
| Cập nhật Q | Q(s_t, a_t) ← Q(s_t, a_t) + α·[r_t + γ·max_a Q(s_{t+1}, a) − Q(s_t, a_t)], một lần mỗi vòng lặp | Mục 3.1, Eq. (6); Algorithm 2 dòng 30 |
| Tốc độ học | α(t) = 1 − 0,9·(eval / MaxEval), giảm tuyến tính | Mục 3.1, Eq. (7) |
| Hệ số chiết khấu | γ = 0,8 | Mục 3.1 |
| Chính sách | chọn hành động có Q lớn nhất ở trạng thái hiện tại (tham lam, không có thăm dò) | Algorithm 2 dòng 1–2, 14 |
| Giá trị đầu | bảng Q toàn 0 | Algorithm 2 dòng 10 |
| Trạng thái | **không định nghĩa.** Mục 3.1 chỉ nêu S = {s_1, …, s_m} ở dạng tổng quát; Mục 3.2, Table 1 và Algorithm 2 không nói trạng thái là gì hay có bao nhiêu | – |
| Ngân sách trong bài gốc | NP = 50 (và 20), MaxEval = 150.000, dừng khi δ = \|f_mean / f_best − 1\| < 1e−6 hoặc hết MaxEval; 30 lần chạy | Mục 4 |
| DE so sánh trong bài gốc | F = 0,8, Cr = 0,9 | Table 1 |

## 2. Chỗ khác bài gốc trong bài TAIMMA

Chỉ thay những phần chung cho mọi baseline DE (Mục 3.8 của YEU-CAU-DOI-CODE-2026-11-14-TAIMMA).

| Hạng mục | Bài gốc | Bài TAIMMA | Lý do |
|---|---|---|---|
| Biểu diễn | vector thực của biến thiết kế | khoá đều `rng.random((P, p0))`, giải mã top-k, mã hoá lại theo dải | cùng mã hoá cho mọi wrapper (Đ2, TQĐ6) |
| Sửa biên | phản xạ, Eq. (3) | trung điểm theo cá thể đích | Đ1, áp cho mọi biến thể DE |
| Chọn lọc | f(u) ≤ f(x), cực tiểu | f(u) ≥ f(x), cực đại | cùng quy tắc, đổi chiều vì fitness được cực đại hoá |
| Phần thưởng | best_{t+1} < best_t | fitness tốt nhất của quần thể sau chọn lọc lớn hơn hẳn trước chọn lọc | như trên |
| Xử lý ràng buộc | hàm phạt, Eq. (11), ε_1 = 1,5, ε_2 = 3 | không có | mọi cá thể đều khả thi (đúng k gen) |
| Ngân sách và điều kiện dừng | NP = 50, MaxEval = 150.000, hoặc δ < 1e−6 | P = 30, T = 50, không có tiêu chí hội tụ | TQĐ6: mọi wrapper cùng ngân sách |
| eval / MaxEval trong Eq. (7) | số lần gọi hàm mục tiêu | P·(t + 1) / (P·(T + 1)) sau thế hệ t, tức α đi từ 0,965 (t = 1) tới 0,1 (t = 50) | N_eval của bài này đếm panel khác nhau (có cache); dùng số danh nghĩa để α không phụ thuộc cache |

## 3. Chỗ bài gốc để ngỏ hoặc tự mâu thuẫn — chờ giảng viên duyệt

| # | Vấn đề | Cài đặt hiện tại | Phương án khác |
|---|---|---|---|
| 1 | Trạng thái không được định nghĩa | một trạng thái, bảng Q 1 × 7; max_a Q(s_{t+1}, a) lấy trên cùng dòng đó | tự đặt trạng thái (ví dụ theo kết quả vòng trước) thì không còn là thiết lập của bài gốc |
| 2 | Table 1 ghi "QLearning: α = 0.8", còn Mục 3.1 ghi α theo Eq. (7) và "γ = 0.8" | theo Mục 3.1: α theo Eq. (7), γ = 0,8 (đọc dòng của Table 1 là lỗi ký hiệu của γ) | α = 0,8 cố định |
| 3 | Cách phá hoà khi nhiều hành động có cùng Q lớn nhất không được nêu; với Q toàn 0 và chính sách tham lam, lựa chọn này quyết định hành động đầu tiên | chỉ số nhỏ nhất (thứ tự của Table 1), nên thế hệ đầu luôn dùng (F, Cr) = (0,4; 0,7); không tốn số ngẫu nhiên | rút đều trong các hành động hoà (thêm một số ngẫu nhiên mỗi thế hệ) |
| 4 | Algorithm 2 gọi ChooseAction trong vòng `for i = 1 → NP`, còn Mục 3.2 và Fig. 2 nói một tổ hợp mỗi vòng lặp | một tổ hợp mỗi thế hệ. Hai cách cho cùng kết quả vì bảng Q chỉ cập nhật cuối vòng lặp và chính sách tất định | – |

Hệ quả cho các tài liệu khác: Bảng 6 của Review-2026-11-14 và PREREG mô tả qlDE là "Q-learning chọn tổ hợp (F, CR) cho từng cá
thể". Theo toàn văn, qlDE chọn một tổ hợp cho cả quần thể mỗi vòng lặp. Table 1 và Table 2 của bài cần ghi đúng điều này.

Hành vi cần biết khi đọc kết quả: phần thưởng −1 làm Q của hành động vừa dùng âm, nên chính sách tham lam chuyển sang hành động
kế tiếp còn Q = 0. Bộ điều khiển vì thế đi lần lượt qua bảy tổ hợp trong những thế hệ không cải thiện và quay lại tổ hợp từng
được thưởng.

## 4. Thứ tự rút số ngẫu nhiên của cài đặt

Như DE/rand/1/bin: khoá đều `rng.random((P, p0))`; mỗi thế hệ: (r1, r2, r3) cho i = 0..P − 1 bằng
`rng.choice(np.delete(np.arange(P), i), size=3, replace=False)`; `U = rng.random((P, p0))`; `j_rand = rng.integers(0, p0, size=P)`.
Chọn hành động và cập nhật Q không dùng số ngẫu nhiên. `tests/test_baselines.py` khẳng định: với một hành động duy nhất
(0,5; 0,9), `run_qlde` cho đúng kết quả của `run_de_rand1` trên cùng seed.
