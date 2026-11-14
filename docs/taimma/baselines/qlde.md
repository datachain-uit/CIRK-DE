# qlDE — cấu hình trích từ bài gốc (Đ6)

Nguồn: Huynh TN, Do DTT, Lee J (2021) Q-learning-based parameter control in differential evolution for structural
optimization. Applied Soft Computing 107:107464. doi:10.1016/j.asoc.2021.107464

**Trạng thái: CHƯA ĐIỀN.** Toàn văn chưa được đối chiếu. Cho tới khi bảng dưới được điền đủ kèm số mục và số phương trình,
`mpqda.config.QLDE` là `None`, `mpqda.baselines.run_qlde` dừng với `NotImplementedError`, và `scripts/run_taimma.py` từ chối
`--methods qlde`. Không tự đặt tham số nào từ trí nhớ hay từ nguồn thứ cấp (Đ6; quy tắc viết của Review Mục 6.6: không dùng
tài liệu chưa đọc toàn văn để đỡ cho một khẳng định cụ thể). Hạn: trước 20:00 ngày 16/11/2026.

## 1. Cấu hình của bài gốc (điền từ toàn văn)

| Hạng mục | Giá trị trong bài gốc | Mục, phương trình, bảng |
|---|---|---|
| Trạng thái (định nghĩa, số trạng thái) | [ĐIỀN] | [ĐIỀN] |
| Hành động (số hành động; mỗi hành động đổi F, CR ra sao) | [ĐIỀN] | [ĐIỀN] |
| Giá trị hoặc bước của (F, CR); miền giá trị | [ĐIỀN] | [ĐIỀN] |
| Phần thưởng | [ĐIỀN] | [ĐIỀN] |
| Chính sách chọn hành động (ε-greedy, softmax, …) và tham số | [ĐIỀN] | [ĐIỀN] |
| Tốc độ học, hệ số chiết khấu | [ĐIỀN] | [ĐIỀN] |
| Giá trị đầu của bảng Q, F, CR | [ĐIỀN] | [ĐIỀN] |
| Bảng Q dùng chung hay theo cá thể; thời điểm cập nhật | [ĐIỀN] | [ĐIỀN] |
| Chiến lược đột biến và lai ghép của DE | [ĐIỀN] | [ĐIỀN] |
| Cỡ quần thể, số thế hệ trong bài gốc (chỉ để ghi; bài này dùng P = 30, T = 50) | [ĐIỀN] | [ĐIỀN] |

## 2. Chỗ khác bài gốc trong bài TAIMMA

Chỉ thay những phần chung cho mọi baseline DE (Mục 3.8 của YEU-CAU-DOI-CODE-2026-11-14-TAIMMA); mọi chỗ khác ghi thêm vào bảng.

| Hạng mục | Bài gốc | Bài TAIMMA | Lý do |
|---|---|---|---|
| Biểu diễn | vector thực của biến thiết kế | khoá đều `rng.random((P, p0))`, giải mã top-k, mã hoá lại theo dải | cùng mã hoá cho mọi wrapper (Đ2, TQĐ6) |
| Sửa biên | [ĐIỀN] | trung điểm theo cá thể đích | Đ1 |
| Chọn lọc | [ĐIỀN] | u_i thay x_i khi f(u_i) ≥ f(x_i) | như mọi baseline DE |
| Xử lý ràng buộc | [ĐIỀN: bài gốc là tối ưu kết cấu có ràng buộc] | không có ràng buộc (mọi cá thể khả thi) | bài toán đúng k |
| Ngân sách | [ĐIỀN] | P = 30, T = 50 | TQĐ6 |

## 3. Thứ tự rút số ngẫu nhiên của cài đặt

[ĐIỀN khi cài đặt `run_qlde`: thứ tự rút số trong khởi tạo và trong một thế hệ, như docstring của các baseline khác.]

## 4. Việc cần làm sau khi điền

1. Chép cấu hình vào `mpqda/config.py` (`QLDE = {...}`) và PREREG Mục 4, dòng qlDE.
2. Cài đặt `mpqda.baselines.run_qlde` trên `_Population` (khoá đều, mã hoá lại theo dải, sửa biên trung điểm, chọn lọc ≥).
3. Thêm `"qlde"` vào tham số của `tests/test_baselines.py::test_deterministic_exact_k_and_budget` và bỏ
   `test_qlde_stops_until_configured`; xoá dòng qlDE khỏi `CHOICES_TO_CONFIRM`.
