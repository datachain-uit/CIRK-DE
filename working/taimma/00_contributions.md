# Hợp đồng đóng góp — bài TAIMMA 2027

Nguồn: Review-2026-11-14 Mục 6.6, Review-2026-11-14b TQĐ14 và PREREG Mục 1. Mỗi Ci có một tiểu mục phương pháp, một tiểu mục kết
quả, một gạch đầu dòng ở Mục 1 và một mệnh đề ở tóm tắt. Cột "Trạng thái" cập nhật khi có kết quả; chưa có số nào tại 14/11/2026.
Mọi số từ pha B lấy theo B-LL (TQĐ11); B-AUC chỉ ở phụ lục độ nhạy.

| Mã | Đóng góp | RQ | Kịch bản | Giả thuyết | Mục phương pháp | Mục kết quả | Tệp số liệu | Trạng thái |
|---|---|---|---|---|---|---|---|---|
| C1 | Bài toán wrapper dưới ngân sách đúng k với fitness log-loss, kèm bằng chứng fitness AUROC bão hoà | RQ1 | T1, T1b | H1 | 3 (fitness) | 5.1 | `stats_h1_saturation.csv` | chưa chạy |
| C2 | Mã hoá lại theo dải cho random key, giữ panel của vector cơ sở khi đột biến DE với giải mã đúng k; và MPQDA-DE, kết hợp nó với leader lấy từ archive theo Jaccard và Q-learning điều khiển F theo trạng thái từng cá thể, hai thành phần điều chỉnh từ các thiết kế DE đã có cho panel đúng k | RQ2, RQ3 | T1, T2 | H2, H3, H5 (A0, A1, A2a, A2b, A3, A5_A1) | 3 | 5.2, 5.3 | `stats_h2_inner_fitness.csv`, `stats_h3_noninferiority.csv`, `table4_ablation.csv` | chưa chạy |
| C3 | Panel đồng thuận từ archive trong một lần chạy | RQ4 | T1, T3 | H3, H4, H5 (A4) | 3 | 5.4 | `table5_seed_stability.csv`, `stats_h4_stability.csv` | chưa chạy |
| C4 | Đánh giá không rò rỉ, cùng ngân sách, có đăng ký trước, kèm mã và tệp kết quả | – | mọi kịch bản | – | 4 | – | tag `taimma-2027-submission` | mã xong; chờ duyệt PREREG |

## Thành phần: mới hay điều chỉnh (TQĐ14)

| Thành phần | Mới hay điều chỉnh | Trích dẫn bắt buộc | Ghi nhận qua |
|---|---|---|---|
| Mã hoá lại theo dải (Đ2) | mới: dạng mã hoá lại cho DE random key giải mã đúng k; tiền lệ mã hoá lại random key không đổi lời giải là Bosman et al. (2016), với mục đích khác (đa dạng) | Bosman et al. (2016); Bean (1994) | A0 (H5) |
| Panel đồng thuận từ archive trong một lần chạy | mới: gộp các nghiệm trong archive của chính lần chạy, trọng số theo hạng, không tốn thêm lần chạy | Li et al. (2001) GA/KNN; Abeel et al. (2010); Meinshausen và Bühlmann (2010) | A4 (H5), H4 (i) |
| Leader lấy từ archive theo novelty và fitness | điều chỉnh cho bài toán đúng k | Wang et al. (2023, IEEE TCYB 53(7)); JADE, SHADE | A2a, A2b, A5_A1 |
| Q-learning điều khiển F theo trạng thái từng cá thể | điều chỉnh: điều khiển F theo bước ±0,05 | qlDE (2021); DE-RLFR (2019); QDEHHO (2025) | A3 |
| Khởi tạo bằng bản đồ logistic | đã biết | – | A1 |
| DE đúng k với random key và giải mã top-k | đã biết | Khushaba et al. (2011) DEFS; Bean (1994) | – |

Mọi baseline DE cũng dùng mã hoá lại theo dải và sửa biên trung điểm, nên so sánh với baseline không cho biết tác dụng của mã hoá lại;
tác dụng đó chỉ được ghi nhận qua A0.

Quy tắc đọc (PREREG Mục 8): một thành phần chỉ được viết là có đóng góp khi ít nhất một phép thử của nó có ý nghĩa sau Holm (họ 39
phép thử); không đạt thì bài viết "không có hiệu ứng đo được". Nếu H3 không đạt, bài nói rõ và không mô tả MPQDA-DE là cạnh tranh
hay tương đương. Đánh giá song song chỉ là cài đặt, không là đóng góp và không là câu hỏi nghiên cứu (TQĐ5). Bài không viết rằng
B-LL cải thiện kết quả.
