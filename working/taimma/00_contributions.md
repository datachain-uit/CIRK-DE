# Hợp đồng đóng góp — bài TAIMMA 2027

Nguồn: Review-2026-11-14 Mục 6.6 và PREREG Mục 1. Mỗi Ci có một tiểu mục phương pháp, một tiểu mục kết quả, một gạch đầu dòng ở
Mục 1 và một mệnh đề ở tóm tắt. Cột "Trạng thái" cập nhật khi có kết quả; chưa có số nào tại 14/11/2026.

| Mã | Đóng góp | RQ | Kịch bản | Giả thuyết | Mục phương pháp | Mục kết quả | Tệp số liệu | Trạng thái |
|---|---|---|---|---|---|---|---|---|
| C1 | Bài toán wrapper dưới ngân sách đúng k với fitness log-loss, kèm bằng chứng fitness AUROC bão hoà | RQ1 | T1, T1b | H1 | 3 (fitness) | 5.1 | `stats_h1_saturation.csv` | chưa chạy |
| C2 | MPQDA-DE: DE random key đúng k có mã hoá lại theo dải, leader từ archive theo novelty và fitness, Q-learning điều khiển F theo trạng thái từng cá thể | RQ2, RQ3 | T1, T2 | H2, H3, H5 | 3 | 5.2, 5.3 | `stats_h2_inner_fitness.csv`, `stats_h3_noninferiority.csv`, `table4_ablation.csv` | chưa chạy |
| C3 | Panel đồng thuận từ archive | RQ4 | T1, T3 | H3, H4, H5 (A4) | 3 | 5.4 | `table5_seed_stability.csv`, `stats_h4_stability.csv` | chưa chạy |
| C4 | Đánh giá không rò rỉ, cùng ngân sách, có đăng ký trước, kèm mã và tệp kết quả | – | mọi kịch bản | – | 4 | – | tag `taimma-2027-submission` | mã xong; chờ duyệt PREREG và bốn điểm mở của qlDE |

Quy tắc đọc (PREREG Mục 8): một thành phần chỉ được viết là có đóng góp khi ít nhất một phép thử của nó có ý nghĩa sau Holm; không
đạt thì bài viết "không có hiệu ứng đo được". Nếu H3 không đạt, bài nói rõ và không mô tả MPQDA-DE là cạnh tranh hay tương đương.
Đánh giá song song chỉ là cài đặt, không là đóng góp và không là câu hỏi nghiên cứu (TQĐ5).
