# Hợp đồng đóng góp — bài TAIMMA 2027, vòng 3 (Review-2026-11-17, TQĐ28)

Trọng tâm mới: bài không còn hứa "phương pháp mới tốt hơn baseline". Bài là một nghiên cứu chẩn đoán và sửa lỗi: chỉ ra cơ chế
làm v1 tìm kiếm kém, sửa cơ chế đó bằng v2 và kiểm từng thành phần, đo xem lợi ích tìm kiếm có chuyển thành hiệu năng ngoài hay
không, và đánh giá độ ổn định của panel đồng thuận. Mọi khẳng định đi qua quy tắc đọc của PREREG-v2 Mục 8.

Thiết kế tách pha phát triển và pha xác nhận, cổng đăng ký và độ chính xác dự kiến được trình bày ở mục Thực nghiệm như nguyên
tắc thiết kế, không thành một đóng góp đánh số.

Cột "Kết quả" và "Mệnh đề được viết" chỉ điền sau khi có bảng của pha xác nhận (19/11), đúng theo cột "Khẳng định được phép".
Mỗi số ghi nguồn trong `claims_v2.csv` (v2) hoặc `claims.csv` (v1) và vào `03_claims.md`.

| Mã | Thành phần và vấn đề | Thí nghiệm, bảng | Khẳng định được phép | Kết quả | Mệnh đề được viết |
|---|---|---|---|---|---|
| C1 | Chẩn đoán toán tử của v1: mã hoá lại theo dải với dải dưới rộng 1/2 cho vector hiệu đẩy gen ngẫu nhiên vào panel thử | Mô phỏng `docs/taimma/disruption.csv` (Bảng 5 của Review-2026-11-16); leader overlap của S1; `diagnostics_v2_behaviour` | Mô tả cơ chế bằng số đo; không cần kiểm định | Đã có: ở F = 0,5, CR = 0,9, khoảng 34–36 % gen của panel thử là gen ngẫu nhiên và panel thử giữ 47–49 % gen của leader; trên S1, leader overlap của v1 là 0,44 (t = 1) và 0,53 (t = 50) | [sau 19/11] |
| C2 | MPQDA-DE v2: khe (eps = 0,01), gen mới với xác suất 1/k, tournament 3 trên archive cho leader và r1 | HV1 (so với v1 và GA), HV4 (ablation), hành vi theo thế hệ; DE-G là đối chứng nội bộ | "v2 tìm kiếm tốt hơn v1" chỉ ở k mà HV1 so với v1 đạt; một thành phần "có tác dụng" chỉ khi HV4 ghi nhận nó; nếu HV1 so với GA không đạt, báo hiệu và CI bootstrap, không nói tương đương | [sau 19/11] | [sau 19/11] |
| C3 | Quan hệ giữa tìm kiếm và khái quát hoá: lợi ích nội bộ có chuyển ra ngoài không | HV2 trên GPL570 dưới bốn tiêu chí pha B; HV5 trên `gse31210_egfr`; chẩn đoán DEV (mô tả, nhãn "development data, exploratory") | "Đánh đổi" chỉ ở k mà HV1 so với v1, HV5 đều đạt và fitness nội bộ trung bình của v2 trên `gse31210_egfr` cao hơn v1; nếu không thì chỉ báo ước lượng và CI. HV5 có độ mạnh thấp (MDD 0,049 và 0,053 so với hiệu 0,009 trên DEV) | [sau 19/11] | [sau 19/11] |
| C4 | Độ ổn định của panel đồng thuận theo seed | HV3 (GSE37745, seed 0–4) | "Ổn định hơn" chỉ khi HV3 đạt | [sau 19/11] | [sau 19/11] |

Giữ từ v1 (báo cáo trong mục Thực nghiệm, không đánh số lại): bão hoà của fitness AUROC (H1 của v1: 360/360, 360/360, 322/360);
H2–H4 của v1 không đạt; không cặp phương pháp nào khác nhau vượt CD; độ chính xác của thiết kế (nửa độ rộng CI của H3 khoảng
0,13); sai lệch khi pha B chọn C theo AUROC.

Không dùng: "competitive", "superior", "state-of-the-art" nếu không có số liệu và phép thử đã đăng ký đi kèm; không đặt số của
DEV cạnh số xác nhận trong cùng một bảng; không dùng số, bảng hay hình nào của bản ICAART.
