# Phản hồi phản biện ICAART 2027 (nội bộ, không nộp)

Theo Bảng 1 của Review-2026-11-14. Cột "Đã xử lý ở" trỏ tới mã và tài liệu đã có trên nhánh `taimma-2027`; mục và bảng của bài
điền khi bài được viết. Trạng thái tại 14/11/2026: thiết kế và mã đã xử lý, chưa có kết quả thực nghiệm nào.

| # | Góp ý (tóm tắt) | Xử lý trong bản mới | Đã xử lý ở (mã, tài liệu) | Mục, bảng của bài |
|---|---|---|---|---|
| 1 | R2: giao thức chia dữ liệu của GSE131907 không khớp số liệu | Bỏ GSE131907; dùng bản chia theo bệnh nhân của survey (TQĐ2, TQĐ3) | `legacy_icaart/README.md`; `scripts/run_taimma.py` kiểm SHA-256 của `repeated_splits.csv`; PREREG Mục 2 | [điền] Mục 4 |
| 2 | R2, R4: một lần chạy, không có độ lệch hay kiểm định | 15 fold ngoài mỗi cohort, thêm 4 seed trên GSE37745, kiểm định ghép cặp, Friedman–Nemenyi (TQĐ7) | PREREG Mục 5, 7; `scripts/make_taimma_tables.py` | [điền] Mục 4, 5 |
| 3 | R2, R4: mỗi phương pháp một fitness, một chỉ số; thiếu DE thường và baseline DE, Q-learning | Cùng mã hoá, fitness, P, T; thêm DE/rand/1/bin, SHADE, qlDE (TQĐ6) | `mpqda/baselines.py`; `tests/test_baselines.py::test_all_wrappers_share_one_fitness_function`; qlDE chờ `docs/taimma/baselines/qlde.md` | [điền] Table 2 |
| 4 | R2, R4: không có ablation | Ablation từng yếu tố, đăng ký trước (TQĐ5, TQĐ7) | `mpqda.config.VARIANTS`; `tests/test_de.py`; PREREG Mục 3.7, H5 | [điền] Mục 5.3, Table 4 |
| 5 | R4: động lực yếu; RQ về đánh giá song song là hiển nhiên | Động lực từ bão hoà fitness và panel tuỳ ý của wrapper; song song hoá chỉ là cài đặt (TQĐ5) | PREREG Mục 1, 3.6 | [điền] Mục 1 |
| 6 | R4: kích thước tập con gần mật độ khởi tạo; hệ số phạt quá nhỏ; so với gen ngẫu nhiên và không chọn gen | Mã hoá đúng k; random-k và B0 làm mốc (TQĐ5, TQĐ6) | `mpqda/encoding.py`; `scripts/run_taimma_eval.py` (T4) | [điền] Mục 3, Table 3 |
| 7 | R1, R4: chỉ số hành vi không có định nghĩa; kích thước archive; lặp từ quần thể độc lập; độ lệch trong hình | Khoảng cách Jaccard; thuật ngữ theo Eiben và Smith; A ∈ {1, 15, 30}; đường cong anytime trung vị và IQR qua 45 lần chạy (TQĐ7, TQĐ9) | `mpqda/de.py` (`history_row`); PREREG Mục 3.8; `fig2_anytime.pdf` | [điền] Mục 3, 5.5, Fig. 2 |
| 8 | R4: tệp kết quả mở rộng không tồn tại; liên kết Zenodo hỏng | Tag mã kèm tệp kết quả; kiểm mọi liên kết trước khi nộp (TQĐ8) | `mpqda/runlog.py`; `scripts/taimma_jobs.py verify`; tag `taimma-2027-submission` (chưa tạo) | [điền] Mục 4 |
| 9 | R3: biên tập chưa chuyên nghiệp | Viết lại toàn bộ theo LNCS và quy tắc văn phong (TQĐ9) | chưa bắt đầu | toàn bài |
