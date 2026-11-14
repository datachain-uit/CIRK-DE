# Sổ số liệu — bài TAIMMA 2027

Mọi con số trong bài phải có một dòng ở đây. Nguồn duy nhất là `claims.csv` do `scripts/make_taimma_tables.py` sinh từ kết quả
có `traceable = true` tại một commit đã push. Sau mỗi câu có số, LaTeX của bài ghi chú `% source: <tệp>, <cột>, <bộ lọc>`.

Quy tắc: không chép tay số từ log hay notebook; không dùng số của bản ICAART (TQĐ2); số của lần chạy smoke test hoặc có
`traceable = false` không được ghi.

**Trạng thái tại 14/11/2026: chưa có kết quả thực nghiệm nào; bảng dưới còn trống.**

| Mã số liệu (claim_id) | Câu trong bài (mục, đoạn) | Giá trị | Tệp | Cột | Bộ lọc | Commit kết quả |
|---|---|---|---|---|---|---|
| | | | | | | |

## Số liệu dẫn từ survey (không chạy lại trong bài này)

Các số này đến từ kết quả của survey, đã truy vết tại commit `9b2d6a5`; bài trích survey như bản thảo đang phản biện.

| Số | Câu trong bài | Nguồn |
|---|---|---|
| 262/270 lần chạy GA, BPSO đạt fitness AUROC 1,000 (GSE10245 90/90, GSE30219 90/90, GSE37745 82/90) | Mục 1 (động lực) | `vendor/reference/E1_wrapper_feature_selection/<cohort>/selection_per_fold.csv`, cột `best_fitness`, đếm `best_fitness == 1.0` |
