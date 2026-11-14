# Sổ số liệu — bài TAIMMA 2027

Mọi con số trong bài phải có một dòng ở đây. Nguồn duy nhất là `claims.csv` do `scripts/make_taimma_tables.py` sinh từ kết quả
có `traceable = true` tại một commit đã push. Sau mỗi câu có số, LaTeX của bài ghi chú `% source: <tệp>, <cột>, <bộ lọc>`.

Quy tắc: không chép tay số từ log hay notebook; không dùng số của bản ICAART (TQĐ2); số của lần chạy smoke test hoặc có
`traceable = false` không được ghi. Mỗi số ghi pha của nó (cột "Pha"): B-LL (chính), B-AUC (độ nhạy, nhãn `_BAUC`) hoặc pha A;
lấy từ cột `phase_b` của `claims.csv`.

**Trạng thái tại 14/11/2026: chưa có kết quả thực nghiệm nào; bảng dưới còn trống.**

| Mã số liệu (claim_id) | Câu trong bài (mục, đoạn) | Giá trị | Pha | Tệp | Cột | Bộ lọc | Commit kết quả |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Số liệu dẫn từ survey (không chạy lại trong bài này)

Các số này đến từ kết quả của survey; bài trích survey như bản thảo đang phản biện.

| Số | Câu trong bài | Pha | Nguồn |
|---|---|---|---|
| 262/270 lần chạy GA, BPSO đạt fitness AUROC 1,000 (GSE10245 90/90, GSE30219 90/90, GSE37745 82/90) | Mục 1 (động lực) | pha A | survey commit `9b2d6a5`; `vendor/reference/E1_wrapper_feature_selection/<cohort>/selection_per_fold.csv`, cột `best_fitness`, đếm `best_fitness == 1.0` |
| 874/1.125 panel của các selector (77,7 %) nhận C = 0,01; không wrapper 811/945 (85,8 %); GA và BPSO 63/180 (35,0 %) | PREREG Mục 4 (đoạn pha B); Mục 4 của bài (lý do chọn B-LL) | B-AUC của survey | survey `dmkd-revision` commit `c1828d2`, `results/survey_v2/E1_classification/<cohort>/classification_per_fold.csv` của ba cohort GPL570; lọc `model == lr_l2`, `repeat ∈ {0, 1, 2}`, `k ∈ {5, 10}`, bỏ `b0` và `random_k`; đếm `C == 0.01`. Tính lại bằng `python scripts/count_survey_c_choice.py --survey-results <clone c1828d2>/results/survey_v2/E1_classification` ngày 14/11/2026; SHA-256 đầu vào: gse10245 `1c68bf38fac4c6a6bd90841c4d60f4c48523aa76a00843e18b6dcfd8ddeccf1b`, gse30219 `bcdb7d77a5ba4deee4d895eed2808ce26119a136ec23a6d96bf00a95a580d637`, gse37745 `db21c3648700b0d1ec4f95bcf39bdfe5b92a0023e1f95b4bb83b7501bbedbe2d` |
