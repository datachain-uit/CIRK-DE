# Sổ số liệu — bài TAIMMA 2027

Mọi con số trong bài phải có một dòng ở đây. Nguồn duy nhất là `claims.csv` do `scripts/make_taimma_tables.py` sinh từ kết quả
có `traceable = true` tại một commit đã push. Nguồn của từng số chỉ ghi ở đây; source LaTeX không chứa dòng chú thích (TQĐ41,
Review-2026-11-18b).

Quy tắc: không chép tay số từ log hay notebook; không dùng số của bản ICAART (TQĐ2); số của lần chạy smoke test hoặc có
`traceable = false` không được ghi. Mỗi số ghi pha của nó (cột "Pha"): B-LL (chính), B-AUC (độ nhạy, nhãn `_BAUC`) hoặc pha A;
lấy từ cột `phase_b` của `claims.csv`.

**Trạng thái tại 19/11/2026: điền cho bản sửa C1 trên source 18b (Review-2026-11-18b, TQĐ41–TQĐ45).** Nguồn: `claims_v2.csv` (tag
`taimma-2027-v2-results`, SHA-256 `813d64f5954783a7225daeeb1e02f954373bd90a62a0725b95825ed240df07ae`), `claims.csv` của v1 (tag
`taimma-2027-v1-results`; bản trong `Nop-GVHD-2026-11-17-TAIMMA/02_Bang-va-hinh/tables/`, SHA-256
`93a61ca80e0a7cb4a2b3cdc54447078f83110097fed7ee75ec059f99045a33f8`), `claims_optimism.csv` (scripts/make_taimma_optimism.py
ở `e41484e`, chạy ngày 18/11, mô tả, exploratory). Số không có trong ba tệp claims ghi tệp nguồn ở cột "Tệp". Một dòng có thể
gom nhiều số của cùng một câu; `<k>` nghĩa là k = 5 và k = 10.

| Mã số liệu (claim_id) | Câu trong bài (mục, đoạn) | Giá trị | Pha | Tệp | Cột | Bộ lọc | Commit kết quả |
|---|---|---|---|---|---|---|---|
| – | Tóm tắt (câu 3, dải dùng chung với các baseline DE); Mục 1 (gạch C1); Mục 3.3: 34–36 % gen của panel thử là gen ngẫu nhiên, giữ 47–49 % gen của leader, F = 0,5 | 0,34–0,36; 0,47–0,49 | mô phỏng | `docs/taimma/disruption.csv` | `random_genes`, `leader_kept` | CR = 0.9, eps = 0.5, F = 0.5, share ∈ {0, 0.5} | `9a752a1` |
| – | Mục 3.4 (Gapped re-encoding): gen ngẫu nhiên 0, giữ 84–87 % | 0; 0,84–0,87 | mô phỏng | `docs/taimma/disruption.csv` | `random_genes`, `leader_kept` | CR = 0.9, eps = 0.01, F = 0.5 | `9a752a1` |
| – | Mục 5.1: leader overlap ở k = 5, v1 42 % (t = 1), 53 % (t = 50); v2 67 %, 71 % | 0,416; 0,528; 0,674; 0,710 | pha A | `tables_v2/diagnostics_v2_behaviour.csv` | `leader_overlap` | method ∈ {mpqda_de, mpqda_v2}, k = 5, t ∈ {1, 50}, trung bình ba cohort | `95333d0` |
| – | Mục 5.1: v2 nhận 1,0 gen mới mỗi lần thử | 0,999 | pha A | `tables_v2/diagnostics_v2_behaviour.csv` | `n_new_genes` | mpqda_v2, t ≥ 1, trung bình | `95333d0` |
| – | Mục 5.2 (đoạn 3, trung bình bốn k); Mục 6 (Why the consensus panel…, trỏ về 5.2): độ đa dạng cuối 0,39 (v2), 0,76 (v1) | 0,391; 0,756 | pha A | `tables_v2/diagnostics_v2_behaviour.csv` | `diversity` | t = 50, trung bình bốn k (v2: 0,50; 0,36; 0,37; 0,34) | `95333d0` |
| H1:cohort=<c>:n_saturated, n_runs | Mục 5.1: fitness AUROC bão hoà 360/360, 360/360, 322/360 | 360; 360; 322 / 360 | pha A | `claims.csv` (v1) | `n_saturated`, `n_runs` | cohort ∈ {gse10245, gse30219, gse37745} | tag `taimma-2027-v1-results` |
| H2:k=<k>;cohort=<c>:decision_k; H3:k=<k>:h3_holds; H4:k=<k>;test=*:h4_holds; H5:variant=mpqda_a1_uniform/mpqda_a3_fixedF:*:significant | Mục 1 (đoạn 3); Mục 5.1: v1 không hơn DE/rand/1/bin, không đạt H3, H4; Q-learning, khởi tạo hỗn loạn không có tác dụng đo được | rejected (6/6); False; False; False (12/12) | pha A, B-LL | `claims.csv` (v1) | các cột quyết định | như mã | tag `taimma-2027-v1-results` |
| HV1:k=<k>;reference=mpqda_de:mean_diff, ci_low, ci_high, p_holm, share_v2_better | Tóm tắt; Mục 5.2 (đoạn 1); Phụ lục B C2 | k = 5: 0,0058 [0,0019; 0,0099], p_holm 0,012, 0,69; k = 10: 0,0096 [0,0075; 0,0118], p_holm < 0,001, 0,87 | pha A | `claims_v2.csv` | như mã | k ∈ {5, 10}; reference = mpqda_de | `95333d0` |
| HV1:k=<k>;reference=ga:mean_diff, ci_low, ci_high, p_holm | Mục 5.2 (đoạn 2) | k = 5: 0,0019 [−0,0013; 0,0053]; k = 10: −0,0005 [−0,0029; 0,0018]; p_holm 0,46 | pha A | `claims_v2.csv` | như mã | reference = ga | `95333d0` |
| – (hiệu của hai dòng HV1) | Mục 5.2 (đoạn 2): GA hơn v1 khoảng 0,004 và 0,010 | 0,0039; 0,0101 | pha A | `claims_v2.csv` | `mean_diff` | 0,0058 − 0,0019; 0,0096 + 0,0005 | `95333d0` |
| HV4:ablation=mpqda_v2_no_new;k=<k>:mean_diff, p_holm | Mục 5.2 (đoạn 3); Mục 6.2: 0,0698; 0,0372; 0,0175; p < 0,001 | 0,0698; 0,0372; 0,0175 | pha A | `claims_v2.csv` | như mã | k ∈ {2, 5, 10} | `95333d0` |
| HV4:ablation=mpqda_v2_no_gap;k=10:mean_diff; HV4:ablation=mpqda_v2_no_tour;k=10:mean_diff | Mục 5.2 (đoạn 3): 0,0045 và 0,0043 | 0,0045; 0,0043 | pha A | `claims_v2.csv` | `mean_diff` | k = 10 | `95333d0` |
| V2T:cohort=<c>;k=<k>;method=<m>:auroc_mean | Mục 5.3 (đoạn 1); Mục 6.4: AUROC ngoài 0,96–0,98 | min 0,963, max 0,983 | B-LL | `claims_v2.csv` (`table_v2_main.csv`) | `auroc_mean` | k ∈ {5, 10}, trung bình ba cohort | `95333d0` |
| HV2:k=<k>;reference=<r>:p_holm, logloss_half_width (và HV2_BC1, HV2_BAUC, HV2_BLLW) | Tóm tắt; Mục 5.3 (đoạn 2); Phụ lục B C3: p_holm = 1 (16 phép); nửa độ rộng B-LL 0,10–0,14, B-C1 0,06–0,08, B-AUC 0,06–0,10, B-LL-wide 0,19–0,23 | B-LL 0,100–0,138; B-C1 0,062–0,084; B-AUC 0,063–0,104; B-LL-wide 0,188–0,226 | B-LL, B-C1, B-AUC, B-LL-wide | `claims_v2.csv` | như mã | k ∈ {5, 10}; reference ∈ {bpso, mpqda_de} | `95333d0` |
| HV5:cohort=gse31210_egfr;k=<k>:auroc_v1_minus_v2, p_holm, inner_fitness_v2_minus_v1 | Mục 5.3 (đoạn 3); Phụ lục B C3 | −0,008 và 0,037; p_holm 0,81 và 0,30; +0,021 và +0,042 | B-LL | `claims_v2.csv` | như mã | k ∈ {5, 10} | `95333d0` |
| V2FRIEDMAN:method=<m>:mean_rank, cd | Mục 5.3 (đoạn 3): hạng từ 3,83 (BPSO) đến 7,58 (SHADE), v2 4,25, v1 6,50; χ² 19,5, p 0,021; CD 3,91 | như câu | B-LL | `claims_v2.csv` | `mean_rank`, `cd` | 12 khối, 10 phương pháp | `95333d0` |
| HV3:k=<k>;reference=<r>:phi_v2, phi_reference | Tóm tắt; Mục 5.4; Phụ lục B C4: Φ v2 0,042 và 0,029; v1 0,053 và 0,045; GA 0,021 và 0,025 | như câu | pha A | `claims_v2.csv` | như mã | reference ∈ {mpqda_de, ga} | `95333d0` |
| – | Mục 5.4 (đoạn 2); Mục 6.3: số gen khác nhau qua năm seed, k = 5: v2 23,0, v1 22,5, GA 23,9; k = 10: 47,1, 46,0, 47,6 | như câu | pha A, mô tả | `tables_v2/genes_across_seeds_gse37745.csv` (`scripts/genes_across_seeds.py`) | `mean` | gse37745, k ∈ {5, 10} | S1, S3, C1, C3 |
| – | Mục 5.5: AUROC ngoài với nhãn xáo trộn 0,3915 (v2), 0,435 (v2 best), 0,454 (DE-G); fitness nội bộ −0,577 | như câu | T0-v2 | `02_cong/T0v2_report.json` | `shuffle_mean_outer_auroc`, `shuffle_mean_inner_fitness` | – | `95333d0` (S0v2) |
| – | Mục 5.5: hiệu chuẩn, panel ngẫu nhiên cùng cỡ có AUROC ngoài trung bình 0,478 trên 11 hoán vị; báo động nhầm 2/11 hoán vị với v2 | 0,4780; 2/11 | mô tả | `06_calibration/calib_summary.csv` (SHA-256 `dfc3e663…`) | `random_mean_auroc`; `share_b_outside_range` | b = all (0,4780000222); output = mpqda_v2, b = all (0,1818 = 2/11) | `95333d0` (CAL) |
| B7V2:cohort=<c>;k=<k>;method=mpqda_v2:* | Mục 5.5: 0/60 panel đồng thuận của v2 có gen giới tính; AUROC giới tính 0,649 và 0,581 | 0/60; 0,649; 0,581 | mô tả | `04_b7_confounders/b7_confounders_v1v2_panels.csv`, `b7_confounders_v1v2_dataset.json` | `n_with_sex_gene`; `sex_only_auroc` | method = mpqda_v2 | `95333d0` |
| OPT:phase_b=B-C1;method=<m>;k=<k>:optimism | Mục 6.1; Fig. 2: optimism 0,10–0,13 (mọi wrapper, k = 5 và 10) | 0,099–0,134 | B-C1, exploratory | `claims_optimism.csv` | `optimism` | k ∈ {5, 10} | `e41484e` |
| OPTGAIN:phase_b=B-C1;k=<k>;reference=mpqda_de:optimism_diff | Mục 6.1: optimism v2 − v1 −0,007 và 0,008 | −0,0074; 0,0079 | B-C1, exploratory | `claims_optimism.csv` | `optimism_diff` | reference = mpqda_de | `e41484e` |
| – | Mục 6.1: nửa độ rộng CI ngoài 0,06–0,14 (B-C1 và B-LL) | 0,062–0,138 | B-C1, B-LL | `claims_v2.csv` (HV2, HV2_BC1) | `logloss_half_width` | – | `95333d0` |
| – | Mục 6 (Development data): v2 − v1 +0,0162, tốt hơn ở 72 % của 50 đơn vị, Wilcoxon một phía p = 3,6·10⁻⁴; v2 − GA −0,0035; log-loss ngoài B-LL: v1 0,352, DE/rand/1/bin 0,382, DE-G 0,416, v2 0,425, GA 0,476 | như câu | development data, mô tả | `docs/taimma/PREREG-v2.md` Mục 2.2 (DEV_results.zip, SHA-256 `be24f71d…`) | – | M1 = v2 | `20f838c` (DEV) |
| – | Mục 4.1; Mục 1: n = 55, 141, 169, 224 (126 đột biến), 493 (58); 44 đến 136 mẫu train | như câu | – | `docs/taimma/PREREG-v2.md` Mục 2.1 | – | 44 = 4/5 × 55; 136 = ⌈4/5 × 169⌉ | – |
| – | Mục 4.4; Mục 6.4: nửa độ rộng dự kiến 0,13; MDD 0,049 và 0,053 | như câu | – | `docs/taimma/PREREG-v2.md` Mục 7 | – | – | tag `taimma-2027-prereg-v2` |
| – | Mục 3.3 (câu cuối): khoá thường, 75–82 % gen của panel thử là gen ngẫu nhiên, giữ 1–11 % gen của leader, F = 0,5 (exploratory, E1) | 0,7455–0,8200; 0,006–0,108 | mô phỏng, khám phá | `docs/taimma/disruption_plain.csv` (SHA-256 `e26be0fd…`) | `random_genes`, `leader_kept` | encoding = plain, CR = 0.9, F = 0.5, share ∈ {0, 0.5} | `f3d26d1` |
| H5:variant=mpqda_a0_noband;k=<k>:Q1_inner_fitness | Mục 3.3 (câu cuối): bỏ dải khỏi v1 làm fitness nội bộ giảm 0,015 đến 0,038 (ablation A0, họ H5 của v1, có ý nghĩa ở cả ba k) | 0,0150; 0,0365; 0,0382 | pha A | `table4_ablation.csv` (v1) | `mean_diff`, `significant` | variant = mpqda_a0_noband, k ∈ {2, 5, 10}, Q1_inner_fitness | tag `taimma-2027-v1-results` |
| – | Mục 5.2 (đoạn 3, câu cuối): DE-G − DE/rand/1/bin, k = 10: 0,0068 [0,0039; 0,0097]; k = 5: CI [−0,0041; 0,0036] (exploratory, E2; quy tắc đọc TQĐ42: CI không loại 0 ở cả hai k nên dùng câu "cannot be attributed to the representation alone") | k = 5: −0,0002 [−0,0041; 0,0036], p một phía 0,59; k = 10: 0,0068 [0,0039; 0,0097], p 3,1·10⁻⁶ | pha A, khám phá | `tables_v2/explore_deg_vs_de.csv` | `mean_diff`, `ci_low`, `ci_high`, `p_one_sided` | k ∈ {5, 10}; k = 2 và 50 chỉ mô tả, không vào bài | `f3d26d1` |

## Số liệu dẫn từ survey (không chạy lại trong bài này)

Các số này đến từ kết quả của survey; bài trích survey như bản thảo đang phản biện.

| Số | Câu trong bài | Pha | Nguồn |
|---|---|---|---|
| 262/270 lần chạy GA, BPSO đạt fitness AUROC 1,000 (GSE10245 90/90, GSE30219 90/90, GSE37745 82/90) | Mục 1 (động lực) | pha A | survey commit `9b2d6a5`; `vendor/reference/E1_wrapper_feature_selection/<cohort>/selection_per_fold.csv`, cột `best_fitness`, đếm `best_fitness == 1.0` |
| 874/1.125 panel của các selector (77,7 %) nhận C = 0,01; không wrapper 811/945 (85,8 %); GA và BPSO 63/180 (35,0 %) | PREREG Mục 4 (đoạn pha B); Mục 4 của bài (lý do chọn B-LL) | B-AUC của survey | survey `dmkd-revision` commit `c1828d2`, `results/survey_v2/E1_classification/<cohort>/classification_per_fold.csv` của ba cohort GPL570; lọc `model == lr_l2`, `repeat ∈ {0, 1, 2}`, `k ∈ {5, 10}`, bỏ `b0` và `random_k`; đếm `C == 0.01`. Tính lại bằng `python scripts/count_survey_c_choice.py --survey-results <clone c1828d2>/results/survey_v2/E1_classification` ngày 14/11/2026; SHA-256 đầu vào: gse10245 `1c68bf38fac4c6a6bd90841c4d60f4c48523aa76a00843e18b6dcfd8ddeccf1b`, gse30219 `bcdb7d77a5ba4deee4d895eed2808ce26119a136ec23a6d96bf00a95a580d637`, gse37745 `db21c3648700b0d1ec4f95bcf39bdfe5b92a0023e1f95b4bb83b7501bbedbe2d` |
