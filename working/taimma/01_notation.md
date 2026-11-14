# Bảng ký hiệu — bài TAIMMA 2027

Dùng chung cho Mục 3 của bài (Table ký hiệu), PREREG và mã. Thuật ngữ theo Eiben và Smith (2015). Cột "Trong mã" chỉ tên biến
hoặc hằng số tương ứng để người đọc đối chiếu.

## Bài toán và dữ liệu

| Ký hiệu | Nghĩa | Trong mã |
|---|---|---|
| X_tr, y_tr | phần train của một fold ngoài, sau lọc sơ bộ; nhãn 0 = LUAD, 1 = LUSC | `X_tr`, `y_tr` (`scripts/run_taimma.py`) |
| n | số mẫu train của fold ngoài | `n_train` |
| p0 | số gen sau lọc sơ bộ theo phương sai, p0 = 5.000 | `p0`, `C.PREFILTER_P0` |
| k | ngân sách: số gen của panel, k ∈ {2, 5, 10, 50} | `k` |
| S ⊂ {1, …, p0}, \|S\| = k | panel (kiểu hình) | `panel` |
| J | số fold ngoài mỗi cohort, J = 15 (3 lần lặp × 5 fold) | `M.N_OUTER_FOLDS` |

## Biểu diễn và toán tử

| Ký hiệu | Nghĩa | Trong mã |
|---|---|---|
| x_i ∈ (0, 1)^p0 | cá thể i: vector khoá (kiểu gen) | `X[i]` |
| S(x) = topk(x) | k chỉ số có khoá lớn nhất; hoà thì chỉ số nhỏ hơn | `encoding.decode_topk` |
| band(x) | mã hoá lại theo dải: gen của S(x) vào (0,5; 1), gen khác vào (0; 0,5), giữ thứ tự trong nhóm | `encoding.band_encode` |
| repair(v, x) | sửa biên trung điểm: v_j > 1 thành (1 + x_j)/2, v_j < 0 thành x_j/2 | `encoding.repair_midpoint` |
| P, T | cỡ quần thể và số thế hệ, P = 30, T = 50 | `M.P`, `M.T` |
| F_i, F_i′ | hệ số tỉ lệ của cá thể i, và giá trị thử ở thế hệ hiện tại, F ∈ [0,1; 1,0] | `ctrl.F`, `F_trial` |
| CR | xác suất lai ghép nhị thức, CR = 0,9 | `M.CR` |
| v_i, u_i | vector đột biến và vector thử | `V`, `Ut` |
| j_rand,i | toạ độ luôn lấy từ v_i trong lai ghép | `jr` |
| r1, r2 | hai cá thể khác nhau và khác i | `R` |
| L_i | leader của cá thể i, rút từ archive | `XL[i]` |
| z_i(t), μ | bản đồ logistic z(t + 1) = μ·z(t)·(1 − z(t)), μ = 3,99 | `encoding.init_chaotic`, `M.MU` |

## Fitness và ngân sách đánh giá

| Ký hiệu | Nghĩa | Trong mã |
|---|---|---|
| f(S) | fitness: −(1/3)·Σ log-loss trên 3 fold nội bộ (chính) hoặc AUROC trung bình (T1b) | `fitness.InnerCVFitness` |
| N_eval | số panel khác nhau đã chấm; N_eval ≤ P·(T + 1) + 1 = 1.531 | `n_eval` |

## Archive, leader và đồng thuận

| Ký hiệu | Nghĩa | Trong mã |
|---|---|---|
| A, \|A\| ≤ A_max | archive, A_max = 15 | `archive.Archive`, `M.A_MAX` |
| d(S_a, S_b) | khoảng cách Jaccard 1 − \|S_a ∩ S_b\| / \|S_a ∪ S_b\| | `encoding.jaccard_distance` |
| δ | ngưỡng gần, δ = 0,2 | `M.DELTA` |
| nov(a) | novelty: trung bình d tới 3 thành viên gần nhất | `Archive.novelty` |
| w_a | trọng số đồng thuận: \|A\| − hạng fitness của a + 1 | `Archive.consensus` |
| π_j | tần suất có trọng số của gen j trong archive | `pi` |
| S_c, S_best | panel đồng thuận (đầu ra chính) và panel của cá thể tốt nhất | `consensus`, `best` |

## Q-learning điều khiển F

| Ký hiệu | Nghĩa | Trong mã |
|---|---|---|
| s_i ∈ {0, 1} | trạng thái của cá thể i: 0 nếu lần thử trước cải thiện, 1 nếu không | `ctrl.s` |
| a ∈ {−0,05; 0; +0,05} | hành động cộng vào F_i | `M.ACTIONS` |
| Q ∈ R^{2×3} | bảng Q dùng chung | `ctrl.Q` |
| τ, η, γ | nhiệt độ softmax, tốc độ học, chiết khấu: 1; 0,1; 0,9 | `M.TAU`, `M.ETA`, `M.GAMMA` |
| r_i | phần thưởng: 1 nếu f(u_i) > f(x_i), ngược lại 0 | `improved` |

## Đánh giá ngoài và thống kê

| Ký hiệu | Nghĩa | Trong mã |
|---|---|---|
| d_j | hiệu ghép cặp trên fold ngoài j | `make_taimma_tables.paired` |
| ρ | tỉ số test/train của Nadeau–Bengio, ρ = 1/4 | `C.NB_TEST_TRAIN_RATIO` |
| Φ | độ ổn định Nogueira | `survey_v2.stability.nogueira` |
| m* | baseline wrapper có log-loss ngoài trung bình thấp nhất (H3) | `make_taimma_tables.comparator` |
| C ∈ {0,01; 0,1; 1; 10} | hằng số điều chuẩn của LR-L2 ở pha B (khác C = 1 cố định của fitness pha A) | `C.C_GRID` |
| C* | giá trị C có log-loss OOF nội bộ (3 fold) nhỏ nhất, hoà giữ C nhỏ hơn (B-LL, TQĐ11) | `mpqda.phase_b.evaluate_panel_ll` |
| B-LL, B-AUC | pha B chọn C theo log-loss nội bộ (chính) và theo AUROC nội bộ của survey (độ nhạy) | `taimma/T5LL/`, `taimma/T5/` |

Kiểm trùng ký hiệu: C* và C chưa dùng ở các mục trên; CR (lai ghép) và C1–C4 (đóng góp) là ký hiệu khác, không viết tắt thành C.
