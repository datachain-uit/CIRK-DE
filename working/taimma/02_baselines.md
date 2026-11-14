# Baseline — nháp Table 1 (so sánh cơ chế) và Table 2 (cài đặt)

Nguồn của từng dòng: mã trong `mpqda/baselines.py` và `vendor/survey_v2/wrappers.py`, PREREG Mục 4. Mọi wrapper dùng cùng đối
tượng fitness, cùng fold nội bộ (chế độ log-loss), cùng P = 30, T = 50 và cùng bản chia ngoài.

## Table 1 (Mục 2 của bài): so sánh cơ chế

| Phương pháp | Mã hoá đúng k | Khởi tạo | Điều khiển tham số | Archive | Đầu ra |
|---|---|---|---|---|---|
| GA (Siedlecki và Sklansky, 1989) | tập chỉ số kích thước k | đều | cố định | không | cá thể tốt nhất |
| BPSO (Kennedy và Eberhart, 1997) | nhị phân, sửa về đúng k | đều | quán tính giảm tuyến tính | không | gbest |
| DE/rand/1/bin (Storn và Price, 1997) | random key, top-k | khoá đều | cố định F = 0,5, CR = 0,9 | không | cá thể tốt nhất |
| SHADE (Tanabe và Fukunaga, 2013) | random key, top-k | khoá đều | bộ nhớ lịch sử thành công, theo cá thể | archive ngoài (đa dạng cho đột biến) | cá thể tốt nhất |
| qlDE (Huynh, Do và Lee, 2021) | random key, top-k | khoá đều | Q-learning chọn một (F, Cr) cho cả quần thể mỗi thế hệ | không | cá thể tốt nhất |
| jDE (Brest et al., 2006), mức 2 | random key, top-k | khoá đều | tự thích nghi theo cá thể | không | cá thể tốt nhất |
| JADE (Zhang và Sanderson, 2009), mức 2 | random key, top-k | khoá đều | cập nhật μ_F, μ_CR | archive ngoài | cá thể tốt nhất |
| MPQDA-DE (bài này) | random key, top-k, mã hoá lại theo dải | bản đồ logistic | Q-learning điều khiển F theo trạng thái từng cá thể | archive theo Jaccard, chọn leader | panel đồng thuận |

Fitness: mọi dòng dùng log-loss CV nội bộ của LR-L2 (chính) và AUROC (T1b). Bài gốc của các baseline dùng hàm mục tiêu khác;
bài này chỉ giữ cơ chế tìm kiếm của chúng.

## Table 2 (Mục 4 của bài): cài đặt

| Phương pháp | Tham số | Ghi chú |
|---|---|---|
| GA | tournament 3, elitism 2, lai ghép giữ phần giao rồi bổ sung ngẫu nhiên, đột biến 1/k mỗi gen | mã survey nguyên trạng |
| BPSO | w 0,9 → 0,4, c1 = c2 = 2, v_max = 6, lấy mẫu sigmoid, sửa về đúng k | mã survey nguyên trạng |
| DE/rand/1/bin | F = 0,5, CR = 0,9 | |
| SHADE | H = 30, M_CR = M_F = 0,5 ban đầu, CR ~ N(M_CR; 0,1), F ~ Cauchy(M_F; 0,1), p ~ U(2/P; 0,2), archive ngoài cỡ P | trung bình có trọng số và trung bình Lehmer có trọng số |
| qlDE | 7 tổ hợp (F, Cr) của Table 1 bài gốc, phần thưởng ±1, α = 1 − 0,9·eval/MaxEval, γ = 0,8, chính sách tham lam, một trạng thái | bốn điểm bài gốc để ngỏ: `docs/taimma/baselines/qlde.md` Mục 3 |
| jDE | τ1 = τ2 = 0,1, F = 0,1 + 0,9·rand khi rút lại, F = 0,5, CR = 0,9 ban đầu | chỉ khi `RUN_TIER2 = True` |
| JADE | p = 0,05 (ít nhất 2), c = 0,1, archive ngoài cỡ P, μ_CR = μ_F = 0,5 ban đầu | chỉ khi `RUN_TIER2 = True` |
| MPQDA-DE | CR = 0,9; F ~ U(0,3; 0,8) ban đầu, F ∈ [0,1; 1,0]; A_max = 15, δ = 0,2, 3 láng giềng; trọng số leader 0,7/0,3, loại bỏ 0,5/0,5; hành động {−0,05; 0; +0,05}, η = 0,1, γ = 0,9, τ = 1; μ = 3,99 | |

Chung cho họ DE: sửa biên trung điểm theo cá thể đích, lai ghép nhị thức có j_rand, chọn lọc ≥.

## Tham chiếu không wrapper (T4)

L1-LR, stability selection, F-test, mRMR, SVM-RFE, abess (mã survey nguyên trạng), B0 (LR-L2 trên mọi gen) và random-k (100 panel
ngẫu nhiên mỗi fold trong phần lọc sơ bộ).

## Kiểm tra tính mới (tra cứu ngày 14/11/2026)

Phạm vi: tìm kiếm web (Google Scholar không được dùng) và đọc toàn văn hai bài; **chưa phải tổng quan có hệ thống**. "Không
tìm thấy" dưới đây chỉ có nghĩa là các lượt tra này không thấy, không phải bằng chứng rằng chưa ai làm. Trước khi viết Mục 1
và Mục 2 của bài, cần tra thêm trên Scopus hoặc Google Scholar với các cụm từ ở cột cuối.

| Điểm của MPQDA-DE | Công trình gần nhất đã thấy | Khác ở đâu | Kết luận tạm | Tra thêm |
|---|---|---|---|---|
| Mã hoá lại theo dải cho random key, để vector hiệu không xoá panel của leader (Đ2) | Bean (1994); Londe et al. (2025) tổng quan RKGA (arXiv:2506.02120); Whalen (2018), luận văn DE random key cho chọn đặc trưng trong dự đoán hệ gen | RKGA lai ghép bằng cách chép khoá của cha mẹ, không có vector hiệu nên không gặp vấn đề này; không nguồn nào đã thấy mã hoá lại khoá theo panel | chưa thấy ai làm; điểm kỹ thuật dễ bảo vệ nhất | "random key" + "differential evolution" + "re-encoding" / "key normalization" / "rank-based keys"; đọc toàn văn Whalen (2018) |
| Q-learning theo trạng thái từng cá thể điều khiển F | **Huang C, Wang M, Heidari AA, Wang Z, Chen H (2025)** QDEHHO, J King Saud Univ Comput Inf Sci 37:280, doi:10.1007/s44443-025-00303-z (đọc toàn văn); DE-RLFR (mỗi cá thể là một agent, trạng thái theo hạng fitness, hành động là chiến lược đột biến; chỉ thấy qua mô tả thứ cấp ở arXiv:2501.12881); qlDE (2021) | QDEHHO: mỗi cá thể chọn ε-greedy một trong 4 chiến lược đột biến và 5 bộ (F, CR), nhưng bảng Q **không có trạng thái** (chỉ đánh chỉ số theo hành động, Eq. 17–19), phần thưởng là mức cải thiện liên tục, mã hoá nhị phân sigmoid, fitness = độ chính xác KNN + tỉ lệ đặc trưng. DE-RLFR có trạng thái theo cá thể nhưng chọn chiến lược đột biến, không điều khiển F. qlDE: một tổ hợp cho cả quần thể | phần "trạng thái theo cá thể" đã có ở DE-RLFR; cái riêng còn lại hẹp: trạng thái là kết quả lần thử trước của chính cá thể, hành động là bước tăng giảm F | tìm toàn văn DE-RLFR (Li et al.); "per-individual state" Q-learning DE parameter |
| Leader lấy từ archive theo novelty và fitness | Wang P, Xue B, Liang J, Zhang M (2023) niching DE đa mục tiêu cho chọn đặc trưng (đã có trong Bảng 11 của Review): dùng độ đông đúc để tỉa archive tinh hoa và chọn leader toàn cục; JADE, SHADE: archive ngoài chỉ cho vector hiệu | Wang et al. đa mục tiêu, kích thước tự do; MPQDA-DE đơn mục tiêu, đúng k, novelty theo Jaccard giữa panel | ý tưởng "leader từ archive có xét đa dạng" đã có; không nên trình bày là mới, chỉ là cách áp vào bài toán đúng k | "novelty" + "archive" + "base vector" differential evolution |
| Panel đồng thuận từ archive | Abeel et al. (2010), Meinshausen và Bühlmann (2010): gộp tập chọn qua nhiều mẫu con hoặc nhiều lần chạy | MPQDA-DE gộp các nghiệm gần tối ưu **trong một lần chạy**, có trọng số theo hạng fitness, không tốn thêm lần chạy | chưa thấy ai làm trong DE đúng k; đóng góp phải dựa vào H4 | "ensemble" / "consensus" + "evolutionary feature selection" + "archive" + "stability" |

QDEHHO phải có trong Table 1 và Mục 2 của bài. Đây là công trình mới nhất đã thấy dùng Q-learning trong DE để chọn gen trên dữ
liệu biểu hiện gen nhiều chiều (13 bộ, 2.000–15.000 đặc trưng). Đánh giá của bài đó chia 8:2 rồi CV 10 fold, P = 20, T = 50,
fitness có hệ số phạt kích thước. Đây là đúng kiểu thiết kế mà R4 đã phê bình ở bản ICAART, nên phần khác biệt về giao thức
(đúng k, fitness log-loss, cùng ngân sách, không rò rỉ, đăng ký trước) là điểm tựa an toàn hơn phần khác biệt về cơ chế.
QDEHHO không cần đưa vào baseline (TQĐ6 đã chốt danh sách), nhưng nếu phản biện hỏi thì đây là ứng viên mức 2.

Hệ quả cho cách viết, tính tới giờ: hai đóng góp có thể nêu là mới ở mức cơ chế là mã hoá lại theo dải và panel đồng thuận
trong một lần chạy. Archive làm nguồn leader và Q-learning điều khiển F nên trình bày là "điều chỉnh cho bài toán đúng k" của
các ý đã có, trích Wang et al. (2023), qlDE, QDEHHO và DE-RLFR. C1 (bão hoà fitness AUROC) và C4 (giao thức) không phụ thuộc
vào tra cứu này.
