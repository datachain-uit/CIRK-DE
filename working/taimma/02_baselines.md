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
| qlDE (Huynh, Do và Lee, 2021) | random key, top-k | khoá đều | Q-learning chọn một trong bảy tổ hợp (F, CR) cho cả quần thể mỗi thế hệ; chính sách tham lam, một trạng thái, phần thưởng ±1 theo fitness tốt nhất của quần thể (TQĐ12) | không | cá thể tốt nhất |
| jDE (Brest et al., 2006), mức 2 | random key, top-k | khoá đều | tự thích nghi theo cá thể | không | cá thể tốt nhất |
| JADE (Zhang và Sanderson, 2009), mức 2 | random key, top-k | khoá đều | cập nhật μ_F, μ_CR | archive ngoài | cá thể tốt nhất |
| MPQDA-DE (bài này) | random key, top-k, mã hoá lại theo dải | bản đồ logistic | Q-learning điều khiển F theo trạng thái từng cá thể | archive theo Jaccard, chọn leader | panel đồng thuận |

Công trình liên quan, chỉ vào Table 1 và Mục 2, không chạy (TQĐ14):

| Công trình | Biểu diễn | Điều khiển | Archive | Đầu ra | Liên quan tới |
|---|---|---|---|---|---|
| DEFS (Khushaba, Al-Ani và Al-Jumaily, 2011) | k chỉ số đặc trưng định trước | cố định | không | cá thể tốt nhất | DE đúng k |
| DE-RLFR (Li, Shi, Yue, Shang và Qu, 2019) | thực, đa mục tiêu đa mốt | Q-learning, mỗi cá thể là một agent, trạng thái theo hạng fitness, hành động là ba chiến lược đột biến | – | tập Pareto | Q-learning theo cá thể |
| QDEHHO (Huang, Wang, Heidari, Wang và Chen, 2025) | nhị phân, kích thước tự do, có phạt | Q-learning chọn chiến lược và bộ (F, CR) theo cá thể | – | cá thể tốt nhất | Q-learning trong DE chọn gen |
| Wang, Xue, Liang và Zhang (2023, IEEE TCYB 53(7)) | thực có ngưỡng, kích thước tự do, đa mục tiêu | – | subarchive với crowding distance, archive dẫn hướng tìm kiếm | tập Pareto | leader lấy từ archive có xét đa dạng |
| GA/KNN (Li, Weinberg, Darden và Pedersen, 2001) | tập con kích thước cố định | – | – | xếp hạng gen theo tần suất qua nhiều nghiệm GA gần tối ưu từ nhiều lần chạy | panel đồng thuận |
| Peng, Li, Yu, Heidari, Chen và Liang (2024), tuỳ chọn | nhị phân | Q-learning điều khiển đột biến của Harris hawk | – | cá thể tốt nhất | Q-learning điều khiển metaheuristic chọn gen |

Fitness: mọi dòng của bảng đầu dùng log-loss CV nội bộ của LR-L2 (chính) và AUROC (T1b). Bài gốc của các baseline dùng hàm mục
tiêu khác; bài này chỉ giữ cơ chế tìm kiếm của chúng. Mục 4 của bài nêu hai chỗ bài gốc qlDE để ngỏ: trạng thái (dùng một trạng
thái) và cách phá hoà (chỉ số nhỏ nhất).

## Table 2 (Mục 4 của bài): cài đặt

| Phương pháp | Tham số | Ghi chú |
|---|---|---|
| GA | tournament 3, elitism 2, lai ghép giữ phần giao rồi bổ sung ngẫu nhiên, đột biến 1/k mỗi gen | mã survey nguyên trạng |
| BPSO | w 0,9 → 0,4, c1 = c2 = 2, v_max = 6, lấy mẫu sigmoid, sửa về đúng k | mã survey nguyên trạng |
| DE/rand/1/bin | F = 0,5, CR = 0,9 | |
| SHADE | H = 30, M_CR = M_F = 0,5 ban đầu, CR ~ N(M_CR; 0,1), F ~ Cauchy(M_F; 0,1), p ~ U(2/P; 0,2), archive ngoài cỡ P | trung bình có trọng số và trung bình Lehmer có trọng số |
| qlDE | 7 tổ hợp (F, Cr) của Table 1 bài gốc, phần thưởng ±1, α = 1 − 0,9·eval/MaxEval, γ = 0,8, chính sách tham lam, một trạng thái | bốn điểm bài gốc để ngỏ: `docs/taimma/baselines/qlde.md` Mục 3 |
| jDE | τ1 = τ2 = 0,1, F = 0,1 + 0,9·rand khi rút lại, F = 0,5, CR = 0,9 ban đầu | chạy (`RUN_TIER2 = True`, TQĐ13); T1, T1b |
| JADE | p = 0,05 (ít nhất 2), c = 0,1, archive ngoài cỡ P, μ_CR = μ_F = 0,5 ban đầu | chạy (`RUN_TIER2 = True`, TQĐ13); T1, T1b |
| MPQDA-DE | CR = 0,9; F ~ U(0,3; 0,8) ban đầu, F ∈ [0,1; 1,0]; A_max = 15, δ = 0,2, 3 láng giềng; trọng số leader 0,7/0,3, loại bỏ 0,5/0,5; hành động {−0,05; 0; +0,05}, η = 0,1, γ = 0,9, τ = 1; μ = 3,99 | |

Chung cho họ DE: sửa biên trung điểm theo cá thể đích, lai ghép nhị thức có j_rand, chọn lọc ≥.

## Tham chiếu không wrapper (T4)

L1-LR, stability selection, F-test, mRMR, SVM-RFE, abess (mã survey nguyên trạng), B0 (LR-L2 trên mọi gen) và random-k (100 panel
ngẫu nhiên mỗi fold trong phần lọc sơ bộ).

## Kiểm tra tính mới (tra cứu ngày 14/11/2026, hai lượt; bổ sung theo Review-2026-11-14b cùng ngày)

Phạm vi: lượt 1 là tìm kiếm web; lượt 2 tra OpenAlex và Semantic Scholar (API mở). Google Scholar chặn truy cập tự động
(HTTP 429, trang "unusual traffic") nên chưa tra được. Đã đọc toàn văn qlDE, QDEHHO và các mục 3.3–3.4 của Bosman et al.
(2016); các bài khác mới đọc tóm tắt. **Chưa phải tổng quan có hệ thống.** "Chưa thấy" chỉ có nghĩa là các lượt tra này không
thấy. Trước khi viết Mục 1–2 vẫn nên tra lại trên Google Scholar hoặc Scopus từ máy của người viết.

| Điểm của MPQDA-DE | Công trình gần nhất đã thấy | Khác ở đâu | Kết luận tạm |
|---|---|---|---|
| Biểu diễn đúng k gen trong DE | **Khushaba RN, Al-Ani A, Al-Jumaily A (2011)** DEFS, Expert Syst Appl 38(9):11515–11526, doi:10.1016/j.eswa.2011.03.028 (219 trích dẫn); Bean (1994) random key | DEFS là DE cho tập con có số đặc trưng định trước, sửa gen trùng bằng cơ chế thống kê. Theo hiểu biết của người tra (chưa đọc toàn văn), mỗi cá thể DEFS là k chỉ số thực làm tròn, không phải random key trên toàn bộ p0 gen | DE chọn đúng k đặc trưng **đã có**; phải trích DEFS và không nêu "DE đúng k" là mới. Cần đọc toàn văn DEFS để mô tả chính xác |
| Mã hoá lại theo dải (Đ2) | **Bosman PAN, Luong NH, Thierens D (2016)** Permutation GOMEA, GECCO, 637–644, doi:10.1145/2908812.2908917, Mục 3.3 "random rescaling" và Mục 3.4 "re-encoding" (đã đọc) | Bosman et al. gán lại khoá ngẫu nhiên mới cho cả quần thể đầu mỗi thế hệ, giữ nguyên hoán vị, để **duy trì đa dạng** trong GOM, bài toán hoán vị. MPQDA-DE gán lại khoá **tất định theo hạng vào hai dải tách nhau ở 0,5** (trong panel, ngoài panel), sau mỗi lần nhận cá thể, để **vector hiệu của DE truyền được panel của leader** khi giải mã top-k trên p0 lớn | ý tưởng "mã hoá lại random key mà không đổi lời giải" **đã có**; phần riêng là cấu trúc hai dải và lý do (vector hiệu trong DE xoá panel khi p0 ≫ k), có số đo hỗ trợ (A0). Phải trích Bosman et al. (2016) và nêu khác biệt |
| Q-learning theo trạng thái từng cá thể điều khiển F | **Li Z, Shi L, Yue C, Shang Z, Qu B (2019)** DE-RLFR, Swarm Evol Comput 49:234–244, doi:10.1016/j.swevo.2019.06.010: mỗi cá thể là một agent, trạng thái theo hạng fitness, hành động là 3 chiến lược đột biến. **Cao Y, Peng H (2026)** Q-LSHADE-PS, Int J Comput Sci Inf Technol 8(5):9–30, doi:10.62051/ijcsit.v8n5.02: mỗi cá thể có bảng Q riêng, trạng thái theo mức trì trệ, chọn chiến lược đột biến. **Huang C, Wang M, Heidari AA, Wang Z, Chen H (2025)** QDEHHO, doi:10.1007/s44443-025-00303-z: Q-learning chọn chiến lược và bộ (F, CR) theo cá thể, không có trạng thái, cho chọn gen. Tang L, Dong Y, Liu J (2015) IEEE TEVC 19(4):560–574: F, CR theo cá thể dựa trên hạng fitness. qlDE (2021) | Các bài trên chọn **chiến lược đột biến** (hoặc bộ tham số rời rạc); MPQDA-DE điều chỉnh **F theo bước ±0,05**, trạng thái là kết quả lần thử trước của chính cá thể, một bảng Q dùng chung | Q-learning theo trạng thái từng cá thể **đã có** (DE-RLFR 2019, Q-LSHADE-PS 2026). Chỉ nên trình bày là biến thể điều khiển F; ablation A3 quyết định có nêu được hiệu ứng hay không |
| Leader lấy từ archive theo novelty và fitness | **Wang P, Xue B, Liang J, Zhang M (2023)** Multiobjective differential evolution for feature selection in classification, IEEE Trans Cybern 53(7):4579–4593, doi:10.1109/TCYB.2021.3128540: subarchive với crowding distance, archive dẫn hướng quá trình tìm kiếm (theo tóm tắt, chưa đọc toàn văn). Bài cùng nhóm trên IEEE TEVC 27(2) (niching DE đa mục tiêu) là bài khác; lượt 2 đã gán nhầm ý này cho bài TEVC. JADE, SHADE: archive ngoài cho vector hiệu | Wang et al. đa mục tiêu, kích thước tự do; MPQDA-DE đơn mục tiêu, đúng k, novelty theo Jaccard | ý tưởng **đã có**; trích bài TCYB cho ý archive dẫn hướng. Chỉ là cách áp vào bài toán đúng k |
| Panel đồng thuận từ archive trong một lần chạy | **Li L, Weinberg CR, Darden TA, Pedersen LG (2001)** GA/KNN, Bioinformatics 17(12):1131–1142, doi:10.1093/bioinformatics/17.12.1131: xếp hạng gen theo tần suất xuất hiện trong nhiều nghiệm GA gần tối ưu, mỗi nghiệm từ một lần chạy riêng; Abeel et al. (2010), Meinshausen và Bühlmann (2010): gộp tập chọn qua nhiều mẫu con; Mandal AK et al. (2024) IEEE Access 12:62341–62357, doi:10.1109/access.2024.3390684: gộp năm bộ lọc **trước** khi DE tìm kiếm | GA/KNN là tiền lệ trực tiếp của việc gộp nhiều nghiệm metaheuristic gần tối ưu; các bài trên gộp qua nhiều lần chạy, nhiều mẫu con hoặc trước tìm kiếm. MPQDA-DE gộp các nghiệm đã có trong archive **của chính một lần chạy**, trọng số theo hạng fitness, không tốn thêm lần chạy và chỉ thêm một lần đánh giá | điểm riêng **hẹp lại**: không phải "gộp nghiệm gần tối ưu" (GA/KNN đã có) mà là gộp trong một lần chạy, trọng số theo hạng, không thêm lần chạy. Phải trích GA/KNN. Đóng góp phải dựa vào H4 (i) và A4 |
| C1: fitness log-loss thay AUROC cho wrapper đúng k, kèm bằng chứng bão hoà | không thấy bài nào đo bão hoà fitness AUROC của wrapper hay đề xuất log-loss làm fitness wrapper (tra "wrapper" + "log-loss" / "proper scoring rule" / "saturation") | – | **chưa thấy ai làm**; là đóng góp thực nghiệm, độc lập với thuật toán |

Hệ quả cho cách viết (TQĐ14, Review-2026-11-14b; thay kết luận của lượt 1 và lượt 2):
- Bài chỉ nêu là mới hai điểm cơ chế: mã hoá lại theo dải cho DE random key giải mã đúng k gen (trích Bosman et al. 2016 là tiền
  lệ mã hoá lại random key; không viết "lần đầu mã hoá lại random key"), và panel đồng thuận lấy từ archive trong một lần chạy
  (trích GA/KNN 2001; nêu được nếu H4 (i) hoặc A4 đạt). C1 là đóng góp thực nghiệm, độc lập với thuật toán.
- Mã hoá lại theo dải chỉ được ghi nhận qua A0, vì mọi baseline DE cũng dùng nó; A0 vào họ Holm chính của H5 (39 phép thử).
- DE đúng k (DEFS 2011), Q-learning theo cá thể (qlDE 2021, DE-RLFR 2019, QDEHHO 2025; Q-LSHADE-PS 2026 nếu dùng), leader từ
  archive (Wang et al. 2023, TCYB): trình bày là điều chỉnh của các ý đã có cho bài toán đúng k. QDEHHO không vào baseline.
- QDEHHO là công trình gần nhất về ứng dụng (Q-learning trong DE chọn gen trên dữ liệu biểu hiện gen). Giao thức của nó (chia
  8:2, phạt kích thước, KNN) là kiểu R4 đã phê bình, nên khác biệt về giao thức (C4) là điểm tựa an toàn.

Việc còn lại (T-Y8, trước khi viết Mục 2 ngày 18–19/11): đọc toàn văn DEFS, DE-RLFR (lượt này mới đọc qua mô tả thứ cấp,
arXiv:2501.12881) và hai bài của Wang et al. (TCYB 53(7), TEVC 27(2)), rồi trích đúng bài cho từng ý; đọc GA/KNN để mô tả chính
xác cách gộp; tra lại trên Google Scholar từ trình duyệt của người viết.
