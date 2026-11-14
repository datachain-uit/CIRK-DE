# legacy_icaart — không dùng, số không tái lập được

Thư mục này giữ nguyên trạng mã của bản nộp ICAART 2027 (commit `db163e7`, 30/06/2026): notebook, `tools_*.py`,
`data/selected_indices/`, `reports/` và tệp trích văn bản bài tham khảo. Các tệp chỉ được chuyển chỗ, không sửa nội dung.

**Không dùng cho bài TAIMMA 2027.** Theo TQĐ2 (Review-2026-11-14):

- Không số liệu, bảng hay hình nào của bản ICAART được dùng lại, và GSE131907 không được dùng.
- Số liệu của bản ICAART không tái lập được từ kho này: kho không có mã tạo `X_train`, `X_val`, `X_test`; một số con số trong
  bài không có trong notebook nào; ít nhất một notebook có output không do mã của nó sinh ra (Review Mục 3.1, 3.2).
- Mã ở đây không được nhập (import) từ `mpqda/`, `scripts/` hay `tests/`.

Mã của bài TAIMMA nằm ở `mpqda/`, `scripts/`, `tests/`, `vendor/` và `docs/taimma/`.
