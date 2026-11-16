"""Cổng đăng ký của MPQDA-DE v2 (Review-2026-11-16, TQĐ24), chép từ taimma_ref/registration_ref.py.

scripts/run_taimma.py gọi check_run(dataset, method, M.V2_CHOSEN) cho mọi cặp (bộ dữ liệu, phương pháp) trước khi dựng
đơn vị đầu tiên. Trên dữ liệu xác nhận (ba cohort GPL570 và hai bộ B7):
  * ứng viên M1, M2, M3 không bao giờ được chạy;
  * mpqda_v2, các biến thể ablation và de_g chỉ chạy khi cấu hình v2 đã chốt (V2_CHOSEN), HEAD mang tag đăng ký và cây
    làm việc không có thay đổi chưa commit ở tệp được theo dõi;
  * v1 và các baseline không bị chặn (cổng G-R chạy lại chúng ở commit đăng ký).
Không có biến môi trường hay cờ dòng lệnh nào bỏ qua cổng, kể cả chế độ smoke (smoke dùng tên bộ giả, không thuộc danh sách).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_V2_TAG = "taimma-2027-prereg-v2"
CONFIRMATORY_DATASETS = ("gse10245", "gse30219", "gse37745", "tcga_luad_egfr", "gse31210_egfr")
CANDIDATES = ("mpqda_v2_m1", "mpqda_v2_m2", "mpqda_v2_m3")
REGISTERED = ("mpqda_v2", "mpqda_v2_no_gap", "mpqda_v2_no_new", "mpqda_v2_no_tour", "mpqda_v2_no_adapt", "de_g")


class RegistrationError(RuntimeError):
    """Lần chạy bị cổng đăng ký từ chối."""


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def check_run(dataset: str, method: str, chosen: str | None, git=git) -> None:
    """Không trả gì nếu được chạy; RegistrationError nếu bị chặn."""
    if dataset not in CONFIRMATORY_DATASETS:
        return
    if method in CANDIDATES:
        raise RegistrationError(f"{method} là ứng viên của pha phát triển, không chạy trên {dataset}")
    if method not in REGISTERED:
        return
    if chosen is None:
        raise RegistrationError(f"{method} trên {dataset}: V2_CHOSEN chưa chốt")
    if PREREG_V2_TAG not in git("tag", "--points-at", "HEAD").split():
        raise RegistrationError(f"{method} trên {dataset}: HEAD chưa mang tag {PREREG_V2_TAG}")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise RegistrationError(f"{method} trên {dataset}: cây làm việc có thay đổi chưa commit")
