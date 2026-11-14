"""MPQDA-DE cho panel đúng k gen (bài TAIMMA 2027).

Gói này dùng mã survey chép nguyên trạng trong `vendor/` (`survey_v2`, `scripts.provenance`). Khi nạp gói, `ROOT/vendor`
rồi `ROOT` được đưa vào `sys.path`, để `survey_v2` và `scripts.*` của survey nạp từ `vendor/` còn script mới nạp từ gốc kho.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
for _p in (VENDOR, ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))
