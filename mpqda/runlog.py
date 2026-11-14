"""Truy vết cho script ở gốc kho (Mục 2): thay `survey_v2.runlog.start_run`.

`REPO` của bản vendor là `vendor/`, nên `is_tracked` của survey báo sai cho script nằm ngoài `vendor/`. Ở đây provenance
và kiểm tra truy vết chạy với `repo = ROOT`. Ghi config vẫn dùng `write_config` của survey; thư mục kết quả là
`$KLTN_DATA_ROOT/results/survey_v2/taimma/<kịch bản>/<cohort>/` (chung cho pha A và pha B).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts.provenance import ProvenanceError, provenance
from survey_v2 import runlog as survey_runlog
from survey_v2.runlog import DATA_ROOT, ENV_ALLOW_UNPUSHED, all_versions, check_traceable, results_dir, write_config

from . import ROOT, VENDOR
from . import config as M

__all__ = ["DATA_ROOT", "all_versions", "results_dir", "write_config", "start_run", "smoke_mode", "vendor_commit",
           "head_commit", "utf8_console"]


def utf8_console() -> None:
    """Windows: console không phải UTF-8 làm lỗi khi in tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def smoke_mode() -> bool:
    return os.environ.get(M.ENV_SMOKE, "").strip().lower() in {"1", "true", "yes"}


def vendor_commit() -> str | None:
    """Commit nguồn của vendor/, đọc từ dòng `| Commit | ... |` của vendor/VENDORED.md."""
    path = VENDOR / "VENDORED.md"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] == "Commit":
            return cells[1]
    return None


def head_commit(repo: Path = ROOT) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def start_run(script, params: dict | None = None) -> dict:
    """provenance(repo_dir=ROOT) + check_traceable(repo=ROOT); trả về dict provenance để stamp vào config.

    Dừng (mã 2) nếu còn thay đổi chưa commit ở tệp đã theo dõi, script chưa được Git theo dõi hoặc HEAD chưa có trên
    remote (trừ khi KLTN_ALLOW_UNPUSHED=1; khi đó traceable = false). Ở chế độ smoke, traceable luôn là false."""
    if not os.environ.get("KLTN_DATA_ROOT", "").strip():           # quy tắc chung 3: KLTN_DATA_ROOT bắt buộc
        print("[provenance] DỪNG: đặt KLTN_DATA_ROOT (thư mục dữ liệu và kết quả).", file=sys.stderr)
        raise SystemExit(2)
    try:
        prov = provenance(script=script, repo_dir=ROOT, params=params or {})
    except ProvenanceError as e:
        print(f"[provenance] DỪNG: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    allow = os.environ.get(ENV_ALLOW_UNPUSHED, "").strip().lower() in {"1", "true", "yes"}
    errors = check_traceable(script, repo=ROOT, require_push=True) if prov.get("tracked") else []
    if smoke_mode():
        errors = errors + [f"{M.ENV_SMOKE}=1: lần chạy smoke test, không được trích"]
    prov["traceable"] = bool(prov.get("tracked", False) and not errors)
    prov["traceable_errors"] = errors
    prov["vendor_commit"] = vendor_commit()
    if errors and not (allow or smoke_mode()):
        print("[traceable] DỪNG:\n  - " + "\n  - ".join(errors)
              + f"\n  (tạm bỏ qua bằng {ENV_ALLOW_UNPUSHED}=1; kết quả khi đó KHÔNG được trích)", file=sys.stderr)
        raise SystemExit(2)
    return prov


assert survey_runlog.REPO == VENDOR, "survey_v2 phải được nạp từ vendor/"
