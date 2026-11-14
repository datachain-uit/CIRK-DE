"""Chép nguyên trạng mã survey vào vendor/ và ghi vendor/VENDORED.md (Mục 2 của YEU-CAU-DOI-CODE-2026-11-14-TAIMMA).

    python scripts/vendor_survey.py --survey <bản clone RNA-Seq-FS-Survey> --commit <hash>

Tệp được lấy thẳng từ đối tượng Git của commit nguồn (`git archive`, tắt chuyển đổi cuối dòng), nên giống từng byte với
kho nguồn. Không sửa tay tệp nào trong vendor/; muốn cập nhật thì chạy lại script này với commit mới rồi commit riêng
`vendor: update to <hash>`. `--check` chỉ đối chiếu vendor/ với VENDORED.md, không ghi gì.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
MANIFEST = VENDOR / "VENDORED.md"
SOURCE_REPO = "datachain-uit/RNA-Seq-FS-Survey"
SOURCE_BRANCH = "dmkd-revision"
REFERENCE_COMMIT = "9b2d6a5"          # commit sinh kết quả GA, BPSO của survey (cổng G-T1)
COHORTS = ("gse10245", "gse30219", "gse37745")
TESTS = ("test_e1_classification.py", "test_e1_precomputed_splits.py", "test_provenance.py", "test_published_panels.py",
         "test_runlog.py", "test_survey_v2_methods.py", "test_survey_v2_stats_stability.py", "test_survey_v3_units.py",
         "test_v317_csv_alignment.py", "test_v3_data_layout.py")
PATHS = ("survey_v2", "scripts/provenance.py", "scripts/run_e1_feature_selection.py", "scripts/run_e1_classification.py",
         *(f"tests/{t}" for t in TESTS), "pytest.ini", "requirements-kaggle.txt", "LICENSE")
REFERENCE_SRC = "results/survey_v2/E1_wrapper_feature_selection"
REFERENCE_FILES = ("selected_genes.csv", "selection_per_fold.csv", "config.json")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def vendor_files() -> list[Path]:
    """Mọi tệp trong vendor/ trừ VENDORED.md và thư mục cache, theo thứ tự đường dẫn."""
    out = [p for p in VENDOR.rglob("*") if p.is_file() and p != MANIFEST and not (set(p.relative_to(VENDOR).parts) & SKIP_DIRS)]
    return sorted(out, key=lambda p: p.relative_to(VENDOR).as_posix())


def read_manifest(path: Path = MANIFEST) -> dict:
    """Trả về {"commit": ..., "files": {đường dẫn tương đối: sha256}} từ VENDORED.md."""
    info, files = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) != 2 or set(cells[0]) <= {"-"}:
            continue
        if len(cells[1]) == 64 and all(ch in "0123456789abcdef" for ch in cells[1]):
            files[cells[0]] = cells[1]
        else:
            info[cells[0]] = cells[1]
    return {"commit": info.get("Commit"), "info": info, "files": files}


def check() -> list[str]:
    """Danh sách lệch giữa vendor/ và VENDORED.md; rỗng = khớp."""
    want = read_manifest()["files"]
    have = {p.relative_to(VENDOR).as_posix(): sha256(p) for p in vendor_files()}
    errors = [f"thiếu tệp: {f}" for f in sorted(set(want) - set(have))]
    errors += [f"tệp không có trong VENDORED.md: {f}" for f in sorted(set(have) - set(want))]
    errors += [f"SHA-256 khác: {f}" for f in sorted(set(want) & set(have)) if want[f] != have[f]]
    return errors


def _archive(survey: Path, commit: str, paths) -> tarfile.TarFile:
    raw = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", commit, *paths], cwd=survey, check=True,
                         capture_output=True).stdout
    return tarfile.open(fileobj=io.BytesIO(raw))


def _extract(tar: tarfile.TarFile, dest: Path, strip: int = 0) -> None:
    for member in tar.getmembers():
        if not member.isfile():
            continue
        target = dest.joinpath(*Path(member.name).parts[strip:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tar.extractfile(member).read())


def copy(survey: Path, commit: str) -> None:
    full = subprocess.run(["git", "rev-parse", commit], cwd=survey, check=True, capture_output=True, text=True).stdout.strip()
    date = subprocess.run(["git", "log", "-1", "--format=%cI", full], cwd=survey, check=True, capture_output=True,
                          text=True).stdout.strip()
    for item in VENDOR.iterdir() if VENDOR.exists() else ():
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    _extract(_archive(survey, full, PATHS), VENDOR)
    ref_paths = [f"{REFERENCE_SRC}/{c}/{f}" for c in COHORTS for f in REFERENCE_FILES]
    _extract(_archive(survey, full, ref_paths), VENDOR / "reference", strip=2)
    from datetime import date as _date
    lines = ["# Mã survey được chép nguyên trạng (vendored)", "",
             "Không sửa tệp nào trong thư mục này. Cập nhật bằng `python scripts/vendor_survey.py --survey <clone> --commit <hash>`",
             "rồi commit riêng `vendor: update to <hash>`. `tests/test_vendor_integrity.py` đối chiếu SHA-256 của mọi tệp với bảng dưới.",
             "", "| Mục | Giá trị |", "|---|---|",
             f"| Kho nguồn | `{SOURCE_REPO}` |", f"| Nhánh | `{SOURCE_BRANCH}` |", f"| Commit | `{full}` |",
             f"| Ngày commit nguồn | `{date}` |", f"| Ngày chép | `{_date.today().isoformat()}` |",
             "| Giấy phép | MIT (`vendor/LICENSE`) |",
             f"| reference/ | kết quả GA, BPSO của survey tại commit `{REFERENCE_COMMIT}`, lấy từ `{REFERENCE_SRC}/` của commit nguồn |",
             "", "## Tệp và SHA-256", "", "| Tệp | SHA-256 |", "|---|---|"]
    lines += [f"| `{p.relative_to(VENDOR).as_posix()}` | `{sha256(p)}` |" for p in vendor_files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", type=Path, help="bản clone RNA-Seq-FS-Survey có commit nguồn")
    ap.add_argument("--commit", help="commit nguồn trên nhánh dmkd-revision")
    ap.add_argument("--check", action="store_true", help="chỉ đối chiếu vendor/ với VENDORED.md")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):                        # Windows: console không phải UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if not args.check:
        if not (args.survey and args.commit):
            ap.error("cần --survey và --commit (hoặc --check)")
        copy(args.survey.resolve(), args.commit)
    errors = check()
    print("\n".join(errors) if errors else f"vendor/ khớp VENDORED.md ({len(read_manifest()['files'])} tệp)")
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
