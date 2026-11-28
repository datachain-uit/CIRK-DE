"""Check that vendor/ is unchanged: every file must match the SHA-256 listed in vendor/VENDORED.md.

    python scripts/vendor_survey.py --check

vendor/ is never edited. The command prints the files that are missing, unlisted or changed, and exits with code 2 if
there is any; otherwise it prints the number of files checked.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
MANIFEST = VENDOR / "VENDORED.md"
COHORTS = ("gse10245", "gse30219", "gse37745")
TESTS = ("test_e1_classification.py", "test_e1_precomputed_splits.py", "test_provenance.py", "test_published_panels.py",
         "test_runlog.py", "test_survey_v2_methods.py", "test_survey_v2_stats_stability.py", "test_survey_v3_units.py",
         "test_v317_csv_alignment.py", "test_v3_data_layout.py")
REFERENCE_FILES = ("selected_genes.csv", "selection_per_fold.csv", "config.json")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def vendor_files() -> list[Path]:
    """Every file in vendor/ except VENDORED.md and cache folders, sorted by path."""
    out = [p for p in VENDOR.rglob("*") if p.is_file() and p != MANIFEST and not (set(p.relative_to(VENDOR).parts) & SKIP_DIRS)]
    return sorted(out, key=lambda p: p.relative_to(VENDOR).as_posix())


def read_manifest(path: Path = MANIFEST) -> dict:
    """Return {"commit": ..., "info": {...}, "files": {relative path: sha256}} read from VENDORED.md."""
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
    """Differences between vendor/ and VENDORED.md; an empty list means that they match."""
    want = read_manifest()["files"]
    have = {p.relative_to(VENDOR).as_posix(): sha256(p) for p in vendor_files()}
    errors = [f"missing file: {f}" for f in sorted(set(want) - set(have))]
    errors += [f"file not listed in VENDORED.md: {f}" for f in sorted(set(have) - set(want))]
    errors += [f"SHA-256 differs: {f}" for f in sorted(set(want) & set(have)) if want[f] != have[f]]
    return errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="check vendor/ against VENDORED.md (the only action)")
    ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):                        # Windows consoles may not use UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    errors = check()
    print("\n".join(errors) if errors else f"vendor/ matches VENDORED.md ({len(read_manifest()['files'])} files)")
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
