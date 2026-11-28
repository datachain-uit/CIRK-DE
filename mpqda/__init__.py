"""QRK-DE and CIRK-DE for panels of exactly k genes (TAIMMA 2027 study).

This package uses the vendored code, copied unchanged into `vendor/` (`survey_v2`, `scripts.provenance`). On import, `ROOT/vendor`
then `ROOT` are added to `sys.path`, so that `survey_v2` and the vendored `scripts.*` load from `vendor/` and new scripts load from the repository root.
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
