"""vendor/ giống từng byte với commit nguồn ghi trong VENDORED.md; test của survey chạy lại đạt (log riêng)."""
import os
import subprocess
import sys

import pytest

from conftest import ROOT
from scripts import vendor_survey as V


def test_every_vendor_file_matches_manifest():
    manifest = V.read_manifest()
    assert manifest["commit"] and len(manifest["commit"]) == 40
    assert len(manifest["files"]) >= 40
    assert V.check() == []


def test_manifest_lists_required_parts():
    files = set(V.read_manifest()["files"])
    need = {"survey_v2/wrappers.py", "survey_v2/config.py", "survey_v2/panels/published_panels.csv", "scripts/provenance.py",
            "scripts/run_e1_feature_selection.py", "scripts/run_e1_classification.py", "pytest.ini", "requirements-kaggle.txt",
            "LICENSE", *(f"tests/{t}" for t in V.TESTS),
            *(f"reference/E1_wrapper_feature_selection/{c}/{f}" for c in V.COHORTS for f in V.REFERENCE_FILES)}
    assert need <= files
    assert not (ROOT / "vendor" / "scripts" / "__init__.py").exists() and not (ROOT / "scripts" / "__init__.py").exists()
    assert "vendor/** -text" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


@pytest.mark.slow
def test_survey_tests_pass(tmp_path):
    """pytest vendor/tests -m "not slow"; log ghi vào logs/pytest_vendor.log (thư mục logs/ không được Git theo dõi)."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    for name in ("KLTN_DATA_ROOT", "KLTN_STRICT_N", "KLTN_ALLOW_UNPUSHED", "KLTN_ALLOW_UNTRACKED", "MPQDA_SMOKE"):
        env.pop(name, None)
    res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-m", "not slow", "-q", "-p", "no:cacheprovider",
                          f"--basetemp={tmp_path / 'vendor'}"], cwd=ROOT / "vendor", env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    log = ROOT / "logs" / "pytest_vendor.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
    tail = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-500:]
    missing = [m for m in ("skrebate", "abess") if f"No module named '{m}'" in res.stdout]
    if res.returncode != 0 and missing:
        pytest.xfail(f"máy này thiếu {missing} (có trong requirements-kaggle.txt): {tail}")
    assert res.returncode == 0, res.stdout[-3000:]
