"""B7 (T-Y5, TQĐ17): bộ B7 nằm trong B7_INCLUDED phải có SHA-256 bản chia chờ, và runner dừng khi bản chia khác giá trị đó."""
import re
import shutil

import pytest
from survey_v2 import config as C

from conftest import SMOKE_COHORT, run_script, smoke_env
from mpqda import config as M


def test_every_included_b7_dataset_has_an_expected_split_sha():
    assert set(M.B7_INCLUDED) <= set(C.B7_DATASETS)
    for name in M.B7_INCLUDED:
        assert re.fullmatch(r"[0-9a-f]{64}", M.EXPECTED_SPLIT_SHA256.get(name, "")), name
        assert C.DATASET_DIRS[name].startswith(C.B7_DATA_DIR + "/")
        assert C.cohort_id(name) in (105, 106)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(M.B7_INCLUDED))
def test_runner_stops_on_unexpected_b7_split_sha(smoke_run, name):
    """Dựng bộ B7 giả (bản sao dữ liệu giả của SMOKE_COHORT) ở experiment-data-b7/<bộ>/: bản chia khác giá trị chờ nên
    runner (không ở chế độ smoke) dừng với mã 2 và không ghi kết quả."""
    data = smoke_run["data"]
    src = data / C.DATASET_DIRS[SMOKE_COHORT]
    dst = data / C.DATASET_DIRS[name]
    if not dst.exists():
        shutil.copytree(src, dst)
    env = smoke_env(data)
    env.pop("MPQDA_SMOKE")
    args = ["--scenario", "B7_sha", "--metric", "logloss", "--datasets", name, "--methods", "mpqda_de", "--k", "5",
            "--repeats", "0", "--folds", "0", "--pop", "6", "--generations", "3"]
    res = run_script(smoke_run["repo"], "run_taimma.py", args, env, check=False)
    assert res.returncode == 2 and "SHA-256" in res.stderr and M.EXPECTED_SPLIT_SHA256[name] in res.stderr
    assert not (data / "results" / "survey_v2" / "taimma" / "B7_sha" / name / "selection_per_fold.csv").exists()
