"""Thiết lập chung cho test: đưa ROOT/vendor rồi ROOT vào sys.path (qua gói mpqda) và các hàm dựng dữ liệu giả.

Dữ liệu giả chỉ để kiểm mã; không kết quả nào ở đây được đưa vào bài.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mpqda  # noqa: E402,F401  (đặt vendor/ và ROOT vào sys.path)

SMOKE_COHORT = "gse10245"


def synthetic_xy(n=60, p=400, n_inf=8, seed=0, effect=1.0):
    """Hai lớp 40/(n − 40); n_inf gen đầu mang tín hiệu, dấu xen kẽ."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(40, int), np.ones(n - 40, int)]
    rng.shuffle(y)
    X = rng.normal(8, 1, size=(n, p))
    X[:, :n_inf] += effect * y[:, None] * np.where(np.arange(n_inf) % 2 == 0, 1, -1)
    return X, y


@pytest.fixture
def xy():
    return synthetic_xy()


def make_smoke_data(root: Path, n_repeats: int = 1) -> Path:
    """Dữ liệu giả theo layout của survey (survey_v2.synthetic) kèm repeated_splits.csv cho SMOKE_COHORT."""
    from survey_v2 import config as C
    from survey_v2.data import load_dataset
    from survey_v2.e1_splits import build_repeated_split_manifest
    from survey_v2.synthetic import make_all
    make_all(root, n=120, p=300, n_target=80)
    ds = load_dataset(root, SMOKE_COHORT, strict_n=False)   # dữ liệu giả: bỏ qua KLTN_STRICT_N mà notebook Kaggle đặt
    manifest = build_repeated_split_manifest(ds.sample_ids, ds.groups, ds.y, seeds=C.REPEAT_SEEDS[:n_repeats])
    manifest.to_csv(ds.data_dir / "repeated_splits.csv", index=False)
    return ds.data_dir / "repeated_splits.csv"


def clone_repo(dest: Path) -> Path:
    """Bản sao kho trong một kho Git tạm đã commit, để provenance không vướng cây làm việc đang sửa dở."""
    repo = dest / "repo"
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git", "legacy_icaart", "*.pyc")
    for name in ("mpqda", "scripts", "vendor", "tests"):
        shutil.copytree(ROOT / name, repo / name, ignore=ignore)
    for name in (".gitattributes", ".gitignore"):
        if (ROOT / name).is_file():
            shutil.copy(ROOT / name, repo / name)
    git = ["git", "-c", "user.name=smoke", "-c", "user.email=smoke@example.invalid", "-c", "core.autocrlf=false"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run([*git, "commit", "-q", "-m", "smoke"], cwd=repo, check=True, capture_output=True)
    return repo


def smoke_env(data_root: Path, **extra) -> dict:
    env = dict(os.environ)
    for name in ("KLTN_STRICT_N", "KLTN_ALLOW_UNTRACKED"):
        env.pop(name, None)
    env.update({"KLTN_DATA_ROOT": str(data_root), "KLTN_ALLOW_UNPUSHED": "1", "MPQDA_SMOKE": "1", "PYTHONUTF8": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env.update(extra)
    return env


def run_script(repo: Path, script: str, args, env, check: bool = True):
    res = subprocess.run([sys.executable, str(repo / "scripts" / script), *map(str, args)], cwd=repo, env=env,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and res.returncode != 0:
        raise AssertionError(f"{script} trả mã {res.returncode}\nSTDOUT:\n{res.stdout[-3000:]}\nSTDERR:\n{res.stderr[-3000:]}")
    return res


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory):
    """Một lần chạy trọn run_taimma.py trên dữ liệu giả (P = 6, T = 3); dùng chung cho các test của runner."""
    base = tmp_path_factory.mktemp("smoke")
    repo, data = clone_repo(base), base / "data"
    split = make_smoke_data(data)
    env = smoke_env(data)
    args = ["--scenario", "SMOKE", "--metric", "logloss", "--datasets", SMOKE_COHORT, "--methods", "mpqda_de", "ga", "bpso",
            "de_rand1", "shade", "qlde", "jde", "jade", "mpqda_a3_fixedF", "--k", "2", "5", "--repeats", "0", "--folds", "0", "1",
            "--pop", "6", "--generations", "3", "--workers", "2"]
    first = run_script(repo, "run_taimma.py", args, env)
    out = data / "results" / "survey_v2" / "taimma" / "SMOKE" / SMOKE_COHORT
    return {"repo": repo, "data": data, "env": env, "args": args, "out": out, "first": first, "split": split}
