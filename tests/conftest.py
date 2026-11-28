"""Shared test setup: puts ROOT/vendor then ROOT on sys.path (through the mpqda package) and the synthetic data builders.

Synthetic data only check the code; no result here goes into the paper.
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
import mpqda  # noqa: E402,F401  (puts vendor/ and ROOT on sys.path)

SMOKE_COHORT = "gse10245"


def synthetic_xy(n=60, p=400, n_inf=8, seed=0, effect=1.0):
    """Two classes 40/(n − 40); the first n_inf genes carry signal, with alternating signs."""
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
    """Synthetic data in the layout of the vendored code (survey_v2.synthetic) with repeated_splits.csv for SMOKE_COHORT."""
    from survey_v2 import config as C
    from survey_v2.data import load_dataset
    from survey_v2.e1_splits import build_repeated_split_manifest
    from survey_v2.synthetic import make_all
    make_all(root, n=120, p=300, n_target=80)
    ds = load_dataset(root, SMOKE_COHORT, strict_n=False)   # synthetic data: ignore KLTN_STRICT_N set by the Kaggle notebook
    manifest = build_repeated_split_manifest(ds.sample_ids, ds.groups, ds.y, seeds=C.REPEAT_SEEDS[:n_repeats])
    manifest.to_csv(ds.data_dir / "repeated_splits.csv", index=False)
    return ds.data_dir / "repeated_splits.csv"


def clone_repo(dest: Path) -> Path:
    """Copy of the repository inside a committed temporary Git repository, so that provenance is not affected by an edited working tree."""
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
        raise AssertionError(f"{script} returned code {res.returncode}\nSTDOUT:\n{res.stdout[-3000:]}\nSTDERR:\n{res.stderr[-3000:]}")
    return res


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory):
    """One full run of run_taimma.py on synthetic data (P = 6, T = 3); shared by the runner tests."""
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
