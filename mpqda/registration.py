"""Registration gate of CIRK-DE (PREREG-v2, Section 9).

scripts/run_taimma.py calls check_run(dataset, method, M.V2_CHOSEN) for every (dataset, method) pair before building the
first unit. On the confirmatory data (the three GPL570 cohorts and the two B7 datasets):
  * the candidates M1, M2, M3 are never run;
  * mpqda_v2, the ablation variants and de_g run only when the v2 configuration is fixed (V2_CHOSEN), HEAD carries a registration tag and the
    working tree has no uncommitted change in tracked files;
  * v1 and the baselines are not blocked (gate G-R reruns them at the registration commit).
No environment variable or command-line flag bypasses the gate, smoke mode included (smoke uses synthetic dataset names, not in the list).
Registration tags: the original tag taimma-2027-prereg-v2 (d5e72e4) or the tag of an approved amendment recorded in PREREG-v2
Section 12 (PREREG_V2_AMENDMENTS). Adding a tag to the list is itself an amendment and needs its own commit.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_V2_TAG = "taimma-2027-prereg-v2"                    # original registration, 2026-11-17, commit d5e72e4
PREREG_V2_AMENDMENTS = ("taimma-2027-prereg-v2.1",         # A1: waiver of item (4) of T0-v2 for mpqda_v2
                        "taimma-2027-prereg-v2.2")         # A2: HV6, contribution of the gap by k
REGISTRATION_TAGS = (PREREG_V2_TAG, *PREREG_V2_AMENDMENTS)
CONFIRMATORY_DATASETS = ("gse10245", "gse30219", "gse37745", "tcga_luad_egfr", "gse31210_egfr")
CANDIDATES = ("mpqda_v2_m1", "mpqda_v2_m2", "mpqda_v2_m3")
REGISTERED = ("mpqda_v2", "mpqda_v2_no_gap", "mpqda_v2_no_new", "mpqda_v2_no_tour", "mpqda_v2_no_adapt", "de_g")


class RegistrationError(RuntimeError):
    """Run refused by the registration gate."""


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def registration_tag(git=git) -> str | None:
    """Most recent registration tag carried by HEAD (latest amendment first, then the original tag); None if HEAD carries none."""
    tags = set(git("tag", "--points-at", "HEAD").split())
    return next((t for t in reversed(REGISTRATION_TAGS) if t in tags), None)


def check_run(dataset: str, method: str, chosen: str | None, git=git) -> None:
    """Returns nothing if the run is allowed; RegistrationError if it is blocked."""
    if dataset not in CONFIRMATORY_DATASETS:
        return
    if method in CANDIDATES:
        raise RegistrationError(f"{method} is a development-phase candidate, not run on {dataset}")
    if method not in REGISTERED:
        return
    if chosen is None:
        raise RegistrationError(f"{method} on {dataset}: V2_CHOSEN is not fixed")
    if registration_tag(git) is None:
        raise RegistrationError(f"{method} on {dataset}: HEAD carries no registration tag ({', '.join(REGISTRATION_TAGS)})")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise RegistrationError(f"{method} on {dataset}: the working tree has uncommitted changes")
