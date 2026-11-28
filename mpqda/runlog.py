"""Traceability for scripts at the repository root: replaces `survey_v2.runlog.start_run`.

`REPO` of the vendored copy is `vendor/`, so `is_tracked` of the vendored code is wrong for scripts outside `vendor/`. Here provenance
and the traceability check run with `repo = ROOT`. The config is still written by `write_config` of the vendored code; the result folder is
`$KLTN_DATA_ROOT/results/survey_v2/taimma/<scenario>/<cohort>/` (shared by phase A and phase B).
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
    """Windows: a console that is not UTF-8 fails when printing non-ASCII text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def smoke_mode() -> bool:
    return os.environ.get(M.ENV_SMOKE, "").strip().lower() in {"1", "true", "yes"}


def vendor_commit() -> str | None:
    """Source commit of vendor/, read from the `| Commit | ... |` line of vendor/VENDORED.md."""
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
    """provenance(repo_dir=ROOT) + check_traceable(repo=ROOT); returns the provenance dict to stamp into the config.

    Stops (code 2) if tracked files have uncommitted changes, a script is not tracked by Git or HEAD is not yet on the
    remote (unless KLTN_ALLOW_UNPUSHED=1; then traceable = false). In smoke mode, traceable is always false."""
    if not os.environ.get("KLTN_DATA_ROOT", "").strip():           # general rule 3: KLTN_DATA_ROOT is required
        print("[provenance] STOP: set KLTN_DATA_ROOT (data and results folder).", file=sys.stderr)
        raise SystemExit(2)
    try:
        prov = provenance(script=script, repo_dir=ROOT, params=params or {})
    except ProvenanceError as e:
        print(f"[provenance] DỪNG: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    allow = os.environ.get(ENV_ALLOW_UNPUSHED, "").strip().lower() in {"1", "true", "yes"}
    errors = check_traceable(script, repo=ROOT, require_push=True) if prov.get("tracked") else []
    if smoke_mode():
        errors = errors + [f"{M.ENV_SMOKE}=1: smoke-test run, not citable"]
    prov["traceable"] = bool(prov.get("tracked", False) and not errors)
    prov["traceable_errors"] = errors
    prov["vendor_commit"] = vendor_commit()
    if errors and not (allow or smoke_mode()):
        print("[traceable] DỪNG:\n  - " + "\n  - ".join(errors)
              + f"\n  (bypass temporarily with {ENV_ALLOW_UNPUSHED}=1; results are then NOT citable)", file=sys.stderr)
        raise SystemExit(2)
    return prov


assert survey_runlog.REPO == VENDOR, "survey_v2 must be loaded from vendor/"
