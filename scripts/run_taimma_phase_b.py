"""Phase B-LL of the TAIMMA study (PREREG, Section 4): runs main() of vendor/scripts/run_e1_classification.py unchanged,
replacing only the evaluation function and adding c_selection to config.json. No file in vendor/ is edited.

    python scripts/run_taimma_phase_b.py --criterion logloss <every argument of run_e1_classification.py>
    python scripts/run_taimma_phase_b.py --criterion auroc   <...>      # tests only: identical to the vendored phase B
    python scripts/run_taimma_phase_b.py --criterion c1      <...>      # B-C1: C = 1 fixed (sensitivity, v2)
    python scripts/run_taimma_phase_b.py --criterion llwide  <...>      # B-LL-wide: log-loss on the wide grid

evaluate_panel_ll and make_evaluator are in mpqda/phase_b.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
for _p in (ROOT, VENDOR):                       # result: vendor/ comes before the repository root
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import scripts.run_e1_classification as E  # noqa: E402

from mpqda.phase_b import C_GRID_WIDE, make_evaluator  # noqa: E402

if Path(E.__file__).resolve().parent != (VENDOR / "scripts").resolve():
    raise SystemExit(f"STOP: run_e1_classification loaded from {E.__file__}, not from vendor/scripts")

C_GRID = [0.01, 0.1, 1.0, 10.0]
RULES = {"logloss": "smallest inner 3-fold out-of-fold log-loss (probabilities clipped to [1e-6, 1-1e-6]; ties keep the smaller C)",
         "auroc": "rule of the vendored code: largest inner 3-fold out-of-fold AUROC (ties keep the smaller C)",
         "c1": "C = 1 fixed (the C of the wrapper fitness); no grid; threshold from the inner out-of-fold scores (B-C1)",
         "llwide": "smallest inner 3-fold out-of-fold log-loss over the wide grid 1e-4..1e3 (probabilities clipped to "
                   "[1e-6, 1-1e-6]; ties keep the smaller C; B-LL-wide)"}
GRIDS = {"logloss": C_GRID, "auroc": C_GRID, "c1": [1.0], "llwide": list(C_GRID_WIDE)}
# Kept verbatim: these decision codes are written to config.json of the archived phase B results.
DECISIONS = {"llwide": "TQĐ27"}                 # the other criteria use the default code given at the call below


def use_repo_root_for_tracking() -> None:
    """survey_v2.runlog.is_tracked takes vendor/ as the repository root (REPO = parents[1] of survey_v2), so it treats scripts outside
    vendor/ as untracked by Git. The real Git root is ROOT (vendor/ is a subfolder of the same repository), so the check runs
    with ROOT. The push check (is_pushed) is unchanged because vendor/ and ROOT are the same repository. No file in vendor/ is edited."""
    import survey_v2.runlog as R
    if getattr(R.is_tracked, "_taimma_root", False):
        return
    original = R.is_tracked

    def is_tracked(path, repo=R.REPO) -> bool:
        return original(path, ROOT)
    is_tracked._taimma_root = True
    R.is_tracked = is_tracked


def main(argv=None) -> int:
    use_repo_root_for_tracking()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--criterion", choices=sorted(RULES), default="logloss")
    args, rest = ap.parse_known_args(argv)
    E.evaluate_panel = make_evaluator(args.criterion)     # B0, random-k and the phase A panels all call through this name
    write_config = E.write_config

    def write_config_with_rule(out_dir, prov, config, data_files=()):
        config = dict(config, c_selection={"criterion": args.criterion, "grid": GRIDS[args.criterion],
                                           "rule": RULES[args.criterion],
                                           # kept verbatim: decision code written to config.json of the archived results
                                           "decision": DECISIONS.get(args.criterion, "TQĐ11")})
        return write_config(out_dir, prov, config, data_files=data_files)

    E.write_config = write_config_with_rule
    E.main(rest, script=__file__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
