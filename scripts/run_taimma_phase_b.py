"""Pha B-LL của bài TAIMMA (TQĐ11, Review-2026-11-14b): chạy nguyên main() của vendor/scripts/run_e1_classification.py,
chỉ thay hàm đánh giá và ghi thêm c_selection vào config.json. Không tệp nào trong vendor/ bị sửa.

    python scripts/run_taimma_phase_b.py --criterion logloss <mọi đối số của run_e1_classification.py>
    python scripts/run_taimma_phase_b.py --criterion auroc   <...>      # chỉ để test: trùng hệt pha B của survey

Chép từ taimma_ref/run_taimma_phase_b_ref.py; evaluate_panel_ll và make_evaluator nằm ở mpqda/phase_b.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
for _p in (ROOT, VENDOR):                       # kết quả: vendor/ đứng trước gốc kho
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import scripts.run_e1_classification as E  # noqa: E402

from mpqda.phase_b import make_evaluator  # noqa: E402

if Path(E.__file__).resolve().parent != (VENDOR / "scripts").resolve():
    raise SystemExit(f"DỪNG: run_e1_classification nạp từ {E.__file__}, không phải vendor/scripts")

C_GRID = [0.01, 0.1, 1.0, 10.0]
RULES = {"logloss": "smallest inner 3-fold out-of-fold log-loss (probabilities clipped to [1e-6, 1-1e-6]; ties keep the smaller C)",
         "auroc": "survey rule: largest inner 3-fold out-of-fold AUROC (ties keep the smaller C)"}


def use_repo_root_for_tracking() -> None:
    """survey_v2.runlog.is_tracked lấy vendor/ làm gốc kho (REPO = parents[1] của survey_v2), nên coi script nằm ngoài
    vendor/ là chưa được Git theo dõi. Gốc Git thật là ROOT (vendor/ là thư mục con của cùng một kho), nên phép kiểm chạy
    với ROOT. Kiểm push (is_pushed) giữ nguyên vì vendor/ và ROOT là cùng một kho. Không sửa tệp nào trong vendor/."""
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
    E.evaluate_panel = make_evaluator(args.criterion)     # B0, random-k và panel pha A đều gọi qua tên này
    write_config = E.write_config

    def write_config_with_rule(out_dir, prov, config, data_files=()):
        config = dict(config, c_selection={"criterion": args.criterion, "grid": C_GRID,
                                           "rule": RULES[args.criterion], "decision": "TQĐ11"})
        return write_config(out_dir, prov, config, data_files=data_files)

    E.write_config = write_config_with_rule
    E.main(rest, script=__file__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
