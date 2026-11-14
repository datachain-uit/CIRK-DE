"""Đếm số panel nhận C = 0,01 trong pha B của survey (T-R4, Review-2026-11-14b), để ghi nguồn cho con số trong PREREG Mục 4.

    python scripts/count_survey_c_choice.py --survey-results <clone survey tại c1828d2>/results/survey_v2/E1_classification

Phạm vi: ba cohort GPL570, lần lặp 0–2, LR-L2, k ∈ {5, 10}, mọi selector (không tính b0, random_k). In số đếm tổng, theo nhóm
(không wrapper; GA và BPSO) và SHA-256 của từng tệp đầu vào. Chỉ đọc CSV; không chạy mô hình.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COHORTS = ("gse10245", "gse30219", "gse37745")
WRAPPERS = ("ga", "bpso")


def count(results: Path) -> dict:
    frames, files = [], {}
    for c in COHORTS:
        path = results / c / "classification_per_fold.csv"
        files[f"{c}/classification_per_fold.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
        frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[(ev.model == "lr_l2") & ev.repeat.isin([0, 1, 2]) & ev.k.isin([5, 10]) & ~ev.method.isin(["b0", "random_k"])]
    small = np.isclose(ev["C"].astype(float), 0.01)

    def part(mask):
        return {"n_panels": int(mask.sum()), "n_C_0.01": int((small & mask).sum()),
                "rate": round(float((small & mask).sum() / mask.sum()), 4)}
    wrapper = ev.method.isin(WRAPPERS).to_numpy()
    return {"all_selectors": part(np.ones(len(ev), bool)), "non_wrapper": part(~wrapper), "ga_bpso": part(wrapper),
            "methods": sorted(ev.method.unique()), "inputs_sha256": files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-results", type=Path, required=True)
    args = ap.parse_args(argv)
    print(json.dumps(count(args.survey_results), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
