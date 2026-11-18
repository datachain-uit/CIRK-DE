"""Độ lạc quan của tiêu chí chọn gen trên dữ liệu xác nhận (mô tả, không đăng ký; Review-2026-11-18, TQĐ38).

    python scripts/make_taimma_optimism.py --results <thư mục giải nén chung của S1, S0v2 và C1> --out tables_v2

Với mỗi (cohort GPL570, lần lặp, fold, phương pháp, k), optimism = log-loss ngoài − log-loss nội bộ của panel đầu ra. Log-loss
nội bộ là −output_fitness trong selection_per_fold.csv (trung bình ba fold nội bộ, LR-L2, C = 1, trọng số lớp cân bằng). Log-loss
ngoài lấy từ pha B: B-C1 là so sánh khớp (cùng mô hình, cùng C với fitness, khác ở chỗ huấn luyện trên toàn phần train); B-LL ghi
kèm để đối chiếu với bảng chính. Đầu vào: T1, T1v2 (selection_per_fold.csv); T5C1/T1 (việc v1b của S0v2), T5C1/T1v2, T5LL/T1,
T5LL/T1v2 (classification_per_fold.csv).

  optimism_v2.csv       theo tiêu chí pha B, phương pháp, k: n, log-loss nội bộ, log-loss ngoài, optimism (trung bình, SD)
  optimism_v2_gain.csv  v2 (panel đồng thuận) so với v1 và GA, k ∈ {5, 10}: hiệu log-loss nội bộ, hiệu log-loss ngoài, hiệu
                        optimism, cùng nửa độ rộng CI 95 % Nadeau–Bengio gộp của hiệu log-loss ngoài, để đặt lợi ích tìm kiếm
                        cạnh độ chính xác của phép đo ngoài
  claims_optimism.csv   số dùng trong bài, cùng định dạng claims_v2.csv; optimism_inputs_sha256.csv
Không kiểm định, không vào họ Holm nào. Khi viết, ghi nhãn "exploratory, not registered". Chạy hai lần cho tệp trùng từng byte.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from mpqda import config as M  # noqa: E402
from scripts import make_taimma_tables as T  # noqa: E402

V1, V2, DEG = "mpqda_de", "mpqda_v2", "de_g"
METHODS = [*[n for m in (V1, *M.TIER1, *M.TIER2) for n in M.output_names(m)], V2, V2 + M.BEST_SUFFIX, DEG]
PHASE_B = {"B-C1": "T5C1", "B-LL": "T5LL"}
KEY = ["cohort", "repeat", "fold", "method", "k"]
GAIN_REF = (V1, "ga")
GAIN_K = (5, 10)


def merged(inp: T.Inputs, phase_b: str) -> pd.DataFrame | None:
    """Ghép log-loss nội bộ (pha A) với log-loss ngoài (pha B) theo đơn vị; thiếu tệp nào thì trả None."""
    sel = [inp.scenario(s, "selection_per_fold.csv") for s in ("T1", "T1v2")]
    ev = [inp.scenario(f"{PHASE_B[phase_b]}/{s}", "classification_per_fold.csv", encoding="utf-8-sig") for s in ("T1", "T1v2")]
    if any(f is None for f in (*sel, *ev)):
        return None
    sel = pd.concat(sel, ignore_index=True)[KEY + ["output_fitness"]]
    ev = pd.concat(ev, ignore_index=True)
    ev = ev[ev.model == "lr_l2"][KEY + ["logloss"]]
    m = sel.merge(ev, on=KEY, how="inner", validate="one_to_one")
    m = m[m.method.isin(METHODS)].copy()
    m["inner_logloss"] = -m.output_fitness
    m["outer_logloss"] = m.logloss
    m["optimism"] = m.outer_logloss - m.inner_logloss
    return m.drop(columns=["output_fitness", "logloss"])


def summary(m: pd.DataFrame, phase_b: str) -> pd.DataFrame:
    g = m.groupby(["method", "k"], sort=False)
    out = g.agg(n=("optimism", "size"), inner_logloss=("inner_logloss", "mean"), outer_logloss=("outer_logloss", "mean"),
                optimism=("optimism", "mean"), optimism_sd=("optimism", "std")).reset_index()
    order = {name: i for i, name in enumerate(METHODS)}
    out = out.sort_values(["k", "method"], key=lambda s: s.map(order) if s.name == "method" else s, kind="stable")
    return out.assign(phase_b=phase_b)[["phase_b", "method", "k", "n", "inner_logloss", "outer_logloss", "optimism", "optimism_sd"]]


def gain(m: pd.DataFrame, phase_b: str) -> pd.DataFrame:
    """Hiệu v2 − tham chiếu theo fold (ghép lần lặp, fold trong từng cohort); âm ở log-loss nghĩa là v2 tốt hơn."""
    rows = []
    for k in GAIN_K:
        for ref in GAIN_REF:
            d = {q: [T.paired(m, V2, ref, q, k, c) for c in T.COHORTS] for q in ("inner_logloss", "outer_logloss", "optimism")}
            nb = T.pooled_nb(d["outer_logloss"])
            rows.append({"phase_b": phase_b, "k": k, "reference": ref, "n": int(sum(len(x) for x in d["optimism"])),
                         "inner_logloss_diff": float(np.concatenate(d["inner_logloss"]).mean()),
                         "outer_logloss_diff": float(np.concatenate(d["outer_logloss"]).mean()),
                         "outer_nb_half_width": float((nb["ci_high"] - nb["ci_low"]) / 2),
                         "optimism_diff": float(np.concatenate(d["optimism"]).mean()),
                         "share_v2_higher_optimism": float(np.mean(np.concatenate(d["optimism"]) > 0))})
    return pd.DataFrame(rows)


def build(results: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    inp, summaries, gains = T.Inputs(results), [], []
    for pb in PHASE_B:
        m = merged(inp, pb)
        if m is None:
            continue
        summaries.append(summary(m, pb))
        gains.append(gain(m, pb))
    if not summaries:
        raise SystemExit(f"thiếu đầu vào: {sorted(set(inp.missing))}")
    s, g = pd.concat(summaries, ignore_index=True), pd.concat(gains, ignore_index=True)
    T.write_csv(s, out / "optimism_v2.csv")
    T.write_csv(g, out / "optimism_v2_gain.csv")
    T.write_tex(s[(s.phase_b == "B-C1") & s.k.isin(GAIN_K)].drop(columns=["phase_b"]), out / "optimism_v2.tex",
                "Optimism of the selection criterion (outer minus inner log-loss, B-C1), GPL570; exploratory, not registered.",
                "tab:optimism_v2")
    claims = (T.claims_rows("OPT", s, "optimism_v2.csv", ["n", "inner_logloss", "outer_logloss", "optimism"], ["method", "k"], "")
              + T.claims_rows("OPTGAIN", g, "optimism_v2_gain.csv",
                              ["inner_logloss_diff", "outer_logloss_diff", "outer_nb_half_width", "optimism_diff"], ["k", "reference"], ""))
    for c in claims:                                   # mã số liệu duy nhất: tiêu chí pha B nằm trong cột phase_b và trong mã
        c["claim_id"] = c["claim_id"].replace(":", f":phase_b={c['phase_b']};", 1)
    T.write_csv(pd.DataFrame(claims, columns=T.CLAIM_COLUMNS), out / "claims_optimism.csv")
    T.write_csv(pd.DataFrame(sorted(inp.used.items()), columns=["file", "sha256"]), out / "optimism_inputs_sha256.csv")
    return {"phase_b": sorted(s.phase_b.unique()), "missing": sorted(set(inp.missing))}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tables_v2"))
    args = ap.parse_args(argv)
    res = build(args.results, args.out)
    print("đã ghi optimism_v2, optimism_v2_gain, claims_optimism cho " + ", ".join(res["phase_b"]))
    if res["missing"]:
        print(f"thiếu {len(res['missing'])} tệp, bỏ qua tiêu chí tương ứng: " + ", ".join(res["missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
