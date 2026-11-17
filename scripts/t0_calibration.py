"""Hiệu chuẩn hoán vị cho cổng T0-v2 (Review-2026-11-17b, TQĐ34). Chỉ mô tả, chạy song song với C1–C3 ở commit mang tag
đăng ký; không có ngưỡng và không kết luận tự động.

    python scripts/t0_calibration.py --dataset gse37745 --k 5 --permutations 0 1 2 3 4 5 6 7 8 9 10 \
        --methods mpqda_v2 ga --random-draws 50 --workers 4 --out taimma/T0v2_calib

Lặp lại thiết kế T0 trên 15 fold ngoài (lần lặp 0–2) với nhiều hoán vị nhãn train: b = 0 là hoán vị đã đăng ký
(run_taimma.shuffle_permutation, phải tái tạo đúng t0_shuffle_eval của S0v2), b ≥ 1 là default_rng([cid, split_seed, fold,
SHUFFLE_TAG, b]). Với mỗi (fold, b), mỗi phương pháp chạy qua đúng run_unit của run_taimma.py (cùng fitness, seed tìm kiếm,
seed pha B, nhãn test thật), và R panel ngẫu nhiên trong phần lọc sơ bộ được đánh giá bằng cùng evaluate_panel làm phân bố so
sánh trên đúng hoán vị đó.

Ghi <out>/<dataset>/:
  calib_units.csv    mỗi dòng một (lần lặp, fold, b, đầu ra): AUROC ngoài, fitness nội bộ, r = tương quan giữa nhãn đã
                     hoán vị và nhãn thật của phần train, trung bình và độ lệch chuẩn AUROC của panel ngẫu nhiên, PIT (phân vị
                     của AUROC trong phân bố panel ngẫu nhiên cùng fold, cùng b)
  calib_random.csv   AUROC từng panel ngẫu nhiên
  calib_summary.csv  theo đầu ra và b: AUROC trung bình 15 fold, PIT trung bình, số fold dưới trung vị ngẫu nhiên; dòng b = all:
                     trung bình và độ lệch chuẩn qua các b, tỉ lệ b có trung bình ngoài [0,40; 0,60] (khoảng đã đăng ký)
  config.json        commit, tag đăng ký, tham số
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, variance_prefilter  # noqa: E402

from mpqda import config as M  # noqa: E402
from mpqda.registration import RegistrationError, check_run, registration_tag  # noqa: E402
from mpqda.runlog import DATA_ROOT, head_commit, results_dir  # noqa: E402
from scripts import run_taimma as R  # noqa: E402
from scripts.run_e1_feature_selection import split_indices, split_path  # noqa: E402

RANGE = (0.40, 0.60)                    # khoảng đã đăng ký của mục (4) (PREREG-v2 Mục 9); chỉ để đếm, không dùng làm cổng
RANDOM_TAG = 777                        # seed của panel ngẫu nhiên, như null_shuffle.py của sinh viên (17/11)


def permutation(cid: int, split_seed: int, fold: int, n: int, b: int) -> np.ndarray:
    """b = 0: hoán vị đã đăng ký của T0; b ≥ 1: hoán vị mới, độc lập theo b."""
    if b == 0:
        return R.shuffle_permutation(cid, split_seed, fold, n)
    return np.random.default_rng([cid, split_seed, fold, M.SHUFFLE_TAG, b]).permutation(n)


def prepare(X, y, genes, cid: int, folds, p0: int, perms) -> list[dict]:
    """Một phần tử cho mỗi (lần lặp, fold, b): phần train đã lọc sơ bộ, nhãn đã hoán vị, phần test với nhãn thật."""
    cells = []
    for f in folds:
        tr, te = f["tr"], f["te"]
        pre = variance_prefilter(X[tr], p0)
        X_tr, X_te, y_tr = X[tr][:, pre], X[te][:, pre], y[tr]
        for b in perms:
            y_perm = y_tr[permutation(cid, f["split_seed"], f["fold"], len(tr), b)]
            cells.append({"repeat": f["repeat"], "split_seed": f["split_seed"], "fold": f["fold"], "b": int(b), "cid": cid,
                          "X_tr": X_tr, "y_tr": y_perm, "X_te": X_te, "y_te": y[te], "genes": np.asarray(genes)[pre],
                          "r": float(np.corrcoef(y_perm, y_tr)[0, 1])})
    return cells


def method_task(cell: dict, method: str, k: int, P: int, T: int, cohort: str) -> list[dict]:
    """Chạy một phương pháp trên một (fold, b) bằng run_unit của run_taimma.py; trả các dòng shuffle_eval."""
    with tempfile.TemporaryDirectory() as tmp:
        parts = dict(R.PARTS, **R.SHUFFLE_PART)
        paths = R.checkpoint_paths(Path(tmp), cell["repeat"], cell["fold"], method, parts)
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        R.run_unit({"cohort": cohort, "cid": cell["cid"], "repeat": cell["repeat"], "split_seed": cell["split_seed"],
                    "fold": cell["fold"], "method": method, "seed_index": 0, "k_list": [k], "metric": "logloss",
                    "X_tr": cell["X_tr"], "y_tr": cell["y_tr"], "genes": cell["genes"], "P": P, "T": T, "n_jobs_eval": 1,
                    "label_shuffled": True, "paths": paths, "X_te": cell["X_te"], "y_te": cell["y_te"]})
        ev = pd.read_csv(paths["shuffle_eval"], float_precision="round_trip")
    return [{"repeat": cell["repeat"], "fold": cell["fold"], "b": cell["b"], "output": row.method, "auroc": float(row.auroc),
             "output_fitness": float(row.output_fitness), "r": cell["r"]} for row in ev.itertuples(index=False)]


def random_task(cell: dict, k: int, draws: int) -> list[dict]:
    rng = np.random.default_rng([cell["cid"], cell["split_seed"], cell["fold"], RANDOM_TAG, cell["b"]])
    pb_seed = C.seed_for(cell["cid"], cell["split_seed"], cell["fold"], k, R.PHASE_B_METHOD_ID)
    rows = []
    for d in range(draws):
        panel = rng.choice(cell["X_tr"].shape[1], size=k, replace=False)
        ev = evaluate_panel(cell["X_tr"][:, panel], cell["y_tr"], cell["X_te"][:, panel], cell["y_te"], classifier="lr_l2",
                            seed=pb_seed)
        rows.append({"repeat": cell["repeat"], "fold": cell["fold"], "b": cell["b"], "draw": d, "auroc": float(ev.auroc)})
    return rows


def pit(value: float, null: np.ndarray) -> float:
    return float((np.sum(null < value) + 0.5 * np.sum(null == value)) / len(null))


def summarise(units: pd.DataFrame, rand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["repeat", "fold", "b"]
    stats = rand.groupby(key).auroc.agg(random_mean="mean", random_sd="std", random_median="median").reset_index()
    units = units.merge(stats, on=key, how="left")
    nulls = {k: g.auroc.to_numpy() for k, g in rand.groupby(key)}
    units["pit"] = [pit(a, nulls[(r, f, b)]) for a, r, f, b in zip(units.auroc, units.repeat, units.fold, units.b)]
    units = units.sort_values([*key, "output"], kind="stable").reset_index(drop=True)
    rows = []
    for (out, b), g in units.groupby(["output", "b"], sort=True):
        rows.append({"output": out, "b": str(b), "n_folds": len(g), "mean_auroc": g.auroc.mean(), "mean_pit": g.pit.mean(),
                     "n_below_random_median": int((g.auroc < g.random_median).sum()), "mean_r": g.r.mean(),
                     "random_mean_auroc": g.random_mean.mean()})
    per_b = pd.DataFrame(rows)
    for out, g in per_b.groupby("output", sort=True):
        rows.append({"output": out, "b": "all", "n_folds": int(g.n_folds.sum()), "mean_auroc": g.mean_auroc.mean(),
                     "sd_mean_auroc": g.mean_auroc.std(ddof=1) if len(g) > 1 else np.nan, "mean_pit": g.mean_pit.mean(),
                     "share_b_outside_range": float(((g.mean_auroc < RANGE[0]) | (g.mean_auroc > RANGE[1])).mean()),
                     "n_below_random_median": int(g.n_below_random_median.sum()), "mean_r": g.mean_r.mean(),
                     "random_mean_auroc": g.random_mean_auroc.mean()})
    rand_rows = []
    for b, g in units.drop_duplicates(key).groupby("b", sort=True):
        rand_rows.append({"output": "random_panel", "b": str(b), "n_folds": len(g), "mean_auroc": g.random_mean.mean()})
    return units, pd.concat([pd.DataFrame(rows), pd.DataFrame(rand_rows)], ignore_index=True)


def calibrate(X, y, genes, cid: int, folds, perms, methods, k: int = 5, p0: int = C.PREFILTER_P0, draws: int = 50,
              P: int = M.P, T: int = M.T, workers: int = 1, cohort: str = "gse37745"):
    cells = prepare(X, y, genes, cid, folds, p0, perms)
    jobs = [delayed(method_task)(c, m, k, P, T, cohort) for c in cells for m in methods]
    jobs += [delayed(random_task)(c, k, draws) for c in cells]
    out = Parallel(n_jobs=workers, backend="loky")(jobs) if workers > 1 else [j[0](*j[1], **j[2]) for j in jobs]
    n_method = len(cells) * len(methods)
    units = pd.DataFrame([row for part in out[:n_method] for row in part])
    rand = pd.DataFrame([row for part in out[n_method:] for row in part])
    units, summary = summarise(units, rand)
    return units, rand, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gse37745")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--permutations", nargs="+", type=int, default=list(range(11)))
    ap.add_argument("--methods", nargs="+", default=["mpqda_v2", "ga"])
    ap.add_argument("--random-draws", type=int, default=50)
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=f"{M.RESULT_SUBDIR}/T0v2_calib")
    args = ap.parse_args(argv)
    try:
        for method in args.methods:
            check_run(args.dataset, method, M.V2_CHOSEN)
    except RegistrationError as exc:
        print(f"DỪNG (cổng đăng ký): {exc}", file=sys.stderr)
        return 2
    ds = load_dataset(DATA_ROOT, args.dataset)
    manifest = pd.read_csv(split_path(ds, None))
    folds = []
    for run in manifest[["repeat", "seed", "fold"]].drop_duplicates().sort_values(["repeat", "fold"]).itertuples(index=False):
        if int(run.repeat) in args.repeats:
            tr, te = split_indices(ds, manifest, int(run.repeat), int(run.fold))
            folds.append({"repeat": int(run.repeat), "split_seed": int(run.seed), "fold": int(run.fold), "tr": tr, "te": te})
    units, rand, summary = calibrate(ds.X, ds.y, ds.genes, C.cohort_id(args.dataset), folds, args.permutations, args.methods,
                                     k=args.k, draws=args.random_draws, workers=args.workers, cohort=args.dataset)
    out = results_dir(args.out, args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("calib_units", units), ("calib_random", rand), ("calib_summary", summary)):
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.10g", lineterminator="\n")
    cfg = {"experiment": "T0-v2 permutation calibration (descriptive; Review-2026-11-17b, TQĐ34)", "commit_hash": head_commit(),
           "registration_tag": registration_tag(), **{k: v for k, v in vars(args).items()}}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary[summary.b == "all"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
