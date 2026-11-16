"""Pha A của bài TAIMMA: chọn panel đúng k gen bằng các wrapper, trên phần train của từng fold ngoài (Mục 4.1).

    python scripts/run_taimma.py --scenario T1 --metric logloss --datasets gse10245 gse30219 gse37745 \
        --methods mpqda_de ga bpso de_rand1 shade qlde --k 2 5 10 50 --repeats 0 1 2 \
        [--seed-index 0] [--workers 4] [--n-jobs-eval 1] [--folds 0 1 2] [--shuffle-train-labels] [--pop 30 --generations 50]

Mỗi đơn vị là (lần lặp, fold, phương pháp) và chứa mọi k. Phần test của fold ngoài không được đọc trong pha A, trừ khi
chạy T0 với --shuffle-train-labels: khi đó runner tự đánh giá ngoài trên nhãn test thật và ghi t0_shuffle_eval.csv.
--metric logloss dùng fold nội bộ chung cho mọi wrapper trên cùng fold ngoài; --metric auroc dùng fold theo seed của
phương pháp như survey (Đ4). Không có cờ đổi cách này.
Kết quả: $KLTN_DATA_ROOT/results/survey_v2/taimma/<kịch bản>/<cohort>/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, sha256, variance_prefilter  # noqa: E402
from survey_v2.e1_splits import validate_repeated_split_manifest  # noqa: E402
from scripts.run_e1_feature_selection import GENE_COLUMNS, split_indices, split_path, write_csv_atomic  # noqa: E402

from mpqda import config as M  # noqa: E402
from mpqda.baselines import make_rng, run_method  # noqa: E402
from mpqda.de import HISTORY_COLUMNS  # noqa: E402
from mpqda.fitness import METRICS, InnerCVFitness  # noqa: E402
from mpqda.registration import RegistrationError, check_run  # noqa: E402
from mpqda.registration import git as _git  # noqa: E402
from mpqda.runlog import DATA_ROOT, all_versions, results_dir, smoke_mode, start_run, utf8_console, write_config  # noqa: E402

KEYS = ["cohort", "repeat", "fold", "method", "seed_index", "k"]
METRIC_COLUMNS = ["cohort", "repeat", "seed", "fold", "method", "base_method", "seed_index", "k", "fitness_metric",
                  "inner_fold_scheme", "inner_fold_seed", "search_seed", "n_train", "p0", "n_selected", "exact_k", "n_eval",
                  "best_fitness", "output_fitness", "saturated", "consensus_equals_best", "selection_time_s", "cpu_time_s",
                  "n_jobs_eval", "label_shuffled"]
HISTORY_FILE_COLUMNS = [*KEYS, *HISTORY_COLUMNS]
CONSENSUS_COLUMNS = [*KEYS, "gene", "pi", "in_consensus", "in_best"]
ARCHIVE_COLUMNS = [*KEYS, "member_rank", "fitness", "genes"]
SHUFFLE_COLUMNS = ["cohort", "repeat", "fold", "method", "k", "auroc", "bacc", "mcc", "logloss", "brier", "best_fitness",
                   "output_fitness", "phase_b_seed"]
PARTS = {"metrics": (METRIC_COLUMNS, ["repeat", "fold", "method", "k"]),
         "genes": (GENE_COLUMNS, ["repeat", "fold", "method", "k", "rank"]),
         "history": (HISTORY_FILE_COLUMNS, ["repeat", "fold", "method", "k", "t"]),
         "consensus": (CONSENSUS_COLUMNS, ["repeat", "fold", "method", "k", "gene"]),
         "archive": (ARCHIVE_COLUMNS, ["repeat", "fold", "method", "k", "member_rank"])}
SHUFFLE_PART = {"shuffle_eval": (SHUFFLE_COLUMNS, ["repeat", "fold", "method", "k"])}
AGGREGATE_NAMES = {"metrics": "selection_per_fold.csv", "genes": "selected_genes.csv", "history": "history.csv",
                   "consensus": "consensus_weights.csv", "archive": "archive_final.csv", "shuffle_eval": "t0_shuffle_eval.csv"}
PHASE_B_METHOD_ID = 90                         # mọi phương pháp mới nhận 90 ở pha B của survey (Mục 3.2)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, help="T0, T1, T1b, T2, T3_s1..T3_s4, T6 hoặc tên smoke test")
    ap.add_argument("--metric", required=True, choices=METRICS)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help=f"{M.ALL_METHODS}")
    ap.add_argument("--k", nargs="+", type=int, required=True)
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="mặc định: mọi fold của các lần lặp đã chọn")
    ap.add_argument("--seed-index", type=int, default=0, help="0 cho T0, T1, T1b, T2, T6; 1..4 cho T3")
    ap.add_argument("--workers", type=int, default=1, help="số đơn vị chạy song song (joblib)")
    ap.add_argument("--n-jobs-eval", type=int, default=1, help="> 1 chỉ dùng cho T0 và T6, và chỉ khi --workers 1")
    ap.add_argument("--shuffle-train-labels", action="store_true", help="chỉ T0: hoán vị nhãn phần train ngoài")
    ap.add_argument("--pop", type=int, default=M.P)
    ap.add_argument("--generations", type=int, default=M.T)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--splits-root", type=Path, default=None,
                    help="thư mục chứa <cohort>/repeated_splits.csv; mặc định là thư mục dữ liệu của cohort")
    ap.add_argument("--out", default=None, help=f"mặc định {M.RESULT_SUBDIR}/<kịch bản>")
    ap.add_argument("--nan-outer-test", action="store_true",
                    help=f"chỉ smoke test ({M.ENV_SMOKE}=1): gán NaN cho các dòng test ngoài sau khi nạp dữ liệu")
    args = ap.parse_args(argv)
    unknown = [m for m in args.methods if m not in M.ALL_METHODS]
    if unknown:
        ap.error(f"phương pháp không có: {unknown}")
    if len(set(args.methods)) != len(args.methods):
        ap.error("--methods có tên lặp")
    needs_choice = [m for m in args.methods if m in M.V2_FAMILY and m not in M.V2_CANDIDATES]
    if needs_choice and M.V2_CHOSEN is None:
        ap.error(f"{needs_choice}: mpqda.config.V2_CHOSEN chưa chốt (chỉ ứng viên {list(M.V2_CANDIDATES)} chạy được)")
    if "qlde" in args.methods and M.QLDE is None:
        ap.error("qlde chưa có cấu hình (Đ6): điền docs/taimma/baselines/qlde.md và mpqda.config.QLDE trước")
    if args.n_jobs_eval > 1 and args.workers > 1:
        ap.error("--n-jobs-eval > 1 chỉ dùng với --workers 1")
    if args.nan_outer_test and (not smoke_mode() or args.shuffle_train_labels):
        ap.error(f"--nan-outer-test chỉ dùng cho smoke test ({M.ENV_SMOKE}=1) và không dùng cùng --shuffle-train-labels")
    if args.seed_index < 0:
        ap.error("--seed-index phải ≥ 0")
    args.k = sorted(set(int(k) for k in args.k))
    args.out = args.out or f"{M.RESULT_SUBDIR}/{args.scenario}"
    return args


def search_seed(cid: int, split_seed: int, fold: int, method: str, seed_index: int) -> int:
    """seed_run (Mục 3.2): seed thứ 0 đúng bằng biến `seed` của survey E1; seed thứ s ≥ 1 thêm s vào cuối."""
    parts = [cid, split_seed, fold, M.METHOD_ID[M.base_method(method)]]
    return C.seed_for(*parts) if seed_index == 0 else C.seed_for(*parts, seed_index)


def inner_fold_seed(metric: str, cid: int, split_seed: int, fold: int, seed_run: int) -> int:
    """logloss: fold nội bộ chung theo fold ngoài; auroc: theo seed của phương pháp, như survey (Đ4)."""
    return C.seed_for(cid, split_seed, fold, M.INNER_FOLD_TAG) if metric == "logloss" else seed_run


def shuffle_permutation(cid: int, split_seed: int, fold: int, n_train: int) -> np.ndarray:
    return np.random.default_rng([cid, split_seed, fold, M.SHUFFLE_TAG]).permutation(n_train)


def checkpoint_paths(out: Path, repeat: int, fold: int, method: str, parts) -> dict[str, Path]:
    base = out / "checkpoints" / f"repeat_{repeat}_fold_{fold}_{method}"
    return {part: base.with_name(base.name + f".{part}.csv") for part in parts}


def head_tags() -> list[str]:
    """Tag ở HEAD của kho (ghi vào config.json); rỗng nếu không đọc được."""
    try:
        return sorted(_git("tag", "--points-at", "HEAD").split())
    except Exception:  # noqa: BLE001
        return []


def run_unit(unit: dict) -> str:
    """Một đơn vị (lần lặp, fold, phương pháp): chạy mọi k rồi ghi checkpoint. Chỉ nhận phần train; phần test chỉ có mặt
    khi nhãn train bị xáo trộn (T0) và chỉ dùng sau khi panel đã chọn xong."""
    name, method, cid = unit["cohort"], unit["method"], unit["cid"]
    repeat, split_seed, fold, s = unit["repeat"], unit["split_seed"], unit["fold"], unit["seed_index"]
    X_tr, y_tr, genes = unit["X_tr"], unit["y_tr"], unit["genes"]
    p0 = X_tr.shape[1]
    seed_run = search_seed(cid, split_seed, fold, method, s)
    fold_seed = inner_fold_seed(unit["metric"], cid, split_seed, fold, seed_run)
    rows = {part: [] for part in unit["paths"]}
    key = {"cohort": name, "repeat": repeat, "fold": fold, "method": method, "seed_index": s}
    with threadpool_limits(1):
        for k in unit["k_list"]:
            fitness = InnerCVFitness(X_tr, y_tr, fold_seed=fold_seed, metric=unit["metric"])     # cache riêng theo k
            cpu0, t0 = time.process_time(), time.perf_counter()
            res = run_method(method, fitness, p0, k, make_rng(method, seed_run, k), P=unit["P"], T=unit["T"],
                             n_jobs=unit["n_jobs_eval"])
            elapsed, cpu = time.perf_counter() - t0, time.process_time() - cpu0
            if not np.isfinite([h["best_fitness"] for h in res["history"]] + [res["best_fitness"]]).all():
                raise FloatingPointError(f"{name} repeat {repeat} fold {fold} {method} k={k}: fitness không hữu hạn")
            best = np.sort(np.asarray(res["best"], dtype=int))
            if method in M.TWO_OUTPUTS:
                cons = np.sort(np.asarray(res["consensus"], dtype=int))
                same = bool(np.array_equal(cons, best))
                outputs = [(method, cons, res["consensus_fitness"], same),
                           (method + M.BEST_SUFFIX, best, res["best_fitness"], same)]
            else:
                outputs = [(method, best, res["best_fitness"], "")]
            saturated = bool(res["best_fitness"] >= 1 - M.SATURATION_EPS) if unit["metric"] == "auroc" else ""
            for out_name, panel, out_fit, same in outputs:
                rows["metrics"].append({
                    "cohort": name, "repeat": repeat, "seed": split_seed, "fold": fold, "method": out_name,
                    "base_method": M.base_method(method), "seed_index": s, "k": k, "fitness_metric": unit["metric"],
                    "inner_fold_scheme": "shared" if unit["metric"] == "logloss" else "per_method",
                    "inner_fold_seed": fold_seed, "search_seed": seed_run, "n_train": len(y_tr), "p0": p0,
                    "n_selected": len(panel), "exact_k": bool(len(panel) == k == len(set(panel.tolist()))),
                    "n_eval": res["n_eval"], "best_fitness": res["best_fitness"], "output_fitness": out_fit,
                    "saturated": saturated, "consensus_equals_best": same, "selection_time_s": elapsed, "cpu_time_s": cpu,
                    "n_jobs_eval": unit["n_jobs_eval"], "label_shuffled": unit["label_shuffled"]})
                rows["genes"].extend({"cohort": name, "repeat": repeat, "fold": fold, "method": out_name, "k": k,
                                      "rank": rank, "gene": genes[col]} for rank, col in enumerate(panel, 1))
                if unit["label_shuffled"]:
                    pb_seed = C.seed_for(cid, split_seed, fold, k, PHASE_B_METHOD_ID)
                    ev = evaluate_panel(X_tr[:, panel], y_tr, unit["X_te"][:, panel], unit["y_te"], classifier="lr_l2",
                                        seed=pb_seed)
                    rows["shuffle_eval"].append({
                        "cohort": name, "repeat": repeat, "fold": fold, "method": out_name, "k": k, "auroc": ev.auroc,
                        "bacc": ev.bacc, "mcc": ev.mcc, "logloss": ev.logloss, "brier": ev.brier,
                        "best_fitness": res["best_fitness"], "output_fitness": out_fit, "phase_b_seed": pb_seed})
            rows["history"].extend({**key, "k": k, **h} for h in res["history"])
            if method in M.TWO_OUTPUTS:
                in_c, in_b = set(cons.tolist()), set(best.tolist())
                rows["consensus"].extend({**key, "k": k, "gene": genes[j], "pi": pi, "in_consensus": j in in_c,
                                          "in_best": j in in_b} for j, pi in sorted(res["pi"].items()))
                rows["archive"].extend({**key, "k": k, "member_rank": r, "fitness": f,
                                        "genes": ";".join(str(genes[j]) for j in member)}
                                       for r, (member, f) in enumerate(res["archive"], 1))
    columns = {**PARTS, **SHUFFLE_PART}
    for part, path in unit["paths"].items():
        frame = pd.DataFrame(rows[part], columns=columns[part][0])
        write_csv_atomic(frame, path, columns=columns[part][0])
    return f"[{name}] repeat {repeat} fold {fold} {method}: checkpoint saved"


def aggregate(out: Path, parts) -> dict[str, pd.DataFrame]:
    """Gộp mọi checkpoint thành tệp theo cohort; thứ tự dòng cố định để hai lần chạy cùng seed cho tệp trùng từng byte."""
    columns = {**PARTS, **SHUFFLE_PART}
    frames = {}
    for part in parts:
        cols, order = columns[part]
        files = sorted((out / "checkpoints").glob(f"*.{part}.csv"))
        loaded = [pd.read_csv(f, float_precision="round_trip", keep_default_na=True) for f in files]
        loaded = [d for d in loaded if len(d)]
        frame = pd.concat(loaded, ignore_index=True) if loaded else pd.DataFrame(columns=cols)
        frame = frame[cols].sort_values(order, kind="stable").reset_index(drop=True)
        frame.to_csv(out / AGGREGATE_NAMES[part], index=False)
        frames[part] = frame
    return frames


def main(argv=None, script=__file__) -> int:
    utf8_console()
    args = parse_args(argv)
    try:                                                   # cổng đăng ký của v2 (TQĐ24): trước mọi đơn vị, không cờ nào bỏ qua
        for name in args.datasets:
            for method in args.methods:
                check_run(name, method, M.V2_CHOSEN)
    except RegistrationError as exc:
        print(f"DỪNG (cổng đăng ký): {exc}", file=sys.stderr)
        return 2
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    smoke = smoke_mode()
    parts = dict(PARTS, **(SHUFFLE_PART if args.shuffle_train_labels else {}))
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        path = split_path(ds, args.splits_root)
        split_sha, expected = sha256(path), M.EXPECTED_SPLIT_SHA256.get(name)
        if split_sha != expected and not smoke:
            print(f"[{name}] DỪNG: SHA-256 của {path} là {split_sha}, giá trị chờ là {expected} "
                  f"(mpqda.config.EXPECTED_SPLIT_SHA256).", file=sys.stderr)
            return 2
        manifest = pd.read_csv(path)
        errors = validate_repeated_split_manifest(manifest, n_folds=C.N_FOLDS)
        if errors:
            raise ValueError(f"{name}: invalid repeated_splits.csv: {'; '.join(errors)}")
        cid = C.cohort_id(name)
        runs = manifest[["repeat", "seed", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
        units = []
        for run in runs.itertuples(index=False):
            repeat, split_seed, fold = int(run.repeat), int(run.seed), int(run.fold)
            if repeat not in args.repeats or (args.folds is not None and fold not in args.folds):
                continue
            todo = []
            for method in args.methods:
                paths = checkpoint_paths(out, repeat, fold, method, parts)
                if all(p.is_file() for p in paths.values()):
                    print(f"[{name}] repeat {repeat} fold {fold} {method}: checkpoint exists, skip", flush=True)
                else:
                    todo.append((method, paths))
            if not todo:
                continue
            tr, te = split_indices(ds, manifest, repeat, fold)
            X_all = ds.X
            if args.nan_outer_test:                                # sentinel: phần test ngoài không được đi vào pha A
                X_all = np.array(ds.X, dtype=float, copy=True)
                X_all[te] = np.nan
            pre = variance_prefilter(X_all[tr], args.prefilter)
            X_tr, y_tr = X_all[tr][:, pre], ds.y[tr]
            extra = {}
            if args.shuffle_train_labels:
                y_tr = y_tr[shuffle_permutation(cid, split_seed, fold, len(tr))]
                extra = {"X_te": ds.X[te][:, pre], "y_te": ds.y[te]}     # nhãn test thật; chỉ dùng sau khi chọn panel
            for method, paths in todo:
                units.append({"cohort": name, "cid": cid, "repeat": repeat, "split_seed": split_seed, "fold": fold,
                              "method": method, "seed_index": args.seed_index, "k_list": args.k, "metric": args.metric,
                              "X_tr": X_tr, "y_tr": y_tr, "genes": np.asarray(ds.genes)[pre], "P": args.pop,
                              "T": args.generations, "n_jobs_eval": args.n_jobs_eval,
                              "label_shuffled": bool(args.shuffle_train_labels), "paths": paths, **extra})
        if args.workers > 1 and len(units) > 1:
            for msg in Parallel(n_jobs=args.workers, backend="loky")(delayed(run_unit)(u) for u in units):
                print(msg, flush=True)
        else:
            for u in units:
                print(run_unit(u), flush=True)
        frames = aggregate(out, parts)
        metrics = frames["metrics"]
        cfg = {"experiment": "TAIMMA phase A (wrapper panel selection)", "scenario": args.scenario, "dataset": name,
               "role": C.DATASETS.get(name), "fitness_metric": args.metric,
               "inner_fold_scheme": "shared: seed_for(cid, split_seed, fold, INNER_FOLD_TAG)" if args.metric == "logloss"
               else "per_method: seed_run of the method (survey)",
               "protocol": "precomputed patient-grouped repeated 5x5 outer train/test; label-free variance prefilter and "
                           "wrapper search use outer train only",
               "methods": args.methods, "outputs_in_results": sorted(metrics.method.unique()) if len(metrics) else [],
               "k_grid": args.k, "repeats": args.repeats, "folds": args.folds, "seed_index": args.seed_index,
               "pop": args.pop, "generations": args.generations, "prefilter_p0": args.prefilter,
               "constants": M.constants(), "vendor_commit": prov.get("vendor_commit"),
               "split_file": str(path), "split_sha256": split_sha, "split_sha256_expected": expected,
               "split_sha256_match": split_sha == expected, "label_shuffled": bool(args.shuffle_train_labels),
               "nan_outer_test": bool(args.nan_outer_test), "smoke": smoke, "workers": args.workers,
               "n_jobs_eval": args.n_jobs_eval, "gene_space": ds.notes, "n_samples": ds.n, "n_genes": ds.p,
               "n_rows_selection_per_fold": int(len(metrics)), "versions": all_versions(),
               "choices_to_confirm": list(M.CHOICES_TO_CONFIRM), "v2_chosen": M.V2_CHOSEN,
               "v2_config": {m: M.v2_config(m) for m in args.methods if m in M.V2_FAMILY},
               "head_tags": head_tags()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", path])
        print(f"[{name}] {len(metrics)} dòng selection_per_fold.csv -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
