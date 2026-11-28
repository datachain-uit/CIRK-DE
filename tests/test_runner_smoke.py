"""Smoke test of scripts/run_taimma.py on the synthetic data of survey_v2.synthetic (P = 6, T = 3). Not used for the paper."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import SMOKE_COHORT, run_script, smoke_env

pytestmark = pytest.mark.slow

OUTPUTS = {"mpqda_de", "mpqda_de_best", "mpqda_a3_fixedF", "mpqda_a3_fixedF_best", "ga", "bpso", "de_rand1", "shade",
           "qlde", "jde", "jade"}


def read(out, name):
    return pd.read_csv(out / name, float_precision="round_trip")


def test_files_and_columns(smoke_run):
    from scripts import run_taimma as RT
    out = smoke_run["out"]
    sel, genes, hist = read(out, "selection_per_fold.csv"), read(out, "selected_genes.csv"), read(out, "history.csv")
    cons, arch = read(out, "consensus_weights.csv"), read(out, "archive_final.csv")
    assert list(sel.columns) == RT.METRIC_COLUMNS and list(genes.columns) == RT.GENE_COLUMNS
    assert list(hist.columns) == RT.HISTORY_FILE_COLUMNS
    assert list(cons.columns) == RT.CONSENSUS_COLUMNS and list(arch.columns) == RT.ARCHIVE_COLUMNS
    assert set(sel.method) == OUTPUTS and len(sel) == 2 * len(OUTPUTS) * 2          # 2 folds × outputs × 2 k
    assert sel.exact_k.all() and (sel.n_selected == sel.k).all()
    assert (sel.n_eval <= 6 * (3 + 1) + 1).all() and (sel.fitness_metric == "logloss").all()
    assert (sel.inner_fold_scheme == "shared").all() and not sel.label_shuffled.any()
    # log-loss mode: every wrapper on the same outer fold shares the inner folds; the T2 variants use the seed of mpqda_de
    assert (sel.groupby(["repeat", "fold"]).inner_fold_seed.nunique() == 1).all()
    a3, full = sel[sel.method == "mpqda_a3_fixedF"], sel[sel.method == "mpqda_de"]
    assert a3.search_seed.tolist() == full.search_seed.tolist() and (a3.base_method == "mpqda_de").all()
    assert sel.groupby(["repeat", "fold"]).search_seed.nunique().min() == 8          # 8 original methods, different seeds
    sizes = genes.groupby(["repeat", "fold", "method", "k"]).gene.nunique().reset_index()
    assert (sizes.gene == sizes.k).all() and len(sizes) == len(sel)
    # history: t = 0..T for every run; best fitness never decreases; Q columns only for the full QRK-DE
    for _, g in hist.groupby(["repeat", "fold", "method", "k"]):
        assert g.t.tolist() == [0, 1, 2, 3] and g.best_fitness.is_monotonic_increasing and g.n_eval.is_monotonic_increasing
    assert hist[hist.method == "mpqda_de"].q_00.notna().all() and hist[hist.method == "mpqda_a3_fixedF"].q_00.isna().all()
    assert (hist[hist.method == "mpqda_a3_fixedF"].mean_F == 0.5).all()
    assert set(cons.method) == set(arch.method) == {"mpqda_de", "mpqda_a3_fixedF"}
    assert (cons[cons.in_consensus].groupby(["repeat", "fold", "method", "k"]).gene.count().reset_index()
            .pipe(lambda d: d.gene == d.k)).all()
    assert np.isfinite(sel.best_fitness).all() and np.isfinite(sel.output_fitness).all() and (sel.best_fitness < 0).all()


def test_config_marks_smoke_run_as_not_citable(smoke_run):
    cfg = json.loads((smoke_run["out"] / "config.json").read_text(encoding="utf-8"))
    assert cfg["smoke"] is True and cfg["traceable"] is False and cfg["split_sha256_match"] is False
    assert cfg["commit_hash"] and cfg["vendor_commit"] and cfg["constants"]["P"] == 30 and cfg["pop"] == 6
    assert cfg["choices_to_confirm"] and cfg["fitness_metric"] == "logloss" and cfg["n_rows_selection_per_fold"] == 44
    assert any(k.endswith("expression.h5") for k in cfg["provenance"]["data_sha256"])
    assert any(k.endswith("repeated_splits.csv") for k in cfg["provenance"]["data_sha256"])


def test_rerun_skips_checkpoints_and_is_byte_identical(smoke_run):
    out = smoke_run["out"]
    before = {n: (out / n).read_bytes() for n in ("selected_genes.csv", "history.csv", "selection_per_fold.csv")}
    again = run_script(smoke_run["repo"], "run_taimma.py", smoke_run["args"], smoke_run["env"])
    assert again.stdout.count("checkpoint exists, skip") == 18 and "checkpoint saved" not in again.stdout
    assert all((out / n).read_bytes() == b for n, b in before.items())


def test_same_seed_gives_identical_genes_and_history(smoke_run):
    """Determinism (T0 condition 1) and parallelism (condition 2): rerun into another folder with n_jobs_eval = 2."""
    args = [a for a in smoke_run["args"]]
    args[args.index("--workers") + 1] = "1"
    keep = args[:args.index("--methods") + 1] + ["mpqda_de", "ga"] + args[args.index("--k"):]
    run_script(smoke_run["repo"], "run_taimma.py", [*keep, "--out", "taimma/SMOKE_b", "--n-jobs-eval", "2"], smoke_run["env"])
    other = smoke_run["out"].parents[1] / "SMOKE_b" / SMOKE_COHORT
    for name in ("selected_genes.csv", "history.csv"):
        a, b = read(smoke_run["out"], name), read(other, name)
        a = a[a.method.isin(["mpqda_de", "mpqda_de_best", "ga"])].reset_index(drop=True)
        assert a.to_csv(index=False) == b.to_csv(index=False)
    assert (read(other, "selection_per_fold.csv").n_jobs_eval == 2).all()


def test_stops_on_unexpected_split_sha(smoke_run):
    """Not in smoke mode: if the SHA-256 of repeated_splits.csv differs from the expected value, stop with code 2 and write nothing."""
    env = smoke_env(smoke_run["data"])
    env.pop("MPQDA_SMOKE")
    args = [*smoke_run["args"], "--out", "taimma/SMOKE_sha"]
    res = run_script(smoke_run["repo"], "run_taimma.py", args, env, check=False)
    assert res.returncode == 2 and "SHA-256" in res.stderr
    assert not (smoke_run["out"].parents[1] / "SMOKE_sha" / SMOKE_COHORT / "selection_per_fold.csv").exists()


def test_sentinel_outer_test_rows_never_reach_phase_a(smoke_run):
    """Set the outer test rows to NaN after loading the data: every phase A fitness stays finite and the panels do not change."""
    args = [a for a in smoke_run["args"]]
    keep = args[:args.index("--methods") + 1] + ["mpqda_de", "ga", "shade"] + args[args.index("--k"):]
    run_script(smoke_run["repo"], "run_taimma.py", [*keep, "--out", "taimma/SMOKE_nan", "--nan-outer-test"], smoke_run["env"])
    other = smoke_run["out"].parents[1] / "SMOKE_nan" / SMOKE_COHORT
    sel, hist = read(other, "selection_per_fold.csv"), read(other, "history.csv")
    assert np.isfinite(sel.best_fitness).all() and np.isfinite(sel.output_fitness).all()
    assert np.isfinite(hist.best_fitness).all() and np.isfinite(hist.mean_fitness).all()
    a = read(smoke_run["out"], "selected_genes.csv")
    a = a[a.method.isin(set(sel.method))].reset_index(drop=True)
    assert a.to_csv(index=False) == read(other, "selected_genes.csv").to_csv(index=False)


def test_shuffled_labels_write_outer_evaluation(smoke_run):
    args = ["--scenario", "SMOKE_shuffle", "--metric", "logloss", "--datasets", SMOKE_COHORT, "--methods", "mpqda_de", "ga",
            "--k", "5", "--repeats", "0", "--folds", "0", "--pop", "6", "--generations", "3", "--shuffle-train-labels"]
    run_script(smoke_run["repo"], "run_taimma.py", args, smoke_run["env"])
    out = smoke_run["out"].parents[1] / "SMOKE_shuffle" / SMOKE_COHORT
    ev, sel = read(out, "t0_shuffle_eval.csv"), read(out, "selection_per_fold.csv")
    assert set(ev.method) == {"mpqda_de", "mpqda_de_best", "ga"} and ev.auroc.between(0, 1).all()
    assert sel.label_shuffled.all() and json.loads((out / "config.json").read_text(encoding="utf-8"))["label_shuffled"]


def test_auroc_mode_uses_per_method_inner_folds(smoke_run):
    args = ["--scenario", "SMOKE_auroc", "--metric", "auroc", "--datasets", SMOKE_COHORT, "--methods", "ga", "bpso",
            "--k", "5", "--repeats", "0", "--folds", "0", "--pop", "6", "--generations", "3"]
    run_script(smoke_run["repo"], "run_taimma.py", args, smoke_run["env"])
    sel = read(smoke_run["out"].parents[1] / "SMOKE_auroc" / SMOKE_COHORT, "selection_per_fold.csv")
    assert (sel.inner_fold_scheme == "per_method").all() and (sel.inner_fold_seed == sel.search_seed).all()
    assert sel.inner_fold_seed.nunique() == 2 and sel.saturated.isin([True, False]).all()
