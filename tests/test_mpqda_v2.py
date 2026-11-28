"""Tests of CIRK-DE (v2), mpqda/de_v2.py."""
import numpy as np
import pytest

from mpqda.encoding import band_encode, decode_topk  # noqa: E402
from mpqda.fitness import InnerCVFitness  # noqa: E402

from mpqda.de_v2 import ABLATIONS, CANDIDATES, gapped_band_encode, innovate, run_de_gapped, run_mpqda_v2  # noqa: E402
from scripts.disruption_sim import simulate  # noqa: E402


def small_data(seed=0, n=60, p=300):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (z + 0.5 * rng.normal(size=n) > 0).astype(int)
    X = rng.normal(size=(n, p))
    X[:, :20] += 1.2 * z[:, None]
    return X, y


def test_eps_half_is_v1_band_encode():
    rng = np.random.default_rng(1)
    for k in (2, 5, 10):
        x = rng.random(400)
        assert np.allclose(gapped_band_encode(x, k, 0.5), band_encode(x, k))


def test_gapped_band_keeps_panel_order_and_bands():
    rng = np.random.default_rng(2)
    x = rng.random(500)
    for k in (2, 5, 10):
        g = gapped_band_encode(x, k, 0.01)
        S = decode_topk(x, k)
        assert np.array_equal(decode_topk(g, k), S)
        out = np.setdiff1d(np.arange(500), S)
        assert g[S].min() > 0.5 and g[out].max() < 0.01
        assert np.array_equal(np.argsort(g[S], kind="stable"), np.argsort(x[S], kind="stable"))
        assert np.array_equal(np.argsort(g[out], kind="stable"), np.argsort(x[out], kind="stable"))


def test_innovate_replaces_exactly_n_genes():
    rng = np.random.default_rng(3)
    u = gapped_band_encode(rng.random(300), 10, 0.01)
    S0 = set(decode_topk(u, 10).tolist())
    for _ in range(50):
        v, n = innovate(u, 10, 0.3, rng)
        S1 = set(decode_topk(v, 10).tolist())
        assert len(S1) == 10 and len(S0 - S1) == n and len(S1 - S0) == n
    v, n = innovate(u, 10, 0.0, rng)
    assert n == 0 and np.array_equal(v, u)


def test_gap_removes_random_genes_from_trials():
    df = simulate(reps=40)
    gap = df[(df.eps == 0.01) & (df.CR == 0.9)]
    nogap = df[(df.eps == 0.5) & (df.CR == 0.9)]
    assert gap.random_genes.max() < 0.01                 # gap: no random genes left
    assert nogap.random_genes.min() > 0.1                # v1: 18–69 % of the genes of the trial panel are random
    assert (gap.leader_kept.values > nogap.leader_kept.values).all()


@pytest.mark.parametrize("name", [*CANDIDATES, *ABLATIONS])
def test_run_is_deterministic_exact_k_and_within_budget(name):
    X, y = small_data()
    cfg = {**CANDIDATES["mpqda_v2_m1"], **CANDIDATES.get(name, {}), **ABLATIONS.get(name, {})}
    outs = []
    for _ in range(2):
        fit = InnerCVFitness(X, y, fold_seed=11)
        r = run_mpqda_v2(fit, X.shape[1], 5, np.random.default_rng([5, 5]), P=6, T=3, **cfg)
        outs.append(r)
        assert len(r["consensus"]) == 5 and len(r["best"]) == 5
        assert r["n_eval"] <= 6 * (3 + 1) + 1
        assert len(r["history"]) == 4
    assert np.array_equal(outs[0]["consensus"], outs[1]["consensus"])
    assert outs[0]["best_fitness"] == outs[1]["best_fitness"]
    assert [h["best_fitness"] for h in outs[0]["history"]] == [h["best_fitness"] for h in outs[1]["history"]]


def test_de_gapped_runs():
    X, y = small_data(1)
    fit = InnerCVFitness(X, y, fold_seed=3)
    r = run_de_gapped(fit, X.shape[1], 5, np.random.default_rng([1, 5]), P=6, T=3)
    assert len(r["best"]) == 5 and r["n_eval"] <= 6 * 4
