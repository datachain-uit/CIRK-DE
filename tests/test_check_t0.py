"""Condition (2) of T0 compares values, not numeric types (error met in the S0 trial run on Kaggle on 2026-11-15)."""
import io

import numpy as np
import pandas as pd

from scripts.taimma_jobs import same_values


def _csv(frame):
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)), float_precision="round_trip")


def test_int_column_turned_float_by_ga_rows_still_matches():
    mp = pd.DataFrame({"method": ["mpqda_de"] * 3, "t": [0, 1, 2], "archive_size": [15, 15, 15],
                       "best_fitness": [-0.31, -0.29, -0.2812345678901234]})
    ga = pd.DataFrame({"method": ["ga"] * 3, "t": [0, 1, 2], "best_fitness": [-0.4, -0.35, -0.33]})
    mixed = _csv(pd.concat([mp, ga], ignore_index=True))          # file T0_det_a: column archive_size becomes float
    alone = _csv(mp)                                               # file T0_det_j4: column archive_size is int
    one = mixed[mixed.method == "mpqda_de"].reset_index(drop=True)
    assert one.archive_size.dtype == float and alone.archive_size.dtype != float
    assert one.to_csv(index=False) != alone.to_csv(index=False)    # the old comparison reports a difference
    assert same_values(one, alone)


def test_real_differences_are_detected():
    a = pd.DataFrame({"t": [0, 1], "x": [0.1, 0.2], "s": ["a", "b"]})
    assert same_values(a, a.copy())
    assert not same_values(a, a.assign(x=[0.1, np.nextafter(0.2, 1)]))   # one ulp apart
    assert not same_values(a, a.assign(s=["a", "c"]))
    assert not same_values(a, a.iloc[:1])
    assert same_values(a.assign(x=[np.nan, 0.2]), a.assign(x=[np.nan, 0.2]))
