"""Phase B-LL of the TAIMMA study (PREREG, Section 4).

Identical to survey_v2.classify.evaluate_panel with LR-L2, except for the criterion that selects C: smallest inner OOF log-loss instead of
largest inner OOF AUROC. Threshold, rule M5 (minority class <= SMALL_N_T_MAX samples), seed, standardisation and refit are unchanged.
criterion = "auroc" reproduces evaluate_panel of the vendored code exactly; used to check the code path, not for the numbers of the paper.
criterion = "c1" (B-C1): C = 1 fixed as in the fitness, no grid search; threshold, M5, seed, refit
as in B-LL. Used only for sensitivity analysis.
criterion = "llwide" (B-LL-wide): as B-LL but searched over the wide grid C_GRID_WIDE, ignoring c_grid
of the caller; checks whether the chosen C hits the edge of the 0.01..10 grid. Used only for sensitivity analysis.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import mpqda  # noqa: F401  (puts vendor/ on sys.path)
from survey_v2 import config as C
from survey_v2.classify import EvalResult, _best_threshold, _fit_predict, evaluate_panel, probability_metrics

CRITERIA = ("logloss", "auroc", "c1", "llwide")
C_FIXED = 1.0                                  # B-C1: equal to the C of the fitness (mpqda.config.FITNESS_C)
C_GRID_WIDE = (1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)   # B-LL-wide: two orders of magnitude on each side of the 0.01..10 grid
PROB_CLIP = 1e-6          # same clipping as probability_metrics of the vendored code


def grid_for(criterion: str, c_grid=C.C_GRID) -> tuple:
    """C grid actually searched by the criterion: c1 one value, llwide the wide grid (independent of c_grid), otherwise c_grid."""
    if criterion == "c1":
        return (C_FIXED,)
    if criterion == "llwide":
        return C_GRID_WIDE
    return tuple(c_grid)


def evaluate_panel_ll(Xtr, ytr, Xte, yte, seed: int = 0, c_grid=C.C_GRID, inner_folds: int = C.INNER_CV_FOLDS,
                      n_jobs: int = 1, small_n: bool | None = None, criterion: str = "logloss") -> EvalResult:
    if criterion not in CRITERIA:
        raise ValueError(criterion)
    n_min = int(min(np.sum(ytr == 0), np.sum(ytr == 1)))
    if small_n is None:
        small_n = n_min <= C.SMALL_N_T_MAX
    if small_n or n_min < inner_folds:                       # rule M5 of the vendored code: unchanged
        return evaluate_panel(Xtr, ytr, Xte, yte, classifier="lr_l2", seed=seed, c_grid=c_grid,
                              inner_folds=inner_folds, n_jobs=n_jobs, small_n=small_n)
    inner = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed).split(np.zeros(len(ytr)), ytr))
    best_c, best_score, best_oof = None, -np.inf, None
    for c in grid_for(criterion, c_grid):                    # ascending grid; ties keep the smaller C, as in the vendored code
        oof = np.zeros(len(ytr))
        for tr, va in inner:
            oof[va] = _fit_predict("lr_l2", c, seed, n_jobs, Xtr[tr], ytr[tr], Xtr[va])
        if criterion == "c1":
            score = 0.0                                       # a single C value, no selection
        elif criterion == "auroc":
            score = roc_auc_score(ytr, oof)
        else:                                                 # logloss and llwide
            score = -log_loss(ytr, np.clip(oof, PROB_CLIP, 1 - PROB_CLIP), labels=[0, 1])
        if score > best_score:
            best_c, best_score, best_oof = c, score, oof
    thr = _best_threshold(ytr, best_oof)
    s_te = _fit_predict("lr_l2", best_c, seed, n_jobs, Xtr, ytr, Xte)
    pred = (s_te >= thr).astype(int)
    ll, br = probability_metrics(yte, s_te, True)
    return EvalResult(classifier="lr_l2", auroc=float(roc_auc_score(yte, s_te)),
                      bacc=float(balanced_accuracy_score(yte, pred)), mcc=float(matthews_corrcoef(yte, pred)),
                      threshold=thr, hyperparam=best_c, scores_test=s_te, logloss=ll, brier=br)


def make_evaluator(criterion: str):
    """Same signature as survey_v2.classify.evaluate_panel: LR-L2 goes through evaluate_panel_ll, other classifiers are unchanged."""
    if criterion not in CRITERIA:
        raise ValueError(criterion)

    def _evaluate(Xtr, ytr, Xte, yte, classifier: str = "lr_l2", seed: int = 0, c_grid=C.C_GRID,
                  inner_folds: int = C.INNER_CV_FOLDS, n_jobs: int = 1, small_n: bool | None = None):
        if classifier != "lr_l2":
            return evaluate_panel(Xtr, ytr, Xte, yte, classifier=classifier, seed=seed, c_grid=c_grid,
                                  inner_folds=inner_folds, n_jobs=n_jobs, small_n=small_n)
        return evaluate_panel_ll(Xtr, ytr, Xte, yte, seed=seed, c_grid=c_grid, inner_folds=inner_folds,
                                 n_jobs=n_jobs, small_n=small_n, criterion=criterion)
    _evaluate.criterion = criterion
    return _evaluate
