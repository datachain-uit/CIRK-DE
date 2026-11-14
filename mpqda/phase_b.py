"""Pha B-LL của bài TAIMMA (TQĐ11, Review-2026-11-14b), chép từ taimma_ref/phase_b_ll_ref.py.

Giống hệt survey_v2.classify.evaluate_panel với LR-L2, chỉ khác tiêu chí chọn C: log-loss OOF nội bộ nhỏ nhất thay cho
AUROC OOF nội bộ lớn nhất. Ngưỡng, quy tắc M5 (lớp nhỏ <= SMALL_N_T_MAX mẫu), seed, chuẩn hoá và refit giữ nguyên.
criterion = "auroc" tái tạo đúng evaluate_panel của survey; dùng để kiểm đường mã, không dùng cho số liệu của bài.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import mpqda  # noqa: F401  (đặt vendor/ vào sys.path)
from survey_v2 import config as C
from survey_v2.classify import EvalResult, _best_threshold, _fit_predict, evaluate_panel, probability_metrics

CRITERIA = ("logloss", "auroc")
PROB_CLIP = 1e-6          # cùng mức cắt với probability_metrics của survey


def evaluate_panel_ll(Xtr, ytr, Xte, yte, seed: int = 0, c_grid=C.C_GRID, inner_folds: int = C.INNER_CV_FOLDS,
                      n_jobs: int = 1, small_n: bool | None = None, criterion: str = "logloss") -> EvalResult:
    if criterion not in CRITERIA:
        raise ValueError(criterion)
    n_min = int(min(np.sum(ytr == 0), np.sum(ytr == 1)))
    if small_n is None:
        small_n = n_min <= C.SMALL_N_T_MAX
    if small_n or n_min < inner_folds:                       # quy tắc M5 của survey: giữ nguyên
        return evaluate_panel(Xtr, ytr, Xte, yte, classifier="lr_l2", seed=seed, c_grid=c_grid,
                              inner_folds=inner_folds, n_jobs=n_jobs, small_n=small_n)
    inner = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed).split(np.zeros(len(ytr)), ytr))
    best_c, best_score, best_oof = None, -np.inf, None
    for c in c_grid:                                          # lưới tăng dần; hoà giữ C nhỏ hơn, như survey
        oof = np.zeros(len(ytr))
        for tr, va in inner:
            oof[va] = _fit_predict("lr_l2", c, seed, n_jobs, Xtr[tr], ytr[tr], Xtr[va])
        if criterion == "auroc":
            score = roc_auc_score(ytr, oof)
        else:
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
    """Hàm cùng chữ ký với survey_v2.classify.evaluate_panel: LR-L2 đi qua evaluate_panel_ll, bộ phân loại khác giữ nguyên."""
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
