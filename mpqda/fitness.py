"""Fitness CV nội bộ của LR-L2 trên PHẦN TRAIN của fold ngoài (Mục 3.4; PREREG Mục 3.2).

- metric = "logloss" (chính): −(trung bình log-loss của 3 fold nội bộ), xác suất cắt vào [1e-6; 1 − 1e-6], labels = [0, 1].
- metric = "auroc" (T1b): giữ đúng từng phép tính của `survey_v2.wrappers.InnerCVFitness`.
Lớp không có tham số nào cho dữ liệu test: phần test của fold ngoài không bao giờ đi vào tìm kiếm.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import config as M

METRICS = ("logloss", "auroc")


def score_panel(X, y, folds, key, metric: str, C_lr: float = M.FITNESS_C, lr_seed: int = 0) -> float:
    """Fitness của một panel (bộ chỉ số đã sắp). Chuẩn hoá khớp trên phần train nội bộ; LR liblinear, C = 1, balanced."""
    Xs = X[:, list(key)]
    vals = []
    for tr, va in folds:
        sc = StandardScaler().fit(Xs[tr])
        m = LogisticRegression(C=C_lr, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=lr_seed)
        m.fit(sc.transform(Xs[tr]), y[tr])
        Xva = sc.transform(Xs[va])
        if metric == "auroc":
            vals.append(roc_auc_score(y[va], m.decision_function(Xva)))
        elif metric == "logloss":
            prob = np.clip(m.predict_proba(Xva)[:, 1], M.PROB_CLIP, 1 - M.PROB_CLIP)
            vals.append(-log_loss(y[va], prob, labels=[0, 1]))
        else:
            raise ValueError(metric)
    return float(np.mean(vals))


def _score_chunk(X, y, folds, keys, metric, C_lr, lr_seed):
    with threadpool_limits(1):
        return [score_panel(X, y, folds, k, metric, C_lr, lr_seed) for k in keys]


class InnerCVFitness:
    """fold_seed quyết định fold nội bộ (và random_state của LR): chế độ logloss dùng seed chung theo fold ngoài, chế độ
    auroc dùng seed của phương pháp như survey (Đ4). Cache theo panel đã sắp; n_eval đếm số panel khác nhau đã chấm;
    log ghi mọi lời gọi (panel, giá trị, n_eval sau lời gọi), kể cả lần trúng cache."""

    def __init__(self, X_tr, y_tr, fold_seed: int, metric: str = "logloss", n_folds: int = M.INNER_FOLDS,
                 C_lr: float = M.FITNESS_C):
        if metric not in METRICS:
            raise ValueError(metric)
        self.X, self.y, self.metric, self.C_lr = X_tr, y_tr, metric, C_lr
        self.fold_seed = self.lr_seed = int(fold_seed)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(fold_seed))
        self.folds = list(cv.split(np.zeros(len(y_tr)), y_tr))
        self.n_eval = 0
        self._cache: dict[tuple, float] = {}
        self.log: list[tuple[tuple, float, int]] = []

    @staticmethod
    def key(genes) -> tuple:
        return tuple(sorted(int(g) for g in genes))

    def __call__(self, genes) -> float:        # GA, BPSO của survey gọi từng panel một
        return self.evaluate_many([genes])[0]

    def evaluate_many(self, panels, n_jobs: int = 1) -> list[float]:
        """Panel chưa có trong cache được chấm theo thứ tự xuất hiện đầu tiên, chia thành n_jobs khúc liền nhau (joblib,
        loky), BLAS một luồng trong tiến trình con. Kết quả trùng từng bit giữa các giá trị n_jobs."""
        keys = [self.key(g) for g in panels]
        todo = list(dict.fromkeys(k for k in keys if k not in self._cache))
        if todo:
            if n_jobs == 1 or len(todo) == 1:
                vals = _score_chunk(self.X, self.y, self.folds, todo, self.metric, self.C_lr, self.lr_seed)
            else:
                parts = [p for p in np.array_split(np.arange(len(todo)), n_jobs) if len(p)]
                out = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(_score_chunk)(self.X, self.y, self.folds, [todo[i] for i in part], self.metric,
                                          self.C_lr, self.lr_seed) for part in parts)
                vals = [v for chunk in out for v in chunk]
            for k, v in zip(todo, vals):
                self._cache[k] = v
                self.n_eval += 1
        res = [self._cache[k] for k in keys]
        self.log.extend((k, v, self.n_eval) for k, v in zip(keys, res))
        return res
