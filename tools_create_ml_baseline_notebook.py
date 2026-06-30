import json
from pathlib import Path


NOTEBOOK_PATH = Path(r"D:\feature-selection-experiments\notebooks\ml-baseline-no-feature-selection.ipynb")


def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    code_cell('''try:
    from google.colab import drive
    drive.mount("/content/drive")
except ModuleNotFoundError:
    print("Not running in Google Colab; skipping Drive mount.")
'''),
    code_cell('''from pathlib import Path
import os


def resolve_data_path(filename: str) -> str:
    candidates = []

    env_data_dir = os.environ.get("FEATURE_SELECTION_DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir) / filename)

    cwd = Path.cwd()
    candidates.extend([
        cwd / "data" / filename,
        cwd.parent / "data" / filename,
        Path("/content/data") / filename,
        Path("/content/feature-selection-experiments/data") / filename,
        Path("/content/drive/MyDrive/feature-selection-experiments/data") / filename,
        Path("/content/drive/MyDrive/data") / filename,
        Path(r"D:\\feature-selection-experiments\\data") / filename,
    ])

    for path in candidates:
        if path.exists():
            return str(path)

    search_roots = [Path("/content"), Path("/content/drive/MyDrive")]
    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.glob(f"**/{filename}"))
        for match in matches:
            if match.parent.name == "data":
                return str(match)
        if matches:
            return str(matches[0])

    checked = "\\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Checked:\\n{checked}\\n\\n"
        "Upload data to /content/data, or mount Drive and place it at "
        "/content/drive/MyDrive/feature-selection-experiments/data."
    )


DATA_DIR = Path(resolve_data_path("X_train.csv")).parent
if Path("/content/drive/MyDrive").exists():
    RESULT_DIR = Path("/content/drive/MyDrive/feature-selection-experiments/results/ml-baseline-no-feature-selection")
else:
    RESULT_DIR = Path.cwd() / "results" / "ml-baseline-no-feature-selection"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def output_path(filename: str) -> str:
    return str(RESULT_DIR / filename)


print(f"Using DATA_DIR: {DATA_DIR}")
print(f"Using RESULT_DIR: {RESULT_DIR}")
'''),
    code_cell('''import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
'''),
    code_cell('''X_train = pd.read_csv(resolve_data_path("X_train.csv"))
X_val = pd.read_csv(resolve_data_path("X_val.csv"))
X_test = pd.read_csv(resolve_data_path("X_test.csv"))

y_train = pd.read_csv(resolve_data_path("y_train.csv")).iloc[:, 0].values
y_val = pd.read_csv(resolve_data_path("y_val.csv")).iloc[:, 0].values
y_test = pd.read_csv(resolve_data_path("y_test.csv")).iloc[:, 0].values

unique_classes = np.unique(y_train)
N_CLASSES = len(unique_classes)
CLASS_NAMES = [str(c) for c in unique_classes]

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)
print("N_CLASSES :", N_CLASSES)
print("CLASS_NAMES:", CLASS_NAMES)
'''),
    code_cell('''def compute_metrics(y_true, y_pred, model_name: str, split: str) -> dict:
    return {
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def flatten_classification_report(y_true, y_pred, model_name: str, split: str) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    row = {"model": model_name, "split": split}
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                row[f"{label}_{metric_name}"] = value
        else:
            row[label] = metrics
    return row
'''),
    code_cell('''ML_CONFIGS = [
    (
        "KNN",
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
    ),
    (
        "DecisionTree",
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
    ),
    (
        "RandomForest",
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
    ),
    (
        "GaussianNB",
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    ),
    (
        "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "saga"],
            "multi_class": ["multinomial"],
        },
    ),
    (
        "LinearSVM",
        LinearSVC(max_iter=3000, random_state=42),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    (
        "RBF_SVM",
        SVC(kernel="rbf", random_state=42),
        {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
        },
    ),
]
'''),
    code_cell('''summary_rows = []
report_rows = []
confusion_rows = []
best_params_rows = []

for model_name, estimator, param_grid in ML_CONFIGS:
    print(f"\\n[{model_name}] GridSearchCV on all features...")
    start_time = time.perf_counter()

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    elapsed_sec = time.perf_counter() - start_time
    best_model = grid_search.best_estimator_

    best_params_rows.append({
        "model": model_name,
        "best_params": json.dumps(grid_search.best_params_, ensure_ascii=False),
        "best_cv_f1_macro": grid_search.best_score_,
        "fit_time_sec": elapsed_sec,
        "n_features": X_train.shape[1],
    })

    for split_name, X_split, y_split in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = best_model.predict(X_split)

        metrics_row = compute_metrics(y_split, y_pred, model_name, split_name)
        metrics_row["best_params"] = json.dumps(grid_search.best_params_, ensure_ascii=False)
        metrics_row["best_cv_f1_macro"] = grid_search.best_score_
        metrics_row["fit_time_sec"] = elapsed_sec
        metrics_row["n_features"] = X_train.shape[1]
        summary_rows.append(metrics_row)

        report_row = flatten_classification_report(y_split, y_pred, model_name, split_name)
        report_row["best_params"] = json.dumps(grid_search.best_params_, ensure_ascii=False)
        report_row["best_cv_f1_macro"] = grid_search.best_score_
        report_rows.append(report_row)

        cm = confusion_matrix(y_split, y_pred, labels=unique_classes)
        for true_idx, true_label in enumerate(unique_classes):
            for pred_idx, pred_label in enumerate(unique_classes):
                confusion_rows.append({
                    "model": model_name,
                    "split": split_name,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(cm[true_idx, pred_idx]),
                })

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(output_path(f"cv_results_{model_name}.csv"), index=False, encoding="utf-8-sig")

    print(
        f"  Best CV F1-macro: {grid_search.best_score_:.4f} | "
        f"Best params: {grid_search.best_params_} | "
        f"Time: {elapsed_sec:.2f}s"
    )
'''),
    code_cell('''summary_df = pd.DataFrame(summary_rows)
report_df = pd.DataFrame(report_rows)
confusion_df = pd.DataFrame(confusion_rows)
best_params_df = pd.DataFrame(best_params_rows)

summary_df.to_csv(output_path("ml_baseline_summary_metrics.csv"), index=False, encoding="utf-8-sig")
report_df.to_csv(output_path("ml_baseline_classification_reports.csv"), index=False, encoding="utf-8-sig")
confusion_df.to_csv(output_path("ml_baseline_confusion_matrices.csv"), index=False, encoding="utf-8-sig")
best_params_df.to_csv(output_path("ml_baseline_best_params.csv"), index=False, encoding="utf-8-sig")

display(summary_df.sort_values(["split", "f1_macro"], ascending=[True, False]))
print(f"Saved results to: {RESULT_DIR}")
'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"created {NOTEBOOK_PATH}")
