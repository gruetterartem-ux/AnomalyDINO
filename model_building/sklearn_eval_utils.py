from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(rows: List[Dict[str, object]], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    if not rows:
        output_file.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_default_k_values(feature_dim: int) -> list[int]:
    candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, feature_dim]
    values = sorted({int(k) for k in candidates if 1 <= int(k) <= feature_dim})
    if values[-1] != feature_dim:
        values.append(feature_dim)
    return values


def build_estimator(args: argparse.Namespace, classifier: str):
    class_weight = None if args.class_weight == "none" else args.class_weight
    if classifier == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=args.max_iter,
                        C=args.c_value,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
    if classifier == "svm_linear":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        C=args.c_value,
                        class_weight=class_weight,
                        probability=True,
                        max_iter=args.max_iter,
                    ),
                ),
            ]
        )
    if classifier == "svm_rbf":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=args.c_value,
                        gamma=args.gamma,
                        class_weight=class_weight,
                        probability=True,
                        max_iter=args.max_iter,
                    ),
                ),
            ]
        )
    if classifier == "rf":
        rf_class_weight = None if args.rf_class_weight == "none" else args.rf_class_weight
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=args.rf_max_features,
            class_weight=rf_class_weight,
            random_state=args.random_state,
            n_jobs=1,
        )
    if classifier == "extratrees":
        rf_class_weight = None if args.rf_class_weight == "none" else args.rf_class_weight
        return ExtraTreesClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=args.rf_max_features,
            class_weight=rf_class_weight,
            random_state=args.random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported classifier: {classifier}")


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }


def score_value(metrics: Dict[str, object], score_key: str) -> float:
    if score_key in ("macro_f1", "macro_recall", "accuracy"):
        return float(metrics[score_key])
    report = metrics["classification_report"]
    if score_key == "3d_f1":
        return float(report["3d"]["f1-score"])
    if score_key == "3d_recall":
        return float(report["3d"]["recall"])
    raise ValueError(f"Unsupported score key: {score_key}")


def evaluate_subset(
    X: np.ndarray,
    y: np.ndarray,
    y_labels: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
    classifier: str,
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[Dict[str, object]], list[Dict[str, object]]]:
    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    estimator = build_estimator(args, classifier)
    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[Dict[str, object]] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_val).astype(np.float32)
        else:
            y_proba = np.zeros((len(val_idx), len(class_names)), dtype=np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = fold_metrics(
            y_labels[val_idx],
            np.array(class_names)[y_pred],
            class_names,
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "num_train_rois": int(len(train_idx)),
                "num_val_rois": int(len(val_idx)),
                "num_train_groups": int(pd.Series(groups[train_idx]).nunique()),
                "num_val_groups": int(pd.Series(groups[val_idx]).nunique()),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    if np.any(oof_pred < 0):
        raise RuntimeError("Some out-of-fold predictions were not filled.")

    y_pred_labels = np.array(class_names)[oof_pred]
    overall_metrics = fold_metrics(y_labels, y_pred_labels, class_names)
    oof_rows: list[Dict[str, object]] = []
    for row_index in range(len(y)):
        row = {
            "row_index": int(row_index),
            "true_label": str(y_labels[row_index]),
            "predicted_label": str(y_pred_labels[row_index]),
            "correct": int(y_labels[row_index] == y_pred_labels[row_index]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"proba_{class_name}"] = float(oof_proba[row_index, class_index])
        oof_rows.append(row)

    return overall_metrics, fold_rows, oof_rows
