from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_METRICS_CSV = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_normalmap_bbox_metrics_local_ring_ignore_black\bbox_normalmap_features12\roi_bbox_normalmap_features12.csv"
)
DEFAULT_FEATURE_COLUMNS = [
    "grad_p95",
    "grad_max",
    "dominant_angle_mean_deg",
    "dominant_angle_p95_deg",
    "delta_mean",
    "delta_p95",
    "grad_frac_gt_t1",
    "grad_largest_component_size_t1",
    "normal_total_variance",
    "nz_std",
    "directional_coherence",
    "delta_frac_gt_t2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an SVM on labeled ROI bbox metric features with StratifiedGroupKFold."
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--feature-columns", nargs="*", default=DEFAULT_FEATURE_COLUMNS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    return parser.parse_args()


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


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def default_output_dir(metrics_csv: Path, explicit_output_dir: Path | None, kernel: str) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (metrics_csv.parent / f"svm_{kernel}_groupcv_results").resolve()


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    metrics_csv = args.metrics_csv.resolve()
    output_dir = default_output_dir(metrics_csv, args.output_dir, args.kernel)
    ensure_dir(output_dir)

    table = pd.read_csv(metrics_csv).copy()
    if args.label_column not in table.columns:
        raise ValueError(f"Label column not found in metrics CSV: {args.label_column}")
    table[args.label_column] = table[args.label_column].map(clean_label)
    labeled_table = table.loc[table[args.label_column].isin({"2d", "3d"})].copy()
    if labeled_table.empty:
        raise ValueError("No labeled 2D/3D bounding boxes found in metrics CSV.")

    feature_columns = list(args.feature_columns)
    missing_columns = [column for column in feature_columns if column not in labeled_table.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns in metrics CSV: {missing_columns}")

    if labeled_table[feature_columns].isna().any().any():
        nan_rows = labeled_table.index[labeled_table[feature_columns].isna().any(axis=1)].tolist()
        raise ValueError(f"NaN feature values found in labeled rows: first rows {nan_rows[:10]}")

    class_names = ["2d", "3d"]
    X = labeled_table[feature_columns].to_numpy(dtype=np.float32)
    y_labels = labeled_table[args.label_column].to_numpy()
    y = np.array([0 if label == "2d" else 1 for label in y_labels], dtype=np.int32)
    groups = labeled_table["sample"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: List[Dict[str, object]] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=args.kernel,
                        C=args.C,
                        gamma=args.gamma,
                        class_weight=None if args.class_weight == "none" else args.class_weight,
                        probability=True,
                        random_state=args.random_state + fold_index,
                    ),
                ),
            ]
        )

        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val).astype(np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = fold_metrics(
            np.array(class_names)[y_val],
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

    oof_rows: List[Dict[str, object]] = []
    labeled_table = labeled_table.reset_index(drop=True)
    for row_index, row in labeled_table.iterrows():
        output_row = row.to_dict()
        output_row["predicted_label"] = y_pred_labels[row_index]
        output_row["correct"] = int(y_labels[row_index] == y_pred_labels[row_index])
        output_row["proba_2d"] = float(oof_proba[row_index, 0])
        output_row["proba_3d"] = float(oof_proba[row_index, 1])
        oof_rows.append(output_row)

    fold_metrics_file = output_dir / "fold_metrics.csv"
    oof_predictions_file = output_dir / "oof_predictions.csv"
    summary_file = output_dir / "summary.json"

    write_csv(fold_rows, fold_metrics_file)
    write_csv(oof_rows, oof_predictions_file)
    write_json(
        {
            "metrics_csv": str(metrics_csv),
            "classifier": "svm",
            "kernel": args.kernel,
            "label_column": args.label_column,
            "feature_columns": feature_columns,
            "num_labeled_rois": int(len(labeled_table)),
            "num_groups": int(labeled_table["sample"].astype(str).nunique()),
            "class_names": class_names,
            "class_counts": {
                class_name: int((labeled_table[args.label_column] == class_name).sum())
                for class_name in class_names
            },
            "cv_type": "StratifiedGroupKFold",
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "C": float(args.C),
            "gamma": args.gamma,
            "class_weight": None if args.class_weight == "none" else args.class_weight,
            "overall": overall_metrics,
            "folds": fold_rows,
            "fold_metrics_file": str(fold_metrics_file),
            "oof_predictions_file": str(oof_predictions_file),
        },
        summary_file,
    )

    print(f"Saved fold metrics: {fold_metrics_file}")
    print(f"Saved OOF predictions: {oof_predictions_file}")
    print(f"Saved summary: {summary_file}")
    print(
        f"OOF macro F1: {overall_metrics['macro_f1']:.4f} | "
        f"Accuracy: {overall_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
