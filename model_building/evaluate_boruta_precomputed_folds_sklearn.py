from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from model_building.rbf_svm_utils import compute_metrics
from model_building.boruta_mrmr_prefilter_maxminmean import DEFAULT_FEATURES_DIR, load_inputs
from model_building.sklearn_eval_utils import write_csv, write_json
from model_building.roi_sklearn_groupcv import build_pipeline


DEFAULT_SELECTION_CSV = (
    DEFAULT_FEATURES_DIR
    / "nested_eval_boruta_prefilter1000_relaxed_rbf"
    / "boruta_selected_by_fold.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate precomputed fold-safe Boruta feature selections with a different "
            "outer classifier on the same StratifiedGroupKFold split."
        )
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--classifier", type=str, default="svm_linear", choices=("logreg", "svm_linear", "svm_rbf"))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
    parser.add_argument("--ignore-labels", nargs="*", default=("skip", "unclear", "unknown"))
    parser.add_argument("--fallback-k", type=int, default=32)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None, classifier: str) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / f"nested_eval_boruta_prefilter1000_relaxed_{classifier}").resolve()


def select_indices_for_fold(selection_table: pd.DataFrame, outer_fold: int, fallback_k: int) -> np.ndarray:
    fold_table = selection_table.loc[selection_table["outer_fold"] == int(outer_fold)].copy()
    if fold_table.empty:
        raise ValueError(f"No rows found in selection table for outer_fold={outer_fold}")

    confirmed = fold_table.loc[fold_table["status"] == "confirmed", "feature_index"].astype(int).to_numpy()
    if confirmed.size > 0:
        return np.unique(confirmed)

    tentative = fold_table.loc[fold_table["status"] == "tentative"].copy()
    if tentative.empty:
        raise ValueError(f"No confirmed or tentative features found for outer_fold={outer_fold}")
    tentative = tentative.sort_values(["hit_rate", "mean_importance"], ascending=[False, False], kind="stable")
    return np.unique(tentative["feature_index"].astype(int).to_numpy()[: int(min(len(tentative), max(1, fallback_k)))])


def build_model(classifier: str, c_value: float, max_iter: int, class_weight: str | None) -> Pipeline:
    return build_pipeline(
        classifier=str(classifier),
        c_value=float(c_value),
        max_iter=int(max_iter),
        class_weight=class_weight,
    )


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    selection_csv = args.selection_csv.resolve()
    output_dir = default_output_dir(features_dir, args.output_dir, str(args.classifier))
    ensure_dir(output_dir)

    X, table, y, y_labels, class_names = load_inputs(
        features_dir=features_dir,
        valid_labels=list(args.valid_labels),
        ignore_labels=list(args.ignore_labels),
    )
    groups = table["group_id"].astype(str).to_numpy()
    selection_table = pd.read_csv(selection_csv)

    splitter = StratifiedGroupKFold(
        n_splits=int(args.outer_splits),
        shuffle=True,
        random_state=int(args.random_state),
    )
    class_weight = None if args.class_weight == "none" else str(args.class_weight)

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[Dict[str, object]] = []

    for outer_fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        selected_indices = select_indices_for_fold(
            selection_table=selection_table,
            outer_fold=outer_fold,
            fallback_k=int(args.fallback_k),
        )
        model = build_model(
            classifier=str(args.classifier),
            c_value=float(args.c_value),
            max_iter=int(args.max_iter),
            class_weight=class_weight,
        )
        model.fit(X[train_idx][:, selected_indices], y[train_idx])
        y_pred = model.predict(X[val_idx][:, selected_indices])
        y_proba = model.predict_proba(X[val_idx][:, selected_indices]).astype(np.float32)

        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = compute_metrics(y_labels[val_idx], np.array(class_names)[y_pred], class_names)
        fold_rows.append(
            {
                "fold": int(outer_fold),
                "num_train_rois": int(len(train_idx)),
                "num_val_rois": int(len(val_idx)),
                "num_train_groups": int(pd.Series(groups[train_idx]).nunique()),
                "num_val_groups": int(pd.Series(groups[val_idx]).nunique()),
                "selected_feature_count": int(len(selected_indices)),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "3d_precision": float(metrics["classification_report"]["3d"]["precision"]),
                "3d_recall": float(metrics["classification_report"]["3d"]["recall"]),
                "3d_f1": float(metrics["classification_report"]["3d"]["f1-score"]),
            }
        )

    if np.any(oof_pred < 0):
        raise RuntimeError("OOF predictions are incomplete.")

    y_pred_labels = np.array(class_names)[oof_pred]
    overall_metrics = compute_metrics(y_labels, y_pred_labels, class_names)

    oof_rows: List[Dict[str, object]] = []
    for row_dict, pred_idx, proba in zip(table.to_dict(orient="records"), oof_pred, oof_proba):
        predicted_label = class_names[int(pred_idx)]
        oof_rows.append(
            {
                **row_dict,
                "predicted_label": predicted_label,
                "proba_2d": float(proba[0]),
                "proba_3d": float(proba[1]),
                "predicted_probability": float(proba[1] if predicted_label == "3d" else proba[0]),
                "correct": int(predicted_label == row_dict["label_clean"]),
            }
        )

    write_csv(fold_rows, output_dir / "fold_metrics.csv")
    write_csv(oof_rows, output_dir / "oof_predictions.csv")
    summary = {
        "features_dir": str(features_dir),
        "selection_csv": str(selection_csv),
        "classifier": str(args.classifier),
        "outer_splits": int(args.outer_splits),
        "random_state": int(args.random_state),
        "c_value": float(args.c_value),
        "max_iter": int(args.max_iter),
        "class_weight": class_weight,
        "num_labeled_rois": int(len(table)),
        "num_groups": int(pd.Series(groups).nunique()),
        "class_names": class_names,
        "overall": overall_metrics,
        "folds": fold_rows,
    }
    write_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
