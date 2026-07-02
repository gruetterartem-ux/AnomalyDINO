from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from extract_labeled_roi_overthreshold_multilayer_maxminmean_features import DEFAULT_EXPERIMENT_DIR
from model_building.rbf_svm_utils import build_classifier, compute_metrics


DEFAULT_FEATURES_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_overthreshold_overlap_multilayer_l1to12_maxminmean_features_labeled"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold-safe outer CV for overlap+over-threshold max/min/mean ROI features with "
            "train-only mRMR selection and fixed-k RBF-SVM."
        )
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--fixed-k", type=int, default=384)
    parser.add_argument("--prefilter-top", type=int, default=4096)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--output-dir", type=Path, default=None)
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


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None, fixed_k: int) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / f"nested_eval_mrmr_fixedk{int(fixed_k)}_rbf").resolve()


def load_inputs(features_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    features_file = features_dir / "roi_features_mean.npy"
    table_file = features_dir / "roi_feature_table.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not table_file.exists():
        raise FileNotFoundError(f"Feature table not found: {table_file}")
    features = np.load(features_file).astype(np.float32)
    table = pd.read_csv(table_file)
    if len(features) != len(table):
        raise ValueError(f"Length mismatch: {len(features)} features vs {len(table)} rows")
    return features, table


def safe_abs_corrcoef(matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return corr


def greedy_mrmr_rank(relevance: np.ndarray, redundancy_matrix: np.ndarray, max_select: int) -> np.ndarray:
    selected_local: list[int] = []
    selected_mask = np.zeros(relevance.shape[0], dtype=bool)
    redundancy_sum = np.zeros(relevance.shape[0], dtype=np.float32)
    for iteration in range(max_select):
        if iteration == 0:
            scores = relevance.copy()
        else:
            scores = relevance - (redundancy_sum / float(iteration))
        scores[selected_mask] = -np.inf
        best_local = int(np.argmax(scores))
        selected_local.append(best_local)
        selected_mask[best_local] = True
        redundancy_sum += redundancy_matrix[:, best_local]
    return np.asarray(selected_local, dtype=np.int32)


def rank_mrmr_features(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    fixed_k: int,
    prefilter_top: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    selector_scaler = StandardScaler()
    X_scaled = selector_scaler.fit_transform(X_train_raw).astype(np.float32)
    relevance = mutual_info_classif(X_scaled, y_train, discrete_features=False, random_state=int(random_state))
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    prefilter_top = int(min(max(prefilter_top, fixed_k), X_train_raw.shape[1]))
    prefilter_indices = np.argsort(-relevance, kind="stable")[:prefilter_top]
    redundancy = safe_abs_corrcoef(X_scaled[:, prefilter_indices])
    ranked_local = greedy_mrmr_rank(relevance[prefilter_indices], redundancy, fixed_k)
    ranked_indices = prefilter_indices[ranked_local]
    return ranked_indices.astype(np.int32), relevance


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    output_dir = default_output_dir(features_dir, args.output_dir, int(args.fixed_k))
    ensure_dir(output_dir)

    X, table = load_inputs(features_dir)
    table = table.copy()
    table["label_clean"] = table["label"].astype(str).str.strip().str.lower()
    labeled_mask = table["label_clean"].isin({"2d", "3d"})
    table = table.loc[labeled_mask].reset_index(drop=True)
    X = X[labeled_mask.to_numpy()]

    label_encoder = LabelEncoder()
    label_encoder.fit(["2d", "3d"])
    y_labels = table["label_clean"].to_numpy()
    y = label_encoder.transform(y_labels)
    class_names = list(label_encoder.classes_)
    groups = table["group_id"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=int(args.outer_splits),
        shuffle=True,
        random_state=int(args.random_state),
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_val_labels = y_labels[val_idx]

        ranked_indices, relevance = rank_mrmr_features(
            X_train_raw=X_train,
            y_train=y_train,
            fixed_k=int(args.fixed_k),
            prefilter_top=int(args.prefilter_top),
            random_state=int(args.random_state) + fold,
        )

        model = build_classifier(
            c_value=float(args.svm_c),
            gamma=str(args.svm_gamma),
            class_weight=None if args.class_weight == "none" else str(args.class_weight),
            random_state=int(args.random_state) + fold,
        )
        model.fit(X_train[:, ranked_indices], y_train)
        y_pred = model.predict(X_val[:, ranked_indices])
        y_proba = model.predict_proba(X_val[:, ranked_indices]).astype(np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = compute_metrics(y_val_labels, np.array(class_names)[y_pred], class_names)
        fold_rows.append(
            {
                "fold": int(fold),
                "num_train_rois": int(len(train_idx)),
                "num_val_rois": int(len(val_idx)),
                "num_train_groups": int(pd.Series(groups[train_idx]).nunique()),
                "num_val_groups": int(pd.Series(groups[val_idx]).nunique()),
                "fixed_k": int(args.fixed_k),
                "prefilter_top": int(min(max(int(args.prefilter_top), int(args.fixed_k)), X_train.shape[1])),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "3d_precision": float(metrics["classification_report"]["3d"]["precision"]),
                "3d_recall": float(metrics["classification_report"]["3d"]["recall"]),
                "3d_f1": float(metrics["classification_report"]["3d"]["f1-score"]),
            }
        )
        for rank, feature_index in enumerate(ranked_indices, start=1):
            selected_rows.append(
                {
                    "fold": int(fold),
                    "rank": int(rank),
                    "feature_index": int(feature_index),
                    "mutual_information": float(relevance[feature_index]),
                }
            )

    if np.any(oof_pred < 0):
        raise RuntimeError("OOF predictions are incomplete.")

    y_pred_labels = np.array(class_names)[oof_pred]
    overall_metrics = compute_metrics(y_labels, y_pred_labels, class_names)

    oof_rows: list[dict[str, object]] = []
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
    write_csv(selected_rows, output_dir / "selected_topk_by_fold.csv")
    write_csv(oof_rows, output_dir / "oof_predictions.csv")

    summary = {
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
        "selector": "mrmr_mi_relevance_abs_corr_redundancy",
        "fixed_k": int(args.fixed_k),
        "prefilter_top": int(args.prefilter_top),
        "outer_splits": int(args.outer_splits),
        "class_names": class_names,
        "num_labeled_rois": int(len(table)),
        "num_groups": int(pd.Series(groups).nunique()),
        "overall": overall_metrics,
        "folds": fold_rows,
    }
    write_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
