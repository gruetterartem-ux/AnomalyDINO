from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from model_building.rbf_svm_utils import build_classifier, compute_metrics
from model_building.boruta_mrmr_prefilter_maxminmean import (
    DEFAULT_FEATURES_DIR,
    decode_feature_index,
    format_seconds,
    greedy_mrmr_rank,
    load_inputs,
    safe_abs_corrcoef,
    write_progress,
)
from model_building.sklearn_eval_utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold-safe outer CV for overlap+over-threshold max/min/mean ROI features with "
            "train-only mRMR prefilter + Boruta selection and outer RBF-SVM evaluation."
        )
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--prefilter-k", type=int, default=1000)
    parser.add_argument("--mrmr-prefilter-top", type=int, default=4096)
    parser.add_argument("--boruta-max-iter", type=int, default=100)
    parser.add_argument("--boruta-alpha", type=float, default=0.1)
    parser.add_argument("--shadow-percentile", type=float, default=95.0)
    parser.add_argument("--boruta-patience", type=int, default=40)
    parser.add_argument("--rf-n-estimators", type=int, default=256)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")
    parser.add_argument("--rf-class-weight", type=str, default="balanced_subsample")
    parser.add_argument("--rf-n-jobs", type=int, default=1)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
    parser.add_argument("--ignore-labels", nargs="*", default=("skip", "unclear", "unknown"))
    parser.add_argument("--fallback-k", type=int, default=32)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / "nested_eval_boruta_prefilter1000_relaxed_rbf").resolve()


def build_mrmr_prefilter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prefilter_k: int,
    mrmr_prefilter_top: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler

    feature_dim = int(X_train.shape[1])
    prefilter_k = int(min(max(1, prefilter_k), feature_dim))
    mrmr_prefilter_top = int(min(max(prefilter_k, mrmr_prefilter_top), feature_dim))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)
    relevance = mutual_info_classif(X_scaled, y_train, discrete_features=False, random_state=int(random_state))
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    seed_indices = np.argsort(-relevance, kind="stable")[:mrmr_prefilter_top]
    redundancy = safe_abs_corrcoef(X_scaled[:, seed_indices])
    ranked_local = greedy_mrmr_rank(relevance[seed_indices], redundancy, prefilter_k)
    prefilter_indices = seed_indices[ranked_local]
    return prefilter_indices.astype(np.int32), relevance


def run_boruta_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prefilter_indices: np.ndarray,
    relevance: np.ndarray,
    args: argparse.Namespace,
    outer_fold: int,
    progress_file: Path,
    global_start_time: float,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    X_prefilter = X_train[:, prefilter_indices].astype(np.float32)
    n_features = int(X_prefilter.shape[1])
    statuses = np.zeros(n_features, dtype=np.int8)
    hits = np.zeros(n_features, dtype=np.int32)
    real_importance_sum = np.zeros(n_features, dtype=np.float64)
    real_importance_max = np.zeros(n_features, dtype=np.float64)
    iteration_rows: list[dict[str, object]] = []
    no_change_rounds = 0

    for iteration in range(1, int(args.boruta_max_iter) + 1):
        iter_start = time.time()
        tentative_mask = statuses == 0
        num_tentative = int(tentative_mask.sum())
        num_confirmed = int((statuses == 1).sum())
        num_rejected = int((statuses == -1).sum())

        if num_tentative == 0:
            break

        shadow = X_prefilter[:, tentative_mask].copy()
        rng = np.random.default_rng(int(args.random_state) + outer_fold * 1000 + iteration)
        for column_index in range(shadow.shape[1]):
            rng.shuffle(shadow[:, column_index])

        X_boruta = np.concatenate([X_prefilter, shadow], axis=1)
        rf = RandomForestClassifier(
            n_estimators=int(args.rf_n_estimators),
            max_depth=None if args.rf_max_depth is None else int(args.rf_max_depth),
            min_samples_leaf=int(args.rf_min_samples_leaf),
            max_features=str(args.rf_max_features),
            class_weight=None if args.rf_class_weight == "none" else str(args.rf_class_weight),
            n_jobs=int(args.rf_n_jobs),
            random_state=int(args.random_state) + outer_fold * 1000 + iteration,
        )
        rf.fit(X_boruta, y_train)
        importances = rf.feature_importances_.astype(np.float64)
        real_importances = importances[:n_features]
        shadow_importances = importances[n_features:]
        threshold = float(np.percentile(shadow_importances, float(args.shadow_percentile)))

        hits += (real_importances > threshold).astype(np.int32)
        real_importance_sum += real_importances
        real_importance_max = np.maximum(real_importance_max, real_importances)

        changed = False
        bonferroni = max(1, num_tentative)
        for feature_idx in np.where(tentative_mask)[0]:
            p_accept = binomtest(int(hits[feature_idx]), iteration, 0.5, alternative="greater").pvalue
            p_reject = binomtest(int(hits[feature_idx]), iteration, 0.5, alternative="less").pvalue
            if p_accept < (float(args.boruta_alpha) / float(bonferroni)):
                statuses[feature_idx] = 1
                changed = True
            elif p_reject < (float(args.boruta_alpha) / float(bonferroni)):
                statuses[feature_idx] = -1
                changed = True

        no_change_rounds = 0 if changed else (no_change_rounds + 1)
        iter_seconds = time.time() - iter_start
        elapsed = time.time() - global_start_time
        completed_outer_fraction = (outer_fold - 1) + (iteration / float(max(1, int(args.boruta_max_iter))))
        mean_per_outer = elapsed / completed_outer_fraction if completed_outer_fraction > 0 else None
        remaining_outer = max(0.0, float(int(args.outer_splits)) - completed_outer_fraction)
        eta_seconds = None if mean_per_outer is None else mean_per_outer * remaining_outer

        num_confirmed = int((statuses == 1).sum())
        num_rejected = int((statuses == -1).sum())
        num_tentative = int((statuses == 0).sum())
        iteration_rows.append(
            {
                "outer_fold": int(outer_fold),
                "iteration": int(iteration),
                "shadow_threshold": float(threshold),
                "num_confirmed": num_confirmed,
                "num_rejected": num_rejected,
                "num_tentative": num_tentative,
                "iteration_seconds": float(iter_seconds),
                "no_change_rounds": int(no_change_rounds),
            }
        )
        write_progress(
            {
                "status": "running",
                "phase": "boruta_outer_fold",
                "outer_fold": int(outer_fold),
                "outer_splits": int(args.outer_splits),
                "boruta_iteration": int(iteration),
                "boruta_max_iter": int(args.boruta_max_iter),
                "num_confirmed": num_confirmed,
                "num_rejected": num_rejected,
                "num_tentative": num_tentative,
                "elapsed_seconds": float(elapsed),
                "eta_seconds": eta_seconds,
                "eta_human": format_seconds(eta_seconds),
            },
            progress_file,
        )

        if int(args.boruta_patience) > 0 and no_change_rounds >= int(args.boruta_patience):
            break

    mean_importance = real_importance_sum / float(max(1, len(iteration_rows)))
    rows: list[dict[str, object]] = []
    for local_idx, feature_index in enumerate(prefilter_indices):
        status_value = int(statuses[local_idx])
        if status_value == 1:
            status_name = "confirmed"
        elif status_value == -1:
            status_name = "rejected"
        else:
            status_name = "tentative"
        row = {
            "outer_fold": int(outer_fold),
            "prefilter_rank": int(local_idx + 1),
            "feature_index": int(feature_index),
            "status": status_name,
            "hits": int(hits[local_idx]),
            "hit_rate": float(hits[local_idx] / max(1, len(iteration_rows))),
            "mean_importance": float(mean_importance[local_idx]),
            "max_importance": float(real_importance_max[local_idx]),
            "mutual_information": float(relevance[feature_index]),
        }
        row.update(decode_feature_index(int(feature_index)))
        rows.append(row)

    confirmed_indices = np.asarray([row["feature_index"] for row in rows if row["status"] == "confirmed"], dtype=np.int32)
    if confirmed_indices.size == 0:
        tentative_rows = [row for row in rows if row["status"] == "tentative"]
        tentative_rows.sort(key=lambda row: (float(row["hit_rate"]), float(row["mean_importance"])), reverse=True)
        fallback_k = int(min(max(1, args.fallback_k), len(tentative_rows)))
        confirmed_indices = np.asarray([row["feature_index"] for row in tentative_rows[:fallback_k]], dtype=np.int32)

    stats = {
        "outer_fold": int(outer_fold),
        "num_confirmed": int(sum(1 for row in rows if row["status"] == "confirmed")),
        "num_tentative": int(sum(1 for row in rows if row["status"] == "tentative")),
        "num_rejected": int(sum(1 for row in rows if row["status"] == "rejected")),
        "selected_feature_count_for_model": int(len(confirmed_indices)),
        "iterations_run": int(len(iteration_rows)),
    }
    return confirmed_indices, rows, iteration_rows, stats


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    output_dir = default_output_dir(features_dir, args.output_dir)
    ensure_dir(output_dir)
    progress_file = output_dir / "progress.json"
    global_start_time = time.time()

    X, table, y, y_labels, class_names = load_inputs(
        features_dir=features_dir,
        valid_labels=list(args.valid_labels),
        ignore_labels=list(args.ignore_labels),
    )
    groups = table["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=int(args.outer_splits),
        shuffle=True,
        random_state=int(args.random_state),
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    boruta_feature_rows: list[dict[str, object]] = []
    boruta_iteration_rows: list[dict[str, object]] = []

    write_progress(
        {
            "status": "running",
            "phase": "initializing",
            "outer_splits": int(args.outer_splits),
            "num_samples": int(len(table)),
            "feature_dim": int(X.shape[1]),
            "elapsed_seconds": 0.0,
        },
        progress_file,
    )

    for outer_fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val_labels = y_labels[val_idx]

        prefilter_indices, relevance = build_mrmr_prefilter(
            X_train=X_train,
            y_train=y_train,
            prefilter_k=int(args.prefilter_k),
            mrmr_prefilter_top=int(args.mrmr_prefilter_top),
            random_state=int(args.random_state) + outer_fold,
        )
        selected_indices, rows, iteration_rows, stats = run_boruta_selection(
            X_train=X_train,
            y_train=y_train,
            prefilter_indices=prefilter_indices,
            relevance=relevance,
            args=args,
            outer_fold=outer_fold,
            progress_file=progress_file,
            global_start_time=global_start_time,
        )
        boruta_feature_rows.extend(rows)
        boruta_iteration_rows.extend(iteration_rows)

        model = build_classifier(
            c_value=float(args.svm_c),
            gamma=str(args.svm_gamma),
            class_weight=None if args.class_weight == "none" else str(args.class_weight),
            random_state=int(args.random_state) + outer_fold,
        )
        model.fit(X_train[:, selected_indices], y_train)
        y_pred = model.predict(X_val[:, selected_indices])
        y_proba = model.predict_proba(X_val[:, selected_indices]).astype(np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = compute_metrics(y_val_labels, np.array(class_names)[y_pred], class_names)
        fold_rows.append(
            {
                "fold": int(outer_fold),
                "num_train_rois": int(len(train_idx)),
                "num_val_rois": int(len(val_idx)),
                "num_train_groups": int(pd.Series(groups[train_idx]).nunique()),
                "num_val_groups": int(pd.Series(groups[val_idx]).nunique()),
                "prefilter_k": int(args.prefilter_k),
                "selected_feature_count": int(len(selected_indices)),
                "boruta_confirmed_count": int(stats["num_confirmed"]),
                "boruta_tentative_count": int(stats["num_tentative"]),
                "boruta_rejected_count": int(stats["num_rejected"]),
                "boruta_iterations_run": int(stats["iterations_run"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "3d_precision": float(metrics["classification_report"]["3d"]["precision"]),
                "3d_recall": float(metrics["classification_report"]["3d"]["recall"]),
                "3d_f1": float(metrics["classification_report"]["3d"]["f1-score"]),
            }
        )
        write_csv(fold_rows, output_dir / "fold_metrics.csv")
        write_csv(boruta_feature_rows, output_dir / "boruta_selected_by_fold.csv")

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

    write_csv(oof_rows, output_dir / "oof_predictions.csv")
    write_csv(boruta_iteration_rows, output_dir / "boruta_iteration_log.csv")
    summary = {
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
        "selector": "boruta_shadow_random_forest",
        "prefilter_selector": "mrmr_mi_relevance_abs_corr_redundancy",
        "prefilter_k": int(args.prefilter_k),
        "mrmr_prefilter_top": int(args.mrmr_prefilter_top),
        "boruta_max_iter": int(args.boruta_max_iter),
        "boruta_alpha": float(args.boruta_alpha),
        "shadow_percentile": float(args.shadow_percentile),
        "boruta_patience": int(args.boruta_patience),
        "rf_n_estimators": int(args.rf_n_estimators),
        "class_names": class_names,
        "num_labeled_rois": int(len(table)),
        "num_groups": int(pd.Series(groups).nunique()),
        "overall": overall_metrics,
        "folds": fold_rows,
    }
    write_json(summary, output_dir / "summary.json")
    write_progress(
        {
            "status": "completed",
            "phase": "done",
            "elapsed_seconds": float(time.time() - global_start_time),
            "eta_seconds": 0.0,
            "eta_human": "0s",
            "summary_json": str(output_dir / "summary.json"),
        },
        progress_file,
    )


if __name__ == "__main__":
    main()
