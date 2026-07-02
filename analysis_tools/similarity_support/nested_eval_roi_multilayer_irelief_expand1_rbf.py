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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from extract_labeled_roi_toppercent_multilayer_softmax_patch_features import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LABELS_FILE,
    DEFAULT_ROI_METADATA_CSV,
    aggregate_selected_patch_features,
    build_multilayer_run_context,
    concatenated_patch_features,
    load_labels_table,
    load_multilayer_cache,
    load_patch_scores,
    load_roi_table,
    prepare_labeled_roi_table,
)
from extract_labeled_roi_toppercent_pca_softmax_patch_features import (
    bbox_patch_window,
    select_roi_patches_center_in_box,
    select_roi_patches_overlap,
)
from analysis_tools.similarity_support.fit_roi_irelief_cosine import (
    build_weighted_feature_set,
    estimate_sigma,
    fit_irelief_cosine,
    l2_normalize_rows,
    pairwise_cosine_distance,
)


DEFAULT_RAW_FEATURES_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_top10pct_centerinbox_multilayer_l1to12_softmax_patch_features_labeled"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold-safe nested evaluation for the current best ROI path: "
            "multilayer raw features, expand1 augmentation, I-Relief on train only, "
            "inner top-k selection, and outer RBF-SVM evaluation."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--multilayer-cache-subdir", type=str, default="patch_feature_cache_multilayer_l1to12")
    parser.add_argument("--top-percent", type=float, default=0.10)
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--selection-mode", type=str, default="center_in_box", choices=("center_in_box", "overlap"))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--sigma-quantile", type=float, default=0.5)
    parser.add_argument("--min-sigma", type=float, default=1e-3)
    parser.add_argument("--irelief-max-iter", type=int, default=50)
    parser.add_argument("--irelief-tol", type=float, default=1e-6)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--fixed-k", type=int, default=None)
    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=[8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096],
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="macro_f1",
        choices=("macro_f1", "accuracy", "3d_f1", "3d_recall"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
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


def default_output_dir(experiment_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (experiment_dir / "nested_eval_current_best_expand1_rbf").resolve()


def clip_patch_window(
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[int, int, int, int]:
    row_min = max(0, min(row_min, grid_rows - 1))
    col_min = max(0, min(col_min, grid_cols - 1))
    row_max = max(row_min + 1, min(row_max, grid_rows))
    col_max = max(col_min + 1, min(col_max, grid_cols))
    return row_min, row_max, col_min, col_max


def select_from_patch_window(
    score_grid: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    top_percent: float,
    min_patches: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for patch_row in range(row_min, row_max):
        for patch_col in range(col_min, col_max):
            candidates.append((float(score_grid[patch_row, patch_col]), patch_row, patch_col))
    candidates.sort(key=lambda item: item[0], reverse=True)
    num_candidates = len(candidates)
    top_k = max(int(min_patches), int(np.ceil(float(top_percent) * float(num_candidates))))
    top_k = min(top_k, num_candidates)
    return [(patch_row, patch_col) for _, patch_row, patch_col in candidates[:top_k]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist(),
        "classification_report": report,
    }


def score_value(metrics: dict[str, object], score_key: str) -> float:
    if score_key in ("macro_f1", "accuracy"):
        return float(metrics[score_key])
    if score_key == "3d_f1":
        return float(metrics["classification_report"]["3d"]["f1-score"])
    if score_key == "3d_recall":
        return float(metrics["classification_report"]["3d"]["recall"])
    raise ValueError(f"Unsupported score key: {score_key}")


def build_classifier(c_value: float, gamma: str, class_weight: str | None, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma=gamma,
                    class_weight=class_weight,
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_base_and_expand1_features(
    labeled_rois: pd.DataFrame,
    sample_map: dict[str, dict],
    top_percent: float,
    min_patches: int,
    selection_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_rows: list[np.ndarray] = []
    expand1_rows: list[np.ndarray] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], dict]] = {}

    for row in labeled_rois.itertuples(index=False):
        roi_row = pd.Series(row._asdict())
        sample_name = str(roi_row["sample"]).replace("\\", "/")
        sample_info = sample_map[sample_name]

        if sample_name not in cache:
            features_layers, grid_shape, _, cache_meta = load_multilayer_cache(Path(sample_info["feature_cache_path"]))
            concat_features = concatenated_patch_features(features_layers)
            score_grid = load_patch_scores(sample_info["run_sample"])
            cache[sample_name] = (concat_features, score_grid, grid_shape, cache_meta)

        concat_features, score_grid, grid_shape, cache_meta = cache[sample_name]

        if selection_mode == "overlap":
            selected_base, _, _ = select_roi_patches_overlap(
                roi_row,
                cache_meta,
                score_grid,
                top_percent=float(top_percent),
                min_patches=int(min_patches),
            )
        else:
            selected_base, _, _ = select_roi_patches_center_in_box(
                roi_row,
                cache_meta,
                score_grid,
                top_percent=float(top_percent),
                min_patches=int(min_patches),
            )
        base_feature, _, _ = aggregate_selected_patch_features(
            concat_features,
            score_grid,
            grid_shape,
            selected_base,
        )
        base_rows.append(base_feature)

        row_min, row_max, col_min, col_max = bbox_patch_window(roi_row, cache_meta)
        row_min, row_max, col_min, col_max = clip_patch_window(
            row_min - 1,
            row_max + 1,
            col_min - 1,
            col_max + 1,
            cache_meta["grid_rows"],
            cache_meta["grid_cols"],
        )
        selected_expand = select_from_patch_window(
            score_grid,
            row_min,
            row_max,
            col_min,
            col_max,
            top_percent=float(top_percent),
            min_patches=int(min_patches),
        )
        expand1_feature, _, _ = aggregate_selected_patch_features(
            concat_features,
            score_grid,
            grid_shape,
            selected_expand,
        )
        expand1_rows.append(expand1_feature)

    return np.stack(base_rows, axis=0).astype(np.float32), np.stack(expand1_rows, axis=0).astype(np.float32)


def inner_select_k(
    X_train_raw: np.ndarray,
    X_train_expand1_raw: np.ndarray,
    y_train: np.ndarray,
    y_train_labels: np.ndarray,
    groups_train: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
) -> tuple[int, dict]:
    inner_splitter = StratifiedGroupKFold(
        n_splits=int(args.inner_splits),
        shuffle=True,
        random_state=int(args.random_state),
    )
    k_values = [int(k) for k in args.k_values if 1 <= int(k) <= X_train_raw.shape[1]]
    if not k_values:
        raise ValueError("No valid k values for inner selection.")

    candidate_rows: list[dict[str, object]] = []
    best_score = float("-inf")
    best_accuracy = float("-inf")
    best_k = None

    inner_fold_artifacts: list[dict[str, object]] = []
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splitter.split(X_train_raw, y_train, groups_train), start=1):
        X_inner_train_raw = X_train_raw[inner_train_idx]
        X_inner_val_raw = X_train_raw[inner_val_idx]
        X_inner_train_expand1_raw = X_train_expand1_raw[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        y_inner_val = y_train[inner_val_idx]
        groups_inner_train = groups_train[inner_train_idx]
        y_inner_train_labels = y_train_labels[inner_train_idx]

        features_unit = l2_normalize_rows(X_inner_train_raw)
        sigma = estimate_sigma(
            pairwise_cosine_distance(features_unit),
            quantile=float(args.sigma_quantile),
            min_sigma=float(args.min_sigma),
        )
        weights, trace = fit_irelief_cosine(
            features_unit=features_unit,
            labels=y_inner_train_labels,
            sigma=float(sigma),
            max_iter=int(args.irelief_max_iter),
            tol=float(args.irelief_tol),
        )
        ranked_indices = np.argsort(-weights, kind="stable")

        X_inner_train_weighted = build_weighted_feature_set(features_unit, weights)
        X_inner_val_weighted = build_weighted_feature_set(l2_normalize_rows(X_inner_val_raw), weights)
        X_inner_train_expand1_weighted = build_weighted_feature_set(l2_normalize_rows(X_inner_train_expand1_raw), weights)
        inner_fold_artifacts.append(
            {
                "sigma": float(sigma),
                "ranked_indices": ranked_indices,
                "X_inner_train_weighted": X_inner_train_weighted,
                "X_inner_val_weighted": X_inner_val_weighted,
                "X_inner_train_expand1_weighted": X_inner_train_expand1_weighted,
                "y_inner_train": y_inner_train,
                "y_inner_val": y_inner_val,
            }
        )

    for k in k_values:
        oof_pred = np.full(len(y_train), fill_value=-1, dtype=np.int32)
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splitter.split(X_train_raw, y_train, groups_train), start=1):
            artifact = inner_fold_artifacts[inner_fold - 1]
            selected = artifact["ranked_indices"][:k]

            X_fit = artifact["X_inner_train_weighted"][:, selected]
            X_fit_aug = artifact["X_inner_train_expand1_weighted"][:, selected]
            X_fit = np.vstack([X_fit, X_fit_aug]).astype(np.float32)
            y_fit = np.concatenate([artifact["y_inner_train"], artifact["y_inner_train"]], axis=0)

            model = build_classifier(
                c_value=float(args.svm_c),
                gamma=str(args.svm_gamma),
                class_weight=None if args.class_weight == "none" else str(args.class_weight),
                random_state=int(args.random_state) + inner_fold,
            )
            model.fit(X_fit, y_fit)
            y_pred = model.predict(artifact["X_inner_val_weighted"][:, selected])
            oof_pred[inner_val_idx] = y_pred

        if np.any(oof_pred < 0):
            raise RuntimeError(f"Inner OOF predictions incomplete for k={k}")

        metrics = compute_metrics(y_train_labels, np.array(class_names)[oof_pred], class_names)
        row = {
            "k": int(k),
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "3d_precision": float(metrics["classification_report"]["3d"]["precision"]),
            "3d_recall": float(metrics["classification_report"]["3d"]["recall"]),
            "3d_f1": float(metrics["classification_report"]["3d"]["f1-score"]),
        }
        candidate_rows.append(row)

        current_score = score_value(metrics, str(args.score_key))
        if (
            current_score > best_score
            or (np.isclose(current_score, best_score) and float(metrics["accuracy"]) > best_accuracy)
            or (
                np.isclose(current_score, best_score)
                and np.isclose(float(metrics["accuracy"]), best_accuracy)
                and (best_k is None or int(k) < int(best_k))
            )
        ):
            best_score = float(current_score)
            best_accuracy = float(metrics["accuracy"])
            best_k = int(k)

    if best_k is None:
        raise RuntimeError("Inner k selection failed.")
    return int(best_k), {
        "candidates": candidate_rows,
        "best_k": int(best_k),
        "best_score": float(best_score),
        "best_accuracy": float(best_accuracy),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    output_dir = default_output_dir(experiment_dir, args.output_dir)
    ensure_dir(output_dir)

    roi_table = load_roi_table(roi_metadata_csv)
    labels_table = load_labels_table(labels_file, list(args.valid_labels) if args.valid_labels else None)
    labeled_rois = prepare_labeled_roi_table(roi_table, labels_table, limit=None).copy()
    labeled_rois["label"] = labeled_rois["label"].astype(str).str.strip()
    labeled_rois = labeled_rois[labeled_rois["label"].isin(list(args.valid_labels))].reset_index(drop=True)
    if labeled_rois.empty:
        raise ValueError("No labeled ROIs found after filtering valid labels.")

    sample_map = build_multilayer_run_context(
        experiment_dir,
        seed=int(args.seed),
        cache_subdir=str(args.multilayer_cache_subdir),
    )
    X_base_raw, X_expand1_raw = build_base_and_expand1_features(
        labeled_rois=labeled_rois,
        sample_map=sample_map,
        top_percent=float(args.top_percent),
        min_patches=int(args.min_patches),
        selection_mode=str(args.selection_mode),
    )

    label_encoder = LabelEncoder()
    label_encoder.fit([str(label).lower() for label in args.valid_labels])
    y_labels = labeled_rois["label"].astype(str).str.lower().to_numpy()
    y = label_encoder.transform(y_labels)
    class_names = list(label_encoder.classes_)
    groups = labeled_rois["sample"].astype(str).to_numpy()

    outer_splitter = StratifiedGroupKFold(
        n_splits=int(args.outer_splits),
        shuffle=True,
        random_state=int(args.random_state),
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    fixed_k = None if args.fixed_k is None else int(args.fixed_k)

    for outer_fold, (train_idx, val_idx) in enumerate(outer_splitter.split(X_base_raw, y, groups), start=1):
        X_train_raw = X_base_raw[train_idx]
        X_val_raw = X_base_raw[val_idx]
        X_train_expand1_raw = X_expand1_raw[train_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_train_labels = y_labels[train_idx]
        y_val_labels = y_labels[val_idx]
        groups_train = groups[train_idx]

        if fixed_k is None:
            best_k, inner_info = inner_select_k(
                X_train_raw=X_train_raw,
                X_train_expand1_raw=X_train_expand1_raw,
                y_train=y_train,
                y_train_labels=y_train_labels,
                groups_train=groups_train,
                class_names=class_names,
                args=args,
            )
            for candidate in inner_info["candidates"]:
                inner_rows.append({"outer_fold": int(outer_fold), **candidate})
        else:
            best_k = int(fixed_k)
            inner_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "fixed_k": int(best_k),
                    "selection_mode": "fixed",
                }
            )

        features_unit = l2_normalize_rows(X_train_raw)
        sigma = estimate_sigma(
            pairwise_cosine_distance(features_unit),
            quantile=float(args.sigma_quantile),
            min_sigma=float(args.min_sigma),
        )
        weights, trace = fit_irelief_cosine(
            features_unit=features_unit,
            labels=y_train_labels,
            sigma=float(sigma),
            max_iter=int(args.irelief_max_iter),
            tol=float(args.irelief_tol),
        )
        ranked_indices = np.argsort(-weights, kind="stable")
        selected = ranked_indices[:best_k]

        X_train_weighted = build_weighted_feature_set(features_unit, weights)
        X_val_weighted = build_weighted_feature_set(l2_normalize_rows(X_val_raw), weights)
        X_train_expand1_weighted = build_weighted_feature_set(l2_normalize_rows(X_train_expand1_raw), weights)

        X_fit = np.vstack([
            X_train_weighted[:, selected],
            X_train_expand1_weighted[:, selected],
        ]).astype(np.float32)
        y_fit = np.concatenate([y_train, y_train], axis=0)

        model = build_classifier(
            c_value=float(args.svm_c),
            gamma=str(args.svm_gamma),
            class_weight=None if args.class_weight == "none" else str(args.class_weight),
            random_state=int(args.random_state) + outer_fold,
        )
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_val_weighted[:, selected])
        y_proba = model.predict_proba(X_val_weighted[:, selected]).astype(np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        fold_metrics = compute_metrics(y_val_labels, np.array(class_names)[y_pred], class_names)
        fold_rows.append(
            {
                "fold": int(outer_fold),
                "num_train_rois": int(len(train_idx)),
                "num_val_rois": int(len(val_idx)),
                "num_train_groups": int(pd.Series(groups_train).nunique()),
                "num_val_groups": int(pd.Series(groups[val_idx]).nunique()),
                "best_k_inner": int(best_k),
                "sigma_outer": float(sigma),
                "irelief_iterations_outer": int(len(trace)),
                "accuracy": float(fold_metrics["accuracy"]),
                "macro_precision": float(fold_metrics["macro_precision"]),
                "macro_recall": float(fold_metrics["macro_recall"]),
                "macro_f1": float(fold_metrics["macro_f1"]),
                "3d_precision": float(fold_metrics["classification_report"]["3d"]["precision"]),
                "3d_recall": float(fold_metrics["classification_report"]["3d"]["recall"]),
                "3d_f1": float(fold_metrics["classification_report"]["3d"]["f1-score"]),
            }
        )

    if np.any(oof_pred < 0):
        raise RuntimeError("Some outer-fold OOF predictions were not filled.")

    y_pred_labels = np.array(class_names)[oof_pred]
    overall = compute_metrics(y_labels, y_pred_labels, class_names)

    oof_rows: list[dict[str, object]] = []
    for row_index, row in labeled_rois.reset_index(drop=True).iterrows():
        out = row.to_dict()
        out["true_label"] = y_labels[row_index]
        out["predicted_label"] = y_pred_labels[row_index]
        out["correct"] = int(y_labels[row_index] == y_pred_labels[row_index])
        for class_index, class_name in enumerate(class_names):
            out[f"proba_{class_name}"] = float(oof_proba[row_index, class_index])
        oof_rows.append(out)

    write_csv(fold_rows, output_dir / "fold_metrics.csv")
    write_csv(inner_rows, output_dir / "inner_k_selection.csv")
    write_csv(oof_rows, output_dir / "oof_predictions.csv")
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "roi_metadata_csv": str(roi_metadata_csv),
            "labels_file": str(labels_file),
            "classifier": "svm_rbf",
            "augmentation_policy": "expand1_fixed",
            "selection_mode": str(args.selection_mode),
            "pipeline": (
                "outer_val clean; "
                + (
                    "inner train learns I-Relief and selects top-k; "
                    if fixed_k is None
                    else "inner train learns I-Relief with fixed top-k; "
                )
                + "final outer train refits I-Relief and SVM"
            ),
            "num_labeled_rois": int(len(labeled_rois)),
            "num_groups": int(pd.Series(groups).nunique()),
            "class_names": class_names,
            "class_counts": {
                class_name: int((y_labels == class_name).sum())
                for class_name in class_names
            },
            "outer_splits": int(args.outer_splits),
            "inner_splits": int(args.inner_splits),
            "random_state": int(args.random_state),
            "score_key": str(args.score_key),
            "fixed_k": None if fixed_k is None else int(fixed_k),
            "k_values": [int(k) for k in args.k_values],
            "svm_c": float(args.svm_c),
            "svm_gamma": str(args.svm_gamma),
            "class_weight": None if args.class_weight == "none" else str(args.class_weight),
            "irelief_max_iter": int(args.irelief_max_iter),
            "irelief_tol": float(args.irelief_tol),
            "sigma_quantile": float(args.sigma_quantile),
            "min_sigma": float(args.min_sigma),
            "overall": overall,
            "folds": fold_rows,
            "fold_metrics_file": str(output_dir / "fold_metrics.csv"),
            "inner_k_selection_file": str(output_dir / "inner_k_selection.csv"),
            "oof_predictions_file": str(output_dir / "oof_predictions.csv"),
        },
        output_dir / "summary.json",
    )

    print(f"Saved fold metrics: {output_dir / 'fold_metrics.csv'}")
    print(f"Saved inner k selection: {output_dir / 'inner_k_selection.csv'}")
    print(f"Saved OOF predictions: {output_dir / 'oof_predictions.csv'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    print(
        f"Nested outer OOF macro F1: {overall['macro_f1']:.4f} | "
        f"Accuracy: {overall['accuracy']:.4f} | "
        f"3D recall: {overall['classification_report']['3d']['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
