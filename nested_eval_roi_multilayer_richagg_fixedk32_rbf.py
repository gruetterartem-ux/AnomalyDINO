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
    softmax_query_weights,
)
from fit_roi_irelief_cosine import (
    build_weighted_feature_set,
    estimate_sigma,
    fit_irelief_cosine,
    l2_normalize_rows,
    pairwise_cosine_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold-safe nested evaluation with richer ROI aggregation: top1 patch, "
            "softmax-weighted mean, mean top3, std topk, optional anomaly-score stats, "
            "then I-Relief + fixed top-k=32 + RBF-SVM."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--multilayer-cache-subdir", type=str, default="patch_feature_cache_multilayer_l1to12")
    parser.add_argument("--top-percent", type=float, default=0.10)
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--fixed-k", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--sigma-quantile", type=float, default=0.5)
    parser.add_argument("--min-sigma", type=float, default=1e-3)
    parser.add_argument("--irelief-max-iter", type=int, default=50)
    parser.add_argument("--irelief-tol", type=float, default=1e-6)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
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
    return (experiment_dir / "nested_eval_richagg_expand1_rbf_fixedk32").resolve()


def l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


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


def normalized_entropy(weights: np.ndarray) -> float:
    if weights.size <= 1:
        return 0.0
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    return float(entropy / max(np.log(float(weights.size)), 1e-8))


def build_rich_roi_descriptor(
    feature_matrix: np.ndarray,
    anomaly_scores: np.ndarray,
    include_score_stats: bool,
) -> np.ndarray:
    weights = softmax_query_weights(anomaly_scores)

    top1_patch = feature_matrix[0].astype(np.float32)

    weighted_mean = (feature_matrix * weights[:, None]).sum(axis=0).astype(np.float32)
    weighted_mean = l2_normalize_vector(weighted_mean)

    top3_count = min(3, int(feature_matrix.shape[0]))
    mean_top3 = feature_matrix[:top3_count].mean(axis=0).astype(np.float32)
    mean_top3 = l2_normalize_vector(mean_top3)

    std_topk = feature_matrix.std(axis=0).astype(np.float32)
    std_topk = l2_normalize_vector(std_topk)

    blocks: list[np.ndarray] = [top1_patch, weighted_mean, mean_top3, std_topk]

    if include_score_stats:
        top3_scores = anomaly_scores[:top3_count]
        score_stats = np.array(
            [
                float(anomaly_scores.max()),
                float(top3_scores.mean()),
                float(anomaly_scores.std()),
                float(anomaly_scores.max() - anomaly_scores.min()),
                float(normalized_entropy(weights)),
            ],
            dtype=np.float32,
        )
        score_stats = l2_normalize_vector(score_stats)
        blocks.append(score_stats)

    descriptor = np.concatenate(blocks, axis=0).astype(np.float32)
    return l2_normalize_vector(descriptor)


def aggregate_selected_patch_features_rich(
    patch_features_norm: np.ndarray,
    score_grid: np.ndarray,
    grid_shape: tuple[int, int],
    selected_patches: list[tuple[int, int]],
    include_score_stats: bool,
) -> np.ndarray:
    feature_rows: list[np.ndarray] = []
    anomaly_scores: list[float] = []
    for patch_row, patch_col in selected_patches:
        idx = patch_row * grid_shape[1] + patch_col
        feature_rows.append(patch_features_norm[idx])
        anomaly_scores.append(float(score_grid[patch_row, patch_col]))
    feature_matrix = np.stack(feature_rows, axis=0).astype(np.float32)
    anomaly_array = np.array(anomaly_scores, dtype=np.float32)
    return build_rich_roi_descriptor(feature_matrix, anomaly_array, include_score_stats=include_score_stats)


def build_base_and_expand1_features(
    labeled_rois: pd.DataFrame,
    sample_map: dict[str, dict],
    top_percent: float,
    min_patches: int,
    include_score_stats: bool,
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

        selected_base, _, _ = select_roi_patches_center_in_box(
            roi_row,
            cache_meta,
            score_grid,
            top_percent=float(top_percent),
            min_patches=int(min_patches),
        )
        base_feature = aggregate_selected_patch_features_rich(
            concat_features,
            score_grid,
            grid_shape,
            selected_base,
            include_score_stats=include_score_stats,
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
        expand1_feature = aggregate_selected_patch_features_rich(
            concat_features,
            score_grid,
            grid_shape,
            selected_expand,
            include_score_stats=include_score_stats,
        )
        expand1_rows.append(expand1_feature)

    return np.stack(base_rows, axis=0).astype(np.float32), np.stack(expand1_rows, axis=0).astype(np.float32)


def run_fold_safe_eval(
    *,
    run_name: str,
    labeled_rois: pd.DataFrame,
    sample_map: dict[str, dict],
    y: np.ndarray,
    y_labels: np.ndarray,
    class_names: list[str],
    groups: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    include_score_stats: bool,
) -> dict[str, object]:
    ensure_dir(output_dir)

    X_base_raw, X_expand1_raw = build_base_and_expand1_features(
        labeled_rois=labeled_rois,
        sample_map=sample_map,
        top_percent=float(args.top_percent),
        min_patches=int(args.min_patches),
        include_score_stats=include_score_stats,
    )

    outer_splitter = StratifiedGroupKFold(
        n_splits=int(args.outer_splits),
        shuffle=True,
        random_state=int(args.random_state),
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for outer_fold, (train_idx, val_idx) in enumerate(outer_splitter.split(X_base_raw, y, groups), start=1):
        X_train_raw = X_base_raw[train_idx]
        X_val_raw = X_base_raw[val_idx]
        X_train_expand1_raw = X_expand1_raw[train_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_train_labels = y_labels[train_idx]
        y_val_labels = y_labels[val_idx]
        groups_train = groups[train_idx]

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
        selected = ranked_indices[: int(args.fixed_k)]

        for rank, feature_index in enumerate(selected, start=1):
            selected_rows.append(
                {
                    "fold": int(outer_fold),
                    "rank": int(rank),
                    "feature_index": int(feature_index),
                    "weight": float(weights[feature_index]),
                }
            )

        X_train_weighted = build_weighted_feature_set(features_unit, weights)
        X_val_weighted = build_weighted_feature_set(l2_normalize_rows(X_val_raw), weights)
        X_train_expand1_weighted = build_weighted_feature_set(l2_normalize_rows(X_train_expand1_raw), weights)

        X_fit = np.vstack(
            [
                X_train_weighted[:, selected],
                X_train_expand1_weighted[:, selected],
            ]
        ).astype(np.float32)
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
                "fixed_k": int(args.fixed_k),
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
        raise RuntimeError(f"Some OOF predictions were not filled for run {run_name}.")

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
    write_csv(selected_rows, output_dir / "selected_topk_by_fold.csv")
    write_csv(oof_rows, output_dir / "oof_predictions.csv")
    write_json(
        {
            "run_name": run_name,
            "experiment_dir": str(args.experiment_dir.resolve()),
            "roi_metadata_csv": str(args.roi_metadata_csv.resolve()),
            "labels_file": str(args.labels_file.resolve()),
            "classifier": "svm_rbf",
            "aggregation_blocks": (
                ["top1_patch", "weighted_mean_topk", "mean_top3", "std_topk", "anomaly_score_stats"]
                if include_score_stats
                else ["top1_patch", "weighted_mean_topk", "mean_top3", "std_topk"]
            ),
            "include_score_stats": bool(include_score_stats),
            "weighting_inside_roi": "softmax_over_patch_anomaly_scores",
            "augmentation_policy": "expand1_fixed",
            "pipeline": (
                "outer_val clean; outer_train learns I-Relief on richer ROI descriptor; "
                "fixed top-k features are kept; final outer train refits I-Relief and RBF-SVM"
            ),
            "num_labeled_rois": int(len(labeled_rois)),
            "num_groups": int(pd.Series(groups).nunique()),
            "class_names": class_names,
            "class_counts": {
                class_name: int((y_labels == class_name).sum())
                for class_name in class_names
            },
            "outer_splits": int(args.outer_splits),
            "random_state": int(args.random_state),
            "fixed_k": int(args.fixed_k),
            "svm_c": float(args.svm_c),
            "svm_gamma": str(args.svm_gamma),
            "class_weight": None if args.class_weight == "none" else str(args.class_weight),
            "irelief_max_iter": int(args.irelief_max_iter),
            "irelief_tol": float(args.irelief_tol),
            "sigma_quantile": float(args.sigma_quantile),
            "min_sigma": float(args.min_sigma),
            "feature_dim_raw": int(X_base_raw.shape[1]),
            "overall": overall,
            "folds": fold_rows,
            "fold_metrics_file": str(output_dir / "fold_metrics.csv"),
            "selected_topk_file": str(output_dir / "selected_topk_by_fold.csv"),
            "oof_predictions_file": str(output_dir / "oof_predictions.csv"),
        },
        output_dir / "summary.json",
    )

    print(
        f"[{run_name}] macro F1: {overall['macro_f1']:.4f} | "
        f"Accuracy: {overall['accuracy']:.4f} | "
        f"3D recall: {overall['classification_report']['3d']['recall']:.4f}"
    )
    return {
        "run_name": run_name,
        "include_score_stats": bool(include_score_stats),
        "feature_dim_raw": int(X_base_raw.shape[1]),
        "overall": overall,
        "summary_file": str(output_dir / "summary.json"),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    output_root = default_output_dir(experiment_dir, args.output_dir)
    ensure_dir(output_root)

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

    label_encoder = LabelEncoder()
    label_encoder.fit([str(label).lower() for label in args.valid_labels])
    y_labels = labeled_rois["label"].astype(str).str.lower().to_numpy()
    y = label_encoder.transform(y_labels)
    class_names = list(label_encoder.classes_)
    groups = labeled_rois["sample"].astype(str).to_numpy()

    result_rows: list[dict[str, object]] = []
    for run_name, include_score_stats in (
        ("without_anomaly_score_stats", False),
        ("with_anomaly_score_stats", True),
    ):
        output_dir = output_root / run_name
        result = run_fold_safe_eval(
            run_name=run_name,
            labeled_rois=labeled_rois,
            sample_map=sample_map,
            y=y,
            y_labels=y_labels,
            class_names=class_names,
            groups=groups,
            args=args,
            output_dir=output_dir,
            include_score_stats=include_score_stats,
        )
        result_rows.append(
            {
                "run_name": run_name,
                "include_score_stats": int(include_score_stats),
                "feature_dim_raw": int(result["feature_dim_raw"]),
                "accuracy": float(result["overall"]["accuracy"]),
                "macro_precision": float(result["overall"]["macro_precision"]),
                "macro_recall": float(result["overall"]["macro_recall"]),
                "macro_f1": float(result["overall"]["macro_f1"]),
                "3d_precision": float(result["overall"]["classification_report"]["3d"]["precision"]),
                "3d_recall": float(result["overall"]["classification_report"]["3d"]["recall"]),
                "3d_f1": float(result["overall"]["classification_report"]["3d"]["f1-score"]),
                "summary_file": str(result["summary_file"]),
            }
        )

    write_csv(result_rows, output_root / "comparison.csv")
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "fixed_k": int(args.fixed_k),
            "aggregation_description": "top1_patch + weighted_mean_topk + mean_top3 + std_topk (+ optional anomaly_score_stats)",
            "comparison_file": str(output_root / "comparison.csv"),
            "runs": result_rows,
        },
        output_root / "comparison_summary.json",
    )
    print(f"Saved comparison: {output_root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
