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
    load_labels_table,
    load_multilayer_cache,
    load_patch_scores,
    load_roi_table,
    prepare_labeled_roi_table,
)
from extract_labeled_roi_toppercent_pca_softmax_patch_features import (
    select_roi_patches_center_in_box,
    select_roi_patches_overlap,
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
            "Fold-safe evaluation for ROI aggregation with all overlapping ROI patches, "
            "per-layer mean+max pooling, I-Relief on train only, fixed top-k, and RBF-SVM."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--multilayer-cache-subdir", type=str, default="patch_feature_cache_multilayer_l1to12")
    parser.add_argument("--selection-mode", type=str, default="overlap", choices=("overlap", "center_in_box"))
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
    return (experiment_dir / "nested_eval_allpatch_meanmax_irelief_fixedk32_rbf").resolve()


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


def l2_normalize_along_axis(matrix: np.ndarray, axis: int) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (matrix / norms).astype(np.float32)


def aggregate_roi_mean_max_descriptor(
    features_layers: np.ndarray,
    selected_patches: list[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    layer_features_unit = l2_normalize_along_axis(features_layers, axis=2)
    patch_indices = [(patch_row * grid_shape[1]) + patch_col for patch_row, patch_col in selected_patches]
    selected = layer_features_unit[np.asarray(patch_indices, dtype=np.int32)]
    if selected.ndim != 3:
        raise ValueError(f"Expected selected features with shape (num_patches, num_layers, dim), got {selected.shape}")
    mean_pool = selected.mean(axis=0).astype(np.float32)
    max_pool = selected.max(axis=0).astype(np.float32)
    descriptor = np.concatenate([mean_pool, max_pool], axis=1).reshape(-1).astype(np.float32)
    descriptor_norm = float(np.linalg.norm(descriptor))
    if descriptor_norm <= 1e-8:
        raise ValueError("Encountered zero-norm mean+max descriptor.")
    return (descriptor / descriptor_norm).astype(np.float32)


def select_all_roi_patches(
    roi_row: pd.Series,
    cache_meta: dict,
    score_grid: np.ndarray,
    selection_mode: str,
) -> tuple[list[tuple[int, int]], str, int]:
    if selection_mode == "overlap":
        return select_roi_patches_overlap(
            row=roi_row,
            meta=cache_meta,
            anomaly_grid=score_grid,
            top_percent=1.0,
            min_patches=1,
        )
    return select_roi_patches_center_in_box(
        row=roi_row,
        meta=cache_meta,
        anomaly_grid=score_grid,
        top_percent=1.0,
        min_patches=1,
    )


def feature_index_to_meta(feature_index: int, layer_indices: list[int], layer_dim: int) -> dict[str, object]:
    block_dim = layer_dim * 2
    layer_pos = int(feature_index // block_dim)
    rem = int(feature_index % block_dim)
    pooling = "mean" if rem < layer_dim else "max"
    dim = rem if rem < layer_dim else rem - layer_dim
    return {
        "layer_position": int(layer_pos),
        "layer": int(layer_indices[layer_pos]),
        "pooling": pooling,
        "dim": int(dim),
    }


def build_raw_features(
    labeled_rois: pd.DataFrame,
    sample_map: dict[str, dict],
    selection_mode: str,
) -> tuple[np.ndarray, list[int], int, list[dict[str, object]]]:
    feature_rows: list[np.ndarray] = []
    detail_rows: list[dict[str, object]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], list[int], dict]] = {}
    layer_indices_ref: list[int] | None = None
    layer_dim_ref: int | None = None

    for row in labeled_rois.itertuples(index=False):
        roi_row = pd.Series(row._asdict())
        sample_name = str(roi_row["sample"]).replace("\\", "/")
        sample_info = sample_map[sample_name]

        if sample_name not in cache:
            features_layers, grid_shape, layer_indices, cache_meta = load_multilayer_cache(Path(sample_info["feature_cache_path"]))
            score_grid = load_patch_scores(sample_info["run_sample"])
            cache[sample_name] = (features_layers, score_grid, grid_shape, layer_indices, cache_meta)

        features_layers, score_grid, grid_shape, layer_indices, cache_meta = cache[sample_name]
        if layer_indices_ref is None:
            layer_indices_ref = list(layer_indices)
            layer_dim_ref = int(features_layers.shape[2])
        else:
            if list(layer_indices) != layer_indices_ref:
                raise ValueError(f"Inconsistent layer indices across samples: {layer_indices} vs {layer_indices_ref}")
            if int(features_layers.shape[2]) != layer_dim_ref:
                raise ValueError("Inconsistent layer dimensions across samples.")

        selected_patches, selection_label, num_candidates = select_all_roi_patches(
            roi_row=roi_row,
            cache_meta=cache_meta,
            score_grid=score_grid,
            selection_mode=selection_mode,
        )
        descriptor = aggregate_roi_mean_max_descriptor(
            features_layers=features_layers,
            selected_patches=selected_patches,
            grid_shape=grid_shape,
        )
        feature_rows.append(descriptor)
        detail_rows.append(
            {
                "sample": sample_name,
                "roi_uid": str(roi_row["roi_uid"]),
                "roi_nummer": str(roi_row.get("roi_nummer", f"roi{int(roi_row['roi_index'])}")),
                "selection_mode_effective": selection_label,
                "num_candidate_patches": int(num_candidates),
                "num_selected_patches": int(len(selected_patches)),
            }
        )

    if not feature_rows:
        raise ValueError("No ROI descriptors were built.")
    return (
        np.stack(feature_rows, axis=0).astype(np.float32),
        list(layer_indices_ref or []),
        int(layer_dim_ref or 0),
        detail_rows,
    )


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
    X_raw, layer_indices, layer_dim, roi_detail_rows = build_raw_features(
        labeled_rois=labeled_rois,
        sample_map=sample_map,
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
    selected_rows: list[dict[str, object]] = []

    for outer_fold, (train_idx, val_idx) in enumerate(outer_splitter.split(X_raw, y, groups), start=1):
        X_train_raw = X_raw[train_idx]
        X_val_raw = X_raw[val_idx]
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

        X_train_weighted = build_weighted_feature_set(features_unit, weights)
        X_val_weighted = build_weighted_feature_set(l2_normalize_rows(X_val_raw), weights)

        model = build_classifier(
            c_value=float(args.svm_c),
            gamma=str(args.svm_gamma),
            class_weight=None if args.class_weight == "none" else str(args.class_weight),
            random_state=int(args.random_state) + outer_fold,
        )
        model.fit(X_train_weighted[:, selected], y_train)
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

        for rank, feature_index in enumerate(selected, start=1):
            meta = feature_index_to_meta(int(feature_index), layer_indices, layer_dim)
            selected_rows.append(
                {
                    "fold": int(outer_fold),
                    "rank": int(rank),
                    "feature_index": int(feature_index),
                    "weight": float(weights[feature_index]),
                    **meta,
                }
            )

    if np.any(oof_pred < 0):
        raise RuntimeError("Some outer-fold OOF predictions were not filled.")

    y_pred_labels = np.array(class_names)[oof_pred]
    overall = compute_metrics(y_labels, y_pred_labels, class_names)

    oof_rows: list[dict[str, object]] = []
    detail_df = pd.DataFrame(roi_detail_rows)
    for row_index, row in labeled_rois.reset_index(drop=True).iterrows():
        out = row.to_dict()
        if not detail_df.empty:
            for key in ("selection_mode_effective", "num_candidate_patches", "num_selected_patches"):
                out[key] = detail_df.iloc[row_index][key]
        out["true_label"] = y_labels[row_index]
        out["predicted_label"] = y_pred_labels[row_index]
        out["correct"] = int(y_labels[row_index] == y_pred_labels[row_index])
        for class_index, class_name in enumerate(class_names):
            out[f"proba_{class_name}"] = float(oof_proba[row_index, class_index])
        oof_rows.append(out)

    selection_mode_counts = (
        pd.DataFrame(roi_detail_rows)["selection_mode_effective"].value_counts().sort_index().to_dict()
        if roi_detail_rows
        else {}
    )
    pool_counts = pd.DataFrame(selected_rows)["pooling"].value_counts().sort_index().to_dict() if selected_rows else {}
    layer_counts = (
        pd.DataFrame(selected_rows)["layer"].value_counts().sort_index().to_dict()
        if selected_rows
        else {}
    )

    write_csv(fold_rows, output_dir / "fold_metrics.csv")
    write_csv(selected_rows, output_dir / "selected_topk_by_fold.csv")
    write_csv(oof_rows, output_dir / "oof_predictions.csv")
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "roi_metadata_csv": str(roi_metadata_csv),
            "labels_file": str(labels_file),
            "classifier": "svm_rbf",
            "aggregation": "all_roi_patches_mean_plus_max_pooling",
            "selection_mode": str(args.selection_mode),
            "pipeline": (
                "outer_val clean; each outer train learns I-Relief on mean+max ROI descriptors; "
                "fixed top-k count is applied in every fold; selected feature indices may differ by fold"
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
            "feature_shape": {
                "num_rois": int(X_raw.shape[0]),
                "feature_dim": int(X_raw.shape[1]),
                "num_layers": int(len(layer_indices)),
                "layer_dim": int(layer_dim),
            },
            "selection_mode_effective_counts": selection_mode_counts,
            "selected_pooling_counts": {str(key): int(value) for key, value in pool_counts.items()},
            "selected_layer_counts": {str(key): int(value) for key, value in layer_counts.items()},
            "overall": overall,
            "folds": fold_rows,
            "fold_metrics_file": str(output_dir / "fold_metrics.csv"),
            "selected_topk_by_fold_file": str(output_dir / "selected_topk_by_fold.csv"),
            "oof_predictions_file": str(output_dir / "oof_predictions.csv"),
        },
        output_dir / "summary.json",
    )

    print(f"Saved fold metrics: {output_dir / 'fold_metrics.csv'}")
    print(f"Saved selected features: {output_dir / 'selected_topk_by_fold.csv'}")
    print(f"Saved OOF predictions: {output_dir / 'oof_predictions.csv'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
