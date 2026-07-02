from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

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
from analysis_tools.similarity_support.nested_eval_roi_multilayer_irelief_expand1_rbf import (
    build_classifier,
    clip_patch_window,
    compute_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory sweep over a fixed number of top anomalous ROI patches for the current "
            "multilayer softmax-weighted aggregation path with global I-Relief + fixed-k RBF-SVM."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--multilayer-cache-subdir", type=str, default="patch_feature_cache_multilayer_l1to12")
    parser.add_argument("--selection-mode", type=str, default="overlap", choices=("center_in_box", "overlap"))
    parser.add_argument("--patchcount-start", type=int, default=1)
    parser.add_argument("--patchcount-stop", type=int, default=15)
    parser.add_argument("--fixed-k", type=int, default=32)
    parser.add_argument("--score-key", type=str, default="macro_f1", choices=("macro_f1", "accuracy", "3d_f1", "3d_recall"))
    parser.add_argument("--n-splits", type=int, default=5)
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
    return (experiment_dir / "sweep_roi_topkpatchcount_irelief_fixedk32_rbf").resolve()


def score_value(metrics: dict[str, object], score_key: str) -> float:
    if score_key in ("macro_f1", "accuracy"):
        return float(metrics[score_key])
    if score_key == "3d_f1":
        return float(metrics["classification_report"]["3d"]["f1-score"])
    if score_key == "3d_recall":
        return float(metrics["classification_report"]["3d"]["recall"])
    raise ValueError(f"Unsupported score key: {score_key}")


def feature_index_to_meta(feature_index: int, layer_indices: list[int], layer_dim: int) -> dict[str, object]:
    layer_pos = int(feature_index // layer_dim)
    dim = int(feature_index % layer_dim)
    return {
        "layer_position": int(layer_pos),
        "layer": int(layer_indices[layer_pos]),
        "dim": int(dim),
    }


def build_sample_cache(sample_map: dict[str, dict]) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], list[int], dict]]:
    cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], list[int], dict]] = {}
    for sample_name, sample_info in sample_map.items():
        features_layers, grid_shape, layer_indices, cache_meta = load_multilayer_cache(Path(sample_info["feature_cache_path"]))
        concat_features = concatenated_patch_features(features_layers)
        score_grid = load_patch_scores(sample_info["run_sample"])
        cache[sample_name] = (concat_features, score_grid, grid_shape, layer_indices, cache_meta)
    return cache


def select_topk_from_patch_window(
    score_grid: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    top_patch_count: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for patch_row in range(row_min, row_max):
        for patch_col in range(col_min, col_max):
            candidates.append((float(score_grid[patch_row, patch_col]), patch_row, patch_col))
    candidates.sort(key=lambda item: item[0], reverse=True)
    top_k = min(int(top_patch_count), len(candidates))
    return [(patch_row, patch_col) for _, patch_row, patch_col in candidates[:top_k]]


def build_features_for_patch_count(
    labeled_rois: pd.DataFrame,
    sample_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], list[int], dict]],
    top_patch_count: int,
    selection_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    base_rows: list[np.ndarray] = []
    expand1_rows: list[np.ndarray] = []
    layer_indices_ref: list[int] | None = None
    layer_dim_ref: int | None = None

    for row in labeled_rois.itertuples(index=False):
        roi_row = pd.Series(row._asdict())
        sample_name = str(roi_row["sample"]).replace("\\", "/")
        concat_features, score_grid, grid_shape, layer_indices, cache_meta = sample_cache[sample_name]

        if layer_indices_ref is None:
            layer_indices_ref = list(layer_indices)
            if len(layer_indices_ref) <= 0:
                raise ValueError("Empty layer index list in cache.")
            layer_dim_ref = int(concat_features.shape[1] // len(layer_indices_ref))

        if selection_mode == "overlap":
            selected_base, _, _ = select_roi_patches_overlap(
                roi_row,
                cache_meta,
                score_grid,
                top_percent=0.0,
                min_patches=int(top_patch_count),
            )
        else:
            selected_base, _, _ = select_roi_patches_center_in_box(
                roi_row,
                cache_meta,
                score_grid,
                top_percent=0.0,
                min_patches=int(top_patch_count),
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
        selected_expand = select_topk_from_patch_window(
            score_grid,
            row_min,
            row_max,
            col_min,
            col_max,
            top_patch_count=int(top_patch_count),
        )
        expand1_feature, _, _ = aggregate_selected_patch_features(
            concat_features,
            score_grid,
            grid_shape,
            selected_expand,
        )
        expand1_rows.append(expand1_feature)

    return (
        np.stack(base_rows, axis=0).astype(np.float32),
        np.stack(expand1_rows, axis=0).astype(np.float32),
        list(layer_indices_ref or []),
        int(layer_dim_ref or 0),
    )


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    output_dir = default_output_dir(experiment_dir, args.output_dir)
    ensure_dir(output_dir)

    if not (1 <= int(args.patchcount_start) <= int(args.patchcount_stop)):
        raise ValueError("patchcount-start/patchcount-stop must satisfy 1 <= start <= stop")

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
    sample_cache = build_sample_cache(sample_map)

    y_labels = labeled_rois["label"].astype(str).str.lower().to_numpy()
    class_names = [str(v).lower() for v in args.valid_labels]
    label_lookup = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_lookup[label] for label in y_labels], dtype=np.int32)
    groups = labeled_rois["sample"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=int(args.n_splits),
        shuffle=True,
        random_state=int(args.random_state),
    )

    result_rows: list[dict[str, object]] = []
    best_score = float("-inf")
    best_accuracy = float("-inf")
    best_patch_count = None
    best_summary: dict[str, object] | None = None

    for patch_count in range(int(args.patchcount_start), int(args.patchcount_stop) + 1):
        X_base_raw, X_expand1_raw, layer_indices, layer_dim = build_features_for_patch_count(
            labeled_rois=labeled_rois,
            sample_cache=sample_cache,
            top_patch_count=int(patch_count),
            selection_mode=str(args.selection_mode),
        )

        features_unit = l2_normalize_rows(X_base_raw)
        sigma = estimate_sigma(
            pairwise_cosine_distance(features_unit),
            quantile=float(args.sigma_quantile),
            min_sigma=float(args.min_sigma),
        )
        weights, trace = fit_irelief_cosine(
            features_unit=features_unit,
            labels=y_labels,
            sigma=float(sigma),
            max_iter=int(args.irelief_max_iter),
            tol=float(args.irelief_tol),
        )
        ranked_indices = np.argsort(-weights, kind="stable")
        selected = ranked_indices[: int(args.fixed_k)]

        X_base_weighted = build_weighted_feature_set(features_unit, weights)
        X_expand1_weighted = build_weighted_feature_set(l2_normalize_rows(X_expand1_raw), weights)

        oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
        oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X_base_weighted, y, groups), start=1):
            X_fit = np.vstack(
                [
                    X_base_weighted[train_idx][:, selected],
                    X_expand1_weighted[train_idx][:, selected],
                ]
            ).astype(np.float32)
            y_fit = np.concatenate([y[train_idx], y[train_idx]], axis=0)

            model = build_classifier(
                c_value=float(args.svm_c),
                gamma=str(args.svm_gamma),
                class_weight=None if args.class_weight == "none" else str(args.class_weight),
                random_state=int(args.random_state) + fold,
            )
            model.fit(X_fit, y_fit)
            oof_pred[val_idx] = model.predict(X_base_weighted[val_idx][:, selected])
            oof_proba[val_idx] = model.predict_proba(X_base_weighted[val_idx][:, selected]).astype(np.float32)

        if np.any(oof_pred < 0):
            raise RuntimeError(f"OOF predictions incomplete for patch_count={patch_count}")

        metrics = compute_metrics(y_labels, np.array(class_names)[oof_pred], class_names)
        row = {
            "patch_count": int(patch_count),
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "3d_precision": float(metrics["classification_report"]["3d"]["precision"]),
            "3d_recall": float(metrics["classification_report"]["3d"]["recall"]),
            "3d_f1": float(metrics["classification_report"]["3d"]["f1-score"]),
            "sigma": float(sigma),
            "irelief_iterations": int(len(trace)),
        }
        result_rows.append(row)

        current_score = score_value(metrics, str(args.score_key))
        if (
            current_score > best_score
            or (np.isclose(current_score, best_score) and float(metrics["accuracy"]) > best_accuracy)
            or (
                np.isclose(current_score, best_score)
                and np.isclose(float(metrics["accuracy"]), best_accuracy)
                and (best_patch_count is None or int(patch_count) < int(best_patch_count))
            )
        ):
            best_score = float(current_score)
            best_accuracy = float(metrics["accuracy"])
            best_patch_count = int(patch_count)
            best_summary = {
                "metrics": metrics,
                "sigma": float(sigma),
                "irelief_iterations": int(len(trace)),
                "selected_rows": [
                    {
                        "rank": int(rank),
                        "feature_index": int(feature_index),
                        "weight": float(weights[feature_index]),
                        **feature_index_to_meta(int(feature_index), layer_indices, layer_dim),
                    }
                    for rank, feature_index in enumerate(selected, start=1)
                ],
                "oof_predictions": [
                    {
                        **labeled_rois.iloc[row_index].to_dict(),
                        "true_label": y_labels[row_index],
                        "predicted_label": str(class_names[oof_pred[row_index]]),
                        "correct": int(y_labels[row_index] == class_names[oof_pred[row_index]]),
                        **{
                            f"proba_{class_name}": float(oof_proba[row_index, class_index])
                            for class_index, class_name in enumerate(class_names)
                        },
                    }
                    for row_index in range(len(labeled_rois))
                ],
            }

    if best_patch_count is None or best_summary is None:
        raise RuntimeError("Failed to determine best patch count.")

    write_csv(result_rows, output_dir / "results.csv")
    write_csv(best_summary["selected_rows"], output_dir / "selected_topk_features_best_patchcount.csv")
    write_csv(best_summary["oof_predictions"], output_dir / "oof_predictions_best_patchcount.csv")
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "roi_metadata_csv": str(roi_metadata_csv),
            "labels_file": str(labels_file),
            "aggregation": "top_k_patches_softmax_weighted_multilayer",
            "selection_mode": str(args.selection_mode),
            "score_key": str(args.score_key),
            "patchcount_start": int(args.patchcount_start),
            "patchcount_stop": int(args.patchcount_stop),
            "fixed_k": int(args.fixed_k),
            "num_labeled_rois": int(len(labeled_rois)),
            "num_groups": int(pd.Series(groups).nunique()),
            "class_names": class_names,
            "class_counts": {class_name: int((y_labels == class_name).sum()) for class_name in class_names},
            "best_patch_count": int(best_patch_count),
            "best_score": float(best_score),
            "best_accuracy": float(best_accuracy),
            "best_metrics": best_summary["metrics"],
            "best_sigma": float(best_summary["sigma"]),
            "best_irelief_iterations": int(best_summary["irelief_iterations"]),
            "results_file": str(output_dir / "results.csv"),
            "selected_topk_features_best_patchcount_file": str(output_dir / "selected_topk_features_best_patchcount.csv"),
            "oof_predictions_best_patchcount_file": str(output_dir / "oof_predictions_best_patchcount.csv"),
        },
        output_dir / "summary.json",
    )

    print(f"Saved results: {output_dir / 'results.csv'}")
    print(f"Saved selected features: {output_dir / 'selected_topk_features_best_patchcount.csv'}")
    print(f"Saved best OOF predictions: {output_dir / 'oof_predictions_best_patchcount.csv'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    print(f"Best patch_count: {best_patch_count}")


if __name__ == "__main__":
    main()
