from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from extract_labeled_roi_overthreshold_multilayer_maxstd_features import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LABELS_FILE,
    DEFAULT_MULTILAYER_CACHE_SUBDIR,
    DEFAULT_ROI_METADATA_CSV,
    select_overlap_threshold_patches,
)
from extract_labeled_roi_toppercent_multilayer_softmax_patch_features import (
    build_multilayer_run_context,
    l2_normalize,
    load_labels_table,
    load_multilayer_cache,
    load_patch_scores,
    load_roi_table,
    prepare_labeled_roi_table,
)
from extract_labeled_roi_toppercent_pca_softmax_patch_features import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build labeled ROI feature vectors from all overlap patches whose anomaly score exceeds "
            "the image threshold. Aggregate per layer with elementwise max, min and mean, then concatenate."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--multilayer-cache-subdir", type=str, default=DEFAULT_MULTILAYER_CACHE_SUBDIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--valid-labels", type=str, nargs="*", default=("2D", "3D"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def default_output_dir(experiment_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (
        experiment_dir
        / "roi_overthreshold_overlap_multilayer_l1to12_maxminmean_features_labeled"
    ).resolve()


def aggregate_maxminmean_per_layer(
    features_layers: np.ndarray,
    selected_patches: list[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    patch_indices = [patch_row * grid_shape[1] + patch_col for patch_row, patch_col in selected_patches]
    selected_layers = features_layers[np.asarray(patch_indices, dtype=np.int32)]
    selected_layers = l2_normalize(selected_layers, axis=2)
    max_pool = selected_layers.max(axis=0)
    min_pool = selected_layers.min(axis=0)
    mean_pool = selected_layers.mean(axis=0)
    per_layer = np.concatenate([max_pool, min_pool, mean_pool], axis=1).astype(np.float32)
    return per_layer.reshape(-1).astype(np.float32)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    output_dir = default_output_dir(experiment_dir, args.output_dir)
    ensure_dir(output_dir)

    roi_table = load_roi_table(roi_metadata_csv)
    labels_table = load_labels_table(labels_file, list(args.valid_labels) if args.valid_labels else None)
    labeled_rois = prepare_labeled_roi_table(roi_table, labels_table, args.limit)
    sample_map = build_multilayer_run_context(
        experiment_dir,
        seed=int(args.seed),
        cache_subdir=str(args.multilayer_cache_subdir),
    )

    feature_rows: list[np.ndarray] = []
    table_rows: list[Dict[str, object]] = []
    layer_indices_ref: list[int] | None = None
    feature_dim_ref: int | None = None
    selected_patch_counts: list[int] = []
    candidate_patch_counts: list[int] = []
    fallback_count = 0
    selection_modes: dict[str, int] = {}

    for feature_index, row in enumerate(labeled_rois.itertuples(index=False), start=0):
        roi_row = pd.Series(row._asdict())
        sample_name = str(roi_row["sample"]).replace("\\", "/")
        if sample_name not in sample_map:
            raise KeyError(f"Sample {sample_name!r} is missing from the multilayer cache context.")
        sample_info = sample_map[sample_name]

        features_layers, grid_shape, layer_indices, cache_meta = load_multilayer_cache(Path(sample_info["feature_cache_path"]))
        score_grid = load_patch_scores(sample_info["run_sample"])
        if layer_indices_ref is None:
            layer_indices_ref = list(layer_indices)
            feature_dim_ref = int(features_layers.shape[1] * features_layers.shape[2] * 3)
        elif list(layer_indices) != layer_indices_ref:
            raise ValueError(f"Inconsistent layer indices: {layer_indices} vs {layer_indices_ref}")

        selected_patches, selection_mode, num_candidates = select_overlap_threshold_patches(
            row=roi_row,
            meta=cache_meta,
            anomaly_grid=score_grid,
            image_threshold=float(sample_info["image_threshold"]),
        )
        if selection_mode == "overlap_threshold_fallback_max":
            fallback_count += 1

        combined_feature = aggregate_maxminmean_per_layer(features_layers, selected_patches, grid_shape)

        selected_patch_counts.append(int(len(selected_patches)))
        candidate_patch_counts.append(int(num_candidates))
        selection_modes[selection_mode] = selection_modes.get(selection_mode, 0) + 1

        patch_rows = [int(item[0]) for item in selected_patches]
        patch_cols = [int(item[1]) for item in selected_patches]
        patch_scores = [float(score_grid[patch_row, patch_col]) for patch_row, patch_col in selected_patches]

        feature_rows.append(combined_feature)
        table_rows.append(
            {
                "feature_index": int(feature_index),
                "sample": sample_name,
                "group_id": sample_name,
                "evaluation_group": sample_info["evaluation_group"],
                "image_path": sample_info["image_path"],
                "feature_cache_path": sample_info["feature_cache_path"],
                "anomaly_map_path": sample_info["anomaly_map_path"],
                "image_score": float(sample_info["image_score"]),
                "image_threshold": float(sample_info["image_threshold"]),
                "bildname": roi_row["bildname"],
                "roi_nummer": roi_row["roi_nummer"],
                "roi_uid": roi_row["roi_uid"],
                "roi_index": int(roi_row["roi_index"]),
                "x_min": int(roi_row["x_min"]),
                "y_min": int(roi_row["y_min"]),
                "x_max": int(roi_row["x_max"]),
                "y_max": int(roi_row["y_max"]),
                "crop_path": roi_row.get("crop_path", ""),
                "label": str(roi_row["label"]),
                "notes": str(roi_row.get("notes", "")),
                "detailed_label": str(roi_row.get("detailed_label", "")),
                "grid_rows": int(grid_shape[0]),
                "grid_cols": int(grid_shape[1]),
                "selection_mode": selection_mode,
                "num_overlap_candidates": int(num_candidates),
                "num_selected_patches": int(len(selected_patches)),
                "selected_patch_rows": ";".join(str(v) for v in patch_rows),
                "selected_patch_cols": ";".join(str(v) for v in patch_cols),
                "selected_patch_scores": ";".join(f"{score:.6f}" for score in patch_scores),
                "feature_mode": f"multilayer_maxminmean_overthreshold_overlap_l{layer_indices[0]}to{layer_indices[-1]}",
                "num_layers": int(features_layers.shape[1]),
                "layer_dim": int(features_layers.shape[2]),
                "feature_dim_concat": int(combined_feature.shape[0]),
                "layer_indices": ";".join(str(int(layer)) for layer in layer_indices),
                "aggregation_mode": "max_plus_min_plus_mean",
                "candidate_rule": "overlap_and_score_gt_image_threshold",
            }
        )

    feature_matrix = np.stack(feature_rows, axis=0).astype(np.float32)
    np.save(output_dir / "roi_features_mean.npy", feature_matrix)
    pd.DataFrame(table_rows).to_csv(output_dir / "roi_feature_table.csv", index=False)

    summary = {
        "experiment_dir": str(experiment_dir),
        "roi_metadata_csv": str(roi_metadata_csv),
        "labels_file": str(labels_file),
        "multilayer_cache_subdir": str(args.multilayer_cache_subdir),
        "output_dir": str(output_dir),
        "num_labeled_rois": int(len(table_rows)),
        "num_groups": int(pd.DataFrame(table_rows)["group_id"].nunique()),
        "class_counts": {
            str(label): int(count)
            for label, count in pd.DataFrame(table_rows)["label"].value_counts().sort_index().items()
        },
        "fallback_count": int(fallback_count),
        "selection_modes": selection_modes,
        "selected_patch_count_min": int(min(selected_patch_counts)) if selected_patch_counts else 0,
        "selected_patch_count_max": int(max(selected_patch_counts)) if selected_patch_counts else 0,
        "selected_patch_count_mean": float(np.mean(selected_patch_counts)) if selected_patch_counts else 0.0,
        "candidate_patch_count_min": int(min(candidate_patch_counts)) if candidate_patch_counts else 0,
        "candidate_patch_count_max": int(max(candidate_patch_counts)) if candidate_patch_counts else 0,
        "candidate_patch_count_mean": float(np.mean(candidate_patch_counts)) if candidate_patch_counts else 0.0,
        "num_layers": int(len(layer_indices_ref or [])),
        "layer_indices": list(layer_indices_ref or []),
        "layer_dim": int((feature_dim_ref or 0) // max(3 * len(layer_indices_ref or [1]), 1)),
        "feature_dim_concat": int(feature_dim_ref or 0),
        "aggregation_mode": "max_plus_min_plus_mean",
        "candidate_rule": "overlap_and_score_gt_image_threshold",
    }
    write_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
