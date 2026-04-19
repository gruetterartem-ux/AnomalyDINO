from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from component_memory_bank.data_io import load_run_samples
from extract_labeled_roi_toppercent_pca_softmax_patch_features import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LABELS_FILE,
    DEFAULT_ROI_METADATA_CSV,
    ensure_dir,
    load_labels_table,
    load_roi_table,
    prepare_labeled_roi_table,
    select_roi_patches_center_in_box,
    softmax_query_weights,
    write_json,
)


DEFAULT_MULTILAYER_CACHE_SUBDIR = "patch_feature_cache_multilayer_l1to12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build labeled ROI feature vectors by selecting the top anomalous patch-grid patches "
            "inside each ROI bbox and aggregating concatenated DINOv3 multi-layer patch features "
            "with softmax weights over their anomaly scores."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multilayer-cache-subdir", type=str, default=DEFAULT_MULTILAYER_CACHE_SUBDIR)
    parser.add_argument("--top-percent", type=float, default=0.10)
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--valid-labels", type=str, nargs="*", default=("2D", "3D"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def default_output_dir(experiment_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (
        experiment_dir
        / "roi_top10pct_centerinbox_multilayer_l1to12_softmax_patch_features_labeled"
    ).resolve()


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


def load_multilayer_manifest(cache_manifest_path: Path) -> Dict[str, Dict[str, str]]:
    if not cache_manifest_path.exists():
        raise FileNotFoundError(
            f"Multilayer cache manifest not found: {cache_manifest_path}. "
            "Build it first via cache_run_patch_features_multilayer.py."
        )

    manifest: Dict[str, Dict[str, str]] = {}
    with cache_manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = row["sample"].replace("\\", "/")
            manifest[sample] = row
    return manifest


def build_multilayer_run_context(experiment_dir: Path, seed: int, cache_subdir: str) -> Dict[str, dict]:
    run_samples = load_run_samples(experiment_dir, seed=seed)
    manifest = load_multilayer_manifest(experiment_dir / cache_subdir / f"seed={seed}" / "cache_manifest.csv")

    sample_map: Dict[str, dict] = {}
    for sample in run_samples:
        sample_name = sample.sample.replace("\\", "/")
        if sample_name not in manifest:
            raise KeyError(
                f"Sample {sample_name!r} is in measurements but missing from multilayer cache manifest. "
                "Build the multilayer patch cache for this run first."
            )
        cache_row = manifest[sample_name]
        sample_map[sample_name] = {
            "sample": sample_name,
            "evaluation_group": sample.evaluation_group,
            "image_label": int(sample.image_label),
            "image_score": float(sample.image_score),
            "image_threshold": float(sample.image_threshold),
            "image_path": str(sample.image_path),
            "feature_cache_path": str(cache_row["cache_file"]),
            "anomaly_map_path": str(sample.anomaly_map_path),
            "run_sample": sample,
        }
    return sample_map


def load_multilayer_cache(cache_file: Path) -> tuple[np.ndarray, tuple[int, int], List[int], dict]:
    if not cache_file.exists():
        raise FileNotFoundError(f"Multilayer cache file not found: {cache_file}")
    with np.load(cache_file) as data:
        features_layers = np.asarray(data["features_layers"], dtype=np.float32)
        grid_shape = tuple(int(v) for v in np.asarray(data["grid_size"]).tolist())
        layer_indices = [int(v) for v in np.asarray(data["layer_indices"]).tolist()]
        patch_size = int(np.asarray(data["patch_size"]).reshape(-1)[0])
        resized_w, resized_h = [int(v) for v in np.asarray(data["resized_size"]).tolist()]
        original_w, original_h = [int(v) for v in np.asarray(data["original_size"]).tolist()]
    meta = {
        "grid_rows": int(grid_shape[0]),
        "grid_cols": int(grid_shape[1]),
        "resized_w": int(resized_w),
        "resized_h": int(resized_h),
        "original_w": int(original_w),
        "original_h": int(original_h),
        "patch_size": int(patch_size),
        "cropped_w": int(grid_shape[1]) * int(patch_size),
        "cropped_h": int(grid_shape[0]) * int(patch_size),
        "num_layers": int(features_layers.shape[1]),
        "layer_dim": int(features_layers.shape[2]),
    }
    return features_layers, grid_shape, layer_indices, meta


def l2_normalize(matrix: np.ndarray, axis: int) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (matrix / norms).astype(np.float32)


def concatenated_patch_features(features_layers: np.ndarray) -> np.ndarray:
    per_layer_norm = l2_normalize(features_layers, axis=2)
    concatenated = per_layer_norm.reshape(per_layer_norm.shape[0], -1).astype(np.float32)
    return l2_normalize(concatenated, axis=1)


def load_patch_scores(run_sample) -> np.ndarray:
    if not run_sample.anomaly_map_path.exists():
        raise FileNotFoundError(f"Patch anomaly map not found: {run_sample.anomaly_map_path}")
    return np.load(run_sample.anomaly_map_path).astype(np.float32)


def aggregate_selected_patch_features(
    patch_features_norm: np.ndarray,
    score_grid: np.ndarray,
    grid_shape: tuple[int, int],
    selected_patches: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_rows: List[np.ndarray] = []
    anomaly_scores: List[float] = []
    for patch_row, patch_col in selected_patches:
        idx = patch_row * grid_shape[1] + patch_col
        feature_rows.append(patch_features_norm[idx])
        anomaly_scores.append(float(score_grid[patch_row, patch_col]))
    feature_matrix = np.stack(feature_rows, axis=0).astype(np.float32)
    anomaly_array = np.array(anomaly_scores, dtype=np.float32)
    weights = softmax_query_weights(anomaly_array)
    combined = (feature_matrix * weights[:, None]).sum(axis=0)
    combined_norm = np.linalg.norm(combined)
    if combined_norm <= 1e-8:
        combined = feature_matrix.mean(axis=0)
        combined_norm = np.linalg.norm(combined)
    combined = (combined / max(float(combined_norm), 1e-8)).astype(np.float32)
    return combined, anomaly_array, weights


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

    feature_rows: List[np.ndarray] = []
    table_rows: List[Dict[str, object]] = []
    layer_indices_ref: List[int] | None = None
    feature_dim_ref: int | None = None
    selected_patch_counts: List[int] = []
    candidate_patch_counts: List[int] = []
    selection_modes: Dict[str, int] = {}

    for feature_index, row in enumerate(labeled_rois.itertuples(index=False), start=0):
        roi_row = pd.Series(row._asdict())
        sample_name = str(roi_row["sample"]).replace("\\", "/")
        if sample_name not in sample_map:
            raise KeyError(f"Sample {sample_name!r} is missing from the multilayer cache context.")
        sample_info = sample_map[sample_name]

        features_layers, grid_shape, layer_indices, cache_meta = load_multilayer_cache(Path(sample_info["feature_cache_path"]))
        if layer_indices_ref is None:
            layer_indices_ref = list(layer_indices)
            feature_dim_ref = int(features_layers.shape[1] * features_layers.shape[2])
        else:
            if list(layer_indices) != layer_indices_ref:
                raise ValueError(f"Inconsistent layer_indices across cache files: {layer_indices} vs {layer_indices_ref}")
            if int(features_layers.shape[1] * features_layers.shape[2]) != feature_dim_ref:
                raise ValueError("Inconsistent concatenated feature dimensions across cache files.")

        concat_features = concatenated_patch_features(features_layers)
        score_grid = load_patch_scores(sample_info["run_sample"])
        selected_patches, selection_mode, num_candidates = select_roi_patches_center_in_box(
            roi_row,
            cache_meta,
            score_grid,
            top_percent=float(args.top_percent),
            min_patches=int(args.min_patches),
        )
        combined_feature, anomaly_scores, weights = aggregate_selected_patch_features(
            concat_features,
            score_grid,
            grid_shape,
            selected_patches,
        )

        selected_patch_counts.append(int(len(selected_patches)))
        candidate_patch_counts.append(int(num_candidates))
        selection_modes[selection_mode] = selection_modes.get(selection_mode, 0) + 1

        patch_rows = [int(item[0]) for item in selected_patches]
        patch_cols = [int(item[1]) for item in selected_patches]
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
                "top_percent": float(args.top_percent),
                "min_patches": int(args.min_patches),
                "num_candidate_patches_center_or_fallback": int(num_candidates),
                "num_selected_patches": int(len(selected_patches)),
                "selected_patch_rows": ";".join(str(v) for v in patch_rows),
                "selected_patch_cols": ";".join(str(v) for v in patch_cols),
                "selected_patch_scores": ";".join(f"{float(score):.6f}" for score in anomaly_scores.tolist()),
                "selected_patch_softmax_weights": ";".join(f"{float(weight):.6f}" for weight in weights.tolist()),
                "feature_mode": f"multilayer_concat_l{layer_indices[0]}to{layer_indices[-1]}",
                "query_weight_mode": "softmax",
                "num_layers": int(features_layers.shape[1]),
                "layer_dim": int(features_layers.shape[2]),
                "feature_dim_concat": int(features_layers.shape[1] * features_layers.shape[2]),
                "layer_indices": ";".join(str(int(layer)) for layer in layer_indices),
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
        "top_percent": float(args.top_percent),
        "min_patches": int(args.min_patches),
        "selected_patch_count_min": int(min(selected_patch_counts)) if selected_patch_counts else 0,
        "selected_patch_count_max": int(max(selected_patch_counts)) if selected_patch_counts else 0,
        "selected_patch_count_mean": float(np.mean(selected_patch_counts)) if selected_patch_counts else 0.0,
        "candidate_patch_count_min": int(min(candidate_patch_counts)) if candidate_patch_counts else 0,
        "candidate_patch_count_max": int(max(candidate_patch_counts)) if candidate_patch_counts else 0,
        "candidate_patch_count_mean": float(np.mean(candidate_patch_counts)) if candidate_patch_counts else 0.0,
        "selection_modes": selection_modes,
        "num_layers": int(len(layer_indices_ref or [])),
        "layer_indices": list(layer_indices_ref or []),
        "layer_dim": int((feature_dim_ref or 0) // max(len(layer_indices_ref or [1]), 1)),
        "feature_dim_concat": int(feature_dim_ref or 0),
    }
    write_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
