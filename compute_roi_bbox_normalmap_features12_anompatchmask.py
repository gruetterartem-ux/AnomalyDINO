from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from component_memory_bank.data_io import load_run_args as load_experiment_args
from component_memory_bank.data_io import load_run_samples, load_patch_scores
from compute_roi_bbox_normalmap_features12 import (
    compute_delta_map,
    dominant_normal,
    largest_component_size,
    safe_max,
    safe_mean,
    safe_percentile,
    safe_std,
    scalar_coherence,
)
from compute_roi_normalmap_bbox_metrics import compute_maps, ensure_dir, valid_core_mask
from show_heatmap import hysteresis_component, infer_patch_multiple, resized_image_for_dense_backbone


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_BASE_METRICS_CSV = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_normalmap_bbox_metrics_local_ring_ignore_black"
    / "roi_normalmap_bbox_metrics.csv"
)
DEFAULT_ROI_METADATA_CSV = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1"
    / "seed=0"
    / "roi_metadata.csv"
)


FEATURE_COLUMNS = [
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
        description=(
            "Compute the 12 normal-map bbox features only on the anomalous patch mask inside each ROI "
            "and ignore black pixels."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--base-metrics-csv", type=Path, default=DEFAULT_BASE_METRICS_CSV)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--black-threshold", type=int, default=0)
    parser.add_argument("--gradient-threshold-quantile", type=float, default=0.90)
    parser.add_argument("--delta-threshold-quantile", type=float, default=0.90)
    return parser.parse_args()


def default_output_dir(experiment_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (experiment_dir / "roi_normalmap_bbox_metrics_local_ring_ignore_black" / "bbox_normalmap_features12_anompatchmask").resolve()


def parse_peak_coords(row: pd.Series) -> np.ndarray:
    peak_rows = str(row.get("peak_rows", "")).strip()
    peak_cols = str(row.get("peak_cols", "")).strip()
    if not peak_rows or not peak_cols:
        return np.zeros((0, 2), dtype=np.int32)

    rows = [int(value) for value in peak_rows.split(";") if str(value).strip() != ""]
    cols = [int(value) for value in peak_cols.split(";") if str(value).strip() != ""]
    if len(rows) != len(cols):
        raise ValueError(f"peak_rows / peak_cols length mismatch for {row.get('sample')} roi={row.get('roi_index')}")
    if not rows:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)], axis=1)


def build_roi_patch_mask(score_map: np.ndarray, roi_row: pd.Series) -> np.ndarray:
    grid_h, grid_w = score_map.shape
    available_mask = np.zeros((grid_h, grid_w), dtype=bool)

    row_min = int(roi_row["region_row_min"])
    row_max = int(roi_row["region_row_max"])
    col_min = int(roi_row["region_col_min"])
    col_max = int(roi_row["region_col_max"])
    available_mask[row_min : row_max + 1, col_min : col_max + 1] = True
    available_mask &= score_map > 0.0

    peak_coords = parse_peak_coords(roi_row)
    seed_mask = np.zeros_like(available_mask, dtype=bool)
    for peak_row, peak_col in peak_coords:
        if 0 <= peak_row < grid_h and 0 <= peak_col < grid_w and available_mask[peak_row, peak_col]:
            seed_mask[peak_row, peak_col] = True

    if not seed_mask.any():
        primary_row = int(roi_row["primary_peak_row"])
        primary_col = int(roi_row["primary_peak_col"])
        if 0 <= primary_row < grid_h and 0 <= primary_col < grid_w and available_mask[primary_row, primary_col]:
            seed_mask[primary_row, primary_col] = True

    if not seed_mask.any():
        return available_mask

    high_threshold = float(roi_row["high_threshold"])
    low_threshold = float(roi_row["low_threshold"])
    region_mask = hysteresis_component(
        score_map=score_map,
        seed_mask=seed_mask,
        available_mask=available_mask,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
    return region_mask & available_mask


def patch_edges(
    grid_shape: Tuple[int, int],
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    patch_multiple: int,
) -> Tuple[np.ndarray, np.ndarray]:
    grid_h, grid_w = grid_shape
    orig_w, orig_h = original_size
    resized_w, resized_h = resized_size

    cropped_w = resized_w - (resized_w % patch_multiple)
    cropped_h = resized_h - (resized_h % patch_multiple)

    processed_x_edges = np.rint(np.linspace(0, cropped_w, grid_w + 1)).astype(np.int32)
    processed_y_edges = np.rint(np.linspace(0, cropped_h, grid_h + 1)).astype(np.int32)
    original_x_edges = np.rint(processed_x_edges * orig_w / resized_w).astype(np.int32)
    original_y_edges = np.rint(processed_y_edges * orig_h / resized_h).astype(np.int32)
    return original_x_edges, original_y_edges


def patch_mask_to_pixel_mask(
    roi_patch_mask: np.ndarray,
    image_shape: Tuple[int, int],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    image_h, image_w = image_shape
    pixel_mask = np.zeros((image_h, image_w), dtype=bool)
    active_rows, active_cols = np.where(roi_patch_mask)
    for patch_row, patch_col in zip(active_rows.tolist(), active_cols.tolist()):
        x0 = int(x_edges[patch_col])
        x1 = int(x_edges[patch_col + 1])
        y0 = int(y_edges[patch_row])
        y1 = int(y_edges[patch_row + 1])
        x0 = max(0, min(x0, image_w))
        x1 = max(x0, min(x1, image_w))
        y0 = max(0, min(y0, image_h))
        y1 = max(y0, min(y1, image_h))
        if x1 > x0 and y1 > y0:
            pixel_mask[y0:y1, x0:x1] = True
    return pixel_mask


def build_sample_index(experiment_dir: Path, seed: int) -> Dict[str, object]:
    samples = load_run_samples(experiment_dir, seed=seed)
    return {sample.sample.replace("\\", "/"): sample for sample in samples}


def load_tables(base_metrics_csv: Path, roi_metadata_csv: Path) -> pd.DataFrame:
    base_table = pd.read_csv(base_metrics_csv).copy()
    roi_table = pd.read_csv(roi_metadata_csv).copy()

    base_table["sample"] = base_table["sample"].astype(str).str.replace("\\", "/", regex=False)
    roi_table["sample"] = roi_table["sample"].astype(str).str.replace("\\", "/", regex=False)
    base_table["label"] = base_table["label"].fillna("").astype(str).str.strip()

    join_columns = [
        "sample",
        "roi_index",
        "high_threshold",
        "low_threshold",
        "primary_peak_row",
        "primary_peak_col",
        "peak_rows",
        "peak_cols",
    ]
    extra_columns = [
        "region_row_min",
        "region_row_max",
        "region_col_min",
        "region_col_max",
        "region_patch_count",
        "region_max_score",
        "primary_peak_score",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
    merge_columns = ["sample", "roi_index"] + [column for column in join_columns + extra_columns if column not in {"sample", "roi_index"}]
    roi_subset = roi_table[merge_columns].copy()

    merged = base_table.drop(columns=[column for column in extra_columns if column in base_table.columns], errors="ignore").merge(
        roi_subset,
        on=["sample", "roi_index"],
        how="left",
        validate="one_to_one",
    )
    if merged[join_columns].isna().any().any():
        missing = merged.loc[merged[join_columns].isna().any(axis=1), ["sample", "roi_index"]]
        raise ValueError(f"Could not join ROI metadata for some rows: {missing.head(10).to_dict(orient='records')}")
    return merged


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    base_metrics_csv = args.base_metrics_csv.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    output_dir = default_output_dir(experiment_dir, args.output_dir)
    ensure_dir(output_dir)

    run_args = load_experiment_args(experiment_dir)
    resolution = int(run_args["resolution"])
    patch_multiple = infer_patch_multiple(str(run_args["model_name"]))

    table = load_tables(base_metrics_csv, roi_metadata_csv)
    sample_index = build_sample_index(experiment_dir, args.seed)

    feature_records: List[dict] = []
    all_gradient_values: List[np.ndarray] = []
    all_delta_values: List[np.ndarray] = []

    for image_path, image_rows in table.groupby("image_path", sort=True):
        image_path = Path(image_path)
        image_rgb = np.array(Image.open(str(image_path)).convert("RGB"))
        normals, _, gradient_map, _, valid_mask, _ = compute_maps(
            image_rgb,
            black_threshold=args.black_threshold,
        )
        image_h, image_w = valid_mask.shape
        delta_map, _ = compute_delta_map(normals, valid_mask)

        sample_name = str(image_rows.iloc[0]["sample"])
        if sample_name not in sample_index:
            raise KeyError(f"Run sample not found for ROI rows: {sample_name}")
        run_sample = sample_index[sample_name]
        score_map = load_patch_scores(run_sample)

        with Image.open(str(image_path)) as image_handle:
            original_size = image_handle.size
            resized_size = resized_image_for_dense_backbone(image_handle.convert("RGB"), resolution).size
        x_edges, y_edges = patch_edges(score_map.shape, original_size, resized_size, patch_multiple)

        for _, row in image_rows.iterrows():
            roi_patch_mask = build_roi_patch_mask(score_map, row)
            roi_pixel_mask = patch_mask_to_pixel_mask(roi_patch_mask, (image_h, image_w), x_edges, y_edges)
            masked_valid = valid_mask & roi_pixel_mask
            grad_valid = valid_core_mask(masked_valid)
            delta_valid = valid_core_mask(grad_valid)

            normals_valid = normals[masked_valid]
            dominant = dominant_normal(normals_valid)
            if dominant is None or normals_valid.size == 0:
                ang_values = np.zeros((0,), dtype=np.float32)
                ang_map = np.zeros(masked_valid.shape, dtype=np.float32)
            else:
                dots = np.clip(np.einsum("ij,j->i", normals_valid, dominant), -1.0, 1.0)
                ang_values = np.degrees(np.arccos(dots)).astype(np.float32)
                ang_map = np.zeros(masked_valid.shape, dtype=np.float32)
                ang_map[masked_valid] = ang_values

            grad_values = gradient_map[grad_valid].astype(np.float32)
            delta_values = delta_map[delta_valid].astype(np.float32)
            row_label = str(row.get("label", "")).strip().lower()
            is_labeled_row = row_label in {"2d", "3d"}
            if is_labeled_row and grad_values.size > 0:
                all_gradient_values.append(grad_values)
            if is_labeled_row and delta_values.size > 0:
                all_delta_values.append(delta_values)

            feature_records.append(
                {
                    "row_data": row.to_dict(),
                    "roi_patch_mask": roi_patch_mask,
                    "roi_pixel_mask": roi_pixel_mask,
                    "masked_valid": masked_valid,
                    "grad_valid": grad_valid,
                    "delta_valid": delta_valid,
                    "grad_values": grad_values,
                    "delta_values": delta_values,
                    "ang_values": ang_values,
                    "ang_map": ang_map.astype(np.float32),
                    "normals_valid": normals_valid.astype(np.float32),
                    "gradient_map": gradient_map.astype(np.float32),
                    "delta_map": delta_map.astype(np.float32),
                }
            )

    if not all_gradient_values or not all_delta_values:
        raise ValueError("Could not derive gradient/delta thresholds from labeled ROI patch masks.")

    t1 = float(np.quantile(np.concatenate(all_gradient_values), args.gradient_threshold_quantile))
    t2 = float(np.quantile(np.concatenate(all_delta_values), args.delta_threshold_quantile))

    rows: List[Dict[str, object]] = []
    for record in feature_records:
        row_dict = dict(record["row_data"])
        masked_valid = record["masked_valid"]
        grad_valid = record["grad_valid"]
        delta_valid = record["delta_valid"]
        gradient_map = record["gradient_map"]
        delta_map = record["delta_map"]
        ang_values = record["ang_values"]
        ang_map = record["ang_map"]
        normals_valid = record["normals_valid"]
        roi_patch_mask = record["roi_patch_mask"]
        roi_pixel_mask = record["roi_pixel_mask"]

        grad_binary = (gradient_map > t1) & grad_valid
        delta_binary = (delta_map > t2) & delta_valid

        row_dict["roi_active_patch_count"] = int(roi_patch_mask.sum())
        row_dict["roi_active_pixel_count"] = int(roi_pixel_mask.sum())
        row_dict["roi_valid_pixel_count"] = int(masked_valid.sum())
        row_dict["roi_ignored_black_pixel_count"] = int(roi_pixel_mask.sum() - masked_valid.sum())
        row_dict["roi_valid_derivative_pixel_count"] = int(grad_valid.sum())
        row_dict["roi_valid_second_derivative_pixel_count"] = int(delta_valid.sum())
        row_dict["grad_p95"] = safe_percentile(record["grad_values"], 95)
        row_dict["grad_max"] = safe_max(record["grad_values"])
        row_dict["dominant_angle_mean_deg"] = safe_mean(ang_values)
        row_dict["dominant_angle_p95_deg"] = safe_percentile(ang_values, 95)
        row_dict["delta_mean"] = safe_mean(record["delta_values"])
        row_dict["delta_p95"] = safe_percentile(record["delta_values"], 95)
        row_dict["grad_frac_gt_t1"] = float(np.mean(grad_binary[grad_valid])) if int(grad_valid.sum()) > 0 else 0.0
        row_dict["grad_largest_component_size_t1"] = largest_component_size(grad_binary)
        row_dict["normal_total_variance"] = (
            float(np.var(normals_valid[:, 0]) + np.var(normals_valid[:, 1]) + np.var(normals_valid[:, 2]))
            if normals_valid.size > 0
            else 0.0
        )
        row_dict["nz_std"] = safe_std(normals_valid[:, 2] if normals_valid.size > 0 else np.zeros((0,), dtype=np.float32))
        row_dict["directional_coherence"] = scalar_coherence(ang_map, masked_valid)
        row_dict["delta_frac_gt_t2"] = float(np.mean(delta_binary[delta_valid])) if int(delta_valid.sum()) > 0 else 0.0
        row_dict["t1_grad_threshold"] = t1
        row_dict["t2_delta_threshold"] = t2
        rows.append(row_dict)

    feature_table = pd.DataFrame(rows)
    output_csv = output_dir / "roi_bbox_normalmap_features12_anompatchmask.csv"
    feature_table.to_csv(output_csv, index=False)

    labeled_mask = feature_table["label"].fillna("").astype(str).str.strip().str.lower().isin({"2d", "3d"})
    summary = {
        "experiment_dir": str(experiment_dir),
        "base_metrics_csv": str(base_metrics_csv),
        "roi_metadata_csv": str(roi_metadata_csv),
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "mask_mode": "roi_anomalous_patch_mask",
        "black_threshold": int(args.black_threshold),
        "gradient_threshold_quantile": float(args.gradient_threshold_quantile),
        "delta_threshold_quantile": float(args.delta_threshold_quantile),
        "t1_grad_threshold": t1,
        "t2_delta_threshold": t2,
        "num_rows": int(feature_table.shape[0]),
        "num_labeled_rows": int(labeled_mask.sum()),
        "feature_columns": FEATURE_COLUMNS,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved masked bbox normalmap features: {output_csv}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
