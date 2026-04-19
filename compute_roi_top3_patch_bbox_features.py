from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from component_memory_bank.data_io import load_run_samples
from compute_roi_normalmap_bbox_metrics import compute_maps, ensure_dir, write_json


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_BASE_METRICS_CSV = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_normalmap_bbox_metrics_local_ring_ignore_black"
    / "roi_normalmap_bbox_metrics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute bbox features from the top-k anomalous patch-grid patches inside each ROI bbox."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--base-metrics-csv", type=Path, default=DEFAULT_BASE_METRICS_CSV)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--black-threshold", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir(base_metrics_csv: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (base_metrics_csv.parent / "top3_patch_features").resolve()


def finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def normalized_vector_or_none(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (vec / norm).astype(np.float32)


def reference_vector_for_row(row: pd.Series) -> np.ndarray | None:
    candidate = np.array(
        [
            finite_float(row.get("reference_nx")),
            finite_float(row.get("reference_ny")),
            finite_float(row.get("reference_nz")),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(candidate)):
        return None
    return normalized_vector_or_none(candidate)


def load_feature_cache_meta(cache_file: Path) -> dict:
    with np.load(cache_file) as data:
        grid_rows, grid_cols = [int(v) for v in data["grid_size"].tolist()]
        resized_w, resized_h = [int(v) for v in data["resized_size"].tolist()]
        original_w, original_h = [int(v) for v in data["original_size"].tolist()]
        patch_size = int(np.asarray(data["patch_size"]).reshape(-1)[0])
    return {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "original_w": original_w,
        "original_h": original_h,
        "patch_size": patch_size,
        "cropped_w": grid_cols * patch_size,
        "cropped_h": grid_rows * patch_size,
    }


def bbox_patch_window(row: pd.Series, meta: dict) -> Tuple[int, int, int, int]:
    x_min = int(row["x_min"])
    y_min = int(row["y_min"])
    x_max = int(row["x_max"])
    y_max = int(row["y_max"])

    x_min_resized = int(np.floor(x_min * meta["resized_w"] / meta["original_w"]))
    x_max_resized = int(np.ceil(x_max * meta["resized_w"] / meta["original_w"]))
    y_min_resized = int(np.floor(y_min * meta["resized_h"] / meta["original_h"]))
    y_max_resized = int(np.ceil(y_max * meta["resized_h"] / meta["original_h"]))

    x_min_resized = max(0, min(x_min_resized, meta["cropped_w"] - 1))
    y_min_resized = max(0, min(y_min_resized, meta["cropped_h"] - 1))
    x_max_resized = max(x_min_resized + 1, min(x_max_resized, meta["cropped_w"]))
    y_max_resized = max(y_min_resized + 1, min(y_max_resized, meta["cropped_h"]))

    patch_col_min = max(0, x_min_resized // meta["patch_size"])
    patch_col_max = min(meta["grid_cols"], int(np.ceil(x_max_resized / meta["patch_size"])))
    patch_row_min = max(0, y_min_resized // meta["patch_size"])
    patch_row_max = min(meta["grid_rows"], int(np.ceil(y_max_resized / meta["patch_size"])))
    return patch_row_min, patch_row_max, patch_col_min, patch_col_max


def patch_original_bounds(patch_row: int, patch_col: int, meta: dict) -> Tuple[int, int, int, int]:
    x0_resized = patch_col * meta["patch_size"]
    y0_resized = patch_row * meta["patch_size"]
    x1_resized = min((patch_col + 1) * meta["patch_size"], meta["cropped_w"])
    y1_resized = min((patch_row + 1) * meta["patch_size"], meta["cropped_h"])

    x0 = max(0, min(meta["original_w"] - 1, int(np.floor(x0_resized * meta["original_w"] / meta["resized_w"]))))
    y0 = max(0, min(meta["original_h"] - 1, int(np.floor(y0_resized * meta["original_h"] / meta["resized_h"]))))
    x1 = max(x0 + 1, min(meta["original_w"], int(np.ceil(x1_resized * meta["original_w"] / meta["resized_w"]))))
    y1 = max(y0 + 1, min(meta["original_h"], int(np.ceil(y1_resized * meta["original_h"] / meta["resized_h"]))))
    return x0, y0, x1, y1


def patch_metrics(
    normals: np.ndarray,
    gradient_map: np.ndarray,
    divergence_map: np.ndarray,
    valid_mask: np.ndarray,
    valid_derivative_mask: np.ndarray,
    patch_bounds_xyxy: Tuple[int, int, int, int],
    reference_vec: np.ndarray | None,
) -> dict:
    x0, y0, x1, y1 = patch_bounds_xyxy
    normals_patch = normals[y0:y1, x0:x1]
    valid_patch = valid_mask[y0:y1, x0:x1]
    valid_deriv_patch = valid_derivative_mask[y0:y1, x0:x1]
    gradient_patch = gradient_map[y0:y1, x0:x1]
    divergence_patch = divergence_map[y0:y1, x0:x1]

    if reference_vec is not None and int(valid_patch.sum()) > 0:
        normals_valid = normals_patch[valid_patch]
        rel_cos = np.clip(np.einsum("ij,j->i", normals_valid, reference_vec), -1.0, 1.0)
        rel_deg = np.degrees(np.arccos(rel_cos)).astype(np.float32)
        relp99 = float(np.percentile(rel_deg, 99))
    else:
        relp99 = 0.0

    gradient_values = gradient_patch[valid_deriv_patch].astype(np.float32)
    divergence_values = divergence_patch[valid_deriv_patch].astype(np.float32)
    gmax = float(np.max(gradient_values)) if gradient_values.size > 0 else 0.0
    div_min = float(np.min(divergence_values)) if divergence_values.size > 0 else 0.0
    div_max = float(np.max(divergence_values)) if divergence_values.size > 0 else 0.0
    return {
        "patch_relp99": relp99,
        "patch_gmax": gmax,
        "patch_divergence_min": div_min,
        "patch_divergence_max": div_max,
        "patch_valid_pixel_count": int(valid_patch.sum()),
        "patch_valid_derivative_pixel_count": int(valid_deriv_patch.sum()),
    }


def mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    base_metrics_csv = args.base_metrics_csv.resolve()
    output_dir = default_output_dir(base_metrics_csv, args.output_dir)
    ensure_dir(output_dir)

    base_table = pd.read_csv(base_metrics_csv).copy()
    samples = load_run_samples(experiment_dir, seed=args.seed)
    sample_map = {sample.sample: sample for sample in samples}

    rows: List[Dict[str, object]] = []
    for sample_name, sample_df in base_table.groupby("sample", sort=True):
        run_sample = sample_map.get(str(sample_name))
        if run_sample is None:
            raise KeyError(f"Sample from metrics CSV not found in run samples: {sample_name}")

        image_rgb = np.array(Image.open(run_sample.image_path).convert("RGB"))
        normals, _, gradient_map, divergence_map, valid_mask, valid_derivative_mask = compute_maps(
            image_rgb,
            black_threshold=args.black_threshold,
        )
        anomaly_grid = np.load(run_sample.anomaly_map_path).astype(np.float32)
        meta = load_feature_cache_meta(run_sample.feature_cache_path)

        if anomaly_grid.shape != (meta["grid_rows"], meta["grid_cols"]):
            raise ValueError(
                f"Grid mismatch for {sample_name}: anomaly grid {anomaly_grid.shape} vs cache {(meta['grid_rows'], meta['grid_cols'])}"
            )

        for _, row in sample_df.iterrows():
            reference_vec = reference_vector_for_row(row)
            row_min, row_max, col_min, col_max = bbox_patch_window(row, meta)

            patch_candidates: List[Tuple[float, int, int]] = []
            for patch_row in range(row_min, row_max):
                for patch_col in range(col_min, col_max):
                    patch_candidates.append((float(anomaly_grid[patch_row, patch_col]), patch_row, patch_col))
            patch_candidates.sort(key=lambda item: item[0], reverse=True)
            selected = patch_candidates[: max(1, int(args.top_k))]

            rel_values: List[float] = []
            gmax_values: List[float] = []
            divmin_values: List[float] = []
            divmax_values: List[float] = []
            selected_scores: List[float] = []

            for patch_score, patch_row, patch_col in selected:
                bounds = patch_original_bounds(patch_row, patch_col, meta)
                metrics = patch_metrics(
                    normals=normals,
                    gradient_map=gradient_map,
                    divergence_map=divergence_map,
                    valid_mask=valid_mask,
                    valid_derivative_mask=valid_derivative_mask,
                    patch_bounds_xyxy=bounds,
                    reference_vec=reference_vec,
                )
                rel_values.append(metrics["patch_relp99"])
                gmax_values.append(metrics["patch_gmax"])
                divmin_values.append(metrics["patch_divergence_min"])
                divmax_values.append(metrics["patch_divergence_max"])
                selected_scores.append(patch_score)

            row_dict = row.to_dict()
            row_dict["topk_patch_count"] = int(len(selected))
            row_dict["topk_patch_anomaly_mean"] = mean_or_zero(selected_scores)
            row_dict["topk_patch_anomaly_max"] = float(max(selected_scores)) if selected_scores else 0.0
            row_dict["topk_patch_relp99_mean"] = mean_or_zero(rel_values)
            row_dict["topk_patch_gmax_mean"] = mean_or_zero(gmax_values)
            row_dict["topk_patch_divergence_min_mean"] = mean_or_zero(divmin_values)
            row_dict["topk_patch_divergence_max_mean"] = mean_or_zero(divmax_values)
            rows.append(row_dict)

    feature_table = pd.DataFrame(rows)
    output_csv = output_dir / "roi_top3_patch_bbox_features.csv"
    feature_table.to_csv(output_csv, index=False)

    summary = {
        "experiment_dir": str(experiment_dir),
        "base_metrics_csv": str(base_metrics_csv),
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "num_rows": int(feature_table.shape[0]),
        "num_labeled_rows": int(feature_table["label"].fillna("").astype(str).str.strip().isin(["2D", "3D", "2d", "3d"]).sum()),
        "top_k": int(args.top_k),
        "feature_columns": [
            "topk_patch_relp99_mean",
            "topk_patch_gmax_mean",
            "topk_patch_divergence_min_mean",
            "topk_patch_divergence_max_mean",
        ],
    }
    write_json(summary, output_dir / "summary.json")

    print(f"Saved top-k patch bbox features: {output_csv}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
