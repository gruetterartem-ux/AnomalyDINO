from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from compute_roi_normalmap_bbox_metrics import compute_maps, ensure_dir, valid_core_mask


DEFAULT_BASE_METRICS_CSV = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_normalmap_bbox_metrics_local_ring_ignore_black\roi_normalmap_bbox_metrics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the 12 patch-style normalmap features directly on ROI bounding boxes."
    )
    parser.add_argument("--base-metrics-csv", type=Path, default=DEFAULT_BASE_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--black-threshold", type=int, default=0)
    parser.add_argument("--gradient-threshold-quantile", type=float, default=0.90)
    parser.add_argument("--delta-threshold-quantile", type=float, default=0.90)
    return parser.parse_args()


def default_output_dir(base_metrics_csv: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (base_metrics_csv.parent / "bbox_normalmap_features12").resolve()


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def dominant_normal(normals_valid: np.ndarray) -> np.ndarray | None:
    if normals_valid.size == 0:
        return None
    median_vec = np.median(normals_valid, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(median_vec))
    if norm <= 1e-8 or not np.isfinite(norm):
        mean_vec = np.mean(normals_valid, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_vec))
        if norm <= 1e-8 or not np.isfinite(norm):
            return None
        return mean_vec / norm
    return median_vec / norm


def compute_delta_map(normals: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dnx_dx = np.gradient(normals[..., 0], axis=1)
    dny_dx = np.gradient(normals[..., 1], axis=1)
    dnz_dx = np.gradient(normals[..., 2], axis=1)
    dnx_dy = np.gradient(normals[..., 0], axis=0)
    dny_dy = np.gradient(normals[..., 1], axis=0)
    dnz_dy = np.gradient(normals[..., 2], axis=0)

    lap_nx = np.gradient(dnx_dx, axis=1) + np.gradient(dnx_dy, axis=0)
    lap_ny = np.gradient(dny_dx, axis=1) + np.gradient(dny_dy, axis=0)
    lap_nz = np.gradient(dnz_dx, axis=1) + np.gradient(dnz_dy, axis=0)
    delta_mag = np.sqrt(lap_nx**2 + lap_ny**2 + lap_nz**2).astype(np.float32)

    valid_derivative_mask = valid_core_mask(valid_mask)
    valid_second_mask = valid_core_mask(valid_derivative_mask)
    return delta_mag, valid_second_mask.astype(bool)


def scalar_coherence(scalar_patch: np.ndarray, valid_mask: np.ndarray) -> float:
    if scalar_patch.size == 0 or int(valid_mask.sum()) < 4:
        return 0.0
    scalar = scalar_patch.copy()
    scalar[~valid_mask] = 0.0
    gx = np.gradient(scalar, axis=1)
    gy = np.gradient(scalar, axis=0)
    valid_grad = valid_core_mask(valid_mask)
    if int(valid_grad.sum()) == 0:
        return 0.0
    gx_v = gx[valid_grad]
    gy_v = gy[valid_grad]
    jxx = float(np.mean(gx_v * gx_v))
    jyy = float(np.mean(gy_v * gy_v))
    jxy = float(np.mean(gx_v * gy_v))
    trace = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = max(trace * trace - 4.0 * det, 0.0)
    sqrt_disc = float(np.sqrt(disc))
    lam1 = 0.5 * (trace + sqrt_disc)
    lam2 = 0.5 * (trace - sqrt_disc)
    return float((lam1 - lam2) / (lam1 + lam2 + 1e-8))


def largest_component_size(binary_mask: np.ndarray) -> int:
    if binary_mask.size == 0 or not np.any(binary_mask):
        return 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def bbox_slices(row: pd.Series) -> Tuple[slice, slice]:
    x_min = int(row["x_min"])
    y_min = int(row["y_min"])
    x_max = int(row["x_max"])
    y_max = int(row["y_max"])
    return slice(y_min, y_max + 1), slice(x_min, x_max + 1)


def main() -> None:
    args = parse_args()
    base_metrics_csv = args.base_metrics_csv.resolve()
    output_dir = default_output_dir(base_metrics_csv, args.output_dir)
    ensure_dir(output_dir)

    table = pd.read_csv(base_metrics_csv).copy()
    table["label"] = table["label"].fillna("").astype(str).str.strip()
    labeled_mask = table["label"].str.lower().isin(["2d", "3d"])
    labeled_table = table.loc[labeled_mask].copy()
    if labeled_table.empty:
        raise ValueError("No labeled 2D/3D ROI rows found.")

    feature_records: List[dict] = []
    all_gradient_values: List[np.ndarray] = []
    all_delta_values: List[np.ndarray] = []

    for image_path, image_rows in table.groupby("image_path", sort=True):
        image_rgb = np.array(Image.open(str(image_path)).convert("RGB"))
        normals, _, gradient_map, _, valid_mask, valid_derivative_mask = compute_maps(
            image_rgb,
            black_threshold=args.black_threshold,
        )
        delta_map, valid_second_mask = compute_delta_map(normals, valid_mask)

        for _, row in image_rows.iterrows():
            ys, xs = bbox_slices(row)
            normals_patch = normals[ys, xs]
            valid_patch = valid_mask[ys, xs]
            grad_patch = gradient_map[ys, xs]
            grad_valid_patch = valid_derivative_mask[ys, xs]
            delta_patch = delta_map[ys, xs]
            delta_valid_patch = valid_second_mask[ys, xs]

            normals_valid = normals_patch[valid_patch]
            dominant = dominant_normal(normals_valid)
            if dominant is None or normals_valid.size == 0:
                ang_values = np.zeros((0,), dtype=np.float32)
                ang_map = np.zeros(valid_patch.shape, dtype=np.float32)
            else:
                dots = np.clip(np.einsum("ij,j->i", normals_valid, dominant), -1.0, 1.0)
                ang_values = np.degrees(np.arccos(dots)).astype(np.float32)
                ang_map = np.zeros(valid_patch.shape, dtype=np.float32)
                ang_map[valid_patch] = ang_values

            grad_values = grad_patch[grad_valid_patch].astype(np.float32)
            delta_values = delta_patch[delta_valid_patch].astype(np.float32)
            row_label = str(row.get("label", "")).strip().lower()
            is_labeled_row = row_label in {"2d", "3d"}
            if is_labeled_row and grad_values.size > 0:
                all_gradient_values.append(grad_values)
            if is_labeled_row and delta_values.size > 0:
                all_delta_values.append(delta_values)

            feature_records.append(
                {
                    "row_data": row.to_dict(),
                    "valid_patch": valid_patch,
                    "grad_valid_patch": grad_valid_patch,
                    "delta_valid_patch": delta_valid_patch,
                    "grad_patch": grad_patch.astype(np.float32),
                    "delta_patch": delta_patch.astype(np.float32),
                    "normals_patch": normals_patch.astype(np.float32),
                    "ang_values": ang_values,
                    "ang_map": ang_map.astype(np.float32),
                }
            )

    if not all_gradient_values or not all_delta_values:
        raise ValueError("Could not derive gradient/delta thresholds from labeled boxes.")

    t1 = float(np.quantile(np.concatenate(all_gradient_values), args.gradient_threshold_quantile))
    t2 = float(np.quantile(np.concatenate(all_delta_values), args.delta_threshold_quantile))

    rows: List[Dict[str, object]] = []
    for record in feature_records:
        row_dict = dict(record["row_data"])
        normals_patch = record["normals_patch"]
        valid_patch = record["valid_patch"]
        grad_patch = record["grad_patch"]
        grad_valid_patch = record["grad_valid_patch"]
        delta_patch = record["delta_patch"]
        delta_valid_patch = record["delta_valid_patch"]
        ang_values = record["ang_values"]
        ang_map = record["ang_map"]

        grad_values = grad_patch[grad_valid_patch]
        delta_values = delta_patch[delta_valid_patch]
        normals_valid = normals_patch[valid_patch]

        grad_binary = (grad_patch > t1) & grad_valid_patch
        delta_binary = (delta_patch > t2) & delta_valid_patch

        row_dict["grad_p95"] = safe_percentile(grad_values, 95)
        row_dict["grad_max"] = safe_max(grad_values)
        row_dict["dominant_angle_mean_deg"] = safe_mean(ang_values)
        row_dict["dominant_angle_p95_deg"] = safe_percentile(ang_values, 95)
        row_dict["delta_mean"] = safe_mean(delta_values)
        row_dict["delta_p95"] = safe_percentile(delta_values, 95)
        row_dict["grad_frac_gt_t1"] = float(np.mean(grad_binary[grad_valid_patch])) if int(grad_valid_patch.sum()) > 0 else 0.0
        row_dict["grad_largest_component_size_t1"] = largest_component_size(grad_binary)
        row_dict["normal_total_variance"] = (
            float(np.var(normals_valid[:, 0]) + np.var(normals_valid[:, 1]) + np.var(normals_valid[:, 2]))
            if normals_valid.size > 0
            else 0.0
        )
        row_dict["nz_std"] = safe_std(normals_valid[:, 2] if normals_valid.size > 0 else np.zeros((0,), dtype=np.float32))
        row_dict["directional_coherence"] = scalar_coherence(ang_map, valid_patch)
        row_dict["delta_frac_gt_t2"] = float(np.mean(delta_binary[delta_valid_patch])) if int(delta_valid_patch.sum()) > 0 else 0.0
        row_dict["t1_grad_threshold"] = t1
        row_dict["t2_delta_threshold"] = t2
        rows.append(row_dict)

    feature_table = pd.DataFrame(rows)
    output_csv = output_dir / "roi_bbox_normalmap_features12.csv"
    feature_table.to_csv(output_csv, index=False)

    summary = {
        "base_metrics_csv": str(base_metrics_csv),
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "num_rows": int(feature_table.shape[0]),
        "num_labeled_rows": int(labeled_table.shape[0]),
        "gradient_threshold_quantile": float(args.gradient_threshold_quantile),
        "delta_threshold_quantile": float(args.delta_threshold_quantile),
        "t1_grad_threshold": t1,
        "t2_delta_threshold": t2,
        "feature_columns": [
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
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved bbox normalmap features: {output_csv}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
