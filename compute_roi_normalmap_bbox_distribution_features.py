from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from compute_roi_normalmap_bbox_metrics import compute_maps, ensure_dir, write_csv, write_json


DEFAULT_BASE_METRICS_CSV = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_normalmap_bbox_metrics_local_ring_ignore_black\roi_normalmap_bbox_metrics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute richer bbox distribution/top-k/fraction features on normalmaps for existing ROI boxes."
        )
    )
    parser.add_argument("--base-metrics-csv", type=Path, default=DEFAULT_BASE_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--black-threshold", type=int, default=0)
    return parser.parse_args()


def default_output_dir(base_metrics_csv: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (base_metrics_csv.parent / "distribution_features").resolve()


def finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def safe_top_mean(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return 0.0
    count = max(1, int(math.ceil(values.size * fraction)))
    partition_index = max(0, values.size - count)
    top_values = np.partition(values, partition_index)[partition_index:]
    return float(np.mean(top_values))


def safe_fraction_ge(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values >= threshold))


def normalized_vector_or_none(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (vec / norm).astype(np.float32)


def reference_vector_for_row(row: pd.Series, normals_bbox_valid: np.ndarray) -> np.ndarray | None:
    candidate = np.array(
        [
            finite_float(row.get("reference_nx")),
            finite_float(row.get("reference_ny")),
            finite_float(row.get("reference_nz")),
        ],
        dtype=np.float32,
    )
    if np.all(np.isfinite(candidate)):
        normalized = normalized_vector_or_none(candidate)
        if normalized is not None:
            return normalized
    if normals_bbox_valid.size == 0:
        return None
    fallback = np.median(normals_bbox_valid, axis=0)
    return normalized_vector_or_none(fallback)


def add_distribution_features(row_dict: Dict[str, object], prefix: str, values: np.ndarray, thresholds: List[float]) -> None:
    row_dict[f"{prefix}_p90"] = safe_percentile(values, 90)
    row_dict[f"{prefix}_p95"] = safe_percentile(values, 95)
    row_dict[f"{prefix}_p99"] = safe_percentile(values, 99)
    row_dict[f"{prefix}_top1pct_mean"] = safe_top_mean(values, 0.01)
    row_dict[f"{prefix}_top5pct_mean"] = safe_top_mean(values, 0.05)
    for threshold in thresholds:
        key = str(threshold).replace(".", "p").replace("-", "m")
        row_dict[f"{prefix}_frac_ge_{key}"] = safe_fraction_ge(values, threshold)


def compute_feature_rows(base_table: pd.DataFrame, black_threshold: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    per_sample_tables = {sample: sample_df.reset_index(drop=True) for sample, sample_df in base_table.groupby("sample", sort=True)}

    for sample, sample_df in per_sample_tables.items():
        image_path = Path(str(sample_df.iloc[0]["image_path"]))
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        normals, _, gradient_map, divergence_map, valid_mask, valid_derivative_mask = compute_maps(
            image_rgb,
            black_threshold=black_threshold,
        )

        for _, row in sample_df.iterrows():
            x_min = int(row["x_min"])
            y_min = int(row["y_min"])
            x_max = int(row["x_max"])
            y_max = int(row["y_max"])

            bbox_normals = normals[y_min : y_max + 1, x_min : x_max + 1]
            bbox_valid = valid_mask[y_min : y_max + 1, x_min : x_max + 1]
            bbox_valid_derivative = valid_derivative_mask[y_min : y_max + 1, x_min : x_max + 1]
            bbox_gradient = gradient_map[y_min : y_max + 1, x_min : x_max + 1]
            bbox_divergence = divergence_map[y_min : y_max + 1, x_min : x_max + 1]

            normals_valid = bbox_normals[bbox_valid]
            ref_vec = reference_vector_for_row(row, normals_valid)
            if ref_vec is None or normals_valid.size == 0:
                relative_inclination = np.zeros((0,), dtype=np.float32)
            else:
                rel_cos = np.clip(np.einsum("ij,j->i", normals_valid, ref_vec), -1.0, 1.0)
                relative_inclination = np.degrees(np.arccos(rel_cos)).astype(np.float32)

            gradient_values = bbox_gradient[bbox_valid_derivative].astype(np.float32)
            divergence_values = bbox_divergence[bbox_valid_derivative].astype(np.float32)
            divergence_negative = np.maximum(-divergence_values, 0.0).astype(np.float32)
            divergence_positive = np.maximum(divergence_values, 0.0).astype(np.float32)

            row_dict = row.to_dict()
            row_dict["distribution_valid_pixel_count"] = int(bbox_valid.sum())
            row_dict["distribution_valid_derivative_pixel_count"] = int(bbox_valid_derivative.sum())

            add_distribution_features(
                row_dict,
                prefix="relative_inclination_deg",
                values=relative_inclination,
                thresholds=[2.0, 4.0, 6.0, 10.0],
            )
            add_distribution_features(
                row_dict,
                prefix="gradient",
                values=gradient_values,
                thresholds=[0.02, 0.04, 0.06, 0.10],
            )

            row_dict["divergence_negative_top1pct_mean"] = safe_top_mean(divergence_negative, 0.01)
            row_dict["divergence_negative_top5pct_mean"] = safe_top_mean(divergence_negative, 0.05)
            row_dict["divergence_negative_frac_ge_0p01"] = safe_fraction_ge(divergence_negative, 0.01)
            row_dict["divergence_negative_frac_ge_0p02"] = safe_fraction_ge(divergence_negative, 0.02)
            row_dict["divergence_negative_frac_ge_0p05"] = safe_fraction_ge(divergence_negative, 0.05)

            row_dict["divergence_positive_top1pct_mean"] = safe_top_mean(divergence_positive, 0.01)
            row_dict["divergence_positive_top5pct_mean"] = safe_top_mean(divergence_positive, 0.05)
            row_dict["divergence_positive_frac_ge_0p01"] = safe_fraction_ge(divergence_positive, 0.01)
            row_dict["divergence_positive_frac_ge_0p02"] = safe_fraction_ge(divergence_positive, 0.02)
            row_dict["divergence_positive_frac_ge_0p05"] = safe_fraction_ge(divergence_positive, 0.05)

            rows.append(row_dict)

    return rows


def main() -> None:
    args = parse_args()
    base_metrics_csv = args.base_metrics_csv.resolve()
    output_dir = default_output_dir(base_metrics_csv, args.output_dir)
    ensure_dir(output_dir)

    base_table = pd.read_csv(base_metrics_csv).copy()
    required_columns = {
        "sample",
        "image_path",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    }
    missing = sorted(required_columns - set(base_table.columns))
    if missing:
        raise ValueError(f"Missing required columns in base metrics CSV: {missing}")

    feature_rows = compute_feature_rows(base_table, black_threshold=args.black_threshold)
    feature_table = pd.DataFrame(feature_rows)

    output_csv = output_dir / "roi_normalmap_bbox_distribution_features.csv"
    feature_table.to_csv(output_csv, index=False)

    feature_columns = [
        "relative_inclination_deg_p90",
        "relative_inclination_deg_p95",
        "relative_inclination_deg_p99",
        "relative_inclination_deg_top1pct_mean",
        "relative_inclination_deg_top5pct_mean",
        "relative_inclination_deg_frac_ge_2p0",
        "relative_inclination_deg_frac_ge_4p0",
        "relative_inclination_deg_frac_ge_6p0",
        "relative_inclination_deg_frac_ge_10p0",
        "gradient_p90",
        "gradient_p95",
        "gradient_p99",
        "gradient_top1pct_mean",
        "gradient_top5pct_mean",
        "gradient_frac_ge_0p02",
        "gradient_frac_ge_0p04",
        "gradient_frac_ge_0p06",
        "gradient_frac_ge_0p1",
        "divergence_negative_top1pct_mean",
        "divergence_negative_top5pct_mean",
        "divergence_negative_frac_ge_0p01",
        "divergence_negative_frac_ge_0p02",
        "divergence_negative_frac_ge_0p05",
        "divergence_positive_top1pct_mean",
        "divergence_positive_top5pct_mean",
        "divergence_positive_frac_ge_0p01",
        "divergence_positive_frac_ge_0p02",
        "divergence_positive_frac_ge_0p05",
    ]

    summary = {
        "base_metrics_csv": str(base_metrics_csv),
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "num_rows": int(feature_table.shape[0]),
        "num_labeled_rows": int(feature_table["label"].fillna("").astype(str).str.strip().isin(["2D", "3D", "2d", "3d"]).sum()),
        "feature_columns": feature_columns,
        "black_threshold": int(args.black_threshold),
    }
    write_json(summary, output_dir / "summary.json")

    print(f"Saved bbox distribution features: {output_csv}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
