from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from component_memory_bank.data_io import load_patch_features, load_patch_scores, load_run_samples


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_ROI_METADATA_CSV = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1"
    / "seed=0"
    / "roi_metadata.csv"
)
DEFAULT_LABELS_FILE = Path(r"C:\ai\AnomalyDINO\labeling_tables\dinov3_res688_roi_labels.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build labeled ROI feature vectors by selecting the top anomalous patch-grid patches "
            "inside each ROI bbox, debiasing them with positional PCA, and aggregating them with "
            "softmax weights over their anomaly scores."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-percent", type=float, default=0.10)
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--pca-components", type=int, default=2)
    parser.add_argument("--pca-power-iterations", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--valid-labels", type=str, nargs="*", default=("2D", "3D"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_table_file(table_file: Path) -> pd.DataFrame:
    suffix = table_file.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(table_file)
    if suffix in {".csv", ".txt", ".tsv"}:
        return pd.read_csv(table_file, sep=None, engine="python")
    raise ValueError(f"Unsupported table format: {table_file}")


def normalize_bildname(value: object) -> str:
    return str(value).strip().replace("\\", "/").split("/")[-1]


def normalize_roi_nummer(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("roi"):
        return text
    if text.isdigit():
        return f"roi{text}"
    return text


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def default_output_dir(experiment_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (
        experiment_dir
        / "roi_top10pct_centerinbox_pca2_softmax_patch_features_labeled"
    ).resolve()


def load_roi_table(roi_metadata_csv: Path) -> pd.DataFrame:
    if not roi_metadata_csv.exists():
        raise FileNotFoundError(f"ROI metadata not found: {roi_metadata_csv}")
    table = pd.read_csv(roi_metadata_csv).copy()
    table["sample"] = table["sample"].astype(str).str.replace("\\", "/", regex=False)
    table["bildname"] = table["sample"].map(normalize_bildname)
    table["roi_nummer"] = "roi" + table["roi_index"].astype(int).astype(str)
    table["roi_uid"] = table["sample"] + "__roi_" + table["roi_index"].astype(int).map(lambda idx: f"{idx:03d}")
    return table


def load_labels_table(labels_file: Path, valid_labels: List[str] | None) -> pd.DataFrame:
    labels = load_table_file(labels_file)
    if not {"bildname", "roi_nummer", "label"}.issubset(labels.columns):
        raise ValueError("Labels file must contain bildname, roi_nummer, and label columns.")

    labels = labels.copy()
    labels["bildname"] = labels["bildname"].map(normalize_bildname)
    labels["roi_nummer"] = labels["roi_nummer"].map(normalize_roi_nummer)
    labels["label"] = labels["label"].map(clean_label)
    labels = labels[labels["label"] != ""].copy()

    if "Genaues Label" in labels.columns:
        labels = labels.rename(columns={"Genaues Label": "detailed_label"})
    if "detailed_label" not in labels.columns:
        labels["detailed_label"] = ""
    labels["detailed_label"] = labels["detailed_label"].fillna("").astype(str).str.strip()

    if valid_labels is not None:
        valid_label_set = {label.lower(): label for label in valid_labels}
        labels["label_lower"] = labels["label"].str.lower()
        labels = labels[labels["label_lower"].isin(valid_label_set.keys())].copy()
        labels["label"] = labels["label_lower"].map(valid_label_set)
        labels = labels.drop(columns=["label_lower"])

    if labels.duplicated(["bildname", "roi_nummer"]).any():
        raise ValueError("Labels file contains duplicate bildname + roi_nummer entries.")
    return labels


def prepare_labeled_roi_table(
    roi_table: pd.DataFrame,
    labels_table: pd.DataFrame,
    limit: int | None,
) -> pd.DataFrame:
    merged = roi_table.merge(labels_table, on=["bildname", "roi_nummer"], how="inner", suffixes=("", "_label"))
    if merged.empty:
        raise ValueError("No labeled ROIs matched the ROI metadata.")
    merged = merged.sort_values(["bildname", "roi_index"]).reset_index(drop=True)
    if limit is not None:
        merged = merged.iloc[:limit].copy()
    return merged


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


def patch_center_original(patch_row: int, patch_col: int, meta: dict) -> Tuple[float, float]:
    cx_resized = (patch_col + 0.5) * meta["patch_size"]
    cy_resized = (patch_row + 0.5) * meta["patch_size"]
    cx = float(cx_resized * meta["original_w"] / meta["resized_w"])
    cy = float(cy_resized * meta["original_h"] / meta["resized_h"])
    return cx, cy


def patch_overlaps_bbox(patch_row: int, patch_col: int, meta: dict, x_min: int, y_min: int, x_max: int, y_max: int) -> bool:
    px0, py0, px1, py1 = patch_original_bounds(patch_row, patch_col, meta)
    overlap_w = min(px1, x_max) - max(px0, x_min)
    overlap_h = min(py1, y_max) - max(py0, y_min)
    return overlap_w > 0 and overlap_h > 0


def patch_distance_to_bbox_center(
    patch_row: int,
    patch_col: int,
    meta: dict,
    bbox_center_x: float,
    bbox_center_y: float,
) -> float:
    cx, cy = patch_center_original(patch_row, patch_col, meta)
    return float((cx - bbox_center_x) ** 2 + (cy - bbox_center_y) ** 2)


def _orthonormalize_patch_basis(candidates: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num_positions, num_components, feature_dim = candidates.shape
    basis = np.zeros((num_positions, num_components, feature_dim), dtype=np.float32)

    v1 = candidates[:, 0, :].astype(np.float32)
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    mask1 = n1[:, 0] > eps
    if np.any(mask1):
        basis[mask1, 0, :] = v1[mask1] / n1[mask1]

    for comp_idx in range(1, num_components):
        vk = candidates[:, comp_idx, :].astype(np.float32)
        for prev_idx in range(comp_idx):
            prev = basis[:, prev_idx, :]
            proj = np.sum(vk * prev, axis=1, keepdims=True)
            vk = vk - proj * prev
        nk = np.linalg.norm(vk, axis=1, keepdims=True)
        maskk = nk[:, 0] > eps
        if np.any(maskk):
            basis[maskk, comp_idx, :] = vk[maskk] / nk[maskk]
    return basis


def build_run_context(experiment_dir: Path, seed: int) -> List[dict]:
    run_samples = load_run_samples(experiment_dir, seed=seed)
    rows: List[dict] = []
    for sample in run_samples:
        rows.append(
            {
                "sample": sample.sample,
                "evaluation_group": sample.evaluation_group,
                "image_label": int(sample.image_label),
                "image_score": float(sample.image_score),
                "image_threshold": float(sample.image_threshold),
                "image_path": str(sample.image_path),
                "feature_cache_path": str(sample.feature_cache_path),
                "anomaly_map_path": str(sample.anomaly_map_path),
                "run_sample": sample,
            }
        )
    rows.sort(key=lambda item: item["sample"])
    return rows


def build_positional_reference(
    run_rows: List[dict],
) -> dict:
    good_rows = [row for row in run_rows if int(row["image_label"]) == 0]
    if not good_rows:
        raise ValueError("Keine good-Bilder fuer die Positionsreferenz gefunden.")

    sum_valid: np.ndarray | None = None
    count_valid: np.ndarray | None = None
    sum_all: np.ndarray | None = None
    count_all: np.ndarray | None = None
    grid_shape_ref: tuple[int, int] | None = None

    for row in good_rows:
        features, grid_shape = load_patch_features(row["run_sample"])
        score_grid = load_patch_scores(row["run_sample"])
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features_norm = (features / norms).astype(np.float64)

        if grid_shape_ref is None:
            grid_shape_ref = tuple(int(v) for v in grid_shape)
            feature_dim = int(features.shape[1])
            num_patches = int(features.shape[0])
            sum_valid = np.zeros((num_patches, feature_dim), dtype=np.float64)
            count_valid = np.zeros((num_patches,), dtype=np.int32)
            sum_all = np.zeros((num_patches, feature_dim), dtype=np.float64)
            count_all = np.zeros((num_patches,), dtype=np.int32)
        elif tuple(int(v) for v in grid_shape) != grid_shape_ref:
            raise ValueError(f"Inkonsistentes Grid in Referenzbildern: {grid_shape} vs {grid_shape_ref}")

        valid_mask = score_grid.reshape(-1) <= float(row["image_threshold"])
        assert sum_valid is not None and count_valid is not None and sum_all is not None and count_all is not None
        sum_all += features_norm
        count_all += 1
        if np.any(valid_mask):
            sum_valid[valid_mask] += features_norm[valid_mask]
            count_valid[valid_mask] += 1

    assert sum_valid is not None and count_valid is not None and sum_all is not None and count_all is not None
    mean_vectors = np.divide(
        sum_valid,
        np.maximum(count_valid[:, None], 1),
        out=np.zeros_like(sum_valid),
        where=count_valid[:, None] > 0,
    )
    fallback_vectors = np.divide(
        sum_all,
        np.maximum(count_all[:, None], 1),
        out=np.zeros_like(sum_all),
        where=count_all[:, None] > 0,
    )
    missing_mask = count_valid <= 0
    if np.any(missing_mask):
        mean_vectors[missing_mask] = fallback_vectors[missing_mask]

    vector_norms = np.linalg.norm(mean_vectors, axis=1, keepdims=True)
    vector_norms = np.maximum(vector_norms, 1e-8)
    mean_vectors_norm = (mean_vectors / vector_norms).astype(np.float32)
    return {
        "reference_vectors": mean_vectors_norm,
        "grid_shape": list(grid_shape_ref) if grid_shape_ref is not None else None,
        "num_good_images": len(good_rows),
        "count_valid_mean": float(count_valid.mean()),
    }


def build_positional_pca_reference(
    run_rows: List[dict],
    num_components: int,
    num_power_iterations: int,
) -> dict:
    mean_reference = build_positional_reference(run_rows)
    mean_vectors = np.asarray(mean_reference["reference_vectors"], dtype=np.float32)
    good_rows = [row for row in run_rows if int(row["image_label"]) == 0]
    if not good_rows:
        raise ValueError("Keine good-Bilder fuer die PCA-Referenz gefunden.")

    num_positions, feature_dim = mean_vectors.shape
    rng = np.random.default_rng(0)
    init = rng.standard_normal((num_positions, num_components, feature_dim), dtype=np.float32)
    basis = _orthonormalize_patch_basis(init)

    valid_counts = np.zeros((num_positions,), dtype=np.int32)
    for _ in range(num_power_iterations):
        accum = np.zeros((num_positions, num_components, feature_dim), dtype=np.float64)
        valid_counts[:] = 0
        for row in good_rows:
            features, _ = load_patch_features(row["run_sample"])
            score_grid = load_patch_scores(row["run_sample"])
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            features_norm = (features / norms).astype(np.float32)

            valid_mask = (score_grid.reshape(-1) <= float(row["image_threshold"])).astype(np.float32)
            projection_mean = np.sum(features_norm * mean_vectors, axis=1, keepdims=True)
            residual = features_norm - projection_mean * mean_vectors
            proj = np.einsum("pd,pkd->pk", residual, basis, optimize=True)
            accum += residual[:, None, :] * proj[:, :, None] * valid_mask[:, None, None]
            valid_counts += valid_mask.astype(np.int32)
        basis = _orthonormalize_patch_basis(accum.astype(np.float32))

    return {
        "mean_reference": mean_reference,
        "basis": basis,
        "num_components": int(num_components),
        "num_power_iterations": int(num_power_iterations),
        "num_good_images": len(good_rows),
        "count_valid_mean": float(valid_counts.mean()),
    }


def pca_subspace_debiased_features(
    features_norm: np.ndarray,
    mean_reference_vectors: np.ndarray,
    pca_basis: np.ndarray,
) -> np.ndarray:
    projection_mean = np.sum(features_norm * mean_reference_vectors, axis=1, keepdims=True)
    residual = features_norm - projection_mean * mean_reference_vectors
    coeff = np.einsum("pd,pkd->pk", residual, pca_basis, optimize=True)
    residual = residual - np.einsum("pk,pkd->pd", coeff, pca_basis, optimize=True)
    residual_norm = np.linalg.norm(residual, axis=1, keepdims=True)
    fallback_mask = residual_norm.squeeze(-1) <= 1e-8
    residual_norm = np.maximum(residual_norm, 1e-8)
    debiased = (residual / residual_norm).astype(np.float32)
    if np.any(fallback_mask):
        debiased[fallback_mask] = features_norm[fallback_mask]
    return debiased


def softmax_query_weights(anomaly_scores: np.ndarray) -> np.ndarray:
    if anomaly_scores.size == 0:
        raise ValueError("No anomaly scores provided for weighting")
    if anomaly_scores.size == 1:
        return np.array([1.0], dtype=np.float32)
    score_min = float(anomaly_scores.min())
    score_max = float(anomaly_scores.max())
    if score_max <= score_min:
        return np.full(anomaly_scores.shape, 1.0 / anomaly_scores.size, dtype=np.float32)
    logits = (anomaly_scores - score_min) / max(score_max - score_min, 1e-8)
    logits = logits - float(logits.max())
    exp_logits = np.exp(logits).astype(np.float32)
    return (exp_logits / max(float(exp_logits.sum()), 1e-8)).astype(np.float32)


def aggregate_selected_patch_features(
    features_norm: np.ndarray,
    score_grid: np.ndarray,
    grid_shape: tuple[int, int],
    selected_patches: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch_features: List[np.ndarray] = []
    anomaly_scores: List[float] = []
    for patch_row, patch_col in selected_patches:
        idx = patch_row * grid_shape[1] + patch_col
        patch_features.append(features_norm[idx])
        anomaly_scores.append(float(score_grid[patch_row, patch_col]))
    feature_matrix = np.stack(patch_features, axis=0).astype(np.float32)
    anomaly_array = np.array(anomaly_scores, dtype=np.float32)
    weights = softmax_query_weights(anomaly_array)
    combined = (feature_matrix * weights[:, None]).sum(axis=0)
    combined_norm = np.linalg.norm(combined)
    if combined_norm <= 1e-8:
        combined = feature_matrix.mean(axis=0)
        combined_norm = np.linalg.norm(combined)
    combined = (combined / max(float(combined_norm), 1e-8)).astype(np.float32)
    return combined, anomaly_array, weights


def _collect_roi_patch_candidates(
    row: pd.Series,
    meta: dict,
    anomaly_grid: np.ndarray,
) -> tuple[list[tuple[float, int, int]], list[tuple[float, int, int]], list[tuple[float, float, int, int]]]:
    x_min = int(row["x_min"])
    y_min = int(row["y_min"])
    x_max = int(row["x_max"])
    y_max = int(row["y_max"])
    bbox_center_x = 0.5 * (x_min + x_max)
    bbox_center_y = 0.5 * (y_min + y_max)
    row_min, row_max, col_min, col_max = bbox_patch_window(row, meta)

    center_candidates: List[Tuple[float, int, int]] = []
    overlap_candidates: List[Tuple[float, int, int]] = []
    nearest_candidates: List[Tuple[float, float, int, int]] = []

    for patch_row in range(row_min, row_max):
        for patch_col in range(col_min, col_max):
            score = float(anomaly_grid[patch_row, patch_col])
            cx, cy = patch_center_original(patch_row, patch_col, meta)
            center_in_box = (x_min <= cx < x_max) and (y_min <= cy < y_max)
            if center_in_box:
                center_candidates.append((score, patch_row, patch_col))
            if patch_overlaps_bbox(patch_row, patch_col, meta, x_min, y_min, x_max, y_max):
                overlap_candidates.append((score, patch_row, patch_col))
            dist2 = patch_distance_to_bbox_center(patch_row, patch_col, meta, bbox_center_x, bbox_center_y)
            nearest_candidates.append((dist2, score, patch_row, patch_col))

    return center_candidates, overlap_candidates, nearest_candidates


def select_roi_patches_center_in_box(
    row: pd.Series,
    meta: dict,
    anomaly_grid: np.ndarray,
    top_percent: float,
    min_patches: int,
) -> tuple[list[tuple[int, int]], str, int]:
    center_candidates, overlap_candidates, nearest_candidates = _collect_roi_patch_candidates(
        row=row,
        meta=meta,
        anomaly_grid=anomaly_grid,
    )

    candidate_list: List[Tuple[float, int, int]]
    selection_mode: str
    if center_candidates:
        candidate_list = center_candidates
        selection_mode = "center_in_box"
    elif overlap_candidates:
        candidate_list = overlap_candidates
        selection_mode = "overlap_fallback"
    else:
        nearest_candidates.sort(key=lambda item: (item[0], -item[1]))
        _, score, patch_row, patch_col = nearest_candidates[0]
        candidate_list = [(score, patch_row, patch_col)]
        selection_mode = "nearest_center_fallback"

    candidate_list.sort(key=lambda item: item[0], reverse=True)
    num_candidates = len(candidate_list)
    top_k = max(int(min_patches), int(math.ceil(float(top_percent) * float(num_candidates))))
    top_k = min(top_k, num_candidates)
    selected = [(patch_row, patch_col) for _, patch_row, patch_col in candidate_list[:top_k]]
    return selected, selection_mode, int(num_candidates)


def select_roi_patches_overlap(
    row: pd.Series,
    meta: dict,
    anomaly_grid: np.ndarray,
    top_percent: float,
    min_patches: int,
) -> tuple[list[tuple[int, int]], str, int]:
    _, overlap_candidates, nearest_candidates = _collect_roi_patch_candidates(
        row=row,
        meta=meta,
        anomaly_grid=anomaly_grid,
    )

    candidate_list: List[Tuple[float, int, int]]
    selection_mode: str
    if overlap_candidates:
        candidate_list = overlap_candidates
        selection_mode = "overlap"
    else:
        nearest_candidates.sort(key=lambda item: (item[0], -item[1]))
        _, score, patch_row, patch_col = nearest_candidates[0]
        candidate_list = [(score, patch_row, patch_col)]
        selection_mode = "nearest_center_fallback"

    candidate_list.sort(key=lambda item: item[0], reverse=True)
    num_candidates = len(candidate_list)
    top_k = max(int(min_patches), int(math.ceil(float(top_percent) * float(num_candidates))))
    top_k = min(top_k, num_candidates)
    selected = [(patch_row, patch_col) for _, patch_row, patch_col in candidate_list[:top_k]]
    return selected, selection_mode, int(num_candidates)


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

    run_rows = build_run_context(experiment_dir, seed=args.seed)
    sample_map = {row["sample"]: row for row in run_rows}
    pca_reference = build_positional_pca_reference(
        run_rows,
        num_components=int(args.pca_components),
        num_power_iterations=int(args.pca_power_iterations),
    )
    mean_reference_vectors = np.asarray(pca_reference["mean_reference"]["reference_vectors"], dtype=np.float32)
    pca_basis = np.asarray(pca_reference["basis"], dtype=np.float32)

    feature_rows: List[Dict[str, object]] = []
    feature_vectors: List[np.ndarray] = []

    for sample_name, sample_df in labeled_rois.groupby("sample", sort=True):
        run_row = sample_map.get(str(sample_name))
        if run_row is None:
            raise KeyError(f"Sample from ROI metadata not found in run samples: {sample_name}")

        run_sample = run_row["run_sample"]
        features, grid_shape = load_patch_features(run_sample)
        score_grid = load_patch_scores(run_sample)
        meta = load_feature_cache_meta(run_sample.feature_cache_path)
        if tuple(int(v) for v in grid_shape) != (meta["grid_rows"], meta["grid_cols"]):
            raise ValueError(f"Grid mismatch for {sample_name}: {grid_shape} vs {(meta['grid_rows'], meta['grid_cols'])}")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features_norm = (features / norms).astype(np.float32)
        debiased_features = pca_subspace_debiased_features(
            features_norm,
            mean_reference_vectors,
            pca_basis,
        )

        for _, row in sample_df.iterrows():
            selected_patches, selection_mode, num_candidate_patches = select_roi_patches_center_in_box(
                row=row,
                meta=meta,
                anomaly_grid=score_grid,
                top_percent=float(args.top_percent),
                min_patches=int(args.min_patches),
            )
            combined_feature, anomaly_scores, weights = aggregate_selected_patch_features(
                debiased_features,
                score_grid,
                tuple(int(v) for v in grid_shape),
                selected_patches,
            )

            feature_index = len(feature_vectors)
            feature_vectors.append(combined_feature.astype(np.float32))
            feature_rows.append(
                {
                    "feature_index": int(feature_index),
                    "sample": str(sample_name),
                    "group_id": str(sample_name),
                    "evaluation_group": str(run_row["evaluation_group"]),
                    "image_path": str(run_row["image_path"]),
                    "feature_cache_path": str(run_row["feature_cache_path"]),
                    "anomaly_map_path": str(run_row["anomaly_map_path"]),
                    "image_score": float(run_row["image_score"]),
                    "image_threshold": float(run_row["image_threshold"]),
                    "bildname": str(row["bildname"]),
                    "roi_nummer": str(row["roi_nummer"]),
                    "roi_uid": str(row["roi_uid"]),
                    "roi_index": int(row["roi_index"]),
                    "x_min": int(row["x_min"]),
                    "y_min": int(row["y_min"]),
                    "x_max": int(row["x_max"]),
                    "y_max": int(row["y_max"]),
                    "crop_path": str(row.get("crop_path", "")),
                    "label": str(row["label"]),
                    "notes": str(row.get("detailed_label", "")),
                    "detailed_label": str(row.get("detailed_label", "")),
                    "grid_rows": int(grid_shape[0]),
                    "grid_cols": int(grid_shape[1]),
                    "selection_mode": selection_mode,
                    "top_percent": float(args.top_percent),
                    "min_patches": int(args.min_patches),
                    "num_candidate_patches_center_or_fallback": int(num_candidate_patches),
                    "num_selected_patches": int(len(selected_patches)),
                    "selected_patch_rows": ";".join(str(patch_row) for patch_row, _ in selected_patches),
                    "selected_patch_cols": ";".join(str(patch_col) for _, patch_col in selected_patches),
                    "selected_patch_scores": ";".join(f"{float(score):.6f}" for score in anomaly_scores.tolist()),
                    "selected_patch_softmax_weights": ";".join(f"{float(weight):.6f}" for weight in weights.tolist()),
                    "feature_mode": f"pca_subspace_k{int(args.pca_components)}",
                    "query_weight_mode": "softmax",
                }
            )

    if not feature_vectors:
        raise ValueError("No labeled ROI features were extracted.")

    features_array = np.stack(feature_vectors, axis=0).astype(np.float32)
    feature_table = pd.DataFrame(feature_rows)
    features_file = output_dir / "roi_features_mean.npy"
    table_file = output_dir / "roi_feature_table.csv"
    np.save(features_file, features_array)
    feature_table.to_csv(table_file, index=False)

    class_counts = feature_table["label"].value_counts().to_dict()
    summary = {
        "experiment_dir": str(experiment_dir),
        "roi_metadata_csv": str(roi_metadata_csv),
        "labels_file": str(labels_file),
        "seed": int(args.seed),
        "top_percent": float(args.top_percent),
        "min_patches": int(args.min_patches),
        "selection_rule": "center_in_box_with_fallback",
        "pca_components": int(args.pca_components),
        "pca_power_iterations": int(args.pca_power_iterations),
        "num_good_reference_images": int(pca_reference["num_good_images"]),
        "reference_valid_mean": float(pca_reference["count_valid_mean"]),
        "num_labeled_rois": int(len(feature_table)),
        "num_groups": int(feature_table["group_id"].astype(str).nunique()),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "features_file": str(features_file),
        "table_file": str(table_file),
    }
    write_json(summary, output_dir / "summary.json")

    print(f"Saved ROI features: {features_file}")
    print(f"Saved ROI feature table: {table_file}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
