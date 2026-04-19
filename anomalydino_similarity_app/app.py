from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from component_memory_bank.data_io import load_patch_features, load_patch_scores, load_run_args, load_run_samples


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)

EXPORT_SUBDIR = "similarity_query_exports"
ALBEDO_CACHE_SUBDIR = "albedo_patch_feature_cache"
MULTILAYER_CACHE_SUBDIR = "patch_feature_cache_multilayer_l1to12"
IRELIEF_SUBDIR = (
    "roi_top10pct_centerinbox_pca2_softmax_patch_features_labeled"
    r"\irelief_cosine_weighted_features"
)
IRELIEF_MULTILAYER_SUBDIR = (
    "roi_top10pct_centerinbox_multilayer_l1to12_softmax_patch_features_labeled"
    r"\irelief_cosine_weighted_features"
)


def _parse_experiment_dir_from_argv() -> str:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []
    if not argv:
        return str(DEFAULT_EXPERIMENT_DIR)
    return argv[0]


def _clean_sample_name(sample: str) -> str:
    return str(sample).replace("\\", "/")


def _safe_name(text: str) -> str:
    safe = str(text).replace("\\", "__").replace("/", "__").replace(":", "_")
    safe = safe.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in safe)


@st.cache_data(show_spinner=False)
def load_run_context(experiment_dir_str: str, seed: int) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    run_args = load_run_args(experiment_dir)
    samples = load_run_samples(experiment_dir, seed=seed)
    rows: list[dict[str, Any]] = []
    sample_index: dict[str, dict[str, Any]] = {}
    for sample in samples:
        normalized = _clean_sample_name(sample.sample)
        row = {
            "sample": normalized,
            "evaluation_group": sample.evaluation_group,
            "image_label": int(sample.image_label),
            "image_score": float(sample.image_score),
            "image_threshold": float(sample.image_threshold),
            "image_path": str(sample.image_path),
            "feature_cache_path": str(sample.feature_cache_path),
            "anomaly_map_path": str(sample.anomaly_map_path),
        }
        rows.append(row)
        sample_index[normalized] = row
    rows.sort(key=lambda item: item["sample"])
    return {
        "experiment_dir": str(experiment_dir),
        "run_args": run_args,
        "samples": rows,
        "sample_index": sample_index,
    }


@st.cache_data(show_spinner=False)
def load_albedo_cache_manifest(experiment_dir_str: str, seed: int) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    manifest_path = experiment_dir / ALBEDO_CACHE_SUBDIR / f"seed={seed}" / "cache_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Albedo-Cache-Manifest nicht gefunden: {manifest_path}. "
            "Erzeuge zuerst den Albedo-Patch-Cache."
        )

    rows: list[dict[str, Any]] = []
    sample_index: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for raw_row in csv.DictReader(handle):
            sample = _clean_sample_name(raw_row["sample"])
            row = {
                "sample": sample,
                "image_path": raw_row["image_path"],
                "cache_file": raw_row["cache_file"],
                "grid_h": int(raw_row["grid_h"]),
                "grid_w": int(raw_row["grid_w"]),
                "feature_dim": int(raw_row["feature_dim"]),
                "patch_size": int(raw_row["patch_size"]),
                "resized_width": int(raw_row["resized_width"]),
                "resized_height": int(raw_row["resized_height"]),
                "original_width": int(raw_row["original_width"]),
                "original_height": int(raw_row["original_height"]),
            }
            rows.append(row)
            sample_index[sample] = row

    rows.sort(key=lambda item: item["sample"])
    return {
        "manifest_path": str(manifest_path),
        "samples": rows,
        "sample_index": sample_index,
    }


@st.cache_data(show_spinner=False)
def load_multilayer_cache_manifest(experiment_dir_str: str, seed: int) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    manifest_path = experiment_dir / MULTILAYER_CACHE_SUBDIR / f"seed={seed}" / "cache_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Multi-Layer-Cache-Manifest nicht gefunden: {manifest_path}. "
            "Erzeuge zuerst den Multi-Layer-Patch-Cache."
        )

    rows: list[dict[str, Any]] = []
    sample_index: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for raw_row in csv.DictReader(handle):
            sample = _clean_sample_name(raw_row["sample"])
            row = {
                "sample": sample,
                "image_path": raw_row["image_path"],
                "cache_file": raw_row["cache_file"],
                "grid_h": int(raw_row["grid_h"]),
                "grid_w": int(raw_row["grid_w"]),
                "patch_size": int(raw_row["patch_size"]),
                "resized_width": int(raw_row["resized_width"]),
                "resized_height": int(raw_row["resized_height"]),
                "original_width": int(raw_row["original_width"]),
                "original_height": int(raw_row["original_height"]),
                "num_layers": int(raw_row["num_layers"]),
                "layer_dim": int(raw_row["layer_dim"]),
                "feature_dim_concat": int(raw_row["feature_dim_concat"]),
                "layer_indices": tuple(int(v) for v in str(raw_row["layer_indices"]).split(";") if str(v).strip()),
            }
            rows.append(row)
            sample_index[sample] = row

    rows.sort(key=lambda item: item["sample"])
    return {
        "manifest_path": str(manifest_path),
        "samples": rows,
        "sample_index": sample_index,
    }


@st.cache_data(show_spinner=False)
def load_albedo_sample_assets(experiment_dir_str: str, seed: int, sample_name: str) -> dict[str, Any]:
    manifest = load_albedo_cache_manifest(experiment_dir_str, seed)
    if sample_name not in manifest["sample_index"]:
        raise KeyError(f"Sample nicht im Albedo-Cache gefunden: {sample_name}")

    sample_info = manifest["sample_index"][sample_name]
    cache_file = Path(sample_info["cache_file"])
    if not cache_file.exists():
        raise FileNotFoundError(f"Albedo-Cache-Datei nicht gefunden: {cache_file}")

    with np.load(cache_file) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        grid_shape = tuple(int(v) for v in np.asarray(data["grid_size"]).tolist())
        patch_size = int(np.asarray(data["patch_size"]).reshape(-1)[0])

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    features_norm = (features / norms).astype(np.float32)

    return {
        "sample": sample_name,
        "features_norm": features_norm,
        "grid_shape": grid_shape,
        "patch_size": patch_size,
        "image_path": sample_info["image_path"],
    }


def _normalize_multilayer_patch_features(features_layers: np.ndarray) -> np.ndarray:
    norms_layer = np.linalg.norm(features_layers, axis=2, keepdims=True)
    norms_layer = np.maximum(norms_layer, 1e-8)
    per_layer_norm = (features_layers / norms_layer).astype(np.float32)
    concatenated = per_layer_norm.reshape(per_layer_norm.shape[0], -1).astype(np.float32)
    norms_concat = np.linalg.norm(concatenated, axis=1, keepdims=True)
    norms_concat = np.maximum(norms_concat, 1e-8)
    return (concatenated / norms_concat).astype(np.float32)


@st.cache_data(show_spinner=False)
def load_multilayer_sample_assets(experiment_dir_str: str, seed: int, sample_name: str) -> dict[str, Any]:
    manifest = load_multilayer_cache_manifest(experiment_dir_str, seed)
    if sample_name not in manifest["sample_index"]:
        raise KeyError(f"Sample nicht im Multi-Layer-Cache gefunden: {sample_name}")

    sample_info = manifest["sample_index"][sample_name]
    cache_file = Path(sample_info["cache_file"])
    if not cache_file.exists():
        raise FileNotFoundError(f"Multi-Layer-Cache-Datei nicht gefunden: {cache_file}")

    with np.load(cache_file) as data:
        features_layers = np.asarray(data["features_layers"], dtype=np.float32)
        grid_shape = tuple(int(v) for v in np.asarray(data["grid_size"]).tolist())
        patch_size = int(np.asarray(data["patch_size"]).reshape(-1)[0])
        layer_indices = tuple(int(v) for v in np.asarray(data["layer_indices"]).tolist())

    features_norm = _normalize_multilayer_patch_features(features_layers)

    return {
        "sample": sample_name,
        "features_norm": features_norm,
        "grid_shape": grid_shape,
        "patch_size": patch_size,
        "image_path": sample_info["image_path"],
        "layer_indices": layer_indices if layer_indices else tuple(sample_info["layer_indices"]),
        "num_layers": int(sample_info["num_layers"]),
        "feature_dim": int(sample_info["feature_dim_concat"]),
        "layer_dim": int(sample_info["layer_dim"]),
    }


def _load_normalized_features_for_reference(
    experiment_dir_str: str,
    seed: int,
    row: dict[str, Any],
    feature_source: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    class _SampleProxy:
        def __init__(self, image_path: str, feature_cache_path: str, anomaly_map_path: str) -> None:
            self.image_path = Path(image_path)
            self.feature_cache_path = Path(feature_cache_path)
            self.anomaly_map_path = Path(anomaly_map_path)

    if feature_source == "normal":
        proxy = _SampleProxy(
            image_path=row["image_path"],
            feature_cache_path=row["feature_cache_path"],
            anomaly_map_path=row["anomaly_map_path"],
        )
        features, grid_shape = load_patch_features(proxy)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return (features / norms).astype(np.float32), tuple(int(v) for v in grid_shape)

    if feature_source == "albedo":
        assets = load_albedo_sample_assets(experiment_dir_str, seed, row["sample"])
        return assets["features_norm"], tuple(int(v) for v in assets["grid_shape"])

    raise ValueError(f"Unsupported feature_source: {feature_source}")


def _build_positional_reference_impl(
    experiment_dir_str: str,
    seed: int,
    feature_source: str,
) -> dict[str, Any]:
    context = load_run_context(experiment_dir_str, seed)
    good_rows = [row for row in context["samples"] if int(row["image_label"]) == 0]
    if not good_rows:
        raise ValueError("Keine good-Bilder fuer die Positionsreferenz gefunden.")

    sum_valid: np.ndarray | None = None
    count_valid: np.ndarray | None = None
    sum_all: np.ndarray | None = None
    count_all: np.ndarray | None = None
    grid_shape_ref: tuple[int, int] | None = None

    class _SampleProxy:
        def __init__(self, image_path: str, feature_cache_path: str, anomaly_map_path: str) -> None:
            self.image_path = Path(image_path)
            self.feature_cache_path = Path(feature_cache_path)
            self.anomaly_map_path = Path(anomaly_map_path)

    for row in good_rows:
        proxy = _SampleProxy(
            image_path=row["image_path"],
            feature_cache_path=row["feature_cache_path"],
            anomaly_map_path=row["anomaly_map_path"],
        )
        features_norm, grid_shape = _load_normalized_features_for_reference(
            experiment_dir_str,
            seed,
            row,
            feature_source=feature_source,
        )
        score_grid = load_patch_scores(proxy)
        if grid_shape_ref is None:
            grid_shape_ref = tuple(int(v) for v in grid_shape)
            feature_dim = int(features_norm.shape[1])
            num_patches = int(features_norm.shape[0])
            sum_valid = np.zeros((num_patches, feature_dim), dtype=np.float64)
            count_valid = np.zeros((num_patches,), dtype=np.int32)
            sum_all = np.zeros((num_patches, feature_dim), dtype=np.float64)
            count_all = np.zeros((num_patches,), dtype=np.int32)
        elif tuple(int(v) for v in grid_shape) != grid_shape_ref:
            raise ValueError(f"Inkonsistentes Grid in Referenzbildern: {grid_shape} vs {grid_shape_ref}")

        features_norm = features_norm.astype(np.float64)
        flat_scores = score_grid.reshape(-1)
        valid_mask = flat_scores <= float(row["image_threshold"])

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
        "num_positions_with_valid_mask": int(np.sum(count_valid > 0)),
        "count_valid_min": int(count_valid.min()),
        "count_valid_max": int(count_valid.max()),
        "count_valid_mean": float(count_valid.mean()),
    }


@st.cache_data(show_spinner="Baue positionsbereinigte Referenz aus guten Bildern...")
def load_positional_reference(experiment_dir_str: str, seed: int) -> dict[str, Any]:
    return _build_positional_reference_impl(experiment_dir_str, seed, feature_source="normal")


@st.cache_data(show_spinner="Baue Albedo-Positionsreferenz aus guten Bildern...")
def load_albedo_positional_reference(experiment_dir_str: str, seed: int) -> dict[str, Any]:
    return _build_positional_reference_impl(experiment_dir_str, seed, feature_source="albedo")


@st.cache_data(show_spinner="Lade I-Relief-Gewichtung...")
def load_irelief_scale_weights(experiment_dir_str: str) -> dict[str, Any]:
    return _load_irelief_scale_weights_from_subdir(experiment_dir_str, IRELIEF_SUBDIR)


@st.cache_data(show_spinner="Lade Multi-Layer-I-Relief-Gewichtung...")
def load_multilayer_irelief_scale_weights(experiment_dir_str: str) -> dict[str, Any]:
    return _load_irelief_scale_weights_from_subdir(experiment_dir_str, IRELIEF_MULTILAYER_SUBDIR)


def _load_irelief_scale_weights_from_subdir(experiment_dir_str: str, subdir: str) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    scale_path = experiment_dir / subdir / "irelief_feature_scale_sqrt.npy"
    weight_path = experiment_dir / subdir / "irelief_feature_weights.npy"
    summary_path = experiment_dir / subdir / "summary.json"
    if not scale_path.exists():
        raise FileNotFoundError(f"I-Relief-Scale-Datei nicht gefunden: {scale_path}")

    scale = np.load(scale_path).astype(np.float32)
    if scale.ndim != 1:
        raise ValueError(f"I-Relief scale must be 1D, got shape {scale.shape}")

    result: dict[str, Any] = {
        "scale_sqrt": scale,
        "scale_path": str(scale_path),
        "feature_dim": int(scale.shape[0]),
        "subdir": subdir,
    }
    if weight_path.exists():
        weights = np.load(weight_path).astype(np.float32)
        if weights.ndim == 1 and weights.shape[0] == scale.shape[0]:
            result["weights"] = weights
    if summary_path.exists():
        result["summary_path"] = str(summary_path)
        result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return result


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


def _build_positional_pca_reference_impl(
    experiment_dir_str: str,
    seed: int,
    feature_source: str,
    num_components: int = 2,
    num_power_iterations: int = 3,
) -> dict[str, Any]:
    if num_components < 1:
        raise ValueError("num_components must be >= 1")

    if feature_source == "normal":
        mean_reference = load_positional_reference(experiment_dir_str, seed)
    elif feature_source == "albedo":
        mean_reference = load_albedo_positional_reference(experiment_dir_str, seed)
    else:
        raise ValueError(f"Unsupported feature_source: {feature_source}")
    mean_vectors = np.asarray(mean_reference["reference_vectors"], dtype=np.float32)
    context = load_run_context(experiment_dir_str, seed)
    good_rows = [row for row in context["samples"] if int(row["image_label"]) == 0]
    if not good_rows:
        raise ValueError("Keine good-Bilder fuer die PCA-Referenz gefunden.")

    num_positions, feature_dim = mean_vectors.shape
    rng = np.random.default_rng(0)
    init = rng.standard_normal((num_positions, num_components, feature_dim), dtype=np.float32)
    basis = _orthonormalize_patch_basis(init)

    class _SampleProxy:
        def __init__(self, image_path: str, feature_cache_path: str, anomaly_map_path: str) -> None:
            self.image_path = Path(image_path)
            self.feature_cache_path = Path(feature_cache_path)
            self.anomaly_map_path = Path(anomaly_map_path)

    valid_counts = np.zeros((num_positions,), dtype=np.int32)
    for _ in range(num_power_iterations):
        accum = np.zeros((num_positions, num_components, feature_dim), dtype=np.float64)
        valid_counts[:] = 0
        for row in good_rows:
            proxy = _SampleProxy(
                image_path=row["image_path"],
                feature_cache_path=row["feature_cache_path"],
                anomaly_map_path=row["anomaly_map_path"],
            )
            score_grid = load_patch_scores(proxy)
            features_norm, _ = _load_normalized_features_for_reference(
                experiment_dir_str,
                seed,
                row,
                feature_source=feature_source,
            )

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
        "num_positions_with_valid_mask": int(np.sum(valid_counts > 0)),
        "count_valid_min": int(valid_counts.min()),
        "count_valid_max": int(valid_counts.max()),
        "count_valid_mean": float(valid_counts.mean()),
    }


@st.cache_data(show_spinner="Baue kleinen PCA-Subspace aus guten Bildern...")
def load_positional_pca_reference(
    experiment_dir_str: str,
    seed: int,
    num_components: int = 2,
    num_power_iterations: int = 3,
) -> dict[str, Any]:
    return _build_positional_pca_reference_impl(
        experiment_dir_str,
        seed,
        feature_source="normal",
        num_components=num_components,
        num_power_iterations=num_power_iterations,
    )


@st.cache_data(show_spinner="Baue kleinen Albedo-PCA-Subspace aus guten Bildern...")
def load_albedo_positional_pca_reference(
    experiment_dir_str: str,
    seed: int,
    num_components: int = 2,
    num_power_iterations: int = 3,
) -> dict[str, Any]:
    return _build_positional_pca_reference_impl(
        experiment_dir_str,
        seed,
        feature_source="albedo",
        num_components=num_components,
        num_power_iterations=num_power_iterations,
    )


def _prepare_display_image(image_path: Path, smaller_edge_size: int, patch_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    resize_transform = transforms.Resize(
        size=smaller_edge_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    image = resize_transform(image)
    image_np = np.array(image)
    cropped_h = image_np.shape[0] - image_np.shape[0] % patch_size
    cropped_w = image_np.shape[1] - image_np.shape[1] % patch_size
    return image_np[:cropped_h, :cropped_w].copy()


def _grid_edges(image_shape: tuple[int, int, int], grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    rows, cols = grid_shape
    row_edges = np.linspace(0, h, rows + 1).round().astype(int)
    col_edges = np.linspace(0, w, cols + 1).round().astype(int)
    return row_edges, col_edges


def _draw_patch_grid(image_rgb: np.ndarray, grid_shape: tuple[int, int], color=(220, 220, 220)) -> np.ndarray:
    canvas = image_rgb.copy()
    row_edges, col_edges = _grid_edges(canvas.shape, grid_shape)
    for y in row_edges:
        cv2.line(canvas, (0, int(y)), (canvas.shape[1] - 1, int(y)), color, 1, lineType=cv2.LINE_AA)
    for x in col_edges:
        cv2.line(canvas, (int(x), 0), (int(x), canvas.shape[0] - 1), color, 1, lineType=cv2.LINE_AA)
    return canvas


def _resize_grid(grid: np.ndarray, image_shape: tuple[int, int, int], interpolation: int) -> np.ndarray:
    h, w = image_shape[:2]
    return cv2.resize(grid.astype(np.float32), (w, h), interpolation=interpolation)


def _blend(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(base_rgb, 1.0 - alpha, overlay_rgb, alpha, 0.0)


def _mark_query_patch(
    canvas: np.ndarray,
    grid_shape: tuple[int, int],
    row: int,
    col: int,
    label: str | None = None,
    box_color: tuple[int, int, int] = (255, 255, 255),
    marker_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    out = canvas.copy()
    row_edges, col_edges = _grid_edges(out.shape, grid_shape)
    y0, y1 = row_edges[row], row_edges[row + 1]
    x0, x1 = col_edges[col], col_edges[col + 1]
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), box_color, 2)
    cv2.drawMarker(
        out,
        (cx, cy),
        marker_color,
        markerType=cv2.MARKER_CROSS,
        markerSize=max(10, min(x1 - x0, y1 - y0) - 2),
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    if label:
        cv2.putText(
            out,
            label,
            (x0 + 3, min(y1 - 4, y0 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return out


def _render_marked_patches(
    image_rgb: np.ndarray,
    grid_shape: tuple[int, int],
    selected_patches: list[tuple[int, int]] | None,
) -> np.ndarray:
    canvas = _draw_patch_grid(image_rgb, grid_shape)
    if selected_patches:
        for index, (row, col) in enumerate(selected_patches, start=1):
            canvas = _mark_query_patch(
                canvas,
                grid_shape,
                row,
                col,
                label=str(index),
            )
    return canvas


def _render_anomaly_overlay(
    image_rgb: np.ndarray,
    score_grid: np.ndarray,
    marked_patches: list[tuple[int, int]] | None,
) -> np.ndarray:
    resized = _resize_grid(score_grid, image_rgb.shape, cv2.INTER_LINEAR)
    resized = resized - resized.min()
    denom = float(resized.max())
    if denom > 0:
        resized = resized / denom
    heat = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = _blend(image_rgb, heat, 0.45)
    overlay = _draw_patch_grid(overlay, score_grid.shape)
    if marked_patches:
        for index, (row, col) in enumerate(marked_patches, start=1):
            overlay = _mark_query_patch(overlay, score_grid.shape, row, col, label=str(index))
    return overlay


def _render_similarity_overlay(
    image_rgb: np.ndarray,
    sim_grid: np.ndarray,
    marked_patches: list[tuple[int, int]] | None,
    top_matches: list[tuple[int, int]] | None = None,
    valid_mask_grid: np.ndarray | None = None,
) -> np.ndarray:
    resized = _resize_grid(sim_grid, image_rgb.shape, cv2.INTER_LINEAR)
    valid_mask_resized = None
    if valid_mask_grid is not None:
        valid_mask_resized = _resize_grid(valid_mask_grid.astype(np.float32), image_rgb.shape, cv2.INTER_NEAREST) >= 0.5
        finite_values = resized[valid_mask_resized]
    else:
        finite_values = resized.reshape(-1)

    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size > 0:
        sim_min = float(finite_values.min())
        sim_max = float(finite_values.max())
    else:
        sim_min = 0.0
        sim_max = 0.0

    if sim_max > sim_min:
        normalized = (resized - sim_min) / (sim_max - sim_min)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = np.zeros_like(resized, dtype=np.float32)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = _blend(image_rgb, heat, 0.50)
    if valid_mask_resized is None:
        overlay = blended
    else:
        overlay = image_rgb.copy()
        overlay[valid_mask_resized] = blended[valid_mask_resized]
        overlay[~valid_mask_resized] = (image_rgb[~valid_mask_resized] * 0.35).astype(np.uint8)
    overlay = _draw_patch_grid(overlay, sim_grid.shape)
    if marked_patches:
        for index, (row, col) in enumerate(marked_patches, start=1):
            overlay = _mark_query_patch(overlay, sim_grid.shape, row, col, label=str(index))

    if top_matches:
        row_edges, col_edges = _grid_edges(overlay.shape, sim_grid.shape)
        for rank, (row, col) in enumerate(top_matches, start=1):
            y0, y1 = row_edges[row], row_edges[row + 1]
            x0, x1 = col_edges[col], col_edges[col + 1]
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
            cv2.putText(
                overlay,
                str(rank),
                (x0 + 3, min(y1 - 4, y0 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
    return overlay


def _pixel_to_patch(
    image_shape: tuple[int, int, int],
    grid_shape: tuple[int, int],
    click_payload: dict[str, Any],
) -> tuple[int, int] | None:
    if not click_payload:
        return None

    width = int(click_payload.get("width", image_shape[1]))
    height = int(click_payload.get("height", image_shape[0]))
    if width <= 0 or height <= 0:
        return None

    x = float(click_payload.get("x", 0.0)) * image_shape[1] / width
    y = float(click_payload.get("y", 0.0)) * image_shape[0] / height
    x = min(max(x, 0.0), image_shape[1] - 1)
    y = min(max(y, 0.0), image_shape[0] - 1)

    rows, cols = grid_shape
    row_edges = np.linspace(0, image_shape[0], rows + 1).round().astype(int)
    col_edges = np.linspace(0, image_shape[1], cols + 1).round().astype(int)
    row = int(np.searchsorted(row_edges, y, side="right") - 1)
    col = int(np.searchsorted(col_edges, x, side="right") - 1)
    row = min(max(row, 0), rows - 1)
    col = min(max(col, 0), cols - 1)
    return row, col


def _patch_crop(image_rgb: np.ndarray, grid_shape: tuple[int, int], row: int, col: int, pad: int = 0) -> np.ndarray:
    row_edges, col_edges = _grid_edges(image_rgb.shape, grid_shape)
    y0, y1 = row_edges[row], row_edges[row + 1]
    x0, x1 = col_edges[col], col_edges[col + 1]
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(image_rgb.shape[0], y1 + pad)
    x1 = min(image_rgb.shape[1], x1 + pad)
    return image_rgb[y0:y1, x0:x1].copy()


def _resize_keep_aspect(image_rgb: np.ndarray, target_height: int) -> np.ndarray:
    if image_rgb.shape[0] == target_height:
        return image_rgb.copy()
    scale = target_height / image_rgb.shape[0]
    target_width = max(1, int(round(image_rgb.shape[1] * scale)))
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _make_titled_panel(image_rgb: np.ndarray, title: str, target_height: int | None = None) -> np.ndarray:
    content = image_rgb.copy()
    if target_height is not None:
        content = _resize_keep_aspect(content, target_height)
    title_bar_h = 34
    panel = np.full((content.shape[0] + title_bar_h, content.shape[1], 3), 24, dtype=np.uint8)
    panel[title_bar_h:, :, :] = content
    cv2.putText(
        panel,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return panel


def _stack_row(images: list[np.ndarray], pad: int = 12, bg_color: int = 18) -> np.ndarray:
    if not images:
        raise ValueError("stack_row requires at least one image")
    max_h = max(img.shape[0] for img in images)
    total_w = sum(img.shape[1] for img in images) + pad * (len(images) - 1)
    canvas = np.full((max_h, total_w, 3), bg_color, dtype=np.uint8)
    x = 0
    for img in images:
        y = (max_h - img.shape[0]) // 2
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
        x += img.shape[1] + pad
    return canvas


def _stack_col(images: list[np.ndarray], pad: int = 12, bg_color: int = 18) -> np.ndarray:
    if not images:
        raise ValueError("stack_col requires at least one image")
    max_w = max(img.shape[1] for img in images)
    total_h = sum(img.shape[0] for img in images) + pad * (len(images) - 1)
    canvas = np.full((total_h, max_w, 3), bg_color, dtype=np.uint8)
    y = 0
    for img in images:
        x = (max_w - img.shape[1]) // 2
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
        y += img.shape[0] + pad
    return canvas


def _build_match_strip(
    target_assets: dict[str, Any],
    matches: list[tuple[int, int, float]],
    max_items: int = 6,
) -> np.ndarray | None:
    tiles: list[np.ndarray] = []
    for rank, (row, col, similarity) in enumerate(matches[:max_items], start=1):
        crop = _patch_crop(target_assets["image_rgb"], target_assets["grid_shape"], row, col, pad=0)
        title = f"#{rank} r={row} c={col} sim={similarity:.3f}"
        tiles.append(_make_titled_panel(crop, title, target_height=140))
    if not tiles:
        return None
    return _stack_row(tiles, pad=10, bg_color=18)


def _export_current_view(
    experiment_dir_str: str,
    seed: int,
    query_assets: dict[str, Any],
    target_assets: dict[str, Any],
    query_patches: list[tuple[int, int]],
    query_details: list[dict[str, Any]],
    query_panel: np.ndarray,
    similarity_panel: np.ndarray,
    query_strip: np.ndarray | None,
    target_anomaly_panel: np.ndarray | None,
    match_strip: np.ndarray | None,
    top_rows: list[dict[str, Any]],
    sim_grid: np.ndarray,
    target_filter_mode: str,
    feature_mode: str,
    feature_source_mode: str,
    fusion_alpha_normal: float,
    fusion_alpha_albedo: float,
) -> tuple[Path, Path]:
    export_dir = Path(experiment_dir_str).resolve() / EXPORT_SUBDIR / f"seed={seed}"
    export_dir.mkdir(parents=True, exist_ok=True)

    query_panel_sheet = _make_titled_panel(
        query_panel,
        f"Query: {query_assets['sample']} | patches={len(query_patches)}",
        target_height=420,
    )
    similarity_panel_sheet = _make_titled_panel(
        similarity_panel,
        f"Target Similarity: {target_assets['sample']}",
        target_height=420,
    )
    upper_row = _stack_row([query_panel_sheet, similarity_panel_sheet], pad=12, bg_color=18)

    lower_panels = [
    ]
    if query_strip is not None:
        lower_panels.append(query_strip)
    if target_anomaly_panel is not None:
        lower_panels.append(_make_titled_panel(target_anomaly_panel, "Target Anomaly Map", target_height=180))
    if match_strip is not None:
        lower_panels.append(match_strip)
    lower_row = _stack_row(lower_panels, pad=12, bg_color=18)

    sheet = _stack_col([upper_row, lower_row], pad=12, bg_color=18)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = (
        f"{timestamp}__q_{_safe_name(query_assets['sample'])}__n{len(query_patches):02d}"
        f"__t_{_safe_name(target_assets['sample'])}__src_{_safe_name(feature_source_mode)}"
    )
    png_path = export_dir / f"{stem}.png"
    json_path = export_dir / f"{stem}.json"

    Image.fromarray(sheet).save(png_path)
    metadata = {
        "timestamp": timestamp,
        "experiment_dir": str(Path(experiment_dir_str).resolve()),
        "seed": int(seed),
        "query_sample": query_assets["sample"],
        "target_sample": target_assets["sample"],
        "query_patches": query_details,
        "query_group": query_assets["evaluation_group"],
        "target_group": target_assets["evaluation_group"],
        "query_weight_mode": "softmax",
        "target_filter_mode": target_filter_mode,
        "feature_mode": feature_mode,
        "feature_source_mode": feature_source_mode,
        "fusion_alpha_normal": float(fusion_alpha_normal),
        "fusion_alpha_albedo": float(fusion_alpha_albedo),
        "query_image_score": float(query_assets["image_score"]),
        "query_image_threshold": float(query_assets["image_threshold"]),
        "target_image_score": float(target_assets["image_score"]),
        "target_image_threshold": float(target_assets["image_threshold"]),
        "grid_shape_query": list(query_assets["grid_shape"]),
        "grid_shape_target": list(target_assets["grid_shape"]),
        "similarity_min": float(sim_grid.min()),
        "similarity_max": float(sim_grid.max()),
        "top_matches": top_rows,
        "png_path": str(png_path),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_path = export_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    return png_path, json_path


@st.cache_data(show_spinner=False)
def load_sample_assets(experiment_dir_str: str, seed: int, sample_name: str) -> dict[str, Any]:
    context = load_run_context(experiment_dir_str, seed)
    sample_info = context["sample_index"][sample_name]
    run_args = context["run_args"]

    class _SampleProxy:
        def __init__(self, image_path: str, feature_cache_path: str, anomaly_map_path: str) -> None:
            self.image_path = Path(image_path)
            self.feature_cache_path = Path(feature_cache_path)
            self.anomaly_map_path = Path(anomaly_map_path)

    proxy = _SampleProxy(
        image_path=sample_info["image_path"],
        feature_cache_path=sample_info["feature_cache_path"],
        anomaly_map_path=sample_info["anomaly_map_path"],
    )

    features, grid_shape = load_patch_features(proxy)
    score_grid = load_patch_scores(proxy)
    with np.load(proxy.feature_cache_path) as data:
        patch_size = int(data["patch_size"].item())
    display_image = _prepare_display_image(proxy.image_path, int(run_args["resolution"]), patch_size)

    if features.shape[0] != grid_shape[0] * grid_shape[1]:
        raise ValueError(
            f"Feature/grid mismatch for {sample_name}: features={features.shape}, grid={grid_shape}"
        )
    if tuple(score_grid.shape) != tuple(grid_shape):
        raise ValueError(
            f"Score grid/grid mismatch for {sample_name}: scores={score_grid.shape}, grid={grid_shape}"
        )

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    features_norm = (features / norms).astype(np.float32)

    return {
        "sample": sample_name,
        "image_rgb": display_image,
        "grid_shape": tuple(int(v) for v in grid_shape),
        "score_grid": score_grid.astype(np.float32),
        "features_norm": features_norm,
        "patch_size": patch_size,
        "evaluation_group": sample_info["evaluation_group"],
        "image_score": float(sample_info["image_score"]),
        "image_threshold": float(sample_info["image_threshold"]),
    }


def _softmax_query_weights(anomaly_scores: np.ndarray) -> np.ndarray:
    if anomaly_scores.size == 0:
        raise ValueError("No anomaly scores provided for query weighting")
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


def _query_feature_from_selected(
    features_norm: np.ndarray,
    score_grid: np.ndarray,
    grid_shape: tuple[int, int],
    selected_patches: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch_features: list[np.ndarray] = []
    anomaly_scores: list[float] = []
    for row, col in selected_patches:
        idx = row * grid_shape[1] + col
        patch_features.append(features_norm[idx])
        anomaly_scores.append(float(score_grid[row, col]))
    feature_matrix = np.stack(patch_features, axis=0).astype(np.float32)
    anomaly_array = np.array(anomaly_scores, dtype=np.float32)
    weights = _softmax_query_weights(anomaly_array)
    combined = (feature_matrix * weights[:, None]).sum(axis=0)
    combined_norm = np.linalg.norm(combined)
    if combined_norm <= 1e-8:
        combined = feature_matrix.mean(axis=0)
        combined_norm = np.linalg.norm(combined)
    combined = (combined / max(float(combined_norm), 1e-8)).astype(np.float32)
    return combined, anomaly_array, weights


def _debiased_features(
    features_norm: np.ndarray,
    reference_vectors: np.ndarray,
) -> np.ndarray:
    if features_norm.shape != reference_vectors.shape:
        raise ValueError(
            f"Feature/reference shape mismatch: {features_norm.shape} vs {reference_vectors.shape}"
        )
    projection = np.sum(features_norm * reference_vectors, axis=1, keepdims=True)
    residual = features_norm - projection * reference_vectors
    residual_norm = np.linalg.norm(residual, axis=1, keepdims=True)
    fallback_mask = residual_norm.squeeze(-1) <= 1e-8
    residual_norm = np.maximum(residual_norm, 1e-8)
    debiased = (residual / residual_norm).astype(np.float32)
    if np.any(fallback_mask):
        debiased[fallback_mask] = features_norm[fallback_mask]
    return debiased


def _pca_subspace_debiased_features(
    features_norm: np.ndarray,
    mean_reference_vectors: np.ndarray,
    pca_basis: np.ndarray,
) -> np.ndarray:
    if features_norm.shape != mean_reference_vectors.shape:
        raise ValueError(
            f"Feature/reference shape mismatch: {features_norm.shape} vs {mean_reference_vectors.shape}"
        )
    if features_norm.shape[0] != pca_basis.shape[0] or features_norm.shape[1] != pca_basis.shape[2]:
        raise ValueError(
            f"Feature/PCA basis shape mismatch: {features_norm.shape} vs {pca_basis.shape}"
        )

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


def _apply_feature_mode(
    features_norm: np.ndarray,
    feature_mode: str,
    reference_vectors: np.ndarray | None = None,
    pca_basis: np.ndarray | None = None,
) -> np.ndarray:
    if feature_mode == "raw":
        return features_norm
    if feature_mode == "debiased":
        if reference_vectors is None:
            raise ValueError("reference_vectors fehlen fuer debiased mode")
        return _debiased_features(features_norm, reference_vectors)
    if feature_mode.startswith("pca_subspace_k"):
        if reference_vectors is None or pca_basis is None:
            raise ValueError(f"reference_vectors oder pca_basis fehlen fuer {feature_mode}")
        return _pca_subspace_debiased_features(features_norm, reference_vectors, pca_basis)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def _apply_irelief_reweight(
    features_norm: np.ndarray,
    scale_sqrt_weights: np.ndarray,
) -> np.ndarray:
    if features_norm.shape[1] != scale_sqrt_weights.shape[0]:
        raise ValueError(
            f"I-Relief scale dimension mismatch: features={features_norm.shape} vs scale={scale_sqrt_weights.shape}"
        )
    weighted = features_norm * scale_sqrt_weights[None, :]
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    fallback_mask = norms.squeeze(-1) <= 1e-8
    norms = np.maximum(norms, 1e-8)
    weighted = (weighted / norms).astype(np.float32)
    if np.any(fallback_mask):
        weighted[fallback_mask] = features_norm[fallback_mask]
    return weighted


def _similarity_grid_from_query(
    query_feature: np.ndarray,
    target_features_norm: np.ndarray,
    target_grid_shape: tuple[int, int],
) -> np.ndarray:
    sims = target_features_norm @ query_feature
    return sims.reshape(target_grid_shape).astype(np.float32)


def _top_matches(
    sim_grid: np.ndarray,
    top_k: int,
    exclude_patches: list[tuple[int, int]] | None = None,
    valid_mask: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    rows, cols = sim_grid.shape
    flat = sim_grid.reshape(-1).astype(np.float32)
    exclude_indices: set[int] = set()
    if exclude_patches:
        exclude_indices = {row * cols + col for row, col in exclude_patches}
    if valid_mask is not None:
        valid_flat = valid_mask.reshape(-1).astype(bool)
        flat = flat.copy()
        flat[~valid_flat] = -np.inf
    order = np.argsort(-flat)
    matches: list[tuple[int, int, float]] = []
    for idx in order:
        if not np.isfinite(flat[int(idx)]):
            continue
        if int(idx) in exclude_indices:
            continue
        row = int(idx // cols)
        col = int(idx % cols)
        matches.append((row, col, float(flat[idx])))
        if len(matches) >= top_k:
            break
    return matches


def main() -> None:
    st.set_page_config(page_title="AnomalyDINO Similarity Explorer", layout="wide")
    st.title("AnomalyDINO Similarity Explorer")

    default_experiment_dir = _parse_experiment_dir_from_argv()
    experiment_dir_str = st.sidebar.text_input("Experiment Directory", value=default_experiment_dir)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=999, value=0, step=1)

    try:
        context = load_run_context(experiment_dir_str, int(seed))
    except Exception as exc:
        st.error(f"Run konnte nicht geladen werden: {exc}")
        return

    sample_rows = context["samples"]
    groups = ["all"] + sorted({row["evaluation_group"] for row in sample_rows})

    query_group = st.sidebar.selectbox("Query Group", groups, index=0)
    if query_group == "all":
        query_rows = sample_rows
    else:
        query_rows = [row for row in sample_rows if row["evaluation_group"] == query_group]
    query_samples = [row["sample"] for row in query_rows]
    if not query_samples:
        st.warning("Keine Query-Samples fuer den gewaehlten Filter gefunden.")
        return

    query_sample = st.sidebar.selectbox("Query Sample", query_samples, index=0)
    target_equals_query = st.sidebar.checkbox("Target = Query", value=True)

    if target_equals_query:
        target_sample = query_sample
    else:
        target_group = st.sidebar.selectbox("Target Group", groups, index=0)
        if target_group == "all":
            target_rows = sample_rows
        else:
            target_rows = [row for row in sample_rows if row["evaluation_group"] == target_group]
        target_samples = [row["sample"] for row in target_rows]
        if not target_samples:
            st.warning("Keine Target-Samples fuer den gewaehlten Filter gefunden.")
            return
        default_target_index = target_samples.index(query_sample) if query_sample in target_samples else 0
        target_sample = st.sidebar.selectbox("Target Sample", target_samples, index=default_target_index)

    top_k = int(st.sidebar.slider("Top aehnliche Patches", min_value=1, max_value=12, value=6))
    feature_source_mode = st.sidebar.selectbox(
        "Feature Source",
        options=["normal", "albedo", "fused"],
        format_func=lambda value: {
            "normal": "Normalmap DINOv3",
            "albedo": "Albedo DINOv3",
            "fused": "Fusion: Normalmap + Albedo",
        }[value],
        index=0,
    )
    fusion_alpha_normal = 1.0
    fusion_alpha_albedo = 0.0
    if feature_source_mode == "fused":
        fusion_alpha_normal = float(
            st.sidebar.slider("Normalmap weight", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
        )
        fusion_alpha_albedo = float(1.0 - fusion_alpha_normal)
    elif feature_source_mode == "albedo":
        fusion_alpha_normal = 0.0
        fusion_alpha_albedo = 1.0
    show_anomaly_overlay = st.sidebar.checkbox("Target Anomaly-Map anzeigen", value=True)
    show_top_match_boxes = st.sidebar.checkbox("Top-Matches im Similarity-Overlay markieren", value=True)
    anomalous_only_target = st.sidebar.checkbox(
        "Nur target patches ueber Bildthreshold vergleichen",
        value=False,
    )
    positions_debiased = st.sidebar.checkbox(
        "Positionsbias mit good-reference entfernen",
        value=False,
    )
    pca_subspace_k = int(st.sidebar.selectbox(
        "PCA-Subspace-Debias",
        options=[0, 2, 3],
        format_func=lambda value: {
            0: "aus",
            2: "k=2",
            3: "k=3",
        }[value],
        index=0,
    ))
    pca_subspace_debiased = pca_subspace_k > 0
    use_irelief_patch_reweight = st.sidebar.checkbox(
        "I-Relief-Reweight auf Normalmap anwenden",
        value=False,
    )
    use_multilayer_irelief_normal = st.sidebar.checkbox(
        "Multi-Layer (Layer 1-12) + I-Relief auf Normalmap verwenden",
        value=False,
    )
    if pca_subspace_debiased:
        positions_debiased = False

    try:
        query_assets = load_sample_assets(experiment_dir_str, int(seed), query_sample)
        target_assets = query_assets if target_sample == query_sample else load_sample_assets(
            experiment_dir_str,
            int(seed),
            target_sample,
        )
    except Exception as exc:
        st.error(f"Sample konnte nicht geladen werden: {exc}")
        return

    query_albedo_assets = None
    target_albedo_assets = None
    if feature_source_mode != "normal":
        try:
            query_albedo_assets = load_albedo_sample_assets(experiment_dir_str, int(seed), query_sample)
            target_albedo_assets = (
                query_albedo_assets
                if target_sample == query_sample
                else load_albedo_sample_assets(experiment_dir_str, int(seed), target_sample)
            )
        except Exception as exc:
            st.error(f"Albedo-Sample konnte nicht geladen werden: {exc}")
            return
        if tuple(query_albedo_assets["grid_shape"]) != tuple(query_assets["grid_shape"]):
            st.error(
                f"Grid-Mismatch Query normal/albedo: {query_assets['grid_shape']} vs {query_albedo_assets['grid_shape']}"
            )
            return
        if tuple(target_albedo_assets["grid_shape"]) != tuple(target_assets["grid_shape"]):
            st.error(
                f"Grid-Mismatch Target normal/albedo: {target_assets['grid_shape']} vs {target_albedo_assets['grid_shape']}"
            )
            return

    query_multilayer_assets = None
    target_multilayer_assets = None
    if use_multilayer_irelief_normal and feature_source_mode in ("normal", "fused"):
        try:
            query_multilayer_assets = load_multilayer_sample_assets(experiment_dir_str, int(seed), query_sample)
            target_multilayer_assets = (
                query_multilayer_assets
                if target_sample == query_sample
                else load_multilayer_sample_assets(experiment_dir_str, int(seed), target_sample)
            )
        except Exception as exc:
            st.error(f"Multi-Layer-Normalmap-Sample konnte nicht geladen werden: {exc}")
            return
        if tuple(query_multilayer_assets["grid_shape"]) != tuple(query_assets["grid_shape"]):
            st.error(
                "Grid-Mismatch Query normal/multilayer: "
                f"{query_assets['grid_shape']} vs {query_multilayer_assets['grid_shape']}"
            )
            return
        if tuple(target_multilayer_assets["grid_shape"]) != tuple(target_assets["grid_shape"]):
            st.error(
                "Grid-Mismatch Target normal/multilayer: "
                f"{target_assets['grid_shape']} vs {target_multilayer_assets['grid_shape']}"
            )
            return

    normal_reference_info = None
    normal_pca_reference_info = None
    albedo_reference_info = None
    albedo_pca_reference_info = None
    irelief_info = None
    irelief_scale_sqrt = None
    multilayer_irelief_info = None
    multilayer_irelief_scale_sqrt = None
    feature_mode = "raw"
    normal_feature_mode = feature_mode
    if pca_subspace_debiased:
        feature_mode = f"pca_subspace_k{pca_subspace_k}"
        if feature_source_mode in ("normal", "fused"):
            if use_multilayer_irelief_normal:
                normal_feature_mode = "raw"
            else:
                normal_feature_mode = feature_mode
                try:
                    normal_pca_reference_info = load_positional_pca_reference(
                        experiment_dir_str,
                        int(seed),
                        num_components=pca_subspace_k,
                        num_power_iterations=3,
                    )
                except Exception as exc:
                    st.error(f"Normalmap-PCA-Subspace-Referenz konnte nicht geladen werden: {exc}")
                    return
                normal_reference_info = normal_pca_reference_info["mean_reference"]
        if feature_source_mode in ("albedo", "fused"):
            try:
                albedo_pca_reference_info = load_albedo_positional_pca_reference(
                    experiment_dir_str,
                    int(seed),
                    num_components=pca_subspace_k,
                    num_power_iterations=3,
                )
            except Exception as exc:
                st.error(f"Albedo-PCA-Subspace-Referenz konnte nicht geladen werden: {exc}")
                return
            albedo_reference_info = albedo_pca_reference_info["mean_reference"]
    elif positions_debiased:
        feature_mode = "debiased"
        if feature_source_mode in ("normal", "fused"):
            if use_multilayer_irelief_normal:
                normal_feature_mode = "raw"
            else:
                normal_feature_mode = feature_mode
                try:
                    normal_reference_info = load_positional_reference(experiment_dir_str, int(seed))
                except Exception as exc:
                    st.error(f"Normalmap-Positionsreferenz konnte nicht geladen werden: {exc}")
                    return
        if feature_source_mode in ("albedo", "fused"):
            try:
                albedo_reference_info = load_albedo_positional_reference(experiment_dir_str, int(seed))
            except Exception as exc:
                st.error(f"Albedo-Positionsreferenz konnte nicht geladen werden: {exc}")
                return
    else:
        normal_feature_mode = "raw"

    use_irelief_patch_reweight_effective = use_irelief_patch_reweight and not use_multilayer_irelief_normal
    if use_irelief_patch_reweight_effective:
        try:
            irelief_info = load_irelief_scale_weights(experiment_dir_str)
            irelief_scale_sqrt = np.asarray(irelief_info["scale_sqrt"], dtype=np.float32)
        except Exception as exc:
            st.error(f"I-Relief-Gewichtung konnte nicht geladen werden: {exc}")
            return
    if use_multilayer_irelief_normal and feature_source_mode in ("normal", "fused"):
        try:
            multilayer_irelief_info = load_multilayer_irelief_scale_weights(experiment_dir_str)
            multilayer_irelief_scale_sqrt = np.asarray(multilayer_irelief_info["scale_sqrt"], dtype=np.float32)
        except Exception as exc:
            st.error(f"Multi-Layer-I-Relief-Gewichtung konnte nicht geladen werden: {exc}")
            return

    feature_mode_display = feature_mode
    if use_multilayer_irelief_normal and feature_source_mode == "normal":
        feature_mode_display = "raw + multilayer_l1to12_irelief"
    elif use_multilayer_irelief_normal and feature_source_mode == "fused":
        feature_mode_display = f"normal:raw+multilayer_l1to12_irelief | albedo:{feature_mode}"
    elif use_irelief_patch_reweight_effective and feature_source_mode in ("normal", "fused"):
        feature_mode_display = f"{feature_mode_display} + irelief_patch_reweight"

    normal_reference_vectors = None
    normal_pca_basis = None
    if normal_reference_info is not None:
        normal_reference_vectors = np.asarray(normal_reference_info["reference_vectors"], dtype=np.float32)
    if normal_pca_reference_info is not None:
        normal_pca_basis = np.asarray(normal_pca_reference_info["basis"], dtype=np.float32)

    albedo_reference_vectors = None
    albedo_pca_basis = None
    if albedo_reference_info is not None:
        albedo_reference_vectors = np.asarray(albedo_reference_info["reference_vectors"], dtype=np.float32)
    if albedo_pca_reference_info is not None:
        albedo_pca_basis = np.asarray(albedo_pca_reference_info["basis"], dtype=np.float32)

    query_normal_features_for_similarity = None
    target_normal_features_for_similarity = None
    if feature_source_mode in ("normal", "fused"):
        query_normal_source_features = (
            query_multilayer_assets["features_norm"]
            if use_multilayer_irelief_normal
            else query_assets["features_norm"]
        )
        target_normal_source_features = (
            query_normal_source_features
            if target_sample == query_sample
            else (
                target_multilayer_assets["features_norm"]
                if use_multilayer_irelief_normal
                else target_assets["features_norm"]
            )
        )
        query_normal_features_for_similarity = _apply_feature_mode(
            query_normal_source_features,
            feature_mode=normal_feature_mode,
            reference_vectors=normal_reference_vectors,
            pca_basis=normal_pca_basis,
        )
        if use_multilayer_irelief_normal:
            assert multilayer_irelief_scale_sqrt is not None
            query_normal_features_for_similarity = _apply_irelief_reweight(
                query_normal_features_for_similarity,
                multilayer_irelief_scale_sqrt,
            )
        elif use_irelief_patch_reweight_effective:
            assert irelief_scale_sqrt is not None
            query_normal_features_for_similarity = _apply_irelief_reweight(
                query_normal_features_for_similarity,
                irelief_scale_sqrt,
            )
        target_normal_features_for_similarity = (
            query_normal_features_for_similarity
            if target_sample == query_sample
            else _apply_feature_mode(
                target_normal_source_features,
                feature_mode=normal_feature_mode,
                reference_vectors=normal_reference_vectors,
                pca_basis=normal_pca_basis,
            )
        )
        if target_sample != query_sample:
            if use_multilayer_irelief_normal:
                assert multilayer_irelief_scale_sqrt is not None
                target_normal_features_for_similarity = _apply_irelief_reweight(
                    target_normal_features_for_similarity,
                    multilayer_irelief_scale_sqrt,
                )
            elif use_irelief_patch_reweight_effective:
                assert irelief_scale_sqrt is not None
                target_normal_features_for_similarity = _apply_irelief_reweight(
                    target_normal_features_for_similarity,
                    irelief_scale_sqrt,
                )

    query_albedo_features_for_similarity = None
    target_albedo_features_for_similarity = None
    if feature_source_mode in ("albedo", "fused"):
        assert query_albedo_assets is not None and target_albedo_assets is not None
        query_albedo_features_for_similarity = _apply_feature_mode(
            query_albedo_assets["features_norm"],
            feature_mode=feature_mode,
            reference_vectors=albedo_reference_vectors,
            pca_basis=albedo_pca_basis,
        )
        target_albedo_features_for_similarity = (
            query_albedo_features_for_similarity
            if target_sample == query_sample
            else _apply_feature_mode(
                target_albedo_assets["features_norm"],
                feature_mode=feature_mode,
                reference_vectors=albedo_reference_vectors,
                pca_basis=albedo_pca_basis,
            )
        )

    query_state_key = f"query_patches::{query_sample}"
    if query_state_key not in st.session_state:
        st.session_state[query_state_key] = []

    left, right = st.columns([1.1, 1.1])

    with left:
        st.subheader(f"Query: {query_sample}")
        st.caption(
            f"group={query_assets['evaluation_group']} | grid={query_assets['grid_shape'][0]}x{query_assets['grid_shape'][1]} | "
            f"image_score={query_assets['image_score']:.5f} | threshold={query_assets['image_threshold']:.5f}"
        )
        source_caption = f"feature_source={feature_source_mode} | feature_mode={feature_mode_display}"
        if feature_source_mode == "fused":
            source_caption += f" | normal_w={fusion_alpha_normal:.2f} | albedo_w={fusion_alpha_albedo:.2f}"
        st.caption(source_caption)
        reference_bits: list[str] = []
        if use_multilayer_irelief_normal and feature_source_mode in ("normal", "fused") and query_multilayer_assets is not None:
            reference_bits.append(f"multilayer_layers={query_multilayer_assets['num_layers']}")
            reference_bits.append(f"multilayer_dim={query_multilayer_assets['feature_dim']}")
        if feature_source_mode in ("normal", "fused") and normal_reference_info is not None:
            reference_bits.append(f"normal_ref_images={normal_reference_info['num_good_images']}")
            reference_bits.append(f"normal_valid_ref_mean={normal_reference_info['count_valid_mean']:.1f}")
        if feature_source_mode in ("albedo", "fused") and albedo_reference_info is not None:
            reference_bits.append(f"albedo_ref_images={albedo_reference_info['num_good_images']}")
            reference_bits.append(f"albedo_valid_ref_mean={albedo_reference_info['count_valid_mean']:.1f}")
        if use_multilayer_irelief_normal and feature_source_mode in ("normal", "fused") and multilayer_irelief_info is not None:
            reference_bits.append(f"multilayer_irelief_dim={multilayer_irelief_info['feature_dim']}")
        elif use_irelief_patch_reweight_effective and feature_source_mode in ("normal", "fused") and irelief_info is not None:
            reference_bits.append(f"irelief_dim={irelief_info['feature_dim']}")
        if reference_bits:
            st.caption(" | ".join(reference_bits))
        if use_multilayer_irelief_normal and (positions_debiased or pca_subspace_debiased) and feature_source_mode in ("normal", "fused"):
            st.caption("Multi-Layer-I-Relief auf Normalmap laeuft bewusst ohne Positionskorrektur; Albedo bleibt unveraendert.")
        if use_irelief_patch_reweight and use_multilayer_irelief_normal and feature_source_mode in ("normal", "fused"):
            st.caption("Das alte Layer-12-I-Relief-Reweight ist in diesem Modus ohne Wirkung.")
        if use_irelief_patch_reweight and feature_source_mode == "albedo":
            st.caption("I-Relief ist im reinen Albedo-Modus ohne Wirkung.")
        query_panel = _render_marked_patches(
            query_assets["image_rgb"],
            query_assets["grid_shape"],
            st.session_state[query_state_key],
        )
        click_payload = streamlit_image_coordinates(
            query_panel,
            key=f"similarity_click_{query_sample}",
            use_column_width="always",
            cursor="crosshair",
        )
        if click_payload:
            click_token = click_payload.get("unix_time")
            last_click_key = f"last_similarity_click::{query_sample}"
            if click_token != st.session_state.get(last_click_key):
                patch_coords = _pixel_to_patch(query_panel.shape, query_assets["grid_shape"], click_payload)
                if patch_coords is not None:
                    selected = [tuple(item) for item in st.session_state.get(query_state_key, [])]
                    if patch_coords in selected:
                        selected = [item for item in selected if item != patch_coords]
                    else:
                        selected.append(patch_coords)
                    st.session_state[query_state_key] = selected
                st.session_state[last_click_key] = click_token
                st.rerun()
        clear_col, hint_col = st.columns([0.35, 0.65])
        with clear_col:
            if st.button("Clear Query", use_container_width=True):
                st.session_state[query_state_key] = []
                st.rerun()
        with hint_col:
            st.caption("Klick toggelt einen Query-Patch im linken Bild.")

    query_patches = [tuple(item) for item in st.session_state.get(query_state_key, [])]
    if not query_patches:
        st.info("Setze links einen oder mehrere Query-Punkte, um die Similarity-Map zu berechnen.")
        return

    sim_grid_normal = None
    sim_grid_albedo = None
    query_anomaly_scores = None
    query_weights = None

    if feature_source_mode in ("normal", "fused"):
        assert query_normal_features_for_similarity is not None and target_normal_features_for_similarity is not None
        query_feature_normal, query_anomaly_scores, query_weights = _query_feature_from_selected(
            query_normal_features_for_similarity,
            query_assets["score_grid"],
            query_assets["grid_shape"],
            query_patches,
        )
        sim_grid_normal = _similarity_grid_from_query(
            query_feature_normal,
            target_normal_features_for_similarity,
            target_assets["grid_shape"],
        )

    if feature_source_mode in ("albedo", "fused"):
        assert query_albedo_features_for_similarity is not None and target_albedo_features_for_similarity is not None
        query_feature_albedo, albedo_query_anomaly_scores, albedo_query_weights = _query_feature_from_selected(
            query_albedo_features_for_similarity,
            query_assets["score_grid"],
            query_assets["grid_shape"],
            query_patches,
        )
        if query_anomaly_scores is None:
            query_anomaly_scores = albedo_query_anomaly_scores
        if query_weights is None:
            query_weights = albedo_query_weights
        sim_grid_albedo = _similarity_grid_from_query(
            query_feature_albedo,
            target_albedo_features_for_similarity,
            target_assets["grid_shape"],
        )

    if feature_source_mode == "normal":
        assert sim_grid_normal is not None
        sim_grid = sim_grid_normal.astype(np.float32)
    elif feature_source_mode == "albedo":
        assert sim_grid_albedo is not None
        sim_grid = sim_grid_albedo.astype(np.float32)
    else:
        assert sim_grid_normal is not None and sim_grid_albedo is not None
        sim_grid = (
            fusion_alpha_normal * sim_grid_normal + fusion_alpha_albedo * sim_grid_albedo
        ).astype(np.float32)

    assert query_anomaly_scores is not None and query_weights is not None
    target_valid_mask = None
    if anomalous_only_target:
        target_valid_mask = target_assets["score_grid"] >= float(target_assets["image_threshold"])
        if not bool(np.any(target_valid_mask)):
            st.warning("Im Target-Bild liegen keine Patches ueber dem Bildthreshold.")
            return
    exclude_patches = query_patches if target_sample == query_sample else None
    matches = _top_matches(
        sim_grid,
        top_k=top_k,
        exclude_patches=exclude_patches,
        valid_mask=target_valid_mask,
    )
    marked_matches = [(row, col) for row, col, _ in matches] if show_top_match_boxes else None
    marked_target_patches = query_patches if target_sample == query_sample else None

    with right:
        st.subheader(f"Target Similarity: {target_sample}")
        st.caption(
            f"group={target_assets['evaluation_group']} | grid={target_assets['grid_shape'][0]}x{target_assets['grid_shape'][1]} | "
            f"image_score={target_assets['image_score']:.5f} | threshold={target_assets['image_threshold']:.5f}"
        )
        target_caption = f"feature_source={feature_source_mode} | feature_mode={feature_mode_display}"
        if feature_source_mode == "fused":
            target_caption += f" | normal_w={fusion_alpha_normal:.2f} | albedo_w={fusion_alpha_albedo:.2f}"
        st.caption(target_caption)
        similarity_panel = _render_similarity_overlay(
            target_assets["image_rgb"],
            sim_grid,
            marked_patches=marked_target_patches,
            top_matches=marked_matches,
            valid_mask_grid=target_valid_mask,
        )
        st.image(similarity_panel, caption="Cosine Similarity Overlay", use_container_width=True)
        if target_sample == query_sample:
            st.caption(
                f"Query patches: {len(query_patches)} | "
                f"target filter={'threshold' if anomalous_only_target else 'all'} | "
                f"sim range=[{float(sim_grid.min()):.4f}, {float(sim_grid.max()):.4f}]"
            )
        elif matches:
            best_row, best_col, best_sim = matches[0]
            st.caption(
                f"Query patches: {len(query_patches)} | "
                f"target filter={'threshold' if anomalous_only_target else 'all'} | "
                f"best target match: row={best_row}, col={best_col}, sim={best_sim:.4f} | "
                f"sim range=[{float(sim_grid.min()):.4f}, {float(sim_grid.max()):.4f}]"
            )

    lower_left, lower_mid, lower_right = st.columns([0.9, 1.1, 1.1])

    query_details: list[dict[str, Any]] = []
    query_tiles: list[np.ndarray] = []
    for index, ((row, col), anomaly_score, weight) in enumerate(zip(query_patches, query_anomaly_scores, query_weights), start=1):
        query_details.append(
            {
                "rank": index,
                "row": int(row),
                "col": int(col),
                "anomaly_score": float(anomaly_score),
                "weight": float(weight),
            }
        )
        crop = _patch_crop(query_assets["image_rgb"], query_assets["grid_shape"], row, col, pad=0)
        query_tiles.append(
            _make_titled_panel(
                crop,
                f"Q{index} r={row} c={col} a={anomaly_score:.3f} w={weight:.3f}",
                target_height=140,
            )
        )
    query_strip = _stack_row(query_tiles, pad=10, bg_color=18) if query_tiles else None

    with lower_left:
        st.subheader("Query Patches")
        st.dataframe(query_details, use_container_width=True, hide_index=True)

    anomaly_panel = None
    with lower_mid:
        if show_anomaly_overlay:
            st.subheader("Target Anomaly Map")
            anomaly_panel = _render_anomaly_overlay(
                target_assets["image_rgb"],
                target_assets["score_grid"],
                marked_target_patches,
            )
            st.image(anomaly_panel, caption="AnomalyDINO Anomaly Overlay", use_container_width=True)

    top_rows: list[dict[str, Any]] = []
    with lower_right:
        st.subheader("Top Matches in Target")
        for rank, (row, col, similarity) in enumerate(matches, start=1):
            row_info = {
                "rank": rank,
                "sample": target_sample,
                "row": row,
                "col": col,
                "cosine_similarity": similarity,
                "anomaly_score": float(target_assets["score_grid"][row, col]),
            }
            if sim_grid_normal is not None:
                row_info["cosine_similarity_normal"] = float(sim_grid_normal[row, col])
            if sim_grid_albedo is not None:
                row_info["cosine_similarity_albedo"] = float(sim_grid_albedo[row, col])
            top_rows.append(row_info)
        st.dataframe(top_rows, use_container_width=True, hide_index=True)

    if query_strip is not None:
        st.subheader("Query Patch Crops")
        st.image(query_strip, use_container_width=True)

    match_strip = _build_match_strip(target_assets, matches, max_items=min(top_k, 6))
    if matches:
        st.subheader("Top Match Crops")
        cols = st.columns(min(top_k, 6))
        for idx, (row, col, similarity) in enumerate(matches[:6]):
            with cols[idx]:
                crop = _patch_crop(target_assets["image_rgb"], target_assets["grid_shape"], row, col, pad=0)
                st.image(
                    crop,
                    caption=f"#{idx + 1} r={row} c={col}\ncos={similarity:.3f}",
                    use_container_width=True,
                )

    export_col, info_col = st.columns([0.35, 0.65])
    with export_col:
        if st.button("Export Current View", use_container_width=True):
            png_path, json_path = _export_current_view(
                experiment_dir_str=experiment_dir_str,
                seed=int(seed),
                query_assets=query_assets,
                target_assets=target_assets,
                query_patches=query_patches,
                query_details=query_details,
                query_panel=query_panel,
                similarity_panel=similarity_panel,
                query_strip=query_strip,
                target_anomaly_panel=anomaly_panel,
                match_strip=match_strip,
                top_rows=top_rows,
                sim_grid=sim_grid,
                target_filter_mode="threshold" if anomalous_only_target else "all",
                feature_mode=feature_mode_display,
                feature_source_mode=feature_source_mode,
                fusion_alpha_normal=fusion_alpha_normal,
                fusion_alpha_albedo=fusion_alpha_albedo,
            )
            st.session_state["similarity_last_export"] = {
                "png": str(png_path),
                "json": str(json_path),
            }
            st.success(f"Export geschrieben: {png_path.name}")
    with info_col:
        export_dir = Path(experiment_dir_str).resolve() / EXPORT_SUBDIR / f"seed={int(seed)}"
        st.caption(f"Export-Ordner: {export_dir}")
        last_export = st.session_state.get("similarity_last_export")
        if last_export:
            st.code(f"PNG : {last_export['png']}\nJSON: {last_export['json']}", language="text")


if __name__ == "__main__":
    main()
