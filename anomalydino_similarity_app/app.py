from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from PIL import Image
from joblib import load as joblib_load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from streamlit_image_coordinates import streamlit_image_coordinates
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from component_memory_bank.data_io import load_patch_features, load_patch_scores, load_run_args, load_run_samples
from extract_labeled_roi_overthreshold_multilayer_maxminmean_features import (
    aggregate_maxminmean_per_layer,
)
from extract_labeled_roi_overthreshold_multilayer_maxstd_features import (
    select_overlap_threshold_patches,
)
from extract_labeled_roi_toppercent_multilayer_softmax_patch_features import (
    DEFAULT_EXPERIMENT_DIR as ROI_CLASSIFIER_DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LABELS_FILE as ROI_CLASSIFIER_DEFAULT_LABELS_FILE,
    DEFAULT_ROI_METADATA_CSV as ROI_CLASSIFIER_DEFAULT_ROI_METADATA_CSV,
    build_multilayer_run_context as build_roi_classifier_run_context,
    load_labels_table as load_roi_classifier_labels_table,
    load_multilayer_cache,
    load_roi_table as load_roi_classifier_roi_table,
    prepare_labeled_roi_table as prepare_roi_classifier_labeled_roi_table,
)
from fit_roi_irelief_cosine import (
    build_weighted_feature_set,
    estimate_sigma,
    fit_irelief_cosine,
    l2_normalize_rows,
    pairwise_cosine_distance,
)
from sweep_roi_topkpatchcount_irelief_fixedk32_rbf import (
    build_features_for_patch_count as build_top1patch_classifier_features,
    build_sample_cache as build_top1patch_classifier_sample_cache,
)
from show_heatmap import (
    boxes_overlap,
    estimate_local_background,
    hysteresis_component,
    infer_patch_multiple,
    merge_without_valley,
    patch_box_to_image_box,
    peak_candidates,
    region_box,
    region_mass,
    region_peak_details,
    strength_key,
)
from src.backbones import get_model
from src.post_eval import mean_top1p
from src.utils import augment_image, dists2map


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)

EXPORT_SUBDIR = "similarity_query_exports"
ALBEDO_CACHE_SUBDIR = "albedo_patch_feature_cache"
MULTILAYER_CACHE_SUBDIR = "patch_feature_cache_multilayer_l1to12"
TOP10PCT_CLASSIFIER_FEATURE_SUBDIR = "final_all_boxes_top10pct_multilayer_irelief_fixedk32_rbf"
TOP1PATCH_CLASSIFIER_FEATURE_SUBDIR = "final_all_boxes_top1patch_multilayer_irelief_fixedk32_rbf"
BORUTA_COMPONENT_CLASSIFIER_FEATURE_SUBDIR = (
    "final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf"
)
IRELIEF_SUBDIR = (
    "roi_top10pct_centerinbox_pca2_softmax_patch_features_labeled"
    r"\irelief_cosine_weighted_features"
)
IRELIEF_MULTILAYER_SUBDIR = (
    "roi_top10pct_centerinbox_multilayer_l1to12_softmax_patch_features_labeled"
    r"\irelief_cosine_weighted_features"
)
ROI_CLASSIFIER_ROI_METADATA_RELATIVE = ROI_CLASSIFIER_DEFAULT_ROI_METADATA_CSV.relative_to(
    ROI_CLASSIFIER_DEFAULT_EXPERIMENT_DIR
)
COMPONENT_TEST_VIEW_KEY = "component_test_view_results"
ANOMALY_DETECTION_SETTINGS_VIEW_KEY = "anomaly_detection_settings_results"
ANOMALY_DETECTION_UPLOAD_NONCE_KEY = "anomaly_detection_upload_nonce"
ANOMALY_DETECTION_UPLOAD_FLASH_KEY = "anomaly_detection_upload_flash"
ANOMALY_DETECTION_CONFIRM_FLASH_KEY = "anomaly_detection_confirm_flash"
COMPONENT_TEST_UPLOAD_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
APP_SETTINGS_SUBDIR = "app_settings"
ANOMALY_DETECTION_SETTINGS_SUBDIR = "anomaly_detection"
ANOMALY_DETECTION_REFERENCE_SUBDIR = "reference_images"
ANOMALY_DETECTION_TEST_GOOD_SUBDIR = "test_good"
ANOMALY_DETECTION_TEST_BAD_SUBDIR = "test_bad"
ANOMALY_DETECTION_CONFIRMED_CONFIG_FILENAME = "confirmed_threshold_config.json"
COMPONENT_TEST_ROI_SETTINGS = {
    "high_prominence_ratio": 0.5,
    "low_prominence_ratio": 0.2,
    "background_ring_inner": 2,
    "background_ring_outer": 5,
    "min_region_patches": 1,
    "min_prominence": 0.02,
    "min_region_mass": 0.05,
    "block_boxes": True,
    "merge_gap_patches": 3,
    "merge_bridge_ratio": 0.1,
    "max_boxes_per_image": None,
}


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


def _compute_label_rect(
    image_shape: tuple[int, int, int] | tuple[int, int],
    text: str,
    x: int,
    y: int,
    font_scale: float,
    thickness: int,
) -> tuple[int, int, int, int, int]:
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + 8
    box_h = text_h + baseline + 8
    max_x1 = max(0, image_w - box_w - 1)
    box_x1 = int(min(max(0, x), max_x1))
    box_x2 = min(image_w - 1, box_x1 + box_w)
    min_y2 = box_h
    max_y2 = max(min_y2, image_h - 1)
    box_y2 = int(min(max(min_y2, y), max_y2))
    box_y1 = max(0, box_y2 - box_h)
    return box_x1, box_y1, box_x2, box_y2, baseline


def _rects_intersect(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
) -> bool:
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _rect_overlap_area(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
) -> int:
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    return int(inter_w * inter_h)


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    box_x1, box_y1, box_x2, box_y2, baseline = _compute_label_rect(
        image.shape,
        text,
        x,
        y,
        font_scale,
        thickness,
    )
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(
        image,
        text,
        (box_x1 + 4, box_y2 - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_component_roi_label(
    image: np.ndarray,
    text: str,
    box_x0: int,
    box_y0: int,
    box_x1: int,
    box_y1: int,
    color: tuple[int, int, int],
    blocked_rects: list[tuple[int, int, int, int]],
    occupied_label_rects: list[tuple[int, int, int, int]],
) -> None:
    font_scale = 0.45
    thickness = 1
    margin = 4
    box_center_x = int((box_x0 + box_x1) / 2)

    candidate_origins = [
        (box_x0 + 2, box_y0 - margin),
        (box_center_x, box_y0 - margin),
        (box_x1 + margin, box_y0 + 18),
        (box_x0 + 2, box_y1 + 22),
        (box_center_x, box_y1 + 22),
        (box_x1 + margin, box_y1 + 22),
        (max(0, box_x0 - 140), box_y0 + 18),
        (max(0, box_x0 - 140), box_y1 + 22),
    ]

    candidates: list[tuple[int, int, int, int, int]] = []
    for cand_x, cand_y in candidate_origins:
        rect_x1, rect_y1, rect_x2, rect_y2, baseline = _compute_label_rect(
            image.shape,
            text,
            cand_x,
            cand_y,
            font_scale,
            thickness,
        )
        rect = (rect_x1, rect_y1, rect_x2, rect_y2)
        overlaps_roi = any(_rects_intersect(rect, blocked_rect) for blocked_rect in blocked_rects)
        overlaps_label = any(_rects_intersect(rect, label_rect) for label_rect in occupied_label_rects)
        if not overlaps_roi and not overlaps_label:
            draw_label(image, text, cand_x, cand_y, color, font_scale, thickness)
            occupied_label_rects.append(rect)
            return
        penalty = sum(_rect_overlap_area(rect, blocked_rect) for blocked_rect in blocked_rects)
        penalty += sum(_rect_overlap_area(rect, label_rect) for label_rect in occupied_label_rects)
        candidates.append((penalty, cand_x, cand_y, baseline, len(candidates)))

    if not candidates:
        draw_label(image, text, box_x0 + 2, box_y1 + 22, color, font_scale, thickness)
        fallback_rect = _compute_label_rect(image.shape, text, box_x0 + 2, box_y1 + 22, font_scale, thickness)[:4]
        occupied_label_rects.append(fallback_rect)
        return

    _, cand_x, cand_y, _, _ = min(candidates, key=lambda item: (item[0], item[4]))
    draw_label(image, text, cand_x, cand_y, color, font_scale, thickness)
    chosen_rect = _compute_label_rect(image.shape, text, cand_x, cand_y, font_scale, thickness)[:4]
    occupied_label_rects.append(chosen_rect)


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


@st.cache_data(show_spinner="Lade Top10%-Klassifikator-Featureauswahl...")
def load_top10pct_classifier_feature_selection(experiment_dir_str: str) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    selection_path = experiment_dir / TOP10PCT_CLASSIFIER_FEATURE_SUBDIR / "selected_topk_features.csv"
    model_info_path = experiment_dir / TOP10PCT_CLASSIFIER_FEATURE_SUBDIR / "model_info.json"
    summary_path = experiment_dir / TOP10PCT_CLASSIFIER_FEATURE_SUBDIR / "summary.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Top10%-Klassifikator-Featuredatei nicht gefunden: {selection_path}")

    selection_df = pd.read_csv(selection_path)
    if "feature_index" not in selection_df.columns:
        raise ValueError(f"'feature_index' fehlt in {selection_path}")
    feature_indices = selection_df["feature_index"].astype(np.int32).to_numpy()
    if feature_indices.ndim != 1 or feature_indices.size <= 0:
        raise ValueError(f"Ungueltige Featureauswahl in {selection_path}")

    result: dict[str, Any] = {
        "selection_path": str(selection_path),
        "feature_indices": feature_indices,
        "num_selected_features": int(feature_indices.size),
        "subdir": TOP10PCT_CLASSIFIER_FEATURE_SUBDIR,
    }
    if model_info_path.exists():
        result["model_info_path"] = str(model_info_path)
        result["model_info"] = json.loads(model_info_path.read_text(encoding="utf-8"))
    if summary_path.exists():
        result["summary_path"] = str(summary_path)
        result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return result


def _classifier_file_signature(*paths: Path) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        stat = path.stat()
        signature.append((str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def _directory_file_signature(directory: Path) -> tuple[tuple[str, int, int], ...]:
    if not directory.exists():
        return tuple()
    signature: list[tuple[str, int, int]] = []
    for path in sorted((item for item in directory.iterdir() if item.is_file()), key=lambda item: item.name.lower()):
        stat = path.stat()
        signature.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def _file_signature_from_paths(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in sorted(paths, key=lambda item: item.name.lower()):
        stat = path.stat()
        signature.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def _anomaly_detection_settings_root(experiment_dir_str: str) -> Path:
    return Path(experiment_dir_str).resolve() / APP_SETTINGS_SUBDIR / ANOMALY_DETECTION_SETTINGS_SUBDIR


def _anomaly_detection_object_dir(experiment_dir_str: str, object_name: str) -> Path:
    return _anomaly_detection_settings_root(experiment_dir_str) / _safe_name(object_name)


def _anomaly_detection_category_dir(experiment_dir_str: str, object_name: str, category: str) -> Path:
    return _anomaly_detection_object_dir(experiment_dir_str, object_name) / category


def _anomaly_detection_confirmed_config_path(experiment_dir_str: str, object_name: str) -> Path:
    return _anomaly_detection_object_dir(experiment_dir_str, object_name) / ANOMALY_DETECTION_CONFIRMED_CONFIG_FILENAME


def _list_saved_anomaly_detection_images(
    experiment_dir_str: str,
    object_name: str,
    category: str,
) -> list[Path]:
    directory = _anomaly_detection_category_dir(experiment_dir_str, object_name, category)
    allowed_exts = {f".{ext.lower()}" for ext in COMPONENT_TEST_UPLOAD_EXTENSIONS}
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed_exts],
        key=lambda path: path.name.lower(),
    )


def _save_uploaded_images_to_category(
    experiment_dir_str: str,
    object_name: str,
    category: str,
    uploaded_files: list[Any],
) -> list[str]:
    target_dir = _anomaly_detection_category_dir(experiment_dir_str, object_name, category)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_names: list[str] = []
    used_names: set[str] = {
        existing_file.name.lower()
        for existing_file in target_dir.iterdir()
        if existing_file.is_file()
    }
    for uploaded_file in uploaded_files:
        raw_name = Path(str(uploaded_file.name)).name
        stem = _safe_name(Path(raw_name).stem) or "image"
        suffix = Path(raw_name).suffix.lower() or ".png"
        candidate_name = f"{stem}{suffix}"
        counter = 2
        while candidate_name.lower() in used_names:
            candidate_name = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate_name.lower())
        (target_dir / candidate_name).write_bytes(uploaded_file.getvalue())
        saved_names.append(candidate_name)
    return saved_names


def load_anomaly_detection_confirmed_config(experiment_dir_str: str, object_name: str) -> dict[str, Any] | None:
    config_path = _anomaly_detection_confirmed_config_path(experiment_dir_str, object_name)
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_confirmed_reference_paths(
    experiment_dir_str: str,
    object_name: str,
    confirmed_config: dict[str, Any],
) -> list[Path]:
    reference_dir = _anomaly_detection_category_dir(
        experiment_dir_str,
        object_name,
        ANOMALY_DETECTION_REFERENCE_SUBDIR,
    )
    reference_filenames = [str(name) for name in confirmed_config.get("reference_filenames", [])]
    if not reference_filenames:
        raise ValueError("Die bestaetigte Konfiguration enthaelt keine Referenzdateien.")

    resolved_paths: list[Path] = []
    missing_files: list[str] = []
    for filename in reference_filenames:
        candidate_path = reference_dir / filename
        if not candidate_path.exists():
            missing_files.append(filename)
        else:
            resolved_paths.append(candidate_path)
    if missing_files:
        raise FileNotFoundError(
            "Bestaetigte Referenzbilder fehlen: " + ", ".join(missing_files)
        )
    return resolved_paths


def _save_anomaly_detection_confirmed_config(
    experiment_dir_str: str,
    object_name: str,
    threshold: float,
    reference_paths: list[Path],
    evaluation_summary: dict[str, Any],
    live_metrics: dict[str, Any],
) -> Path:
    config_path = _anomaly_detection_confirmed_config_path(experiment_dir_str, object_name)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_name": object_name,
        "threshold": float(threshold),
        "reference_filenames": [path.name for path in reference_paths],
        "num_reference_images": int(len(reference_paths)),
        "evaluation_summary": evaluation_summary,
        "selected_metrics": {
            "precision": float(live_metrics["precision"]),
            "recall": float(live_metrics["recall"]),
            "f1": float(live_metrics["f1"]),
            "accuracy": float(live_metrics["accuracy"]),
            "tp": int(live_metrics["tp"]),
            "fp": int(live_metrics["fp"]),
            "tn": int(live_metrics["tn"]),
            "fn": int(live_metrics["fn"]),
        },
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
    }
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


@st.cache_resource(show_spinner="Baue Referenzbank aus gespeicherten Referenzbildern...")
def load_anomaly_detection_reference_bank(
    experiment_dir_str: str,
    seed: int,
    object_name: str,
    reference_image_paths: tuple[str, ...],
    reference_signature: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    _ = reference_signature
    if not reference_image_paths:
        raise ValueError("Es wurden keine gespeicherten Referenzbilder gefunden.")

    live_config = load_component_test_live_config(experiment_dir_str, int(seed))
    backbone = load_component_test_backbone(experiment_dir_str)
    model = backbone["model"]

    use_rotation = bool(live_config["rotation_by_object"].get(object_name, False))
    use_ref_masking = bool(live_config["mask_ref_images"] and live_config["masking_by_object"].get(object_name, False))

    ref_feature_rows: list[np.ndarray] = []
    resolved_paths = [Path(path_str) for path_str in reference_image_paths]
    for image_path in resolved_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Referenzbild konnte nicht geladen werden: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        variants = augment_image(image_rgb) if use_rotation else [image_rgb]
        for image_variant in variants:
            image_tensor, grid_shape = model.prepare_image(image_variant)
            last_layer_features = np.asarray(model.extract_features(image_tensor), dtype=np.float32)
            valid_mask = model.compute_background_mask(
                last_layer_features,
                grid_shape,
                threshold=10,
                masking_type=use_ref_masking,
            )
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if np.any(valid_mask):
                ref_feature_rows.append(last_layer_features[valid_mask])

    if not ref_feature_rows:
        raise ValueError("Aus den gespeicherten Referenzbildern konnten keine gueltigen Referenzpatches gebildet werden.")

    ref_features = np.concatenate(ref_feature_rows, axis=0).astype(np.float32)
    if str(live_config["knn_metric"]) == "L2_normalized":
        ref_features = _normalize_feature_rows(ref_features)

    return {
        "object_name": object_name,
        "ref_features": ref_features,
        "knn_metric": str(live_config["knn_metric"]),
        "k_neighbors": int(live_config["k_neighbors"]),
        "num_ref_images": int(len(reference_image_paths)),
        "num_ref_patches": int(ref_features.shape[0]),
    }


def _score_image_for_live_anomaly_detection(
    experiment_dir_str: str,
    seed: int,
    object_name: str,
    image_rgb: np.ndarray,
    reference_bank: dict[str, Any],
) -> dict[str, Any]:
    live_config = load_component_test_live_config(experiment_dir_str, int(seed))
    backbone = load_component_test_backbone(experiment_dir_str)
    model = backbone["model"]

    image_tensor, grid_shape = model.prepare_image(image_rgb)
    anomaly_features = np.asarray(model.extract_features(image_tensor), dtype=np.float32)
    masking_enabled = bool(live_config["masking_by_object"].get(object_name, False))
    valid_mask = np.asarray(
        model.compute_background_mask(anomaly_features, grid_shape, threshold=10, masking_type=masking_enabled),
        dtype=bool,
    )

    output_distances = np.zeros(anomaly_features.shape[0], dtype=np.float32)
    if np.any(valid_mask):
        patch_distances = _nearest_reference_distances(
            anomaly_features[valid_mask],
            reference_bank["ref_features"],
            metric=str(reference_bank["knn_metric"]),
            k_neighbors=int(reference_bank["k_neighbors"]),
        )
        output_distances[valid_mask] = patch_distances
    score_grid = output_distances.reshape(tuple(int(v) for v in grid_shape)).astype(np.float32)

    aggregation_statistics = str(live_config["aggregation_statistics"])
    if aggregation_statistics == "meantop1p":
        image_score = float(mean_top1p(output_distances.flatten()))
    elif aggregation_statistics == "max_patch_distance":
        image_score = float(np.max(output_distances))
    else:
        image_score = float(np.max(dists2map(score_grid, image_rgb.shape)))

    return {
        "image_score": float(image_score),
        "score_grid": score_grid,
        "grid_shape": tuple(int(v) for v in grid_shape),
        "num_valid_patches": int(np.count_nonzero(valid_mask)),
    }


def _compute_anomaly_detection_metrics(
    scores: np.ndarray,
    true_is_bad: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    true_is_bad = np.asarray(true_is_bad, dtype=bool).reshape(-1)
    if scores.shape[0] != true_is_bad.shape[0]:
        raise ValueError(f"Scores/Labels passen nicht zusammen: {scores.shape} vs {true_is_bad.shape}")
    pred_is_bad = scores >= float(threshold)

    tp = int(np.sum(pred_is_bad & true_is_bad))
    fp = int(np.sum(pred_is_bad & ~true_is_bad))
    tn = int(np.sum(~pred_is_bad & ~true_is_bad))
    fn = int(np.sum(~pred_is_bad & true_is_bad))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / scores.shape[0]) if scores.shape[0] > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_bad": int(np.sum(pred_is_bad)),
        "predicted_good": int(np.sum(~pred_is_bad)),
    }


def _sweep_anomaly_detection_thresholds(
    scores: np.ndarray,
    true_is_bad: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    true_is_bad = np.asarray(true_is_bad, dtype=bool).reshape(-1)
    if scores.size == 0:
        raise ValueError("Keine Scores fuer den Threshold-Sweep vorhanden.")

    unique_scores = np.unique(scores)
    candidate_thresholds = [float(score) for score in unique_scores]
    candidate_thresholds.append(float(np.nextafter(unique_scores.max(), np.inf)))

    metric_rows = [_compute_anomaly_detection_metrics(scores, true_is_bad, threshold) for threshold in candidate_thresholds]
    sweep_df = pd.DataFrame(metric_rows).sort_values("threshold").reset_index(drop=True)
    best_row = (
        sweep_df.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True])
        .iloc[0]
        .to_dict()
    )
    return sweep_df, best_row


def _sync_state_value(target_key: str, source_key: str) -> None:
    if source_key in st.session_state:
        st.session_state[target_key] = st.session_state[source_key]


@st.cache_resource(show_spinner="Baue Top1-Patch-Klassifikator-Endmodell...")
def load_top1patch_classifier_model(
    experiment_dir_str: str,
    seed: int,
    cache_signature: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    selection_path = experiment_dir / TOP1PATCH_CLASSIFIER_FEATURE_SUBDIR / "selected_topk_features.csv"
    model_info_path = experiment_dir / TOP1PATCH_CLASSIFIER_FEATURE_SUBDIR / "model_info.json"
    summary_path = experiment_dir / TOP1PATCH_CLASSIFIER_FEATURE_SUBDIR / "summary.json"
    classifier_path = experiment_dir / TOP1PATCH_CLASSIFIER_FEATURE_SUBDIR / "classifier_pipeline.joblib"
    scale_sqrt_path = experiment_dir / TOP1PATCH_CLASSIFIER_FEATURE_SUBDIR / "irelief_feature_scale_sqrt.npy"
    if not selection_path.exists():
        raise FileNotFoundError(f"Top1-Patch-Klassifikator-Featuredatei nicht gefunden: {selection_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Top1-Patch-Klassifikator-Modell nicht gefunden: {classifier_path}")
    if not scale_sqrt_path.exists():
        raise FileNotFoundError(f"Top1-Patch-I-Relief-Scale-Datei nicht gefunden: {scale_sqrt_path}")

    selection_df = pd.read_csv(selection_path)
    if "feature_index" not in selection_df.columns:
        raise ValueError(f"'feature_index' fehlt in {selection_path}")
    selected_indices = selection_df["feature_index"].astype(np.int32).to_numpy()
    fixed_k = int(selected_indices.size)
    _ = cache_signature
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    model_info = json.loads(model_info_path.read_text(encoding="utf-8")) if model_info_path.exists() else {}
    model = joblib_load(classifier_path)
    scale_sqrt = np.load(scale_sqrt_path).astype(np.float32)

    selection_mode = str(summary.get("selection_mode", model_info.get("selection_mode", "overlap")))
    patch_count = int(summary.get("patch_count", 1))
    return {
        "model": model,
        "selected_indices": selected_indices,
        "scale_sqrt": scale_sqrt,
        "selection_mode": selection_mode,
        "patch_count": int(patch_count),
        "fixed_k": int(selected_indices.size),
        "sigma": float(summary.get("sigma", np.nan)),
        "irelief_iterations": int(summary.get("irelief_iterations", 0)),
    }


@st.cache_data(show_spinner=False)
def load_component_test_roi_table(experiment_dir_str: str) -> pd.DataFrame:
    experiment_dir = Path(experiment_dir_str).resolve()
    roi_metadata_csv = experiment_dir / ROI_CLASSIFIER_ROI_METADATA_RELATIVE
    table = load_roi_classifier_roi_table(roi_metadata_csv)
    table["sample"] = table["sample"].astype(str).str.replace("\\", "/", regex=False)
    return table


def load_component_test_live_config(
    experiment_dir_str: str,
    seed: int,
    train_good_signatures: tuple[tuple[str, tuple[tuple[str, int, int], ...]], ...] | None = None,
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    run_args = load_run_args(experiment_dir)
    run_samples = load_run_samples(experiment_dir, seed=int(seed))
    preprocess_path = experiment_dir / "preprocess.yaml"
    preprocess_data = yaml.safe_load(preprocess_path.read_text(encoding="utf-8")) if preprocess_path.exists() else {}
    data_root = Path(str(run_args.get("data_root", ""))).resolve()
    _ = train_good_signatures

    object_names = sorted({sample.object_name for sample in run_samples})
    threshold_by_object: dict[str, float] = {}
    for object_name in object_names:
        object_thresholds = [round(float(sample.image_threshold), 8) for sample in run_samples if sample.object_name == object_name]
        if not object_thresholds:
            continue
        counts = pd.Series(object_thresholds).value_counts()
        threshold_by_object[object_name] = float(counts.index[0])

    stored_ref_images_by_object: dict[str, list[str]] = {}
    current_train_good_images_by_object: dict[str, list[str]] = {}
    for object_name in object_names:
        ref_json = experiment_dir / f"reference_images_{object_name}_seed={int(seed)}.json"
        if ref_json.exists():
            stored_ref_images_by_object[object_name] = list(json.loads(ref_json.read_text(encoding="utf-8")))
        elif len(object_names) == 1 and run_args.get("ref_image_names"):
            stored_ref_images_by_object[object_name] = [str(item) for item in run_args["ref_image_names"]]
        else:
            stored_ref_images_by_object[object_name] = []

        train_good_dir = data_root / object_name / "train" / "good"
        if train_good_dir.exists():
            current_train_good_images_by_object[object_name] = sorted(
                [path.name for path in train_good_dir.iterdir() if path.is_file()]
            )
        else:
            current_train_good_images_by_object[object_name] = []

    return {
        "run_args": run_args,
        "object_names": object_names,
        "threshold_by_object": threshold_by_object,
        "stored_ref_images_by_object": stored_ref_images_by_object,
        "current_train_good_images_by_object": current_train_good_images_by_object,
        "masking_by_object": {
            str(key): bool(value) for key, value in dict(preprocess_data.get("masking", {})).items()
        },
        "rotation_by_object": {
            str(key): bool(value) for key, value in dict(preprocess_data.get("rotation", {})).items()
        },
        "mask_ref_images": bool(run_args.get("mask_ref_images", False)),
        "aggregation_statistics": str(run_args.get("aggregation_statistics", "max_anomaly_map")),
        "knn_metric": str(run_args.get("knn_metric", "L2_normalized")),
        "k_neighbors": int(run_args.get("k_neighbors", 1)),
        "data_root": str(data_root),
        "resolution": int(run_args.get("resolution", 688)),
        "model_name": str(run_args.get("model_name", "")),
        "backbone_weights": run_args.get("backbone_weights"),
    }


@st.cache_resource(show_spinner="Lade Backbone fuer frische Inferenz...")
def load_component_test_backbone(experiment_dir_str: str) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    run_args = load_run_args(experiment_dir)
    model_name = str(run_args.get("model_name", ""))
    if not model_name.startswith("dinov3"):
        raise ValueError(
            f"Der externe Bauteil-Test erwartet aktuell DINOv3-Multilayer-Features, erhalten: {model_name!r}"
        )

    requested_device = str(run_args.get("device", "cpu"))
    device = requested_device if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = get_model(
        model_name,
        device=device,
        smaller_edge_size=int(run_args.get("resolution", 688)),
        weights_path=run_args.get("backbone_weights"),
    )
    return {
        "model": model,
        "device": device,
        "model_name": model_name,
        "patch_size": int(model.patch_size),
    }


def _normalize_feature_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (features / norms).astype(np.float32)


@st.cache_resource(show_spinner="Baue Referenzbank fuer frische Inferenz...")
def load_component_test_reference_bank(
    experiment_dir_str: str,
    seed: int,
    object_name: str,
    train_good_signature: tuple[tuple[str, int, int], ...] | None = None,
) -> dict[str, Any]:
    live_config = load_component_test_live_config(
        experiment_dir_str,
        int(seed),
        train_good_signatures=((object_name, train_good_signature or tuple()),),
    )
    backbone = load_component_test_backbone(experiment_dir_str)
    model = backbone["model"]
    _ = train_good_signature

    ref_images = list(live_config["current_train_good_images_by_object"].get(object_name, []))
    if not ref_images:
        raise ValueError(f"Keine Referenzbilder fuer Objekt {object_name!r} gefunden.")

    data_root = Path(live_config["data_root"]).resolve()
    ref_dir = data_root / object_name / "train" / "good"
    if not ref_dir.exists():
        raise FileNotFoundError(f"Train/Good-Ordner nicht gefunden: {ref_dir}")

    use_rotation = bool(live_config["rotation_by_object"].get(object_name, False))
    use_ref_masking = bool(live_config["mask_ref_images"] and live_config["masking_by_object"].get(object_name, False))

    ref_feature_rows: list[np.ndarray] = []
    for ref_name in ref_images:
        image_path = ref_dir / str(ref_name)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Referenzbild konnte nicht geladen werden: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        variants = augment_image(image_rgb) if use_rotation else [image_rgb]
        for image_variant in variants:
            image_tensor, grid_shape = model.prepare_image(image_variant)
            last_layer_features = np.asarray(model.extract_features(image_tensor), dtype=np.float32)
            valid_mask = model.compute_background_mask(
                last_layer_features,
                grid_shape,
                threshold=10,
                masking_type=use_ref_masking,
            )
            ref_feature_rows.append(last_layer_features[np.asarray(valid_mask, dtype=bool)])

    ref_features = np.concatenate(ref_feature_rows, axis=0).astype(np.float32)
    if live_config["knn_metric"] == "L2_normalized":
        ref_features = _normalize_feature_rows(ref_features)

    return {
        "object_name": object_name,
        "ref_features": ref_features,
        "knn_metric": live_config["knn_metric"],
        "k_neighbors": int(live_config["k_neighbors"]),
        "num_ref_images": int(len(ref_images)),
        "num_ref_patches": int(ref_features.shape[0]),
    }


def _nearest_reference_distances(
    query_features: np.ndarray,
    ref_features: np.ndarray,
    metric: str,
    k_neighbors: int,
) -> np.ndarray:
    query_features = np.asarray(query_features, dtype=np.float32)
    ref_features = np.asarray(ref_features, dtype=np.float32)
    if query_features.ndim != 2 or ref_features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrices, got {query_features.shape} and {ref_features.shape}")
    if query_features.shape[1] != ref_features.shape[1]:
        raise ValueError(
            f"Feature dimensions do not match: query={query_features.shape}, ref={ref_features.shape}"
        )
    if ref_features.shape[0] == 0:
        raise ValueError("Reference bank is empty.")

    k_neighbors = max(1, min(int(k_neighbors), int(ref_features.shape[0])))

    if metric == "L2_normalized":
        query_norm = _normalize_feature_rows(query_features)
        similarities = query_norm @ ref_features.T
        if k_neighbors == 1:
            return (1.0 - similarities.max(axis=1)).astype(np.float32)
        topk = np.partition(similarities, similarities.shape[1] - k_neighbors, axis=1)[:, -k_neighbors:]
        return (1.0 - topk.mean(axis=1)).astype(np.float32)

    if metric != "L2":
        raise ValueError(f"Unsupported kNN metric for live inference: {metric!r}")

    query_sq = np.sum(query_features * query_features, axis=1, keepdims=True)
    ref_sq = np.sum(ref_features * ref_features, axis=1, keepdims=True).T
    sq_dists = np.maximum(query_sq + ref_sq - 2.0 * (query_features @ ref_features.T), 0.0).astype(np.float32)
    if k_neighbors == 1:
        return np.sqrt(sq_dists.min(axis=1)).astype(np.float32)
    topk = np.partition(sq_dists, k_neighbors - 1, axis=1)[:, :k_neighbors]
    return np.sqrt(topk.mean(axis=1)).astype(np.float32)


def _build_live_cache_meta(model: Any, image_rgb: np.ndarray, grid_shape: tuple[int, int]) -> dict[str, Any]:
    resized_image = model.resize_transform(Image.fromarray(image_rgb))
    resized_w, resized_h = [int(v) for v in resized_image.size]
    original_h, original_w = [int(v) for v in image_rgb.shape[:2]]
    patch_size = int(model.patch_size)
    return {
        "grid_rows": int(grid_shape[0]),
        "grid_cols": int(grid_shape[1]),
        "resized_w": int(resized_w),
        "resized_h": int(resized_h),
        "original_w": int(original_w),
        "original_h": int(original_h),
        "patch_size": int(patch_size),
        "cropped_w": int(grid_shape[1]) * int(patch_size),
        "cropped_h": int(grid_shape[0]) * int(patch_size),
        "num_layers": 12,
    }


def _extract_live_component_rois(
    sample_name: str,
    object_name: str,
    image_rgb: np.ndarray,
    score_grid: np.ndarray,
    image_score: float,
    image_threshold: float,
    resized_size: tuple[int, int],
    patch_multiple: int,
) -> pd.DataFrame:
    valid_mask = score_grid > 0.0
    consumed_mask = np.zeros_like(valid_mask, dtype=bool)
    rejected_mask = np.zeros_like(valid_mask, dtype=bool)
    box_blocked_mask = np.zeros_like(valid_mask, dtype=bool)
    settings = COMPONENT_TEST_ROI_SETTINGS
    accepted_regions: list[dict[str, Any]] = []

    while True:
        available_mask = valid_mask & ~consumed_mask & ~rejected_mask & ~box_blocked_mask
        peak_candidates_current = peak_candidates(score_grid, available_mask)
        if not peak_candidates_current:
            break

        best_candidate = peak_candidates_current[0]
        peak_score = float(best_candidate["peak_score"])
        if peak_score < float(image_threshold):
            break

        seed_mask = np.asarray(best_candidate["seed_mask"], dtype=bool)
        background_value, ring_mask, background_source = estimate_local_background(
            score_grid,
            seed_mask,
            available_mask=available_mask,
            valid_mask=valid_mask,
            inner_radius=int(settings["background_ring_inner"]),
            outer_radius=int(settings["background_ring_outer"]),
        )
        prominence = float(peak_score - background_value)
        if prominence < float(settings["min_prominence"]):
            rejected_mask |= seed_mask
            continue

        high_threshold = background_value + float(settings["high_prominence_ratio"]) * prominence
        low_threshold = background_value + float(settings["low_prominence_ratio"]) * prominence
        region_mask = hysteresis_component(
            score_grid,
            seed_mask=seed_mask,
            available_mask=available_mask,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        if int(region_mask.sum()) < int(settings["min_region_patches"]):
            rejected_mask |= region_mask
            continue

        region_max_score = float(score_grid[region_mask].max())
        if region_max_score < float(image_threshold):
            rejected_mask |= region_mask
            continue

        current_region_mass = region_mass(score_grid, region_mask, background_value)
        if current_region_mass < float(settings["min_region_mass"]):
            rejected_mask |= region_mask
            continue

        consumed_mask |= region_mask
        merged_mask = region_mask.copy()
        merged_summary = {
            "background_source": background_source,
            "background_value": float(background_value),
            "prominence": float(prominence),
            "high_threshold": float(high_threshold),
            "low_threshold": float(low_threshold),
            "ring_patch_count": int(ring_mask.sum()),
            "proposal_count": 1,
        }

        changed = True
        while changed:
            changed = False
            merged_box = region_box(merged_mask)
            remaining_regions: list[dict[str, Any]] = []
            for accepted_region in accepted_regions:
                if merge_without_valley(
                    score_map=score_grid,
                    valid_mask=valid_mask,
                    region_a=merged_mask,
                    box_a=merged_box,
                    background_a=float(merged_summary["background_value"]),
                    prominence_a=float(merged_summary["prominence"]),
                    region_b=accepted_region["region_mask"],
                    box_b=accepted_region["patch_box"],
                    background_b=float(accepted_region["background_value"]),
                    prominence_b=float(accepted_region["prominence"]),
                    gap=int(settings["merge_gap_patches"]),
                    bridge_ratio=float(settings["merge_bridge_ratio"]),
                ):
                    merged_mask |= accepted_region["region_mask"]
                    merged_summary["proposal_count"] += int(accepted_region["proposal_count"])
                    if float(accepted_region["prominence"]) > float(merged_summary["prominence"]):
                        merged_summary["background_source"] = accepted_region["background_source"]
                        merged_summary["background_value"] = float(accepted_region["background_value"])
                        merged_summary["prominence"] = float(accepted_region["prominence"])
                        merged_summary["high_threshold"] = float(accepted_region["high_threshold"])
                        merged_summary["low_threshold"] = float(accepted_region["low_threshold"])
                        merged_summary["ring_patch_count"] = int(accepted_region["ring_patch_count"])
                    changed = True
                else:
                    remaining_regions.append(accepted_region)
            accepted_regions = remaining_regions

        merged_box = region_box(merged_mask)
        accepted_regions.append(
            {
                "region_mask": merged_mask,
                "patch_box": merged_box,
                **merged_summary,
            }
        )
        if bool(settings["block_boxes"]):
            row_min, row_max, col_min, col_max = merged_box
            box_blocked_mask[row_min : row_max + 1, col_min : col_max + 1] = True

    if accepted_regions:
        accepted_regions.sort(key=lambda accepted_region: strength_key(score_grid, accepted_region), reverse=True)
        non_overlapping_regions: list[dict[str, Any]] = []
        for accepted_region in accepted_regions:
            if any(boxes_overlap(accepted_region["patch_box"], kept_region["patch_box"]) for kept_region in non_overlapping_regions):
                continue
            non_overlapping_regions.append(accepted_region)
        accepted_regions = non_overlapping_regions

    max_boxes_per_image = settings["max_boxes_per_image"]
    if max_boxes_per_image is not None:
        accepted_regions = accepted_regions[: int(max_boxes_per_image)]

    original_h, original_w = [int(v) for v in image_rgb.shape[:2]]
    roi_rows: list[dict[str, Any]] = []
    for roi_index, accepted_region in enumerate(accepted_regions):
        region_mask = np.asarray(accepted_region["region_mask"], dtype=bool)
        region_row_min, region_row_max, region_col_min, region_col_max = accepted_region["patch_box"]
        x0, y0, x1, y1 = patch_box_to_image_box(
            (region_row_min, region_row_max, region_col_min, region_col_max),
            tuple(int(v) for v in score_grid.shape),
            (int(original_w), int(original_h)),
            (int(resized_size[0]), int(resized_size[1])),
            patch_multiple=int(patch_multiple),
        )
        peaks_in_region = region_peak_details(score_grid, available_mask=valid_mask, region_mask=region_mask)
        roi_rows.append(
            {
                "object": object_name,
                "split": "uploaded",
                "sample": sample_name,
                "bildname": sample_name,
                "roi_index": int(roi_index),
                "roi_nummer": f"roi{roi_index}",
                "roi_uid": f"{sample_name}__roi_{roi_index:03d}",
                "image_threshold": float(image_threshold),
                "image_score": float(image_score),
                "x_min": int(x0),
                "y_min": int(y0),
                "x_max": int(x1),
                "y_max": int(y1),
                "region_patch_count": int(region_mask.sum()),
                "region_max_score": float(score_grid[region_mask].max()),
                "region_mean_score": float(score_grid[region_mask].mean()),
                "region_row_min": int(region_row_min),
                "region_row_max": int(region_row_max),
                "region_col_min": int(region_col_min),
                "region_col_max": int(region_col_max),
                "primary_peak_row": int(peaks_in_region[0]["row"]) if peaks_in_region else int(region_row_min),
                "primary_peak_col": int(peaks_in_region[0]["col"]) if peaks_in_region else int(region_col_min),
                "primary_peak_score": float(peaks_in_region[0]["peak_score"]) if peaks_in_region else float(score_grid[region_mask].max()),
                "peak_count_in_region": int(len(peaks_in_region)),
                "crop_path": "",
            }
        )
    return pd.DataFrame(roi_rows)


def _run_component_test_for_external_image(
    experiment_dir_str: str,
    seed: int,
    object_name: str,
    sample_name: str,
    image_rgb: np.ndarray,
    classifier_info: dict[str, Any],
    render_overlays: bool,
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    live_config = load_component_test_live_config(
        experiment_dir_str,
        int(seed),
    )

    backbone = load_component_test_backbone(experiment_dir_str)
    model = backbone["model"]
    confirmed_config = load_anomaly_detection_confirmed_config(experiment_dir_str, object_name)
    if confirmed_config is not None:
        reference_paths = _resolve_confirmed_reference_paths(experiment_dir_str, object_name, confirmed_config)
        reference_bank = load_anomaly_detection_reference_bank(
            experiment_dir_str,
            int(seed),
            object_name,
            tuple(str(path.resolve()) for path in reference_paths),
            _file_signature_from_paths(reference_paths),
        )
        image_threshold = float(confirmed_config["threshold"])
    else:
        data_root = Path(str(load_run_args(experiment_dir).get("data_root", ""))).resolve()
        train_good_dir = data_root / object_name / "train" / "good"
        train_good_signature = _directory_file_signature(train_good_dir)
        if object_name not in live_config["threshold_by_object"]:
            raise KeyError(f"Kein Threshold fuer Objekt {object_name!r} gefunden.")
        reference_bank = load_component_test_reference_bank(
            experiment_dir_str,
            int(seed),
            object_name,
            train_good_signature=train_good_signature,
        )
        image_threshold = float(live_config["threshold_by_object"][object_name])

    image_tensor, grid_shape = model.prepare_image(image_rgb)
    anomaly_features = np.asarray(model.extract_features(image_tensor), dtype=np.float32)
    features_layers, layer_indices = model.extract_multilayer_features(image_tensor, layer_indices=list(range(1, 13)))
    features_layers = np.asarray(features_layers, dtype=np.float32)
    masking_enabled = bool(live_config["masking_by_object"].get(object_name, False))
    valid_mask = np.asarray(
        model.compute_background_mask(anomaly_features, grid_shape, threshold=10, masking_type=masking_enabled),
        dtype=bool,
    )

    query_features = anomaly_features[valid_mask]
    patch_distances = _nearest_reference_distances(
        query_features,
        reference_bank["ref_features"],
        metric=str(reference_bank["knn_metric"]),
        k_neighbors=int(reference_bank["k_neighbors"]),
    )
    output_distances = np.zeros(anomaly_features.shape[0], dtype=np.float32)
    output_distances[valid_mask] = patch_distances
    score_grid = output_distances.reshape(tuple(int(v) for v in grid_shape)).astype(np.float32)

    aggregation_statistics = str(live_config["aggregation_statistics"])
    if aggregation_statistics == "meantop1p":
        image_score = float(mean_top1p(output_distances.flatten()))
    elif aggregation_statistics == "max_patch_distance":
        image_score = float(np.max(output_distances))
    else:
        image_score = float(np.max(dists2map(score_grid, image_rgb.shape)))

    anomalous_patch_count = int(np.count_nonzero(score_grid >= image_threshold))
    io_nio = "NIO" if image_score >= image_threshold else "IO"

    resized_image = model.resize_transform(Image.fromarray(image_rgb))
    resized_size = tuple(int(v) for v in resized_image.size)
    cache_meta = _build_live_cache_meta(model, image_rgb, tuple(int(v) for v in grid_shape))
    sample_assets = {
        "sample": sample_name,
        "evaluation_group": f"uploaded/{object_name}",
        "image_rgb": image_rgb,
        "score_grid": score_grid,
        "grid_shape": tuple(int(v) for v in grid_shape),
        "image_threshold": float(image_threshold),
        "image_score": float(image_score),
        "image_path": sample_name,
    }
    multilayer_assets = {
        "sample": sample_name,
        "features_layers": features_layers,
        "grid_shape": tuple(int(v) for v in grid_shape),
        "layer_indices": tuple(int(v) for v in layer_indices),
        "cache_meta": cache_meta,
        "patch_size": int(model.patch_size),
    }

    roi_prediction_rows: list[dict[str, Any]] = []
    note = ""
    if io_nio == "IO":
        part_label = "IO"
    else:
        roi_rows = _extract_live_component_rois(
            sample_name=sample_name,
            object_name=object_name,
            image_rgb=image_rgb,
            score_grid=score_grid,
            image_score=image_score,
            image_threshold=image_threshold,
            resized_size=resized_size,
            patch_multiple=int(model.patch_size),
        )
        if roi_rows.empty:
            part_label = "NIO_no_roi"
            note = "Bild ist ueber dem Bildthreshold, aber aus der frischen ROI-Extraktion kamen keine ROIs."
        else:
            roi_prediction_rows = _classify_component_sample_rois(
                sample_assets=sample_assets,
                multilayer_assets=multilayer_assets,
                roi_rows=roi_rows,
                classifier_info=classifier_info,
            )
            part_label = "3D" if any(row["predicted_label"] == "3D" for row in roi_prediction_rows) else "2D"

    overlay_rgb = None
    if render_overlays:
        overlay_rgb = _render_component_test_overlay(
            sample_assets=sample_assets,
            roi_prediction_rows=roi_prediction_rows,
            part_label=part_label,
        )

    num_rois = int(len(roi_prediction_rows))
    num_2d_rois = int(sum(row["predicted_label"] == "2D" for row in roi_prediction_rows))
    num_3d_rois = int(sum(row["predicted_label"] == "3D" for row in roi_prediction_rows))
    max_roi_p3d = float(max((row["proba_3d"] for row in roi_prediction_rows), default=0.0))
    max_roi_confidence = float(max((row["predicted_probability"] for row in roi_prediction_rows), default=0.0))

    return {
        "sample": sample_name,
        "evaluation_group": f"uploaded/{object_name}",
        "image_score": float(image_score),
        "image_threshold": float(image_threshold),
        "anomalous_patch_count": int(anomalous_patch_count),
        "io_nio": io_nio,
        "part_label": part_label,
        "num_rois": num_rois,
        "num_2d_rois": num_2d_rois,
        "num_3d_rois": num_3d_rois,
        "max_roi_p3d": max_roi_p3d,
        "max_roi_confidence": max_roi_confidence,
        "note": note,
        "roi_predictions": roi_prediction_rows,
        "overlay_rgb": overlay_rgb,
    }


@st.cache_data(show_spinner=False)
def load_multilayer_sample_layers(experiment_dir_str: str, seed: int, sample_name: str) -> dict[str, Any]:
    manifest = load_multilayer_cache_manifest(experiment_dir_str, seed)
    if sample_name not in manifest["sample_index"]:
        raise KeyError(f"Sample nicht im Multi-Layer-Cache gefunden: {sample_name}")
    sample_info = manifest["sample_index"][sample_name]
    cache_file = Path(sample_info["cache_file"])
    if not cache_file.exists():
        raise FileNotFoundError(f"Multi-Layer-Cache-Datei nicht gefunden: {cache_file}")

    features_layers, grid_shape, layer_indices, cache_meta = load_multilayer_cache(cache_file)
    return {
        "sample": sample_name,
        "features_layers": np.asarray(features_layers, dtype=np.float32),
        "grid_shape": tuple(int(v) for v in grid_shape),
        "layer_indices": tuple(int(v) for v in layer_indices),
        "cache_meta": cache_meta,
        "patch_size": int(sample_info["patch_size"]),
    }


@st.cache_resource(show_spinner="Lade Boruta-Bauteilklassifikator...")
def load_boruta_component_classifier_model(
    experiment_dir_str: str,
    cache_signature: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir_str).resolve()
    classifier_dir = experiment_dir / BORUTA_COMPONENT_CLASSIFIER_FEATURE_SUBDIR
    selection_path = classifier_dir / "selected_features.csv"
    selection_indices_path = classifier_dir / "selected_feature_indices.npy"
    classifier_path = classifier_dir / "classifier_pipeline.joblib"
    model_info_path = classifier_dir / "model_info.json"
    summary_path = classifier_dir / "summary.json"

    if not classifier_path.exists():
        raise FileNotFoundError(f"Boruta-Klassifikator nicht gefunden: {classifier_path}")

    _ = cache_signature
    if selection_indices_path.exists():
        selected_indices = np.load(selection_indices_path).astype(np.int32)
    else:
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Boruta-Featuredatei nicht gefunden: weder {selection_indices_path} noch {selection_path}"
            )
        selection_df = pd.read_csv(selection_path)
        if "feature_index" not in selection_df.columns:
            raise ValueError(f"'feature_index' fehlt in {selection_path}")
        if "status" in selection_df.columns:
            confirmed_df = selection_df.loc[selection_df["status"].astype(str).str.lower() == "confirmed"]
            if not confirmed_df.empty:
                selection_df = confirmed_df
        selected_indices = selection_df["feature_index"].astype(np.int32).to_numpy()
    model = joblib_load(classifier_path)
    model_info = json.loads(model_info_path.read_text(encoding="utf-8")) if model_info_path.exists() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "model": model,
        "selected_indices": selected_indices,
        "model_info": model_info,
        "summary": summary,
        "selected_feature_count": int(selected_indices.size),
        "classifier_dir": str(classifier_dir),
    }


def _roi_patch_box_to_display_box(
    image_rgb: np.ndarray,
    grid_shape: tuple[int, int],
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> tuple[int, int, int, int]:
    row_edges, col_edges = _grid_edges(image_rgb.shape, grid_shape)
    x0 = int(col_edges[col_min])
    y0 = int(row_edges[row_min])
    x1 = int(col_edges[col_max + 1])
    y1 = int(row_edges[row_max + 1])
    return x0, y0, x1, y1


def _classify_component_sample_rois(
    sample_assets: dict[str, Any],
    multilayer_assets: dict[str, Any],
    roi_rows: pd.DataFrame,
    classifier_info: dict[str, Any],
) -> list[dict[str, Any]]:
    if roi_rows.empty:
        return []

    selected_feature_rows: list[np.ndarray] = []
    roi_prediction_rows: list[dict[str, Any]] = []
    score_grid = np.asarray(sample_assets["score_grid"], dtype=np.float32)
    cache_meta = multilayer_assets["cache_meta"]
    grid_shape = tuple(int(v) for v in multilayer_assets["grid_shape"])
    features_layers = np.asarray(multilayer_assets["features_layers"], dtype=np.float32)
    selected_indices = np.asarray(classifier_info["selected_indices"], dtype=np.int32)

    for roi_row in roi_rows.itertuples(index=False):
        row_series = pd.Series(roi_row._asdict())
        selected_patches, selection_mode, num_candidates = select_overlap_threshold_patches(
            row=row_series,
            meta=cache_meta,
            anomaly_grid=score_grid,
            image_threshold=float(sample_assets["image_threshold"]),
        )
        combined_feature = aggregate_maxminmean_per_layer(
            features_layers=features_layers,
            selected_patches=selected_patches,
            grid_shape=grid_shape,
        )
        selected_feature_rows.append(combined_feature[selected_indices].astype(np.float32))
        roi_prediction_rows.append(
            {
                "sample": sample_assets["sample"],
                "roi_index": int(row_series["roi_index"]),
                "roi_nummer": str(row_series["roi_nummer"]),
                "x_min": int(row_series["x_min"]),
                "y_min": int(row_series["y_min"]),
                "x_max": int(row_series["x_max"]),
                "y_max": int(row_series["y_max"]),
                "region_row_min": int(row_series["region_row_min"]),
                "region_row_max": int(row_series["region_row_max"]),
                "region_col_min": int(row_series["region_col_min"]),
                "region_col_max": int(row_series["region_col_max"]),
                "region_patch_count": int(row_series["region_patch_count"]),
                "region_max_score": float(row_series["region_max_score"]),
                "selection_mode": selection_mode,
                "num_overlap_candidates": int(num_candidates),
                "num_selected_patches": int(len(selected_patches)),
                "selected_patch_rows": ";".join(str(int(patch_row)) for patch_row, _ in selected_patches),
                "selected_patch_cols": ";".join(str(int(patch_col)) for _, patch_col in selected_patches),
            }
        )

    X = np.stack(selected_feature_rows, axis=0).astype(np.float32)
    model = classifier_info["model"]
    pred_labels = model.predict(X)
    pred_proba = model.predict_proba(X).astype(np.float32)

    for row_dict, pred_label_raw, proba in zip(roi_prediction_rows, pred_labels, pred_proba):
        pred_label = "3D" if int(pred_label_raw) == 1 else "2D"
        predicted_probability = float(proba[1] if pred_label == "3D" else proba[0])
        row_dict["predicted_label"] = pred_label
        row_dict["predicted_probability"] = predicted_probability
        row_dict["proba_2d"] = float(proba[0])
        row_dict["proba_3d"] = float(proba[1])

    return roi_prediction_rows


def _render_component_test_overlay(
    sample_assets: dict[str, Any],
    roi_prediction_rows: list[dict[str, Any]],
    part_label: str,
) -> np.ndarray:
    overlay = sample_assets["image_rgb"].copy()
    grid_shape = tuple(int(v) for v in sample_assets["grid_shape"])
    _ = part_label
    roi_boxes: list[tuple[dict[str, Any], tuple[int, int, int, int]]] = []

    for row_dict in roi_prediction_rows:
        x0, y0, x1, y1 = _roi_patch_box_to_display_box(
            image_rgb=overlay,
            grid_shape=grid_shape,
            row_min=int(row_dict["region_row_min"]),
            row_max=int(row_dict["region_row_max"]),
            col_min=int(row_dict["region_col_min"]),
            col_max=int(row_dict["region_col_max"]),
        )
        roi_boxes.append((row_dict, (x0, y0, x1, y1)))

    blocked_rects = [box for _, box in roi_boxes]
    occupied_label_rects: list[tuple[int, int, int, int]] = []

    for row_dict, (x0, y0, x1, y1) in roi_boxes:
        pred_label = str(row_dict["predicted_label"])
        color = (40, 180, 40) if pred_label == "2D" else (0, 80, 220)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        label_text = (
            f"{row_dict['roi_nummer']} {pred_label} "
            f"{float(row_dict['predicted_probability']) * 100:.1f}%"
        )
        _draw_component_roi_label(
            overlay,
            label_text,
            x0,
            y0,
            x1,
            y1,
            color,
            blocked_rects,
            occupied_label_rects,
        )
    return overlay


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


def _apply_feature_subset(
    features_norm: np.ndarray,
    feature_indices: np.ndarray,
    renormalize: bool = True,
) -> np.ndarray:
    if feature_indices.ndim != 1:
        raise ValueError(f"feature_indices muessen 1D sein, erhalten: {feature_indices.shape}")
    if feature_indices.size <= 0:
        raise ValueError("feature_indices duerfen nicht leer sein")
    if int(np.min(feature_indices)) < 0 or int(np.max(feature_indices)) >= features_norm.shape[1]:
        raise ValueError(
            f"Feature-Subset ausserhalb des gueltigen Bereichs: "
            f"features={features_norm.shape}, min={int(np.min(feature_indices))}, max={int(np.max(feature_indices))}"
        )
    reduced = features_norm[:, feature_indices].astype(np.float32)
    if not renormalize:
        return reduced
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    fallback_mask = norms.squeeze(-1) <= 1e-8
    norms = np.maximum(norms, 1e-8)
    reduced = (reduced / norms).astype(np.float32)
    if np.any(fallback_mask):
        reduced[fallback_mask] = 0.0
    return reduced


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


def _run_component_test_for_sample(
    experiment_dir_str: str,
    seed: int,
    sample_name: str,
    roi_table: pd.DataFrame,
    classifier_info: dict[str, Any],
    render_overlays: bool,
) -> dict[str, Any]:
    sample_assets = load_sample_assets(experiment_dir_str, seed, sample_name)
    score_grid = np.asarray(sample_assets["score_grid"], dtype=np.float32)
    image_threshold = float(sample_assets["image_threshold"])
    image_score = float(sample_assets["image_score"])
    anomalous_patch_count = int(np.count_nonzero(score_grid >= image_threshold))
    io_nio = "NIO" if image_score >= image_threshold else "IO"
    sample_roi_rows = (
        roi_table.loc[roi_table["sample"] == sample_name]
        .sort_values(["roi_index"])
        .reset_index(drop=True)
    )

    roi_prediction_rows: list[dict[str, Any]] = []
    note = ""
    if io_nio == "IO":
        part_label = "IO"
    else:
        if sample_roi_rows.empty:
            part_label = "NIO_no_roi"
            note = "Bild ist ueber dem Bildthreshold, aber es wurden keine vorliegenden ROIs gefunden."
        else:
            multilayer_assets = load_multilayer_sample_layers(experiment_dir_str, seed, sample_name)
            roi_prediction_rows = _classify_component_sample_rois(
                sample_assets=sample_assets,
                multilayer_assets=multilayer_assets,
                roi_rows=sample_roi_rows,
                classifier_info=classifier_info,
            )
            part_label = "3D" if any(row["predicted_label"] == "3D" for row in roi_prediction_rows) else "2D"

    num_rois = int(len(roi_prediction_rows))
    num_2d_rois = int(sum(row["predicted_label"] == "2D" for row in roi_prediction_rows))
    num_3d_rois = int(sum(row["predicted_label"] == "3D" for row in roi_prediction_rows))
    max_roi_p3d = float(max((row["proba_3d"] for row in roi_prediction_rows), default=0.0))
    max_roi_confidence = float(max((row["predicted_probability"] for row in roi_prediction_rows), default=0.0))

    overlay_rgb = None
    if render_overlays:
        overlay_rgb = _render_component_test_overlay(
            sample_assets=sample_assets,
            roi_prediction_rows=roi_prediction_rows,
            part_label=part_label,
        )

    return {
        "sample": sample_name,
        "evaluation_group": str(sample_assets["evaluation_group"]),
        "image_score": image_score,
        "image_threshold": image_threshold,
        "anomalous_patch_count": anomalous_patch_count,
        "io_nio": io_nio,
        "part_label": part_label,
        "num_rois": num_rois,
        "num_2d_rois": num_2d_rois,
        "num_3d_rois": num_3d_rois,
        "max_roi_p3d": max_roi_p3d,
        "max_roi_confidence": max_roi_confidence,
        "note": note,
        "roi_predictions": roi_prediction_rows,
        "overlay_rgb": overlay_rgb,
    }


def _save_anomaly_detection_evaluation_artifacts(
    experiment_dir_str: str,
    object_name: str,
    score_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    evaluation_dir = _anomaly_detection_object_dir(experiment_dir_str, object_name) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    scores_path = evaluation_dir / "image_scores.csv"
    sweep_path = evaluation_dir / "threshold_sweep.csv"
    summary_path = evaluation_dir / "summary.json"
    score_df.to_csv(scores_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "scores_csv": str(scores_path),
        "threshold_sweep_csv": str(sweep_path),
        "summary_json": str(summary_path),
    }


def render_anomaly_detection_settings_mode(experiment_dir_str: str, seed: int) -> None:
    st.markdown("### Anomaly Detection")
    st.caption(
        "Diese Seite bewertet nur die Bildentscheidung `good` vs. `bad` (`IO/NIO`). "
        "ROI-Extraktion und 2D/3D-Klassifikation sind hier bewusst nicht Teil der Evaluierung."
    )

    try:
        live_config = load_component_test_live_config(experiment_dir_str, int(seed))
    except Exception as exc:
        st.error(f"Anomaly-Detection-Einstellungen konnten nicht geladen werden: {exc}")
        return

    object_names = list(live_config.get("object_names", []))
    if not object_names:
        st.warning("Im aktuellen Run wurden keine Objekte gefunden.")
        return

    selected_object_name = st.selectbox(
        "Objekt",
        options=object_names,
        index=0,
        key="anomaly_detection_settings_object",
    )

    flash_message = st.session_state.pop(ANOMALY_DETECTION_UPLOAD_FLASH_KEY, "")
    if flash_message:
        st.success(str(flash_message))
    confirm_flash_message = st.session_state.pop(ANOMALY_DETECTION_CONFIRM_FLASH_KEY, "")
    if confirm_flash_message:
        st.success(str(confirm_flash_message))

    upload_nonce = int(st.session_state.get(ANOMALY_DETECTION_UPLOAD_NONCE_KEY, 0))

    with st.form("anomaly_detection_settings_upload_form"):
        st.caption(
            "Neue Uploads werden zur jeweiligen Kategorie hinzugefuegt. "
            "Du kannst denselben Bereich also mehrfach nacheinander befuellen."
        )
        upload_col_ref, upload_col_good, upload_col_bad = st.columns(3)
        with upload_col_ref:
            reference_uploads = st.file_uploader(
                "Referenzbilder",
                type=COMPONENT_TEST_UPLOAD_EXTENSIONS,
                accept_multiple_files=True,
                key=f"anomaly_detection_reference_uploads_{upload_nonce}",
            )
        with upload_col_good:
            test_good_uploads = st.file_uploader(
                "Testbilder good",
                type=COMPONENT_TEST_UPLOAD_EXTENSIONS,
                accept_multiple_files=True,
                key=f"anomaly_detection_test_good_uploads_{upload_nonce}",
            )
        with upload_col_bad:
            test_bad_uploads = st.file_uploader(
                "Testbilder bad",
                type=COMPONENT_TEST_UPLOAD_EXTENSIONS,
                accept_multiple_files=True,
                key=f"anomaly_detection_test_bad_uploads_{upload_nonce}",
            )
        save_uploads = st.form_submit_button("Uploads speichern", use_container_width=True)

    if save_uploads:
        saved_sections: list[str] = []
        if reference_uploads:
            saved_names = _save_uploaded_images_to_category(
                experiment_dir_str,
                selected_object_name,
                ANOMALY_DETECTION_REFERENCE_SUBDIR,
                reference_uploads,
            )
            saved_sections.append(f"Referenzbilder: {len(saved_names)}")
        if test_good_uploads:
            saved_names = _save_uploaded_images_to_category(
                experiment_dir_str,
                selected_object_name,
                ANOMALY_DETECTION_TEST_GOOD_SUBDIR,
                test_good_uploads,
            )
            saved_sections.append(f"Test good: {len(saved_names)}")
        if test_bad_uploads:
            saved_names = _save_uploaded_images_to_category(
                experiment_dir_str,
                selected_object_name,
                ANOMALY_DETECTION_TEST_BAD_SUBDIR,
                test_bad_uploads,
            )
            saved_sections.append(f"Test bad: {len(saved_names)}")

        if not saved_sections:
            st.warning("Es wurden keine neuen Bilder hochgeladen.")
        else:
            st.session_state.pop(ANOMALY_DETECTION_SETTINGS_VIEW_KEY, None)
            st.session_state[ANOMALY_DETECTION_UPLOAD_NONCE_KEY] = upload_nonce + 1
            st.session_state[ANOMALY_DETECTION_UPLOAD_FLASH_KEY] = "Gespeichert: " + ", ".join(saved_sections)
            st.rerun()

    reference_paths = _list_saved_anomaly_detection_images(
        experiment_dir_str,
        selected_object_name,
        ANOMALY_DETECTION_REFERENCE_SUBDIR,
    )
    test_good_paths = _list_saved_anomaly_detection_images(
        experiment_dir_str,
        selected_object_name,
        ANOMALY_DETECTION_TEST_GOOD_SUBDIR,
    )
    test_bad_paths = _list_saved_anomaly_detection_images(
        experiment_dir_str,
        selected_object_name,
        ANOMALY_DETECTION_TEST_BAD_SUBDIR,
    )

    count_cols = st.columns(3)
    count_cols[0].metric("Referenzbilder", str(len(reference_paths)))
    count_cols[1].metric("Test good", str(len(test_good_paths)))
    count_cols[2].metric("Test bad", str(len(test_bad_paths)))

    confirmed_config = load_anomaly_detection_confirmed_config(experiment_dir_str, selected_object_name)
    if confirmed_config is not None:
        st.caption(
            "Aktive bestaetigte Konfiguration: "
            f"Threshold={float(confirmed_config['threshold']):.6f} | "
            f"Referenzbilder={int(confirmed_config.get('num_reference_images', 0))} | "
            f"bestaetigt am {confirmed_config.get('confirmed_at', '-')}"
        )

    with st.expander("Gespeicherte Dateien", expanded=False):
        file_cols = st.columns(3)
        file_cols[0].write([path.name for path in reference_paths] or ["Keine Referenzbilder gespeichert."])
        file_cols[1].write([path.name for path in test_good_paths] or ["Keine good-Testbilder gespeichert."])
        file_cols[2].write([path.name for path in test_bad_paths] or ["Keine bad-Testbilder gespeichert."])

    with st.form("anomaly_detection_evaluation_form"):
        st.caption(
            "Die Evaluierung berechnet fuer alle gespeicherten Testbilder frische Bildscores, "
            "bestimmt daraus den besten Bildthreshold nach F1 und speichert die Ergebnisse lokal ab."
        )
        start_evaluation = st.form_submit_button("Evaluierung starten", use_container_width=True)

    if start_evaluation:
        if not reference_paths:
            st.warning("Speichere zuerst mindestens ein Referenzbild.")
            return
        if not test_good_paths or not test_bad_paths:
            st.warning("Speichere sowohl good- als auch bad-Testbilder.")
            return

        try:
            reference_bank = load_anomaly_detection_reference_bank(
                experiment_dir_str,
                int(seed),
                selected_object_name,
                tuple(str(path.resolve()) for path in reference_paths),
                _file_signature_from_paths(reference_paths),
            )
        except Exception as exc:
            st.error(f"Referenzbank konnte nicht aufgebaut werden: {exc}")
            return

        progress_bar = st.progress(0.0)
        status_box = st.empty()
        evaluation_rows: list[dict[str, Any]] = []
        test_items = (
            [(path, "good", 0) for path in test_good_paths]
            + [(path, "bad", 1) for path in test_bad_paths]
        )

        for index, (image_path, true_label, true_is_bad) in enumerate(test_items, start=1):
            status_box.info(f"Evaluiere {index}/{len(test_items)}: {image_path.name}")
            try:
                image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                score_result = _score_image_for_live_anomaly_detection(
                    experiment_dir_str,
                    int(seed),
                    selected_object_name,
                    image_rgb,
                    reference_bank,
                )
                evaluation_rows.append(
                    {
                        "sample": image_path.name,
                        "path": str(image_path.resolve()),
                        "split": true_label,
                        "true_label": true_label,
                        "true_is_bad": int(true_is_bad),
                        "image_score": float(score_result["image_score"]),
                        "num_valid_patches": int(score_result["num_valid_patches"]),
                    }
                )
            except Exception as exc:
                evaluation_rows.append(
                    {
                        "sample": image_path.name,
                        "path": str(image_path.resolve()),
                        "split": true_label,
                        "true_label": true_label,
                        "true_is_bad": int(true_is_bad),
                        "image_score": float("nan"),
                        "num_valid_patches": 0,
                        "error": str(exc),
                    }
                )
            progress_bar.progress(index / float(len(test_items)))

        status_box.empty()
        progress_bar.empty()

        score_df = pd.DataFrame(evaluation_rows)
        valid_score_df = score_df.loc[score_df["image_score"].notna()].copy()
        if valid_score_df.empty:
            st.error("Keine gueltigen Bildscores berechnet. Die Evaluierung kann nicht fortgesetzt werden.")
            return

        sweep_df, best_row = _sweep_anomaly_detection_thresholds(
            valid_score_df["image_score"].to_numpy(dtype=np.float64),
            valid_score_df["true_is_bad"].to_numpy(dtype=np.int32),
        )
        best_threshold = float(best_row["threshold"])
        best_metrics = _compute_anomaly_detection_metrics(
            valid_score_df["image_score"].to_numpy(dtype=np.float64),
            valid_score_df["true_is_bad"].to_numpy(dtype=np.int32),
            best_threshold,
        )
        summary = {
            "object_name": selected_object_name,
            "num_reference_images": int(len(reference_paths)),
            "num_test_good": int(len(test_good_paths)),
            "num_test_bad": int(len(test_bad_paths)),
            "num_scored_images": int(valid_score_df.shape[0]),
            "num_failed_images": int(score_df.shape[0] - valid_score_df.shape[0]),
            "best_threshold": float(best_threshold),
            "best_precision": float(best_metrics["precision"]),
            "best_recall": float(best_metrics["recall"]),
            "best_f1": float(best_metrics["f1"]),
            "best_accuracy": float(best_metrics["accuracy"]),
            "reference_patches": int(reference_bank["num_ref_patches"]),
            "reference_images_used": int(reference_bank["num_ref_images"]),
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        artifact_paths = _save_anomaly_detection_evaluation_artifacts(
            experiment_dir_str,
            selected_object_name,
            score_df,
            sweep_df,
            summary,
        )

        st.session_state[ANOMALY_DETECTION_SETTINGS_VIEW_KEY] = {
            "object_name": selected_object_name,
            "score_rows": score_df.to_dict(orient="records"),
            "threshold_sweep_rows": sweep_df.to_dict(orient="records"),
            "best_threshold": float(best_threshold),
            "summary": summary,
            "artifact_paths": artifact_paths,
            "run_id": summary["evaluated_at"],
        }
        st.success(
            f"Evaluierung abgeschlossen. Bester Bildthreshold nach F1: {best_threshold:.6f} "
            f"(F1={best_metrics['f1']:.4f})"
        )

    stored = st.session_state.get(ANOMALY_DETECTION_SETTINGS_VIEW_KEY)
    if not stored or stored.get("object_name") != selected_object_name:
        return

    score_df = pd.DataFrame(stored["score_rows"])
    sweep_df = pd.DataFrame(stored["threshold_sweep_rows"])
    best_threshold = float(stored["best_threshold"])
    summary = dict(stored["summary"])
    valid_score_df = score_df.loc[score_df["image_score"].notna()].copy()

    st.markdown("### Evaluierung")
    summary_cols = st.columns(5)
    summary_cols[0].metric("Bester Threshold", f"{best_threshold:.6f}")
    summary_cols[1].metric("Bestes F1", f"{float(summary['best_f1']) * 100:.1f}%")
    summary_cols[2].metric("Bestes Precision", f"{float(summary['best_precision']) * 100:.1f}%")
    summary_cols[3].metric("Bestes Recall", f"{float(summary['best_recall']) * 100:.1f}%")
    summary_cols[4].metric("Gescorete Bilder", str(int(summary["num_scored_images"])))

    score_min = float(min(valid_score_df["image_score"].min(), best_threshold))
    score_max = float(max(valid_score_df["image_score"].max(), best_threshold))
    slider_key = f"anomaly_detection_threshold_slider_{selected_object_name}_{stored['run_id']}"
    numeric_key = f"anomaly_detection_threshold_input_{selected_object_name}_{stored['run_id']}"
    if score_max > score_min:
        threshold_cols = st.columns([2.0, 1.0])
        slider_step = max((score_max - score_min) / 5000.0, 1e-6)
        if slider_key not in st.session_state:
            st.session_state[slider_key] = float(best_threshold)
        if numeric_key not in st.session_state:
            st.session_state[numeric_key] = float(best_threshold)
        with threshold_cols[0]:
            slider_threshold = st.slider(
                "Bildthreshold",
                min_value=float(score_min),
                max_value=float(score_max),
                value=float(st.session_state[slider_key]),
                step=float(slider_step),
                format="%.6f",
                key=slider_key,
                on_change=_sync_state_value,
                args=(numeric_key, slider_key),
            )
        with threshold_cols[1]:
            input_threshold = st.number_input(
                "Threshold exakt",
                min_value=float(score_min),
                max_value=float(score_max),
                value=float(st.session_state[numeric_key]),
                step=float(slider_step),
                format="%.6f",
                key=numeric_key,
                on_change=_sync_state_value,
                args=(slider_key, numeric_key),
            )
        current_threshold = float(input_threshold)
    else:
        current_threshold = st.number_input(
            "Bildthreshold",
            value=float(best_threshold),
            format="%.6f",
            key=slider_key,
        )

    live_metrics = _compute_anomaly_detection_metrics(
        valid_score_df["image_score"].to_numpy(dtype=np.float64),
        valid_score_df["true_is_bad"].to_numpy(dtype=np.int32),
        float(current_threshold),
    )
    live_cols = st.columns(5)
    live_cols[0].metric("Precision", f"{live_metrics['precision'] * 100:.1f}%")
    live_cols[1].metric("Recall", f"{live_metrics['recall'] * 100:.1f}%")
    live_cols[2].metric("F1", f"{live_metrics['f1'] * 100:.1f}%")
    live_cols[3].metric("Accuracy", f"{live_metrics['accuracy'] * 100:.1f}%")
    live_cols[4].metric("Predicted bad", str(int(live_metrics["predicted_bad"])))

    confusion_cols = st.columns(4)
    confusion_cols[0].metric("TP", str(int(live_metrics["tp"])))
    confusion_cols[1].metric("FP", str(int(live_metrics["fp"])))
    confusion_cols[2].metric("TN", str(int(live_metrics["tn"])))
    confusion_cols[3].metric("FN", str(int(live_metrics["fn"])))

    confirm_threshold = st.button(
        "Diesen Threshold übernehmen",
        use_container_width=True,
        key=f"anomaly_detection_confirm_button_{selected_object_name}_{stored['run_id']}",
    )
    if confirm_threshold:
        config_path = _save_anomaly_detection_confirmed_config(
            experiment_dir_str=experiment_dir_str,
            object_name=selected_object_name,
            threshold=float(current_threshold),
            reference_paths=reference_paths,
            evaluation_summary=summary,
            live_metrics=live_metrics,
        )
        st.session_state[ANOMALY_DETECTION_CONFIRM_FLASH_KEY] = (
            f"Threshold {float(current_threshold):.6f} für {selected_object_name} übernommen "
            f"und gespeichert in {config_path}"
        )
        st.rerun()

    chart_df = sweep_df[["threshold", "precision", "recall", "f1"]].copy().set_index("threshold")
    st.line_chart(chart_df, use_container_width=True)

    display_df = score_df.copy()
    valid_pred_mask = display_df["image_score"].notna().to_numpy(dtype=bool)
    pred_is_bad = np.zeros(display_df.shape[0], dtype=bool)
    pred_is_bad[valid_pred_mask] = (
        display_df.loc[valid_pred_mask, "image_score"].to_numpy(dtype=np.float64) >= float(current_threshold)
    )
    true_is_bad = display_df["true_is_bad"].to_numpy(dtype=np.int32).astype(bool)
    display_df["predicted_label"] = np.where(
        valid_pred_mask,
        np.where(pred_is_bad, "bad", "good"),
        "ERROR",
    )
    display_df["outcome"] = np.where(
        ~valid_pred_mask,
        "ERROR",
        np.where(
            pred_is_bad & true_is_bad,
            "TP",
            np.where(
                pred_is_bad & ~true_is_bad,
                "FP",
                np.where(~pred_is_bad & true_is_bad, "FN", "TN"),
            ),
        ),
    )
    st.dataframe(
        display_df[
            [
                "sample",
                "split",
                "image_score",
                "predicted_label",
                "outcome",
                "num_valid_patches",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    artifact_paths = stored.get("artifact_paths", {})
    if artifact_paths:
        st.caption(
            "Artefakte: "
            f"scores={artifact_paths.get('scores_csv', '')} | "
            f"sweep={artifact_paths.get('threshold_sweep_csv', '')} | "
            f"summary={artifact_paths.get('summary_json', '')}"
        )


def render_settings_mode(context: dict[str, Any], experiment_dir_str: str, seed: int) -> None:
    _ = context
    st.subheader("Einstellungen")
    settings_tabs = st.tabs(["Anomaly Detection"])
    with settings_tabs[0]:
        render_anomaly_detection_settings_mode(experiment_dir_str, seed)


def render_component_test_mode(context: dict[str, Any], experiment_dir_str: str, seed: int) -> None:
    st.subheader("Bauteil-Test")
    st.caption(
        "Der Bauteil-Test kann vorhandene Run-Bilder verwenden oder externe Bilder mit frischer "
        "End-to-End-Inferenz durch AnomalyDINO, ROI-Extraktion und Boruta-Klassifikation schicken."
    )

    sample_rows = context["samples"]
    groups = ["all"] + sorted({row["evaluation_group"] for row in sample_rows})
    source_mode = st.radio(
        "Bildquelle",
        options=["Run-Bilder", "Externe Uploads"],
        horizontal=True,
        key="component_test_source_mode",
    )

    selected_samples: list[str] = []
    selected_object_name = ""
    uploaded_files = []
    live_config: dict[str, Any] | None = None
    if source_mode == "Externe Uploads":
        try:
            experiment_dir = Path(experiment_dir_str).resolve()
            data_root = Path(str(load_run_args(experiment_dir).get("data_root", ""))).resolve()
            run_object_names = sorted({sample.object_name for sample in load_run_samples(experiment_dir, seed=int(seed))})
            train_good_signatures = tuple(
                (
                    object_name,
                    _directory_file_signature(data_root / object_name / "train" / "good"),
                )
                for object_name in run_object_names
            )
            live_config = load_component_test_live_config(
                experiment_dir_str,
                int(seed),
                train_good_signatures=train_good_signatures,
            )
        except Exception as exc:
            st.error(f"Live-Inferenz konnte nicht vorbereitet werden: {exc}")
            return

    with st.form("component_test_form"):
        config_col, option_col = st.columns([1.2, 1.0])
        with config_col:
            if source_mode == "Run-Bilder":
                test_group = st.selectbox("Sample Group", groups, index=0, key="component_test_group")
                if test_group == "all":
                    filtered_rows = sample_rows
                else:
                    filtered_rows = [row for row in sample_rows if row["evaluation_group"] == test_group]
                sample_options = [row["sample"] for row in filtered_rows]
                selected_samples = st.multiselect(
                    "Zu testende Bilder",
                    options=sample_options,
                    default=[],
                    key="component_test_samples",
                )
            else:
                object_names = list((live_config or {}).get("object_names", []))
                if not object_names:
                    st.warning("Im aktuellen Run wurden keine Objekte fuer frische Inferenz gefunden.")
                else:
                    selected_object_name = st.selectbox(
                        "Objekt",
                        object_names,
                        index=0,
                        key="component_test_external_object",
                    )
                    uploaded_files = st.file_uploader(
                        "Externe Bilder hochladen",
                        type=COMPONENT_TEST_UPLOAD_EXTENSIONS,
                        accept_multiple_files=True,
                        key="component_test_uploads",
                    )
                    confirmed_config = load_anomaly_detection_confirmed_config(experiment_dir_str, selected_object_name)
                    if confirmed_config is not None:
                        try:
                            confirmed_reference_paths = _resolve_confirmed_reference_paths(
                                experiment_dir_str,
                                selected_object_name,
                                confirmed_config,
                            )
                            object_threshold = float(confirmed_config["threshold"])
                            ref_count = int(len(confirmed_reference_paths))
                            source_label = "Einstellungen"
                        except Exception as exc:
                            object_threshold = float("nan")
                            ref_count = 0
                            source_label = f"Einstellungen ungueltig: {exc}"
                    else:
                        object_threshold = float((live_config or {}).get("threshold_by_object", {}).get(selected_object_name, np.nan))
                        ref_count = len((live_config or {}).get("current_train_good_images_by_object", {}).get(selected_object_name, []))
                        source_label = "Run-Default"
                    st.caption(
                        f"Frische Inferenz mit Objekt={selected_object_name} | "
                        f"Threshold={object_threshold:.6f} | Referenzbilder={ref_count} | Quelle={source_label}"
                    )
        with option_col:
            render_overlays = st.checkbox("Overlays rendern", value=False, key="component_test_render_overlays")
            if source_mode == "Run-Bilder":
                st.caption("Wenn aktiv, werden pro Bild ROI-Boxen mit 2D/3D-Labels direkt in der App gerendert.")
            else:
                st.caption(
                    "Frische Inferenz laedt den Backbone, baut die Referenzbank und extrahiert danach "
                    "ROIs fuer jedes hochgeladene Bild."
                )
        start_test = st.form_submit_button("Test starten", use_container_width=True)

    if start_test:
        classifier_dir = Path(experiment_dir_str).resolve() / BORUTA_COMPONENT_CLASSIFIER_FEATURE_SUBDIR
        cache_signature = _classifier_file_signature(
            classifier_dir / "selected_features.csv",
            classifier_dir / "selected_feature_indices.npy",
            classifier_dir / "classifier_pipeline.joblib",
            classifier_dir / "model_info.json",
            classifier_dir / "summary.json",
        )
        try:
            classifier_info = load_boruta_component_classifier_model(experiment_dir_str, cache_signature)
        except Exception as exc:
            st.error(f"Bauteil-Test konnte nicht initialisiert werden: {exc}")
            return

        progress_bar = st.progress(0.0)
        status_box = st.empty()
        results: list[dict[str, Any]] = []

        if source_mode == "Run-Bilder":
            if not selected_samples:
                st.warning("Waehle mindestens ein Bild aus.")
                return
            try:
                roi_table = load_component_test_roi_table(experiment_dir_str)
            except Exception as exc:
                st.error(f"ROI-Tabelle konnte nicht geladen werden: {exc}")
                return

            total = len(selected_samples)
            for index, sample_name in enumerate(selected_samples, start=1):
                status_box.info(f"Teste {index}/{total}: {sample_name}")
                result = _run_component_test_for_sample(
                    experiment_dir_str=experiment_dir_str,
                    seed=int(seed),
                    sample_name=sample_name,
                    roi_table=roi_table,
                    classifier_info=classifier_info,
                    render_overlays=bool(render_overlays),
                )
                results.append(result)
                progress_bar.progress(index / float(total))
        else:
            if not selected_object_name:
                st.warning("Waehle ein Objekt aus.")
                return
            if not uploaded_files:
                st.warning("Lade mindestens ein Bild hoch.")
                return
            try:
                experiment_dir = Path(experiment_dir_str).resolve()
                load_component_test_backbone(experiment_dir_str)
                confirmed_config = load_anomaly_detection_confirmed_config(experiment_dir_str, selected_object_name)
                if confirmed_config is not None:
                    confirmed_reference_paths = _resolve_confirmed_reference_paths(
                        experiment_dir_str,
                        selected_object_name,
                        confirmed_config,
                    )
                    load_anomaly_detection_reference_bank(
                        experiment_dir_str,
                        int(seed),
                        selected_object_name,
                        tuple(str(path.resolve()) for path in confirmed_reference_paths),
                        _file_signature_from_paths(confirmed_reference_paths),
                    )
                else:
                    data_root = Path(str(load_run_args(experiment_dir).get("data_root", ""))).resolve()
                    train_good_signature = _directory_file_signature(data_root / selected_object_name / "train" / "good")
                    load_component_test_reference_bank(
                        experiment_dir_str,
                        int(seed),
                        selected_object_name,
                        train_good_signature=train_good_signature,
                    )
            except Exception as exc:
                st.error(f"Frische Inferenz konnte nicht vorbereitet werden: {exc}")
                return

            total = len(uploaded_files)
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                status_box.info(f"Teste {index}/{total}: {uploaded_file.name}")
                try:
                    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                    image_rgb = np.asarray(image, dtype=np.uint8)
                    result = _run_component_test_for_external_image(
                        experiment_dir_str=experiment_dir_str,
                        seed=int(seed),
                        object_name=selected_object_name,
                        sample_name=str(uploaded_file.name),
                        image_rgb=image_rgb,
                        classifier_info=classifier_info,
                        render_overlays=bool(render_overlays),
                    )
                except Exception as exc:
                    result = {
                        "sample": str(uploaded_file.name),
                        "evaluation_group": f"uploaded/{selected_object_name}",
                        "image_score": float("nan"),
                        "image_threshold": float("nan"),
                        "anomalous_patch_count": 0,
                        "io_nio": "ERROR",
                        "part_label": "ERROR",
                        "num_rois": 0,
                        "num_2d_rois": 0,
                        "num_3d_rois": 0,
                        "max_roi_p3d": 0.0,
                        "max_roi_confidence": 0.0,
                        "note": f"Fehler in frischer Inferenz: {exc}",
                        "roi_predictions": [],
                        "overlay_rgb": None,
                    }
                results.append(result)
                progress_bar.progress(index / float(total))

        status_box.success(f"Test abgeschlossen fuer {len(results)} Bild(er).")
        st.session_state[COMPONENT_TEST_VIEW_KEY] = {
            "results": results,
            "render_overlays": bool(render_overlays),
        }

    stored = st.session_state.get(COMPONENT_TEST_VIEW_KEY)
    if not stored:
        return

    results = list(stored["results"])
    render_overlays = bool(stored["render_overlays"])
    summary_rows = [
        {
            "sample": row["sample"],
            "group": row["evaluation_group"],
            "image_score": row["image_score"],
            "image_threshold": row["image_threshold"],
            "anomalous_patches": row["anomalous_patch_count"],
            "io_nio": row["io_nio"],
            "part_label": row["part_label"],
            "num_rois": row["num_rois"],
            "num_2d_rois": row["num_2d_rois"],
            "num_3d_rois": row["num_3d_rois"],
            "max_roi_p3d": row["max_roi_p3d"],
            "max_roi_confidence": row["max_roi_confidence"],
            "note": row["note"],
        }
        for row in results
    ]

    st.subheader("Bauteil-Ergebnisse")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    for result in results:
        label = f"{result['sample']} -> {result['part_label']}"
        with st.expander(label, expanded=len(results) == 1):
            metric_cols = st.columns(6)
            metric_cols[0].metric("Bildscore", f"{result['image_score']:.5f}")
            metric_cols[1].metric("Threshold", f"{result['image_threshold']:.5f}")
            metric_cols[2].metric("IO/NIO", result["io_nio"])
            metric_cols[3].metric("Bauteil", result["part_label"])
            metric_cols[4].metric("ROIs", str(result["num_rois"]))
            metric_cols[5].metric("3D-ROIs", str(result["num_3d_rois"]))

            if result["note"]:
                st.warning(result["note"])

            if render_overlays and result["overlay_rgb"] is not None:
                st.image(result["overlay_rgb"], caption="ROI-Overlay", use_container_width=True)

            if result["roi_predictions"]:
                st.dataframe(pd.DataFrame(result["roi_predictions"]), use_container_width=True, hide_index=True)
            else:
                st.caption("Keine ROI-Klassifikation fuer dieses Bild.")


def main() -> None:
    st.set_page_config(page_title="AnomalyDINO Explorer", layout="wide")
    st.title("AnomalyDINO Explorer")

    default_experiment_dir = _parse_experiment_dir_from_argv()
    experiment_dir_str = st.sidebar.text_input("Experiment Directory", value=default_experiment_dir)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=999, value=0, step=1)

    try:
        context = load_run_context(experiment_dir_str, int(seed))
    except Exception as exc:
        st.error(f"Run konnte nicht geladen werden: {exc}")
        return

    view_mode = st.radio(
        "Ansicht",
        options=["Similarity Explorer", "Bauteil-Test", "Einstellungen"],
        index=0,
        horizontal=True,
    )
    if view_mode == "Bauteil-Test":
        render_component_test_mode(context, experiment_dir_str, int(seed))
        return
    if view_mode == "Einstellungen":
        render_settings_mode(context, experiment_dir_str, int(seed))
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
    use_top10pct_classifier_top32 = st.sidebar.checkbox(
        "Top10%-Klassifikator: nur die 32 I-Relief-Features auf Normalmap verwenden",
        value=False,
    )
    evaluate_top10pct_classifier = st.sidebar.checkbox(
        "Top1-Patch-Klassifikator auf Query-Patches auswerten",
        value=False,
    )
    if use_top10pct_classifier_top32:
        use_multilayer_irelief_normal = True
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
    need_multilayer_query = (
        (use_multilayer_irelief_normal and feature_source_mode in ("normal", "fused"))
        or use_top10pct_classifier_top32
        or evaluate_top10pct_classifier
    )
    if need_multilayer_query:
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
    top10pct_classifier_feature_info = None
    top10pct_classifier_feature_indices = None
    top10pct_classifier_model_info = None
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
    if use_top10pct_classifier_top32 and feature_source_mode in ("normal", "fused"):
        try:
            top10pct_classifier_feature_info = load_top10pct_classifier_feature_selection(experiment_dir_str)
            top10pct_classifier_feature_indices = np.asarray(
                top10pct_classifier_feature_info["feature_indices"],
                dtype=np.int32,
            )
        except Exception as exc:
            st.error(f"Top10%-Klassifikator-Featureauswahl konnte nicht geladen werden: {exc}")
            return
    if evaluate_top10pct_classifier:
        try:
            classifier_dir = Path(experiment_dir_str).resolve() / TOP1PATCH_CLASSIFIER_FEATURE_SUBDIR
            top1patch_cache_signature = _classifier_file_signature(
                classifier_dir / "selected_topk_features.csv",
                classifier_dir / "classifier_pipeline.joblib",
                classifier_dir / "irelief_feature_scale_sqrt.npy",
                classifier_dir / "summary.json",
                classifier_dir / "model_info.json",
            )
            top10pct_classifier_model_info = load_top1patch_classifier_model(
                experiment_dir_str,
                int(seed),
                top1patch_cache_signature,
            )
        except Exception as exc:
            st.error(f"Top1-Patch-Klassifikator konnte nicht geladen werden: {exc}")
            return

    feature_mode_display = feature_mode
    normal_mode_bits: list[str] = []
    if use_multilayer_irelief_normal:
        normal_mode_bits.append("multilayer_l1to12_irelief")
    if use_top10pct_classifier_top32:
        normal_mode_bits.append("top10pct_classifier_top32")
    normal_mode_display = "raw"
    if normal_mode_bits:
        normal_mode_display = "raw + " + " + ".join(normal_mode_bits)
    if use_multilayer_irelief_normal and feature_source_mode == "normal":
        feature_mode_display = normal_mode_display
    elif use_multilayer_irelief_normal and feature_source_mode == "fused":
        feature_mode_display = f"normal:{normal_mode_display} | albedo:{feature_mode}"
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
            if use_top10pct_classifier_top32:
                assert top10pct_classifier_feature_indices is not None
                query_normal_features_for_similarity = _apply_feature_subset(
                    query_normal_features_for_similarity,
                    top10pct_classifier_feature_indices,
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
                if use_top10pct_classifier_top32:
                    assert top10pct_classifier_feature_indices is not None
                    target_normal_features_for_similarity = _apply_feature_subset(
                        target_normal_features_for_similarity,
                        top10pct_classifier_feature_indices,
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
        if use_top10pct_classifier_top32 and feature_source_mode in ("normal", "fused") and top10pct_classifier_feature_info is not None:
            reference_bits.append(
                f"top10pct_classifier_features={top10pct_classifier_feature_info['num_selected_features']}"
            )
        if evaluate_top10pct_classifier and top10pct_classifier_model_info is not None:
            reference_bits.append(f"top1patch_classifier_k={top10pct_classifier_model_info['fixed_k']}")
        elif use_irelief_patch_reweight_effective and feature_source_mode in ("normal", "fused") and irelief_info is not None:
            reference_bits.append(f"irelief_dim={irelief_info['feature_dim']}")
        if reference_bits:
            st.caption(" | ".join(reference_bits))
        if use_multilayer_irelief_normal and (positions_debiased or pca_subspace_debiased) and feature_source_mode in ("normal", "fused"):
            st.caption("Multi-Layer-I-Relief auf Normalmap laeuft bewusst ohne Positionskorrektur; Albedo bleibt unveraendert.")
        if use_irelief_patch_reweight and use_multilayer_irelief_normal and feature_source_mode in ("normal", "fused"):
            st.caption("Das alte Layer-12-I-Relief-Reweight ist in diesem Modus ohne Wirkung.")
        if use_top10pct_classifier_top32 and feature_source_mode in ("normal", "fused"):
            st.caption("Der Top10%-Featuremodus nutzt weiterhin den Multilayer-Normalmap-Zweig, wendet I-Relief-Reweight an und schneidet danach auf die gespeicherten 32 Top10%-Klassifikator-Merkmale.")
        if evaluate_top10pct_classifier:
            st.caption("Die Query-Auswahl wird zusaetzlich durch den Top1-Patch-Endklassifikator geschickt. Bei einem einzelnen Query-Patch ist die Softmax-Gewichtung automatisch 1.0.")
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
    classifier_result = None

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

    if evaluate_top10pct_classifier:
        assert top10pct_classifier_model_info is not None
        assert query_multilayer_assets is not None
        classifier_query_feature, _, _ = _query_feature_from_selected(
            query_multilayer_assets["features_norm"],
            query_assets["score_grid"],
            query_assets["grid_shape"],
            query_patches,
        )
        classifier_query_feature = _apply_irelief_reweight(
            classifier_query_feature[None, :],
            np.asarray(top10pct_classifier_model_info["scale_sqrt"], dtype=np.float32),
        )[0]
        classifier_query_feature = _apply_feature_subset(
            classifier_query_feature[None, :],
            np.asarray(top10pct_classifier_model_info["selected_indices"], dtype=np.int32),
            renormalize=False,
        )
        classifier_model = top10pct_classifier_model_info["model"]
        classifier_proba = classifier_model.predict_proba(classifier_query_feature.astype(np.float32))[0].astype(np.float32)
        classifier_pred_idx = int(classifier_model.predict(classifier_query_feature.astype(np.float32))[0])
        classifier_pred_label = "3D" if classifier_pred_idx == 1 else "2D"
        classifier_pred_confidence = float(classifier_proba[classifier_pred_idx])
        classifier_result = {
            "predicted_label": classifier_pred_label,
            "predicted_confidence": classifier_pred_confidence,
            "proba_2d": float(classifier_proba[0]),
            "proba_3d": float(classifier_proba[1]),
        }

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
        if classifier_result is not None:
            st.subheader("Top1-Patch Classifier")
            pred_col, conf_col = st.columns(2)
            with pred_col:
                st.metric("Vorhersage", classifier_result["predicted_label"])
            with conf_col:
                st.metric(
                    "Wahrscheinlichkeit der Vorhersage",
                    f"{classifier_result['predicted_confidence']*100:.1f}%",
                )
            st.caption(
                f"p(2D)={classifier_result['proba_2d']:.4f} | "
                f"p(3D)={classifier_result['proba_3d']:.4f}"
            )

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
