import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, label, maximum_filter
from torchvision import transforms


EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)
EIGHT_CONNECTED_BOOL = EIGHT_CONNECTED.astype(bool)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ROI crops from AnomalyDINO patch-distance maps using adaptive peak-centered hysteresis on d_masked."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(
            r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704\16-shot_preprocess=force_no_mask_no_rotation_all16_test_maxpatch_random"
        ),
        help="Experiment directory that contains args.yaml, measurements_seed=*.csv and anomaly_maps/.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed to process.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <experiment-dir>/roi_crops_peak_hysteresis/seed=<seed>.",
    )
    parser.add_argument(
        "--high-prominence-ratio",
        type=float,
        default=0.6,
        help="High hysteresis threshold as b + ratio * (p - b).",
    )
    parser.add_argument(
        "--low-prominence-ratio",
        type=float,
        default=0.3,
        help="Low hysteresis threshold as b + ratio * (p - b).",
    )
    parser.add_argument(
        "--background-ring-inner",
        type=int,
        default=2,
        help="Inner Chebyshev distance in patch cells for the local-background ring.",
    )
    parser.add_argument(
        "--background-ring-outer",
        type=int,
        default=5,
        help="Outer Chebyshev distance in patch cells for the local-background ring.",
    )
    parser.add_argument(
        "--min-region-patches",
        type=int,
        default=1,
        help="Minimum number of patch cells required for an accepted hysteresis region.",
    )
    parser.add_argument(
        "--min-prominence",
        type=float,
        default=0.02,
        help="Minimum local prominence p - b required for an accepted region.",
    )
    parser.add_argument(
        "--min-region-mass",
        type=float,
        default=0.05,
        help="Minimum summed excess score over background within the region.",
    )
    parser.add_argument(
        "--block-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After accepting a region, block its full patch-space bounding box for all later ROIs.",
    )
    parser.add_argument(
        "--merge-gap-patches",
        type=int,
        default=0,
        help="Merge accepted regions whose patch-space bounding boxes touch or come within this many patch cells.",
    )
    parser.add_argument(
        "--merge-bridge-ratio",
        type=float,
        default=0.4,
        help="Only merge nearby regions when they are connected above max(b1,b2) + ratio * min(d1,d2), i.e. without a clear valley.",
    )
    parser.add_argument(
        "--max-boxes-per-image",
        type=int,
        default=None,
        help="Keep only the N strongest final boxes per image. Strength is ranked by region_max_score, then region_mass.",
    )
    parser.add_argument(
        "--include-good",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export ROIs for images from the good split if they exceed the threshold.",
    )
    parser.add_argument(
        "--save-overlay-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one debug image per processed sample with all accepted ROI boxes drawn on the original image.",
    )
    return parser.parse_args()


def load_run_args(experiment_dir: Path) -> Dict:
    args_path = experiment_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Could not find run arguments: {args_path}")

    with args_path.open("r", encoding="utf-8") as handle:
        run_args = yaml.safe_load(handle)

    if run_args.get("aggregation_statistics") != "max_patch_distance":
        raise ValueError(
            "This ROI export expects runs with aggregation_statistics=max_patch_distance, "
            f"got {run_args.get('aggregation_statistics')!r}."
        )

    model_name = str(run_args.get("model_name", ""))
    if not model_name.startswith("dinov2"):
        raise ValueError(
            "This script currently supports DINOv2 runs only, because the ROI mapping reuses "
            "the DINOv2 resize + crop logic."
        )

    return run_args


def infer_patch_multiple(model_name: str) -> int:
    if model_name.startswith("dinov2"):
        return 14
    raise ValueError(f"Could not infer patch size for model {model_name!r}.")


def load_measurements(measurements_file: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not measurements_file.exists():
        raise FileNotFoundError(f"Could not find measurements file: {measurements_file}")

    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    with measurements_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = row["Sample"].replace("\\", "/")
            sample_no_ext = str(Path(sample).with_suffix("")).replace("\\", "/")
            rows[(row["Object"], sample_no_ext)] = row
    return rows


def resized_image_for_dinov2(image: Image.Image, smaller_edge_size: int) -> Image.Image:
    resize = transforms.Resize(
        size=smaller_edge_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    return resize(image)


def component_boxes(binary_map: np.ndarray) -> Iterable[Tuple[int, int, int, int, np.ndarray]]:
    labeled, num_components = label(binary_map.astype(np.uint8), structure=EIGHT_CONNECTED)
    for component_id in range(1, num_components + 1):
        rows, cols = np.where(labeled == component_id)
        if rows.size == 0:
            continue
        yield rows.min(), rows.max(), cols.min(), cols.max(), labeled == component_id


def local_maxima_mask(score_map: np.ndarray, available_mask: np.ndarray) -> np.ndarray:
    masked_scores = np.full(score_map.shape, -np.inf, dtype=np.float32)
    masked_scores[available_mask] = score_map[available_mask]
    local_max = masked_scores == maximum_filter(masked_scores, footprint=EIGHT_CONNECTED, mode="constant", cval=-np.inf)
    local_max &= available_mask
    return local_max


def peak_candidates(score_map: np.ndarray, available_mask: np.ndarray) -> List[Dict[str, object]]:
    peak_mask = local_maxima_mask(score_map, available_mask)
    peak_labels, num_peak_regions = label(peak_mask.astype(np.uint8), structure=EIGHT_CONNECTED)

    candidates: List[Dict[str, object]] = []
    for peak_region_id in range(1, num_peak_regions + 1):
        region_mask = peak_labels == peak_region_id
        if not region_mask.any():
            continue
        peak_score = float(score_map[region_mask].max())
        max_coords = np.argwhere(region_mask & np.isclose(score_map, peak_score))
        candidates.append(
            {
                "label_id": peak_region_id,
                "seed_mask": region_mask,
                "peak_score": peak_score,
                "representative": (int(max_coords[0, 0]), int(max_coords[0, 1])),
                "plateau_size": int(region_mask.sum()),
                "peak_labels": peak_labels,
            }
        )

    candidates.sort(key=lambda candidate: candidate["peak_score"], reverse=True)
    return candidates


def estimate_local_background(
    score_map: np.ndarray,
    seed_mask: np.ndarray,
    available_mask: np.ndarray,
    valid_mask: np.ndarray,
    inner_radius: int,
    outer_radius: int,
) -> Tuple[float, np.ndarray, str]:
    outer = binary_dilation(seed_mask, structure=EIGHT_CONNECTED_BOOL, iterations=outer_radius)
    inner = binary_dilation(seed_mask, structure=EIGHT_CONNECTED_BOOL, iterations=inner_radius)
    ring_mask = outer & ~inner
    ring_valid = ring_mask & available_mask & valid_mask
    if ring_valid.any():
        return float(np.median(score_map[ring_valid])), ring_valid, "ring"

    fallback_valid = available_mask & valid_mask & ~seed_mask
    if fallback_valid.any():
        return float(np.median(score_map[fallback_valid])), ring_valid, "fallback_free"

    return float(np.median(score_map[seed_mask])), ring_valid, "seed_only"


def hysteresis_component(
    score_map: np.ndarray,
    seed_mask: np.ndarray,
    available_mask: np.ndarray,
    high_threshold: float,
    low_threshold: float,
) -> np.ndarray:
    strong_mask = available_mask & (score_map >= high_threshold)
    strong_labels, _ = label(strong_mask.astype(np.uint8), structure=EIGHT_CONNECTED)
    strong_component_ids = np.unique(strong_labels[seed_mask])
    strong_component_ids = strong_component_ids[strong_component_ids > 0]
    if strong_component_ids.size == 0:
        return seed_mask.copy()

    strong_component_mask = np.isin(strong_labels, strong_component_ids)
    weak_mask = available_mask & (score_map >= low_threshold)
    weak_labels, _ = label(weak_mask.astype(np.uint8), structure=EIGHT_CONNECTED)
    weak_component_ids = np.unique(weak_labels[strong_component_mask])
    weak_component_ids = weak_component_ids[weak_component_ids > 0]
    if weak_component_ids.size == 0:
        return strong_component_mask

    return np.isin(weak_labels, weak_component_ids)


def region_peak_details(
    score_map: np.ndarray,
    available_mask: np.ndarray,
    region_mask: np.ndarray,
) -> List[Dict[str, object]]:
    details: List[Dict[str, object]] = []
    peak_mask = local_maxima_mask(score_map, available_mask)
    peak_labels, num_peak_regions = label(peak_mask.astype(np.uint8), structure=EIGHT_CONNECTED)

    for peak_region_id in range(1, num_peak_regions + 1):
        current_peak_mask = peak_labels == peak_region_id
        if not np.any(current_peak_mask & region_mask):
            continue
        peak_score = float(score_map[current_peak_mask].max())
        max_coords = np.argwhere(current_peak_mask & np.isclose(score_map, peak_score))
        details.append(
            {
                "peak_score": peak_score,
                "row": int(max_coords[0, 0]),
                "col": int(max_coords[0, 1]),
                "plateau_size": int(current_peak_mask.sum()),
            }
        )

    details.sort(key=lambda item: item["peak_score"], reverse=True)
    if not details and np.any(region_mask):
        masked_scores = np.full(score_map.shape, -np.inf, dtype=np.float32)
        masked_scores[region_mask] = score_map[region_mask]
        row, col = np.unravel_index(np.argmax(masked_scores), score_map.shape)
        details.append(
            {
                "peak_score": float(score_map[row, col]),
                "row": int(row),
                "col": int(col),
                "plateau_size": int(region_mask.sum()),
            }
        )
    return details


def patch_box_to_image_box(
    patch_box: Tuple[int, int, int, int],
    grid_shape: Tuple[int, int],
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    patch_multiple: int,
) -> Tuple[int, int, int, int]:
    row_min, row_max, col_min, col_max = patch_box
    grid_h, grid_w = grid_shape
    orig_w, orig_h = original_size
    resized_w, resized_h = resized_size

    cropped_w = resized_w - (resized_w % patch_multiple)
    cropped_h = resized_h - (resized_h % patch_multiple)

    processed_x_edges = np.rint(np.linspace(0, cropped_w, grid_w + 1)).astype(int)
    processed_y_edges = np.rint(np.linspace(0, cropped_h, grid_h + 1)).astype(int)
    original_x_edges = np.rint(processed_x_edges * orig_w / resized_w).astype(int)
    original_y_edges = np.rint(processed_y_edges * orig_h / resized_h).astype(int)

    x0 = int(original_x_edges[col_min])
    x1 = int(original_x_edges[col_max + 1])
    y0 = int(original_y_edges[row_min])
    y1 = int(original_y_edges[row_max + 1])

    x0 = max(0, min(x0, orig_w - 1))
    y0 = max(0, min(y0, orig_h - 1))
    x1 = max(x0 + 1, min(x1, orig_w))
    y1 = max(y0 + 1, min(y1, orig_h))

    return x0, y0, x1, y1


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def draw_roi_overlay(
    image: Image.Image,
    roi_entries: List[Dict[str, object]],
    output_path: Path,
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    palette = [
        "red",
        "lime",
        "cyan",
        "yellow",
        "magenta",
        "orange",
        "deepskyblue",
        "chartreuse",
    ]

    for roi_entry in roi_entries:
        color = palette[int(roi_entry["roi_index"]) % len(palette)]
        x0 = int(roi_entry["x_min"])
        y0 = int(roi_entry["y_min"])
        x1 = int(roi_entry["x_max"])
        y1 = int(roi_entry["y_max"])
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=4)
        label_text = (
            f"ROI {int(roi_entry['roi_index']):02d} | "
            f"peaks={int(roi_entry['peak_count_in_region'])} | "
            f"max={float(roi_entry['region_max_score']):.3f}"
        )
        draw.text((x0 + 6, max(4, y0 - 18)), label_text, fill=color)

    ensure_parent(output_path)
    canvas.save(output_path)


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    for png_file in output_dir.rglob("*.png"):
        png_file.unlink()

    metadata_file = output_dir / "roi_metadata.csv"
    if metadata_file.exists():
        metadata_file.unlink()


def region_box(region_mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows, cols = np.where(region_mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def region_mass(score_map: np.ndarray, region_mask: np.ndarray, background_value: float) -> float:
    return float(np.clip(score_map[region_mask] - background_value, a_min=0.0, a_max=None).sum())


def boxes_overlap(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> bool:
    a_row_min, a_row_max, a_col_min, a_col_max = box_a
    b_row_min, b_row_max, b_col_min, b_col_max = box_b
    row_overlap = max(a_row_min, b_row_min) <= min(a_row_max, b_row_max)
    col_overlap = max(a_col_min, b_col_min) <= min(a_col_max, b_col_max)
    return row_overlap and col_overlap


def boxes_within_gap(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
    gap: int,
) -> bool:
    a_row_min, a_row_max, a_col_min, a_col_max = box_a
    b_row_min, b_row_max, b_col_min, b_col_max = box_b
    row_close = not (a_row_max + gap + 1 < b_row_min or b_row_max + gap + 1 < a_row_min)
    col_close = not (a_col_max + gap + 1 < b_col_min or b_col_max + gap + 1 < a_col_min)
    return row_close and col_close


def merge_without_valley(
    score_map: np.ndarray,
    valid_mask: np.ndarray,
    region_a: np.ndarray,
    box_a: Tuple[int, int, int, int],
    background_a: float,
    prominence_a: float,
    region_b: np.ndarray,
    box_b: Tuple[int, int, int, int],
    background_b: float,
    prominence_b: float,
    gap: int,
    bridge_ratio: float,
) -> bool:
    if not boxes_within_gap(box_a, box_b, gap):
        return False

    bridge_threshold = max(background_a, background_b) + bridge_ratio * min(prominence_a, prominence_b)
    row_min = max(0, min(box_a[0], box_b[0]) - gap - 1)
    row_max = min(score_map.shape[0] - 1, max(box_a[1], box_b[1]) + gap + 1)
    col_min = max(0, min(box_a[2], box_b[2]) - gap - 1)
    col_max = min(score_map.shape[1] - 1, max(box_a[3], box_b[3]) + gap + 1)

    local_scores = score_map[row_min : row_max + 1, col_min : col_max + 1]
    local_valid = valid_mask[row_min : row_max + 1, col_min : col_max + 1]
    local_region_a = region_a[row_min : row_max + 1, col_min : col_max + 1]
    local_region_b = region_b[row_min : row_max + 1, col_min : col_max + 1]
    bridge_mask = local_valid & (local_scores >= bridge_threshold)
    if not np.any(local_region_a & bridge_mask) or not np.any(local_region_b & bridge_mask):
        return False

    bridge_labels, _ = label(bridge_mask.astype(np.uint8), structure=EIGHT_CONNECTED)
    component_ids_a = np.unique(bridge_labels[local_region_a & bridge_mask])
    component_ids_b = np.unique(bridge_labels[local_region_b & bridge_mask])
    component_ids_a = component_ids_a[component_ids_a > 0]
    component_ids_b = component_ids_b[component_ids_b > 0]
    if component_ids_a.size == 0 or component_ids_b.size == 0:
        return False
    return np.intersect1d(component_ids_a, component_ids_b).size > 0


def strength_key(score_map: np.ndarray, accepted_region: Dict[str, object]) -> Tuple[float, float, float, int]:
    return (
        float(score_map[accepted_region["region_mask"]].max()),
        region_mass(
            score_map,
            accepted_region["region_mask"],
            float(accepted_region["background_value"]),
        ),
        float(accepted_region["prominence"]),
        int(accepted_region["region_mask"].sum()),
    )


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    run_args = load_run_args(experiment_dir)
    if args.background_ring_inner < 0 or args.background_ring_outer <= args.background_ring_inner:
        raise ValueError("background ring must satisfy 0 <= inner < outer")
    if not (0.0 <= args.low_prominence_ratio <= args.high_prominence_ratio <= 1.0):
        raise ValueError("Prominence ratios must satisfy 0 <= low <= high <= 1")
    if args.min_region_patches < 1:
        raise ValueError("min_region_patches must be at least 1")
    if args.min_prominence < 0:
        raise ValueError("min_prominence must be non-negative")
    if args.min_region_mass < 0:
        raise ValueError("min_region_mass must be non-negative")
    if args.merge_gap_patches < 0:
        raise ValueError("merge_gap_patches must be non-negative")
    if not (0.0 <= args.merge_bridge_ratio <= 1.0):
        raise ValueError("merge_bridge_ratio must satisfy 0 <= ratio <= 1")
    if args.max_boxes_per_image is not None and args.max_boxes_per_image < 1:
        raise ValueError("max_boxes_per_image must be at least 1 when provided")

    measurements = load_measurements(experiment_dir / f"measurements_seed={args.seed}.csv")
    anomaly_maps_root = experiment_dir / "anomaly_maps" / f"seed={args.seed}"
    if not anomaly_maps_root.exists():
        raise FileNotFoundError(f"Could not find anomaly map directory: {anomaly_maps_root}")

    data_root = Path(run_args["data_root"])
    patch_multiple = infer_patch_multiple(str(run_args["model_name"]))
    resolution = int(run_args["resolution"])
    output_dir = args.output_dir or (experiment_dir / "roi_crops_peak_hysteresis" / f"seed={args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)
    overlay_dir = output_dir / "overlay_images"
    if args.save_overlay_images:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: List[Dict[str, object]] = []
    exported_rois = 0
    processed_images = 0

    for npy_file in anomaly_maps_root.rglob("*.npy"):
        rel_path = npy_file.relative_to(anomaly_maps_root)
        if len(rel_path.parts) < 4:
            continue

        object_name = rel_path.parts[0]
        split_name = rel_path.parts[1]
        sample_no_ext = str(Path(*rel_path.parts[2:]).with_suffix("")).replace("\\", "/")
        sample_key = (object_name, sample_no_ext)
        row = measurements.get(sample_key)
        if row is None:
            print(f"Skipping {rel_path}: no matching measurement row found.")
            continue

        if not args.include_good and row.get("Label") == "0":
            continue

        threshold = float(row["Threshold"])
        image_score = float(row["Anomaly_Score"])
        if image_score < threshold:
            continue

        d_masked = np.load(npy_file)
        valid_mask = d_masked > 0.0
        consumed_mask = np.zeros_like(valid_mask, dtype=bool)
        rejected_mask = np.zeros_like(valid_mask, dtype=bool)
        box_blocked_mask = np.zeros_like(valid_mask, dtype=bool)

        sample_rel_path = Path(row["Sample"].replace("\\", "/"))
        image_path = data_root / object_name / split_name / sample_rel_path
        if not image_path.exists():
            print(f"Skipping {rel_path}: source image not found at {image_path}.")
            continue

        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            original_size = image.size
            resized = resized_image_for_dinov2(image, resolution)
            resized_size = resized.size

            accepted_regions: List[Dict[str, object]] = []
            while True:
                available_mask = valid_mask & ~consumed_mask & ~rejected_mask & ~box_blocked_mask
                peak_candidates_current = peak_candidates(d_masked, available_mask)
                if not peak_candidates_current:
                    break

                best_candidate = peak_candidates_current[0]
                peak_score = float(best_candidate["peak_score"])
                if peak_score < threshold:
                    break

                seed_mask = best_candidate["seed_mask"]
                background_value, ring_mask, background_source = estimate_local_background(
                    d_masked,
                    seed_mask,
                    available_mask=available_mask,
                    valid_mask=valid_mask,
                    inner_radius=args.background_ring_inner,
                    outer_radius=args.background_ring_outer,
                )
                prominence = peak_score - background_value
                if prominence < args.min_prominence:
                    rejected_mask |= seed_mask
                    continue

                high_threshold = background_value + args.high_prominence_ratio * prominence
                low_threshold = background_value + args.low_prominence_ratio * prominence
                region_mask = hysteresis_component(
                    d_masked,
                    seed_mask=seed_mask,
                    available_mask=available_mask,
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                )
                if int(region_mask.sum()) < args.min_region_patches:
                    rejected_mask |= region_mask
                    continue

                region_max_score = float(d_masked[region_mask].max())
                if region_max_score < threshold:
                    rejected_mask |= region_mask
                    continue

                current_region_mass = region_mass(d_masked, region_mask, background_value)
                if current_region_mass < args.min_region_mass:
                    rejected_mask |= region_mask
                    continue

                region_row_min, region_row_max, region_col_min, region_col_max = region_box(region_mask)
                crop_patch_box = (region_row_min, region_row_max, region_col_min, region_col_max)
                consumed_mask |= region_mask
                merged_mask = region_mask.copy()
                merged_summary = {
                    "background_source": background_source,
                    "background_value": background_value,
                    "prominence": prominence,
                    "high_threshold": high_threshold,
                    "low_threshold": low_threshold,
                    "ring_patch_count": int(ring_mask.sum()),
                    "proposal_count": 1,
                }

                changed = True
                while changed:
                    changed = False
                    merged_box = region_box(merged_mask)
                    remaining_regions: List[Dict[str, object]] = []
                    for accepted_region in accepted_regions:
                        if merge_without_valley(
                            score_map=d_masked,
                            valid_mask=valid_mask,
                            region_a=merged_mask,
                            box_a=merged_box,
                            background_a=float(merged_summary["background_value"]),
                            prominence_a=float(merged_summary["prominence"]),
                            region_b=accepted_region["region_mask"],
                            box_b=accepted_region["patch_box"],
                            background_b=float(accepted_region["background_value"]),
                            prominence_b=float(accepted_region["prominence"]),
                            gap=args.merge_gap_patches,
                            bridge_ratio=args.merge_bridge_ratio,
                        ):
                            merged_mask |= accepted_region["region_mask"]
                            merged_summary["proposal_count"] += int(accepted_region["proposal_count"])
                            if float(accepted_region["prominence"]) > float(merged_summary["prominence"]):
                                merged_summary["background_source"] = accepted_region["background_source"]
                                merged_summary["background_value"] = accepted_region["background_value"]
                                merged_summary["prominence"] = accepted_region["prominence"]
                                merged_summary["high_threshold"] = accepted_region["high_threshold"]
                                merged_summary["low_threshold"] = accepted_region["low_threshold"]
                                merged_summary["ring_patch_count"] = accepted_region["ring_patch_count"]
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
                if args.block_boxes:
                    region_row_min, region_row_max, region_col_min, region_col_max = merged_box
                    box_blocked_mask[region_row_min : region_row_max + 1, region_col_min : region_col_max + 1] = True

            if accepted_regions:
                accepted_regions.sort(key=lambda accepted_region: strength_key(d_masked, accepted_region), reverse=True)
                non_overlapping_regions: List[Dict[str, object]] = []
                for accepted_region in accepted_regions:
                    if any(
                        boxes_overlap(accepted_region["patch_box"], kept_region["patch_box"])
                        for kept_region in non_overlapping_regions
                    ):
                        continue
                    non_overlapping_regions.append(accepted_region)
                accepted_regions = non_overlapping_regions
                if args.max_boxes_per_image is not None:
                    accepted_regions = accepted_regions[: args.max_boxes_per_image]
                image_overlay_rows: List[Dict[str, object]] = []
                for roi_index, accepted_region in enumerate(accepted_regions):
                    merged_mask = accepted_region["region_mask"]
                    region_row_min, region_row_max, region_col_min, region_col_max = accepted_region["patch_box"]
                    crop_patch_box = (region_row_min, region_row_max, region_col_min, region_col_max)
                    x0, y0, x1, y1 = patch_box_to_image_box(
                        crop_patch_box,
                        d_masked.shape,
                        original_size,
                        resized_size,
                        patch_multiple=patch_multiple,
                    )

                    crop = image.crop((x0, y0, x1, y1))
                    roi_path = output_dir / object_name / split_name / sample_rel_path.parent / (
                        f"{sample_rel_path.stem}__roi_{roi_index:02d}.png"
                    )
                    ensure_parent(roi_path)
                    crop.save(roi_path)

                    peaks_in_region = region_peak_details(d_masked, available_mask=valid_mask, region_mask=merged_mask)
                    primary_peak = peaks_in_region[0]
                    final_region_mass = region_mass(d_masked, merged_mask, float(accepted_region["background_value"]))
                    roi_metadata = {
                        "object": object_name,
                        "split": split_name,
                        "sample": row["Sample"],
                        "roi_index": roi_index,
                        "image_threshold": threshold,
                        "image_score": image_score,
                        "background_ring_inner": args.background_ring_inner,
                        "background_ring_outer": args.background_ring_outer,
                        "background_source": accepted_region["background_source"],
                        "background_value": accepted_region["background_value"],
                        "prominence": accepted_region["prominence"],
                        "min_prominence": args.min_prominence,
                        "high_threshold": accepted_region["high_threshold"],
                        "low_threshold": accepted_region["low_threshold"],
                        "region_mass": final_region_mass,
                        "min_region_mass": args.min_region_mass,
                        "merged_proposal_count": int(accepted_region["proposal_count"]),
                        "merge_gap_patches": args.merge_gap_patches,
                        "merge_bridge_ratio": args.merge_bridge_ratio,
                        "max_boxes_per_image": args.max_boxes_per_image if args.max_boxes_per_image is not None else "",
                        "primary_peak_score": primary_peak["peak_score"],
                        "primary_peak_row": primary_peak["row"],
                        "primary_peak_col": primary_peak["col"],
                        "primary_peak_plateau_size": primary_peak["plateau_size"],
                        "peak_count_in_region": len(peaks_in_region),
                        "peak_rows": ";".join(str(peak["row"]) for peak in peaks_in_region),
                        "peak_cols": ";".join(str(peak["col"]) for peak in peaks_in_region),
                        "peak_scores": ";".join(f"{peak['peak_score']:.6f}" for peak in peaks_in_region),
                        "peak_plateau_sizes": ";".join(str(peak["plateau_size"]) for peak in peaks_in_region),
                        "region_patch_count": int(merged_mask.sum()),
                        "region_max_score": float(d_masked[merged_mask].max()),
                        "region_mean_score": float(d_masked[merged_mask].mean()),
                        "region_row_min": int(region_row_min),
                        "region_row_max": int(region_row_max),
                        "region_col_min": int(region_col_min),
                        "region_col_max": int(region_col_max),
                        "ring_patch_count": int(accepted_region["ring_patch_count"]),
                        "x_min": x0,
                        "y_min": y0,
                        "x_max": x1,
                        "y_max": y1,
                        "crop_path": str(roi_path),
                    }
                    metadata_rows.append(roi_metadata)
                    image_overlay_rows.append(roi_metadata)
                    exported_rois += 1

                if args.save_overlay_images:
                    overlay_path = overlay_dir / object_name / split_name / sample_rel_path.parent / (
                        f"{sample_rel_path.stem}__overlay.png"
                    )
                    draw_roi_overlay(image, image_overlay_rows, overlay_path)
                processed_images += 1
                print(f"Exported {len(accepted_regions)} ROI(s) for {row['Sample']}")

    metadata_file = output_dir / "roi_metadata.csv"
    with metadata_file.open("w", newline="", encoding="utf-8") as handle:
        if metadata_rows:
            writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_rows)
        else:
            handle.write("")

    print(f"Processed images: {processed_images}")
    print(f"Exported ROIs: {exported_rois}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
