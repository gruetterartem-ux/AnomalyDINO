import argparse
import csv
import heapq
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import label, maximum_filter
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ROI crops from AnomalyDINO patch-distance maps using the saved image-level threshold."
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
        help="Optional output directory. Defaults to <experiment-dir>/roi_crops_peak_seeds/seed=<seed>.",
    )
    parser.add_argument(
        "--crop-size-patches",
        type=int,
        default=5,
        help="Fixed square crop size in patch units. Must be an odd number.",
    )
    parser.add_argument(
        "--grow-threshold-ratio",
        type=float,
        default=0.7,
        help="Grow threshold as a ratio of the saved image-level threshold.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=3,
        help="Maximum number of seed groups opened per image.",
    )
    parser.add_argument(
        "--min-seed-distance-patches",
        type=int,
        default=2,
        help="Minimum Chebyshev distance in patch units between opened seed groups.",
    )
    parser.add_argument(
        "--overlap-iou-threshold",
        type=float,
        default=0.5,
        help="Reject a weaker crop if its IoU with an already accepted stronger crop reaches this threshold.",
    )
    parser.add_argument(
        "--include-good",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also export ROIs for images from the good split if they exceed the threshold.",
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
    labeled, num_components = label(binary_map.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    for component_id in range(1, num_components + 1):
        rows, cols = np.where(labeled == component_id)
        if rows.size == 0:
            continue
        yield rows.min(), rows.max(), cols.min(), cols.max(), labeled == component_id


def local_maxima_mask(score_map: np.ndarray) -> np.ndarray:
    neighborhood = np.ones((3, 3), dtype=np.uint8)
    local_max = score_map == maximum_filter(score_map, footprint=neighborhood, mode="constant", cval=-np.inf)
    local_max &= score_map > 0.0
    return local_max


def build_peak_seed_regions(
    score_map: np.ndarray,
    seed_threshold: float,
    max_seeds: int,
    min_seed_distance_patches: int,
) -> Tuple[np.ndarray, Dict[int, float], Dict[int, int], Dict[int, Tuple[int, int]]]:
    peak_mask = local_maxima_mask(score_map)
    peak_labels, num_peak_regions = label(peak_mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))

    peak_regions: List[Tuple[float, int]] = []
    peak_centers: Dict[int, Tuple[float, float]] = {}
    peak_representatives: Dict[int, Tuple[int, int]] = {}
    for peak_region_id in range(1, num_peak_regions + 1):
        region_mask = peak_labels == peak_region_id
        peak_score = float(score_map[region_mask].max())
        peak_regions.append((peak_score, peak_region_id))
        rows, cols = np.where(region_mask)
        peak_centers[peak_region_id] = (float(rows.mean()), float(cols.mean()))
        max_coords = np.argwhere(region_mask & (score_map == peak_score))
        peak_representatives[peak_region_id] = (int(max_coords[0, 0]), int(max_coords[0, 1]))

    peak_regions.sort(key=lambda item: item[0], reverse=True)

    selected_seed_labels = np.zeros_like(peak_labels, dtype=np.int32)
    seed_strengths: Dict[int, float] = {}
    seed_ranks: Dict[int, int] = {}
    seed_centers: Dict[int, Tuple[int, int]] = {}
    selected_centers: List[Tuple[float, float]] = []
    next_seed_id = 1

    for peak_score, peak_region_id in peak_regions:
        if peak_score < seed_threshold:
            break
        if next_seed_id > max_seeds:
            break

        center_row, center_col = peak_centers[peak_region_id]
        too_close = False
        for selected_row, selected_col in selected_centers:
            chebyshev_distance = max(abs(center_row - selected_row), abs(center_col - selected_col))
            if chebyshev_distance <= min_seed_distance_patches:
                too_close = True
                break
        if too_close:
            continue

        selected_seed_labels[peak_labels == peak_region_id] = next_seed_id
        seed_strengths[next_seed_id] = peak_score
        seed_ranks[next_seed_id] = next_seed_id
        seed_centers[next_seed_id] = peak_representatives[peak_region_id]
        selected_centers.append((center_row, center_col))
        next_seed_id += 1

    return selected_seed_labels, seed_strengths, seed_ranks, seed_centers


def grow_seed_regions(
    score_map: np.ndarray,
    seed_labels: np.ndarray,
    seed_strengths: Dict[int, float],
    grow_threshold: float,
) -> np.ndarray:
    grow_mask = score_map >= grow_threshold
    assigned = np.zeros_like(seed_labels, dtype=np.int32)
    priority_queue: List[Tuple[int, float, int, int, int]] = []

    for seed_id, seed_strength in seed_strengths.items():
        for row, col in np.argwhere(seed_labels == seed_id):
            heapq.heappush(priority_queue, (0, -seed_strength, seed_id, int(row), int(col)))

    while priority_queue:
        distance, neg_seed_strength, seed_id, row, col = heapq.heappop(priority_queue)
        if assigned[row, col] != 0:
            continue
        if not grow_mask[row, col] and seed_labels[row, col] == 0:
            continue

        assigned[row, col] = seed_id

        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                next_row = row + row_offset
                next_col = col + col_offset
                if not (0 <= next_row < score_map.shape[0] and 0 <= next_col < score_map.shape[1]):
                    continue
                if assigned[next_row, next_col] != 0:
                    continue
                if not grow_mask[next_row, next_col]:
                    continue
                heapq.heappush(
                    priority_queue,
                    (distance + 1, neg_seed_strength, seed_id, next_row, next_col),
                )

    return assigned


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

    x0_processed = math.floor(col_min * cropped_w / grid_w)
    x1_processed = math.ceil((col_max + 1) * cropped_w / grid_w)
    y0_processed = math.floor(row_min * cropped_h / grid_h)
    y1_processed = math.ceil((row_max + 1) * cropped_h / grid_h)

    x0 = math.floor(x0_processed * orig_w / resized_w)
    x1 = math.ceil(x1_processed * orig_w / resized_w)
    y0 = math.floor(y0_processed * orig_h / resized_h)
    y1 = math.ceil(y1_processed * orig_h / resized_h)

    x0 = max(0, min(x0, orig_w - 1))
    y0 = max(0, min(y0, orig_h - 1))
    x1 = max(x0 + 1, min(x1, orig_w))
    y1 = max(y0 + 1, min(y1, orig_h))

    return x0, y0, x1, y1


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fixed_patch_window(
    center: Tuple[int, int],
    grid_shape: Tuple[int, int],
    crop_size_patches: int,
) -> Tuple[int, int, int, int]:
    if crop_size_patches <= 0 or crop_size_patches % 2 == 0:
        raise ValueError(f"crop_size_patches must be a positive odd number, got {crop_size_patches}")

    center_row, center_col = center
    grid_h, grid_w = grid_shape
    crop_h = min(crop_size_patches, grid_h)
    crop_w = min(crop_size_patches, grid_w)
    radius_h = crop_h // 2
    radius_w = crop_w // 2

    row_min = max(0, center_row - radius_h)
    row_max = row_min + crop_h - 1
    if row_max >= grid_h:
        row_max = grid_h - 1
        row_min = max(0, row_max - crop_h + 1)

    col_min = max(0, center_col - radius_w)
    col_max = col_min + crop_w - 1
    if col_max >= grid_w:
        col_max = grid_w - 1
        col_min = max(0, col_max - crop_w + 1)

    return row_min, row_max, col_min, col_max


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


def box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    a_row_min, a_row_max, a_col_min, a_col_max = box_a
    b_row_min, b_row_max, b_col_min, b_col_max = box_b

    inter_row_min = max(a_row_min, b_row_min)
    inter_row_max = min(a_row_max, b_row_max)
    inter_col_min = max(a_col_min, b_col_min)
    inter_col_max = min(a_col_max, b_col_max)

    if inter_row_min > inter_row_max or inter_col_min > inter_col_max:
        return 0.0

    inter_area = (inter_row_max - inter_row_min + 1) * (inter_col_max - inter_col_min + 1)
    area_a = (a_row_max - a_row_min + 1) * (a_col_max - a_col_min + 1)
    area_b = (b_row_max - b_row_min + 1) * (b_col_max - b_col_min + 1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    run_args = load_run_args(experiment_dir)

    measurements = load_measurements(experiment_dir / f"measurements_seed={args.seed}.csv")
    anomaly_maps_root = experiment_dir / "anomaly_maps" / f"seed={args.seed}"
    if not anomaly_maps_root.exists():
        raise FileNotFoundError(f"Could not find anomaly map directory: {anomaly_maps_root}")

    data_root = Path(run_args["data_root"])
    patch_multiple = infer_patch_multiple(str(run_args["model_name"]))
    resolution = int(run_args["resolution"])
    output_dir = args.output_dir or (experiment_dir / "roi_crops_peak_seeds" / f"seed={args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

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
        seed_threshold = threshold
        grow_threshold = args.grow_threshold_ratio * seed_threshold
        if grow_threshold > seed_threshold:
            raise ValueError(
                f"grow_threshold_ratio must produce a threshold <= seed threshold, got {grow_threshold} > {seed_threshold}"
            )

        seed_labels, seed_strengths, seed_ranks, seed_centers = build_peak_seed_regions(
            d_masked,
            seed_threshold,
            max_seeds=args.max_seeds,
            min_seed_distance_patches=args.min_seed_distance_patches,
        )
        if not seed_strengths:
            continue
        assigned_regions = grow_seed_regions(d_masked, seed_labels, seed_strengths, grow_threshold)

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

            roi_index = 0
            accepted_patch_boxes: List[Tuple[int, int, int, int]] = []
            for seed_id in sorted(seed_strengths):
                region_mask = assigned_regions == seed_id
                if not region_mask.any():
                    continue

                region_row_min, region_row_max, region_col_min, region_col_max = region_box(region_mask)
                crop_patch_box = fixed_patch_window(
                    seed_centers[seed_id],
                    d_masked.shape,
                    crop_size_patches=args.crop_size_patches,
                )

                if any(box_iou(crop_patch_box, accepted_box) >= args.overlap_iou_threshold for accepted_box in accepted_patch_boxes):
                    continue

                row_min, row_max, col_min, col_max = crop_patch_box
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
                accepted_patch_boxes.append(crop_patch_box)

                component_scores = d_masked[region_mask]
                seed_mask = seed_labels == seed_id
                metadata_rows.append(
                    {
                        "object": object_name,
                        "split": split_name,
                        "sample": row["Sample"],
                        "roi_index": roi_index,
                        "seed_threshold": seed_threshold,
                        "grow_threshold": grow_threshold,
                        "min_seed_distance_patches": args.min_seed_distance_patches,
                        "crop_size_patches": args.crop_size_patches,
                        "overlap_iou_threshold": args.overlap_iou_threshold,
                        "image_score": image_score,
                        "seed_id": seed_id,
                        "seed_rank": seed_ranks[seed_id],
                        "seed_max_score": seed_strengths[seed_id],
                        "seed_center_row": seed_centers[seed_id][0],
                        "seed_center_col": seed_centers[seed_id][1],
                        "roi_max_score": float(component_scores.max()),
                        "roi_mean_score": float(component_scores.mean()),
                        "region_patch_count": int(region_mask.sum()),
                        "seed_patch_count": int(seed_mask.sum()),
                        "region_row_min": int(region_row_min),
                        "region_row_max": int(region_row_max),
                        "region_col_min": int(region_col_min),
                        "region_col_max": int(region_col_max),
                        "crop_patch_row_min": int(row_min),
                        "crop_patch_row_max": int(row_max),
                        "crop_patch_col_min": int(col_min),
                        "crop_patch_col_max": int(col_max),
                        "x_min": x0,
                        "y_min": y0,
                        "x_max": x1,
                        "y_max": y1,
                        "crop_path": str(roi_path),
                    }
                )

                roi_index += 1
                exported_rois += 1

            if roi_index > 0:
                processed_images += 1
                print(f"Exported {roi_index} ROI(s) for {row['Sample']}")

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
