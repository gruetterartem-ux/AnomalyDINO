import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import yaml

from hq_sam_segment import load_predictor, mask_bbox, pick_mask, resolve_device, validate_box
from src.backbones import get_model


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_HQ_SAM_BATCH_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "hq_sam_outputs_batch"
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1_seed=0_sam_hq_vit_tiny"
)
DEFAULT_CHECKPOINT = Path(r"C:\ai\AnomalyDINO\checkpoints\hq_sam\sam_hq_vit_tiny.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute HQ-SAM masks for ROI prompts, pool dense DINO patch features over all "
            "patches touched by the mask, and write a label-ready CSV plus features.npy."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="AnomalyDINO experiment directory with args.yaml.",
    )
    parser.add_argument(
        "--hq-sam-batch-dir",
        type=Path,
        default=DEFAULT_HQ_SAM_BATCH_DIR,
        help="HQ-SAM batch directory that contains manifest.csv and combined overlays.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="HQ-SAM checkpoint used to recompute masks.",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_tiny",
        help="HQ-SAM model type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for both DINO and HQ-SAM.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
        help="Optional local checkpoint path for the DINO backbone.",
    )
    parser.add_argument(
        "--multimask-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use multimask HQ-SAM output and keep the mask with the best predicted IoU.",
    )
    parser.add_argument(
        "--hq-token-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the HQ token only.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional limit for how many samples are processed.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Optional per-image patch-feature cache directory. If a sample cache exists, no backbone forward pass is run for that sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <hq-sam-batch-dir>_maskpooled_features.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def make_labeling_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    labeling_keys = [
        "roi_uid",
        "label",
        "notes",
        "sample",
        "group_id",
        "roi_index",
        "object",
        "split",
        "image_path",
        "gallery_overlay_path",
        "combined_overlay_path",
        "roi_crop_path",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "sam_predicted_iou",
        "sam_mask_area_px",
        "selected_patch_count",
        "selected_patch_fraction",
        "region_max_score",
        "region_mass",
        "primary_peak_score",
    ]
    return [{key: row.get(key, "") for key in labeling_keys} for row in rows]


def load_run_args(experiment_dir: Path) -> Dict[str, object]:
    args_path = experiment_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Run arguments not found: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_manifest_rows(manifest_file: Path) -> List[Dict[str, str]]:
    if not manifest_file.exists():
        raise FileNotFoundError(f"HQ-SAM manifest not found: {manifest_file}")
    with manifest_file.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (row["sample"], int(row["roi_index"])))
    return rows


def group_rows_by_sample(rows: List[Dict[str, str]], limit_samples: int | None) -> List[Tuple[str, List[Dict[str, str]]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)
    sample_names = sorted(grouped.keys())
    if limit_samples is not None:
        sample_names = sample_names[:limit_samples]
    return [(sample_name, grouped[sample_name]) for sample_name in sample_names]


def default_output_dir(hq_sam_batch_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (hq_sam_batch_dir.parent / f"{hq_sam_batch_dir.name}_maskpooled_features").resolve()


def sample_to_gallery_overlay_path(hq_sam_batch_dir: Path, sample: str) -> Path:
    sample_path = Path(sample)
    stem = sample_path.stem + ".png"
    gallery_root = hq_sam_batch_dir / "combined_overlay_gallery_flat"
    if sample.startswith("test/bad/"):
        relative_dir = sample_path.relative_to("test/bad").parent
        return gallery_root / relative_dir / stem
    if sample.startswith("good_test/"):
        return gallery_root / "good_test" / stem
    if sample.startswith("good_train_remaining/"):
        return gallery_root / "good_train_remaining" / stem
    if sample.startswith("test/good/"):
        return gallery_root / "good_test" / stem
    return gallery_root / stem


def cache_file_for_sample(feature_cache_dir: Path, object_name: str, sample: str) -> Path:
    return feature_cache_dir / object_name / Path(sample).with_suffix(".npz")


def load_feature_cache(cache_file: Path) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], Tuple[int, int], int]:
    with np.load(cache_file) as payload:
        features = payload["features"].astype(np.float32)
        grid_size = tuple(int(value) for value in payload["grid_size"].tolist())
        resized_size = tuple(int(value) for value in payload["resized_size"].tolist())
        original_size = tuple(int(value) for value in payload["original_size"].tolist())
        patch_size = int(payload["patch_size"].reshape(-1)[0])
    return features, grid_size, resized_size, original_size, patch_size


def save_feature_cache(
    cache_file: Path,
    sample_name: str,
    image_path: Path,
    object_name: str,
    features: np.ndarray,
    grid_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    original_size: Tuple[int, int],
    patch_size: int,
) -> None:
    ensure_dir(cache_file.parent)
    np.savez_compressed(
        cache_file,
        object=object_name,
        sample=sample_name,
        image_path=str(image_path),
        features=features.astype(np.float16),
        grid_size=np.asarray(grid_size, dtype=np.int32),
        resized_size=np.asarray(resized_size, dtype=np.int32),
        original_size=np.asarray(original_size, dtype=np.int32),
        patch_size=np.asarray([int(patch_size)], dtype=np.int32),
    )


def resize_binary_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    resized = mask_image.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def mask_to_patch_selector(mask: np.ndarray, grid_size: Tuple[int, int], patch_size: int) -> np.ndarray:
    grid_h, grid_w = grid_size
    cropped_h = grid_h * patch_size
    cropped_w = grid_w * patch_size
    cropped_mask = mask[:cropped_h, :cropped_w]
    patch_mask = cropped_mask.reshape(grid_h, patch_size, grid_w, patch_size).any(axis=(1, 3))
    return patch_mask.reshape(-1)


def roi_box_patch_selector(
    box_xyxy: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    x_min, y_min, x_max, y_max = box_xyxy
    orig_w, orig_h = original_size
    resized_w, resized_h = resized_size
    grid_h, grid_w = grid_size
    cropped_h = grid_h * patch_size
    cropped_w = grid_w * patch_size

    x_min_resized = int(np.floor(x_min * resized_w / orig_w))
    x_max_resized = int(np.ceil(x_max * resized_w / orig_w))
    y_min_resized = int(np.floor(y_min * resized_h / orig_h))
    y_max_resized = int(np.ceil(y_max * resized_h / orig_h))

    x_min_resized = max(0, min(x_min_resized, cropped_w - 1))
    y_min_resized = max(0, min(y_min_resized, cropped_h - 1))
    x_max_resized = max(x_min_resized + 1, min(x_max_resized, cropped_w))
    y_max_resized = max(y_min_resized + 1, min(y_max_resized, cropped_h))

    patch_mask = np.zeros((grid_h, grid_w), dtype=bool)
    patch_col_min = max(0, x_min_resized // patch_size)
    patch_col_max = min(grid_w, int(np.ceil(x_max_resized / patch_size)))
    patch_row_min = max(0, y_min_resized // patch_size)
    patch_row_max = min(grid_h, int(np.ceil(y_max_resized / patch_size)))
    patch_mask[patch_row_min:patch_row_max, patch_col_min:patch_col_max] = True
    return patch_mask.reshape(-1)


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    hq_sam_batch_dir = args.hq_sam_batch_dir.resolve()
    output_dir = default_output_dir(hq_sam_batch_dir, args.output_dir)
    ensure_dir(output_dir)
    feature_cache_dir = None if args.feature_cache_dir is None else args.feature_cache_dir.resolve()

    run_args = load_run_args(experiment_dir)
    model_name = str(run_args["model_name"])
    resolution = int(run_args["resolution"])
    device = resolve_device(args.device)

    dino_model = get_model(
        model_name,
        device,
        smaller_edge_size=resolution,
        weights_path=args.backbone_weights,
    )
    predictor = load_predictor(
        model_type=args.sam_model_type,
        checkpoint=args.checkpoint.resolve(),
        device=device,
    )

    manifest_rows = load_manifest_rows(hq_sam_batch_dir / "manifest.csv")
    grouped_rows = group_rows_by_sample(manifest_rows, args.limit_samples)

    feature_rows: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []
    feature_index = 0

    for sample_idx, (sample_name, sample_rows) in enumerate(grouped_rows, start=1):
        image_path = Path(sample_rows[0]["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Original image not found: {image_path}")

        with Image.open(image_path) as image_handle:
            base_rgb = image_handle.convert("RGB")

        original_size = base_rgb.size
        cache_used = False
        if feature_cache_dir is not None:
            cache_file = cache_file_for_sample(feature_cache_dir, sample_rows[0]["object"], sample_name)
            if cache_file.exists():
                features, grid_size, resized_size, cached_original_size, cached_patch_size = load_feature_cache(cache_file)
                if cached_original_size != original_size:
                    raise ValueError(
                        f"Original size mismatch for cache {cache_file}: {cached_original_size} vs {original_size}"
                    )
                if cached_patch_size != int(dino_model.patch_size):
                    raise ValueError(
                        f"Patch size mismatch for cache {cache_file}: {cached_patch_size} vs {dino_model.patch_size}"
                    )
                cache_used = True
            else:
                resized_image = dino_model.resize_transform(base_rgb)
                resized_size = resized_image.size
                image_tensor, grid_size = dino_model.prepare_image(base_rgb)
                features = dino_model.extract_features(image_tensor).astype(np.float32)
                save_feature_cache(
                    cache_file=cache_file,
                    sample_name=sample_name,
                    image_path=image_path,
                    object_name=sample_rows[0]["object"],
                    features=features,
                    grid_size=grid_size,
                    resized_size=resized_size,
                    original_size=original_size,
                    patch_size=int(dino_model.patch_size),
                )
        else:
            resized_image = dino_model.resize_transform(base_rgb)
            resized_size = resized_image.size
            image_tensor, grid_size = dino_model.prepare_image(base_rgb)
            features = dino_model.extract_features(image_tensor).astype(np.float32)

        grid_h, grid_w = grid_size
        if features.shape[0] != grid_h * grid_w:
            raise ValueError(
                f"Feature/grid mismatch for {sample_name}: {features.shape[0]} vs {grid_h}x{grid_w}"
            )

        predictor.set_image(np.asarray(base_rgb))
        combined_overlay_path = hq_sam_batch_dir / Path(sample_name).with_suffix("") / "combined_overlay.png"
        gallery_overlay_path = sample_to_gallery_overlay_path(hq_sam_batch_dir, sample_name)

        for row in sorted(sample_rows, key=lambda current: int(current["roi_index"])):
            prompt_box = validate_box(row, base_rgb.width, base_rgb.height)
            box = np.asarray(prompt_box, dtype=np.float32)
            masks, iou_predictions, _ = predictor.predict(
                box=box,
                multimask_output=args.multimask_output,
                hq_token_only=args.hq_token_only,
            )
            best_mask, best_mask_index, best_iou = pick_mask(masks, iou_predictions)
            resized_mask = resize_binary_mask(best_mask, resized_size)
            patch_selector = mask_to_patch_selector(resized_mask, grid_size, dino_model.patch_size)
            fallback_used = False
            if not np.any(patch_selector):
                patch_selector = roi_box_patch_selector(
                    prompt_box,
                    original_size=original_size,
                    resized_size=resized_size,
                    grid_size=grid_size,
                    patch_size=dino_model.patch_size,
                )
                fallback_used = True

            pooled_feature = features[patch_selector].mean(axis=0)
            feature_rows.append(pooled_feature.astype(np.float32))

            bbox = mask_bbox(best_mask)
            metadata_rows.append(
                {
                    "feature_index": feature_index,
                    "roi_uid": f"{sample_name}__roi_{int(row['roi_index']):03d}",
                    "label": "",
                    "notes": "",
                    "sample": sample_name,
                    "group_id": sample_name,
                    "roi_index": int(row["roi_index"]),
                    "object": row.get("object", ""),
                    "split": row.get("split", ""),
                    "image_path": str(image_path),
                    "combined_overlay_path": str(combined_overlay_path),
                    "gallery_overlay_path": str(gallery_overlay_path),
                    "roi_crop_path": row.get("crop_path", ""),
                    "x_min": int(prompt_box[0]),
                    "y_min": int(prompt_box[1]),
                    "x_max": int(prompt_box[2]),
                    "y_max": int(prompt_box[3]),
                    "box_area_px": int(row.get("box_area_px") or 0),
                    "sam_selected_mask_index": int(best_mask_index),
                    "sam_predicted_iou": float(best_iou),
                    "sam_mask_area_px": int(best_mask.sum()),
                    "sam_mask_bbox_x_min": "" if bbox is None else int(bbox[0]),
                    "sam_mask_bbox_y_min": "" if bbox is None else int(bbox[1]),
                    "sam_mask_bbox_x_max": "" if bbox is None else int(bbox[2]),
                    "sam_mask_bbox_y_max": "" if bbox is None else int(bbox[3]),
                    "selected_patch_count": int(patch_selector.sum()),
                    "total_patch_count": int(grid_h * grid_w),
                    "selected_patch_fraction": float(patch_selector.mean()),
                    "mask_to_patch_fallback_used": int(fallback_used),
                    "region_max_score": float(row.get("region_max_score") or 0.0),
                    "region_mass": float(row.get("region_mass") or 0.0),
                    "primary_peak_score": float(row.get("primary_peak_score") or 0.0),
                    "backbone_model_name": model_name,
                    "backbone_resolution": resolution,
                    "backbone_patch_size": int(dino_model.patch_size),
                    "feature_dim": int(features.shape[1]),
                }
            )
            feature_index += 1

        print(
            f"[{sample_idx}/{len(grouped_rows)}] Processed {sample_name} "
            f"({len(sample_rows)} ROI(s), grid={grid_h}x{grid_w}, cache={'hit' if cache_used else 'miss'})"
        )

    feature_array = np.stack(feature_rows, axis=0).astype(np.float32)
    features_file = output_dir / "roi_features_mean.npy"
    labels_table_file = output_dir / "roi_feature_table.csv"
    labeling_table_file = output_dir / "roi_labeling_table.csv"
    summary_file = output_dir / "summary.json"

    np.save(features_file, feature_array)
    write_csv(metadata_rows, labels_table_file)
    write_csv(make_labeling_rows(metadata_rows), labeling_table_file)
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "hq_sam_batch_dir": str(hq_sam_batch_dir),
            "checkpoint": str(args.checkpoint.resolve()),
            "sam_model_type": args.sam_model_type,
            "model_name": model_name,
            "resolution": resolution,
            "num_samples": len(grouped_rows),
            "num_rois": len(metadata_rows),
            "feature_dim": int(feature_array.shape[1]),
            "features_file": str(features_file),
            "labels_table_file": str(labels_table_file),
            "labeling_table_file": str(labeling_table_file),
        },
        summary_file,
    )

    print(f"Saved features: {features_file}")
    print(f"Saved label table: {labels_table_file}")
    print(f"Saved labeling table: {labeling_table_file}")
    print(f"Saved summary: {summary_file}")
    print(f"Feature shape: {feature_array.shape}")


if __name__ == "__main__":
    main()
