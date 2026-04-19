import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image, ImageColor, ImageDraw

warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.registry is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting tiny_vit_.* in registry with segment_anything_hq.*",
    category=UserWarning,
)

from segment_anything_hq import SamPredictor, sam_model_registry


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704\4-shot_preprocess=force_no_mask_no_rotation_all4_test_maxpatch_normalmap_my_own_4_20260410"
)
DEFAULT_ROI_METADATA_CSV = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge2bridge0.15_all"
    / "seed=0"
    / "roi_metadata.csv"
)
MASK_COLORS = [
    "#e63946",
    "#ff7f11",
    "#ffd166",
    "#2a9d8f",
    "#118ab2",
    "#6a4c93",
    "#ef476f",
    "#06d6a0",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HQ-SAM on explicit boxes or AnomalyDINO ROI boxes."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the HQ-SAM checkpoint file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        choices=sorted(sam_model_registry.keys()),
        help="HQ-SAM model type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device. Use cpu if CUDA is not available.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory with args.yaml. Needed to resolve original images from roi_metadata.csv.",
    )
    parser.add_argument(
        "--roi-metadata-csv",
        type=Path,
        default=DEFAULT_ROI_METADATA_CSV,
        help="ROI metadata CSV exported by show_heatmap.py.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Sample path from roi_metadata.csv, e.g. bad/2D/2026_03_11-09_22_05-974.png.",
    )
    parser.add_argument(
        "--all-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process all samples contained in roi_metadata.csv.",
    )
    parser.add_argument(
        "--roi-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of roi_index values from roi_metadata.csv.",
    )
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Optional limit after roi_index sorting. In batch mode this is applied per sample.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional limit for how many samples from roi_metadata.csv are processed.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Direct image path. If set, boxes must be passed with --box.",
    )
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        action="append",
        default=None,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Explicit XYXY box. Can be repeated.",
    )
    parser.add_argument(
        "--multimask-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return multiple masks per prompt and keep the one with the best predicted IoU.",
    )
    parser.add_argument(
        "--hq-token-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the HQ output token only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <experiment-dir>/hq_sam_outputs/<sample> or next to the image.",
    )
    parser.add_argument(
        "--save-per-roi-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-ROI mask/cutout/overlay files in addition to the combined image overview.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_sample_path(sample: str) -> str:
    return str(Path(sample).as_posix())


def slugify_sample(sample: str) -> Path:
    sample_path = Path(sample)
    return sample_path.with_suffix("")


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for HQ-SAM.") from exc
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to cpu.")
            return "cpu"
    return device_name


def load_run_args(experiment_dir: Path) -> Dict[str, object]:
    args_path = experiment_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Run arguments not found: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_roi_rows(roi_metadata_csv: Path) -> List[Dict[str, str]]:
    if not roi_metadata_csv.exists():
        raise FileNotFoundError(f"ROI metadata not found: {roi_metadata_csv}")
    with roi_metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_original_image_path(
    data_root: Path,
    roi_row: Dict[str, str],
) -> Path:
    object_name = roi_row["object"]
    split = roi_row["split"]
    sample = normalize_sample_path(roi_row["sample"])
    sample_path = Path(sample)
    if split == "custom_eval":
        if sample.startswith("good_train_remaining/"):
            image_path = data_root / object_name / "train" / "good" / sample_path.relative_to("good_train_remaining")
        elif sample.startswith("good_test/"):
            image_path = data_root / object_name / "test" / "good" / sample_path.relative_to("good_test")
        elif sample.startswith("test/bad/"):
            image_path = data_root / object_name / "test" / "bad" / sample_path.relative_to("test/bad")
        elif sample.startswith("test/good/"):
            image_path = data_root / object_name / "test" / "good" / sample_path.relative_to("test/good")
        else:
            raise FileNotFoundError(
                f"Could not resolve custom_eval sample {sample!r} for object {object_name!r}."
            )
    else:
        image_path = data_root / object_name / split / sample_path
    if not image_path.exists():
        raise FileNotFoundError(f"Original image not found: {image_path}")
    return image_path


def prompts_from_rows(rows: Sequence[Dict[str, str]], target_sample: str) -> List[Dict[str, object]]:
    prompts: List[Dict[str, object]] = []
    for row in rows:
        prompts.append(
            {
                "source": "roi_metadata",
                "roi_index": int(row["roi_index"]),
                "x_min": int(row["x_min"]),
                "y_min": int(row["y_min"]),
                "x_max": int(row["x_max"]),
                "y_max": int(row["y_max"]),
                "sample": target_sample,
                "object": row["object"],
                "split": row["split"],
                "primary_peak_score": float(row.get("primary_peak_score") or 0.0),
                "region_max_score": float(row.get("region_max_score") or 0.0),
                "region_mass": float(row.get("region_mass") or 0.0),
            }
        )
    return prompts


def collect_roi_prompts(
    data_root: Path,
    roi_metadata_csv: Path,
    sample: str,
    roi_indices: Optional[Sequence[int]],
    max_rois: Optional[int],
) -> Tuple[Path, List[Dict[str, object]]]:
    target_sample = normalize_sample_path(sample)
    rows = [
        row
        for row in load_roi_rows(roi_metadata_csv)
        if normalize_sample_path(row["sample"]) == target_sample
    ]
    if not rows:
        raise ValueError(f"No ROI rows found for sample {target_sample!r} in {roi_metadata_csv}")

    rows.sort(key=lambda row: int(row["roi_index"]))
    if roi_indices is not None:
        wanted = set(roi_indices)
        rows = [row for row in rows if int(row["roi_index"]) in wanted]
        if not rows:
            raise ValueError(
                f"None of the requested roi_index values {sorted(wanted)} were found for {target_sample!r}."
            )
    if max_rois is not None:
        rows = rows[:max_rois]

    image_path = resolve_original_image_path(data_root, rows[0])
    return image_path, prompts_from_rows(rows, target_sample)


def collect_all_roi_prompts(
    data_root: Path,
    roi_metadata_csv: Path,
    roi_indices: Optional[Sequence[int]],
    max_rois: Optional[int],
    limit_samples: Optional[int],
) -> List[Tuple[str, Path, List[Dict[str, object]]]]:
    grouped_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in load_roi_rows(roi_metadata_csv):
        grouped_rows[normalize_sample_path(row["sample"])].append(row)

    sample_names = sorted(grouped_rows.keys())
    if limit_samples is not None:
        sample_names = sample_names[:limit_samples]

    collected: List[Tuple[str, Path, List[Dict[str, object]]]] = []
    for sample_name in sample_names:
        rows = grouped_rows[sample_name]
        rows.sort(key=lambda row: int(row["roi_index"]))
        if roi_indices is not None:
            wanted = set(roi_indices)
            rows = [row for row in rows if int(row["roi_index"]) in wanted]
        if not rows:
            continue
        if max_rois is not None:
            rows = rows[:max_rois]
        image_path = resolve_original_image_path(data_root, rows[0])
        collected.append((sample_name, image_path, prompts_from_rows(rows, sample_name)))
    return collected


def collect_direct_prompts(image_path: Path, boxes: Optional[List[List[int]]]) -> Tuple[Path, List[Dict[str, object]]]:
    if boxes is None:
        raise ValueError("At least one --box is required when using --image.")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompts: List[Dict[str, object]] = []
    for idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        prompts.append(
            {
                "source": "direct_box",
                "roi_index": idx,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "sample": image_path.name,
            }
        )
    return image_path, prompts


def validate_box(box: Dict[str, object], width: int, height: int) -> Tuple[int, int, int, int]:
    x_min = max(0, min(int(box["x_min"]), width - 1))
    y_min = max(0, min(int(box["y_min"]), height - 1))
    x_max = max(0, min(int(box["x_max"]), width))
    y_max = max(0, min(int(box["y_max"]), height))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid box after clipping: {(x_min, y_min, x_max, y_max)}")
    return x_min, y_min, x_max, y_max


def load_predictor(model_type: str, checkpoint: Path, device: str) -> SamPredictor:
    if not checkpoint.exists():
        raise FileNotFoundError(f"HQ-SAM checkpoint not found: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def pick_mask(
    masks: np.ndarray,
    iou_predictions: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape CxHxW, got {masks.shape}")
    best_index = int(np.argmax(iou_predictions))
    best_mask = masks[best_index]
    return best_mask.astype(bool), best_index, float(iou_predictions[best_index])


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def rgba_from_hex(hex_color: str, alpha: int) -> Tuple[int, int, int, int]:
    rgb = ImageColor.getrgb(hex_color)
    return rgb + (alpha,)


def draw_mask_overlay(
    base_rgb: Image.Image,
    mask: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    color_hex: str,
    label_text: str,
) -> Image.Image:
    overlay = base_rgb.convert("RGBA")
    mask_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    mask_pixels = np.zeros((overlay.height, overlay.width, 4), dtype=np.uint8)
    color_rgba = np.array(rgba_from_hex(color_hex, 96), dtype=np.uint8)
    mask_pixels[mask] = color_rgba
    mask_layer = Image.fromarray(mask_pixels, mode="RGBA")
    overlay = Image.alpha_composite(overlay, mask_layer)

    draw = ImageDraw.Draw(overlay)
    x_min, y_min, x_max, y_max = box_xyxy
    draw.rectangle((x_min, y_min, x_max - 1, y_max - 1), outline=ImageColor.getrgb(color_hex), width=3)
    draw.text((x_min + 4, max(0, y_min - 16)), label_text, fill=ImageColor.getrgb(color_hex))
    return overlay


def save_mask_png(mask: np.ndarray, output_file: Path) -> None:
    ensure_dir(output_file.parent)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_file)


def save_cutout_png(base_rgb: Image.Image, mask: np.ndarray, output_file: Path) -> None:
    ensure_dir(output_file.parent)
    rgba = np.zeros((base_rgb.height, base_rgb.width, 4), dtype=np.uint8)
    rgb_np = np.asarray(base_rgb)
    rgba[..., :3] = rgb_np
    rgba[..., 3] = mask.astype(np.uint8) * 255
    Image.fromarray(rgba, mode="RGBA").save(output_file)


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


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


def default_output_dir(
    experiment_dir: Path,
    explicit_output_dir: Optional[Path],
    sample: str,
) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (experiment_dir / "hq_sam_outputs" / slugify_sample(sample)).resolve()


def default_batch_output_dir(
    experiment_dir: Path,
    explicit_output_dir: Optional[Path],
    roi_metadata_csv: Path,
    checkpoint: Path,
) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    roi_parent = roi_metadata_csv.parent.parent.name
    roi_seed = roi_metadata_csv.parent.name
    return (
        experiment_dir
        / "hq_sam_outputs_batch"
        / f"{roi_parent}_{roi_seed}_{checkpoint.stem}"
    ).resolve()


def process_sample(
    predictor: SamPredictor,
    image_path: Path,
    prompts: List[Dict[str, object]],
    sample_name: str,
    output_dir: Path,
    multimask_output: bool,
    hq_token_only: bool,
    save_per_roi_artifacts: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    ensure_dir(output_dir)

    with Image.open(image_path) as image_handle:
        base_rgb = image_handle.convert("RGB")
    base_np = np.asarray(base_rgb)
    predictor.set_image(base_np)

    combined_overlay = base_rgb.convert("RGBA")
    combined_draw = ImageDraw.Draw(combined_overlay)
    manifest_rows: List[Dict[str, object]] = []

    for idx, prompt in enumerate(prompts):
        x_min, y_min, x_max, y_max = validate_box(prompt, base_rgb.width, base_rgb.height)
        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        masks, iou_predictions, _ = predictor.predict(
            box=box,
            multimask_output=multimask_output,
            hq_token_only=hq_token_only,
        )
        best_mask, best_mask_index, best_iou = pick_mask(masks, iou_predictions)
        best_mask_bbox = mask_bbox(best_mask)
        color_hex = MASK_COLORS[idx % len(MASK_COLORS)]
        roi_label = f"roi_{int(prompt['roi_index']):02d}"

        mask_png = None
        cutout_png = None
        overlay_png = None
        if save_per_roi_artifacts:
            roi_output_dir = output_dir / roi_label
            ensure_dir(roi_output_dir)

            mask_png = roi_output_dir / "mask.png"
            cutout_png = roi_output_dir / "cutout.png"
            overlay_png = roi_output_dir / "overlay.png"

            save_mask_png(best_mask, mask_png)
            save_cutout_png(base_rgb, best_mask, cutout_png)
            draw_mask_overlay(base_rgb, best_mask, (x_min, y_min, x_max, y_max), color_hex, roi_label).save(overlay_png)

        combined_mask_pixels = np.zeros((base_rgb.height, base_rgb.width, 4), dtype=np.uint8)
        combined_mask_pixels[best_mask] = np.array(rgba_from_hex(color_hex, 72), dtype=np.uint8)
        combined_overlay = Image.alpha_composite(
            combined_overlay,
            Image.fromarray(combined_mask_pixels, mode="RGBA"),
        )
        combined_draw = ImageDraw.Draw(combined_overlay)
        combined_draw.rectangle((x_min, y_min, x_max - 1, y_max - 1), outline=ImageColor.getrgb(color_hex), width=3)
        combined_draw.text((x_min + 4, max(0, y_min - 16)), roi_label, fill=ImageColor.getrgb(color_hex))

        row = {
            "roi_index": int(prompt["roi_index"]),
            "source": prompt["source"],
            "sample": sample_name,
            "image_path": str(image_path),
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "box_area_px": int((x_max - x_min) * (y_max - y_min)),
            "selected_mask_index": best_mask_index,
            "predicted_iou": best_iou,
            "mask_area_px": int(best_mask.sum()),
            "mask_bbox_x_min": None if best_mask_bbox is None else best_mask_bbox[0],
            "mask_bbox_y_min": None if best_mask_bbox is None else best_mask_bbox[1],
            "mask_bbox_x_max": None if best_mask_bbox is None else best_mask_bbox[2],
            "mask_bbox_y_max": None if best_mask_bbox is None else best_mask_bbox[3],
            "mask_png": "" if mask_png is None else str(mask_png),
            "cutout_png": "" if cutout_png is None else str(cutout_png),
            "overlay_png": "" if overlay_png is None else str(overlay_png),
        }
        for extra_key in ("object", "split", "primary_peak_score", "region_max_score", "region_mass"):
            if extra_key in prompt:
                row[extra_key] = prompt[extra_key]
        manifest_rows.append(row)

    combined_overlay_path = output_dir / "combined_overlay.png"
    combined_overlay.save(combined_overlay_path)

    summary = {
        "image_path": str(image_path),
        "sample": sample_name,
        "num_prompts": len(prompts),
        "combined_overlay": str(combined_overlay_path),
    }
    write_json(summary, output_dir / "summary.json")
    write_csv(manifest_rows, output_dir / "manifest.csv")
    return summary, manifest_rows


def main():
    args = parse_args()
    device = resolve_device(args.device)
    experiment_dir = args.experiment_dir.resolve()
    run_args = load_run_args(experiment_dir)
    data_root = Path(str(run_args["data_root"]))

    using_roi_mode = args.sample is not None
    using_batch_roi_mode = bool(args.all_samples)
    using_direct_mode = args.image is not None
    active_modes = [using_roi_mode, using_batch_roi_mode, using_direct_mode]
    if sum(bool(mode) for mode in active_modes) != 1:
        raise ValueError(
            "Use exactly one input mode: --sample, --all-samples, or --image with --box."
        )

    predictor = load_predictor(
        model_type=args.model_type,
        checkpoint=args.checkpoint.resolve(),
        device=device,
    )

    if using_roi_mode:
        image_path, prompts = collect_roi_prompts(
            data_root=data_root,
            roi_metadata_csv=args.roi_metadata_csv.resolve(),
            sample=args.sample,
            roi_indices=args.roi_indices,
            max_rois=args.max_rois,
        )
        sample_name = normalize_sample_path(args.sample)
        output_dir = default_output_dir(experiment_dir, args.output_dir, sample_name)
        summary, manifest_rows = process_sample(
            predictor=predictor,
            image_path=image_path,
            prompts=prompts,
                sample_name=sample_name,
                output_dir=output_dir,
                multimask_output=args.multimask_output,
                hq_token_only=args.hq_token_only,
                save_per_roi_artifacts=args.save_per_roi_artifacts,
            )
        summary.update(
            {
                "model_type": args.model_type,
                "checkpoint": str(args.checkpoint.resolve()),
                "device": device,
                "multimask_output": args.multimask_output,
                "hq_token_only": args.hq_token_only,
            }
        )
        write_json(summary, output_dir / "summary.json")
        print(f"HQ-SAM finished for {sample_name}")
        print(f"Image: {image_path}")
        print(f"Prompts: {len(prompts)}")
        print(f"Output: {output_dir}")
        return

    if using_batch_roi_mode:
        batches = collect_all_roi_prompts(
            data_root=data_root,
            roi_metadata_csv=args.roi_metadata_csv.resolve(),
            roi_indices=args.roi_indices,
            max_rois=args.max_rois,
            limit_samples=args.limit_samples,
        )
        if not batches:
            raise ValueError("No ROI prompts found for batch mode.")

        root_output_dir = default_batch_output_dir(
            experiment_dir=experiment_dir,
            explicit_output_dir=args.output_dir,
            roi_metadata_csv=args.roi_metadata_csv.resolve(),
            checkpoint=args.checkpoint.resolve(),
        )
        ensure_dir(root_output_dir)

        all_manifest_rows: List[Dict[str, object]] = []
        sample_summaries: List[Dict[str, object]] = []
        total_samples = len(batches)
        total_rois = 0
        for sample_idx, (sample_name, image_path, prompts) in enumerate(batches, start=1):
            sample_output_dir = root_output_dir / slugify_sample(sample_name)
            print(f"[{sample_idx}/{total_samples}] HQ-SAM {sample_name} ({len(prompts)} ROI prompts)")
            sample_summary, manifest_rows = process_sample(
                predictor=predictor,
                image_path=image_path,
                prompts=prompts,
                sample_name=sample_name,
                output_dir=sample_output_dir,
                multimask_output=args.multimask_output,
                hq_token_only=args.hq_token_only,
                save_per_roi_artifacts=args.save_per_roi_artifacts,
            )
            sample_summary.update(
                {
                    "model_type": args.model_type,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "device": device,
                    "multimask_output": args.multimask_output,
                    "hq_token_only": args.hq_token_only,
                }
            )
            all_manifest_rows.extend(manifest_rows)
            sample_summaries.append(sample_summary)
            total_rois += len(prompts)

        batch_summary = {
            "model_type": args.model_type,
            "checkpoint": str(args.checkpoint.resolve()),
            "device": device,
            "multimask_output": args.multimask_output,
            "hq_token_only": args.hq_token_only,
            "roi_metadata_csv": str(args.roi_metadata_csv.resolve()),
            "num_samples": total_samples,
            "num_rois": total_rois,
            "samples": sample_summaries,
        }
        write_csv(all_manifest_rows, root_output_dir / "manifest.csv")
        write_json(batch_summary, root_output_dir / "summary.json")
        print(f"HQ-SAM batch finished")
        print(f"Samples: {total_samples}")
        print(f"ROI prompts: {total_rois}")
        print(f"Output: {root_output_dir}")
        return

    if using_direct_mode:
        image_path, prompts = collect_direct_prompts(args.image.resolve(), args.box)
        sample_name = image_path.name
        output_dir = default_output_dir(experiment_dir, args.output_dir, sample_name)
        summary, manifest_rows = process_sample(
            predictor=predictor,
            image_path=image_path,
            prompts=prompts,
            sample_name=sample_name,
            output_dir=output_dir,
            multimask_output=args.multimask_output,
            hq_token_only=args.hq_token_only,
            save_per_roi_artifacts=args.save_per_roi_artifacts,
        )
        summary.update(
            {
                "model_type": args.model_type,
                "checkpoint": str(args.checkpoint.resolve()),
                "device": device,
                "multimask_output": args.multimask_output,
                "hq_token_only": args.hq_token_only,
            }
        )
        write_json(summary, output_dir / "summary.json")
    print(f"HQ-SAM finished for {sample_name}")
    print(f"Image: {image_path}")
    print(f"Prompts: {len(prompts)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
