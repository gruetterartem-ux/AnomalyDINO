import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


DEFAULT_HQ_SAM_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704\4-shot_preprocess=force_no_mask_no_rotation_all4_test_maxpatch_normalmap_my_own_4_20260410\hq_sam_outputs_batch\roi_crops_peak_hysteresis_h0.5_l0.2_merge2bridge0.15_all_seed=0_sam_hq_vit_tiny_overview_only"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export rectangular ROI crops from HQ-SAM mask bounding boxes with symmetric padding."
    )
    parser.add_argument(
        "--hq-sam-dir",
        type=Path,
        default=DEFAULT_HQ_SAM_DIR,
        help="HQ-SAM batch output directory with manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <hq-sam-dir>_rect_crops_pad<PX>.",
    )
    parser.add_argument(
        "--padding-px",
        type=int,
        default=16,
        help="Symmetric padding added to all four sides of the HQ-SAM mask bounding box.",
    )
    parser.add_argument(
        "--size-multiple",
        type=int,
        default=14,
        help="Force final crop width and height to this multiple. The crop is chosen as close as possible to the padded box while still containing the HQ-SAM mask box.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of ROI rows to export.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_manifest_rows(manifest_file: Path) -> List[Dict[str, str]]:
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    with manifest_file.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_sample_path(sample: str) -> Path:
    return Path(sample.replace("\\", "/"))


def parse_optional_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if value == "":
        return None
    return int(value)


def padded_crop_box(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    padding_px: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    padded_x_min = max(0, x_min - padding_px)
    padded_y_min = max(0, y_min - padding_px)
    padded_x_max = min(width, x_max + padding_px)
    padded_y_max = min(height, y_max + padding_px)
    if padded_x_max <= padded_x_min or padded_y_max <= padded_y_min:
        raise ValueError(
            f"Invalid padded crop box after clipping: {(padded_x_min, padded_y_min, padded_x_max, padded_y_max)}"
        )
    return padded_x_min, padded_y_min, padded_x_max, padded_y_max


def choose_axis_bounds(
    mask_min: int,
    mask_max: int,
    desired_min: int,
    desired_max: int,
    image_size: int,
    size_multiple: int,
) -> Tuple[int, int]:
    desired_size = desired_max - desired_min
    mask_size = mask_max - mask_min
    if desired_size <= 0 or mask_size <= 0:
        raise ValueError(
            f"Invalid sizes for axis selection: desired_size={desired_size}, mask_size={mask_size}"
        )

    if size_multiple <= 1:
        target_size = desired_size
    else:
        shrunken_size = (desired_size // size_multiple) * size_multiple
        if shrunken_size >= mask_size and shrunken_size > 0:
            target_size = shrunken_size
        else:
            target_size = ((mask_size + size_multiple - 1) // size_multiple) * size_multiple

    if target_size > image_size:
        if size_multiple > 1:
            target_size = (image_size // size_multiple) * size_multiple
        else:
            target_size = image_size
    if target_size < mask_size or target_size <= 0:
        target_size = image_size

    allowed_min = max(0, mask_max - target_size)
    allowed_max = min(mask_min, image_size - target_size)
    if allowed_min > allowed_max:
        start = max(0, min(desired_min, image_size - target_size))
        return start, start + target_size

    desired_center = (desired_min + desired_max) / 2.0
    preferred_min = round(desired_center - target_size / 2.0)
    final_min = max(allowed_min, min(preferred_min, allowed_max))
    final_max = final_min + target_size
    return final_min, final_max


def multiple_aligned_crop_box(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    padding_px: int,
    width: int,
    height: int,
    size_multiple: int,
) -> Tuple[int, int, int, int, Tuple[int, int, int, int]]:
    padded_x_min, padded_y_min, padded_x_max, padded_y_max = padded_crop_box(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        padding_px=padding_px,
        width=width,
        height=height,
    )
    crop_x_min, crop_x_max = choose_axis_bounds(
        mask_min=x_min,
        mask_max=x_max,
        desired_min=padded_x_min,
        desired_max=padded_x_max,
        image_size=width,
        size_multiple=size_multiple,
    )
    crop_y_min, crop_y_max = choose_axis_bounds(
        mask_min=y_min,
        mask_max=y_max,
        desired_min=padded_y_min,
        desired_max=padded_y_max,
        image_size=height,
        size_multiple=size_multiple,
    )
    return (
        crop_x_min,
        crop_y_min,
        crop_x_max,
        crop_y_max,
        (padded_x_min, padded_y_min, padded_x_max, padded_y_max),
    )


def build_default_output_dir(hq_sam_dir: Path, padding_px: int, size_multiple: int) -> Path:
    return hq_sam_dir.parent / f"{hq_sam_dir.name}_rect_crops_pad{padding_px}_mul{size_multiple}"


def output_image_path(output_dir: Path, sample: str, roi_index: int) -> Path:
    sample_path = normalized_sample_path(sample)
    return output_dir / sample_path.parent / f"{sample_path.stem}__roi_{roi_index:02d}.png"


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


def main():
    args = parse_args()
    hq_sam_dir = args.hq_sam_dir.resolve()
    manifest_file = hq_sam_dir / "manifest.csv"
    rows = load_manifest_rows(manifest_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows found in {manifest_file}")

    output_dir = (args.output_dir or build_default_output_dir(hq_sam_dir, args.padding_px, args.size_multiple)).resolve()
    ensure_dir(output_dir)

    exported_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    for row_index, row in enumerate(rows, start=1):
        mask_x_min = parse_optional_int(row.get("mask_bbox_x_min", ""))
        mask_y_min = parse_optional_int(row.get("mask_bbox_y_min", ""))
        mask_x_max = parse_optional_int(row.get("mask_bbox_x_max", ""))
        mask_y_max = parse_optional_int(row.get("mask_bbox_y_max", ""))
        if None in (mask_x_min, mask_y_min, mask_x_max, mask_y_max):
            skipped_rows.append(
                {
                    "sample": row["sample"],
                    "roi_index": int(row["roi_index"]),
                    "reason": "missing_mask_bbox",
                }
            )
            continue

        image_path = Path(row["image_path"])
        if not image_path.exists():
            skipped_rows.append(
                {
                    "sample": row["sample"],
                    "roi_index": int(row["roi_index"]),
                    "reason": f"missing_image:{image_path}",
                }
            )
            continue

        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            crop_x_min, crop_y_min, crop_x_max, crop_y_max, padded_box = multiple_aligned_crop_box(
                x_min=mask_x_min,
                y_min=mask_y_min,
                x_max=mask_x_max,
                y_max=mask_y_max,
                padding_px=args.padding_px,
                width=image.width,
                height=image.height,
                size_multiple=args.size_multiple,
            )
            crop = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

        crop_path = output_image_path(output_dir, row["sample"], int(row["roi_index"]))
        ensure_dir(crop_path.parent)
        crop.save(crop_path)

        exported_row: Dict[str, object] = {
            "object": row.get("object", ""),
            "split": row.get("split", ""),
            "sample": row["sample"],
            "roi_index": int(row["roi_index"]),
            "image_path": str(image_path),
            "source_box_x_min": int(row["x_min"]),
            "source_box_y_min": int(row["y_min"]),
            "source_box_x_max": int(row["x_max"]),
            "source_box_y_max": int(row["y_max"]),
            "mask_bbox_x_min": mask_x_min,
            "mask_bbox_y_min": mask_y_min,
            "mask_bbox_x_max": mask_x_max,
            "mask_bbox_y_max": mask_y_max,
            "padding_px": args.padding_px,
            "size_multiple": args.size_multiple,
            "padded_crop_x_min": padded_box[0],
            "padded_crop_y_min": padded_box[1],
            "padded_crop_x_max": padded_box[2],
            "padded_crop_y_max": padded_box[3],
            "crop_x_min": crop_x_min,
            "crop_y_min": crop_y_min,
            "crop_x_max": crop_x_max,
            "crop_y_max": crop_y_max,
            "crop_width": crop_x_max - crop_x_min,
            "crop_height": crop_y_max - crop_y_min,
            "predicted_iou": row.get("predicted_iou", ""),
            "mask_area_px": row.get("mask_area_px", ""),
            "primary_peak_score": row.get("primary_peak_score", ""),
            "region_max_score": row.get("region_max_score", ""),
            "region_mass": row.get("region_mass", ""),
            "crop_path": str(crop_path),
        }
        exported_rows.append(exported_row)

        if row_index % 100 == 0 or row_index == len(rows):
            print(f"Processed {row_index}/{len(rows)} ROI rows")

    metadata_file = output_dir / "roi_metadata.csv"
    skipped_file = output_dir / "skipped_rows.csv"
    summary_file = output_dir / "summary.json"

    write_csv(exported_rows, metadata_file)
    write_csv(skipped_rows, skipped_file)
    write_json(
        {
            "hq_sam_dir": str(hq_sam_dir),
            "manifest_file": str(manifest_file),
            "padding_px": args.padding_px,
            "size_multiple": args.size_multiple,
            "exported_rois": len(exported_rows),
            "skipped_rois": len(skipped_rows),
            "output_dir": str(output_dir),
            "roi_metadata_csv": str(metadata_file),
            "skipped_rows_csv": str(skipped_file),
        },
        summary_file,
    )

    print(f"Exported ROI crops: {len(exported_rows)}")
    print(f"Skipped ROI rows: {len(skipped_rows)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
