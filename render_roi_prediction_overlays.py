import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_ROI_METADATA_CSV = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1"
    / "seed=0"
    / "roi_metadata.csv"
)
DEFAULT_GALLERY_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "hq_sam_outputs_batch"
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1_seed=0_sam_hq_vit_tiny"
    / "combined_overlay_gallery_flat"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render prediction overlays for labeled ROIs by drawing predicted class names and "
            "confidence scores on top of the existing combined HQ-SAM overlays."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment root used only for the default output location.",
    )
    parser.add_argument(
        "--roi-metadata-csv",
        type=Path,
        default=DEFAULT_ROI_METADATA_CSV,
        help="ROI metadata CSV that contains sample, roi_index and box coordinates.",
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=DEFAULT_GALLERY_DIR,
        help="Directory containing the flat combined overlay gallery grouped by category.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to <experiment-dir>/prediction_overlay_sets.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def default_output_root(experiment_dir: Path, explicit_output_root: Path | None) -> Path:
    if explicit_output_root is not None:
        return explicit_output_root.resolve()
    return (experiment_dir / "prediction_overlay_sets").resolve()


def default_prediction_sets(experiment_dir: Path) -> Dict[str, Path]:
    return {
        "raw_cls_logreg": experiment_dir / "cls_roi_features_labeled" / "logreg_groupcv_results" / "oof_predictions.csv",
        "raw_cls_svm_linear": experiment_dir / "cls_roi_features_labeled" / "svm_linear_groupcv_results" / "oof_predictions.csv",
        "raw_cls_svm_rbf": experiment_dir / "cls_roi_features_labeled" / "svm_rbf_groupcv_results" / "oof_predictions.csv",
        "maskpooled_logreg": experiment_dir
        / "hq_sam_outputs_batch"
        / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1_seed=0_sam_hq_vit_tiny_maskpooled_features"
        / "logreg_groupcv_results"
        / "oof_predictions.csv",
        "samcrops_cls_logreg": experiment_dir / "cls_roi_features_labeled_sam_crops" / "logreg_groupcv_results" / "oof_predictions.csv",
        "combined_cls_maskpooled_logreg": experiment_dir / "combined_cls_maskpooled_features" / "logreg_groupcv_results" / "oof_predictions.csv",
    }


def normalize_sample(sample: str) -> str:
    return str(sample).replace("\\", "/")


def roi_uid(sample: str, roi_index: int) -> str:
    return f"{normalize_sample(sample)}__roi_{int(roi_index):03d}"


def load_roi_metadata(roi_metadata_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(roi_metadata_csv)
    table = table.copy()
    table["sample"] = table["sample"].map(normalize_sample)
    table["roi_uid"] = table.apply(lambda row: roi_uid(row["sample"], int(row["roi_index"])), axis=1)
    return table


def category_and_filename_from_sample(sample: str) -> Tuple[str, str]:
    normalized = normalize_sample(sample)
    path = Path(normalized)
    filename = path.name
    if normalized.startswith("test/bad/2D und 3D/"):
        return "2D und 3D", filename
    if normalized.startswith("test/bad/2D/"):
        return "2D", filename
    if normalized.startswith("test/bad/3D/"):
        return "3D", filename
    if normalized.startswith("good_test/"):
        return "good_test", filename
    if normalized.startswith("good_train_remaining/"):
        return "good_train_remaining", filename
    return path.parent.name, filename


def gallery_image_path(gallery_dir: Path, sample: str) -> Path:
    category, filename = category_and_filename_from_sample(sample)
    return gallery_dir / category / filename


def display_label(label: str) -> str:
    upper = str(label).strip().upper()
    if upper == "2D":
        return "2D"
    if upper == "3D":
        return "3D"
    return upper


def class_color(label: str) -> Tuple[int, int, int]:
    normalized = display_label(label)
    if normalized == "2D":
        return (34, 139, 230)
    if normalized == "3D":
        return (245, 140, 32)
    return (220, 220, 220)


def confidence_for_row(row: pd.Series) -> float:
    predicted = str(row["predicted_label"]).strip().lower()
    proba_column = f"proba_{predicted}"
    if proba_column not in row.index:
        raise ValueError(f"Missing probability column {proba_column}")
    return float(row[proba_column])


def load_prediction_table(predictions_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(predictions_csv)
    if "sample" not in table.columns or "roi_index" not in table.columns:
        raise ValueError(f"Prediction file must contain sample and roi_index: {predictions_csv}")
    table = table.copy()
    table["sample"] = table["sample"].map(normalize_sample)
    table["roi_uid"] = table.apply(lambda row: roi_uid(row["sample"], int(row["roi_index"])), axis=1)
    table["confidence"] = table.apply(confidence_for_row, axis=1)
    return table


def resolve_coordinate_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for base_name in ("x_min", "y_min", "x_max", "y_max"):
        if base_name in table.columns:
            continue
        fallback_candidates = [f"{base_name}_x", f"{base_name}_y"]
        for candidate in fallback_candidates:
            if candidate in table.columns:
                table[base_name] = table[candidate]
                break
    return table


def load_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def annotation_text(row: pd.Series) -> str:
    roi_number = f"roi{int(row['roi_index'])}"
    predicted = display_label(row["predicted_label"])
    confidence = row["confidence"] * 100.0
    return f"{roi_number}: {predicted} {confidence:.1f}%"


def text_box(draw: ImageDraw.ImageDraw, position: Tuple[int, int], text: str, font, fill, background):
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x = 3
    pad_y = 2
    rect = (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
    draw.rectangle(rect, fill=background)
    draw.text((x, y), text, font=font, fill=fill)


def draw_prediction_overlay(base_image: Image.Image, rows: List[pd.Series]) -> Image.Image:
    image = base_image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font()

    for row in rows:
        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])
        color = class_color(row["predicted_label"])
        outline = color + (255,)
        translucent = color + (190,)

        draw.rectangle((x_min, y_min, x_max, y_max), outline=outline, width=3)

        text = annotation_text(row)
        anchor_x = x_min + 2
        above_y = y_min - 16
        below_y = y_min + 2
        text_y = above_y if above_y >= 0 else below_y
        text_box(
            draw=draw,
            position=(anchor_x, text_y),
            text=text,
            font=font,
            fill=(255, 255, 255, 255),
            background=translucent,
        )

    return image


def render_prediction_set(
    result_name: str,
    predictions_csv: Path,
    roi_table: pd.DataFrame,
    gallery_dir: Path,
    output_root: Path,
) -> Dict[str, object]:
    prediction_table = load_prediction_table(predictions_csv)
    coord_columns = {"x_min", "y_min", "x_max", "y_max"}
    if coord_columns.issubset(prediction_table.columns):
        merged = prediction_table.copy()
    else:
        merged = prediction_table.merge(
            roi_table[["roi_uid", "sample", "roi_index", "x_min", "y_min", "x_max", "y_max"]],
            on=["roi_uid", "sample", "roi_index"],
            how="left",
        )
    merged = resolve_coordinate_columns(merged)
    if merged[["x_min", "y_min", "x_max", "y_max"]].isna().any().any():
        missing = merged[merged["x_min"].isna()]["roi_uid"].tolist()[:5]
        raise ValueError(f"Missing ROI coordinates after merge for {result_name}: {missing}")

    output_dir = output_root / result_name
    ensure_dir(output_dir)

    rendered_rows: List[Dict[str, object]] = []
    grouped = merged.sort_values(["sample", "roi_index"]).groupby("sample", sort=True)
    for sample, sample_rows in grouped:
        if "gallery_overlay_path" in sample_rows.columns:
            candidate_gallery_path = Path(str(sample_rows.iloc[0]["gallery_overlay_path"]))
            gallery_path = candidate_gallery_path if candidate_gallery_path.exists() else gallery_image_path(gallery_dir, sample)
        else:
            gallery_path = gallery_image_path(gallery_dir, sample)
        if not gallery_path.exists():
            raise FileNotFoundError(f"Combined overlay image not found for sample {sample}: {gallery_path}")

        category, filename = category_and_filename_from_sample(sample)
        output_path = output_dir / category / filename
        ensure_dir(output_path.parent)

        with Image.open(gallery_path) as base_handle:
            overlay = draw_prediction_overlay(base_handle, [row for _, row in sample_rows.iterrows()])
            overlay.save(output_path)

        rendered_rows.append(
            {
                "sample": sample,
                "category": category,
                "num_predicted_rois": int(len(sample_rows)),
                "source_overlay": str(gallery_path),
                "output_overlay": str(output_path),
            }
        )

    manifest_file = output_dir / "manifest.csv"
    summary_file = output_dir / "summary.json"
    write_csv(rendered_rows, manifest_file)
    write_json(
        {
            "result_name": result_name,
            "predictions_csv": str(predictions_csv),
            "rendered_images": len(rendered_rows),
            "rendered_rois": int(len(merged)),
            "output_dir": str(output_dir),
            "manifest_file": str(manifest_file),
        },
        summary_file,
    )
    return {
        "result_name": result_name,
        "predictions_csv": str(predictions_csv),
        "rendered_images": len(rendered_rows),
        "rendered_rois": int(len(merged)),
        "output_dir": str(output_dir),
    }


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    gallery_dir = args.gallery_dir.resolve()
    output_root = default_output_root(experiment_dir, args.output_root)
    ensure_dir(output_root)

    roi_table = load_roi_metadata(roi_metadata_csv)
    results = []
    for result_name, predictions_csv in default_prediction_sets(experiment_dir).items():
        if not predictions_csv.exists():
            print(f"Skipping missing predictions file: {predictions_csv}")
            continue
        summary = render_prediction_set(
            result_name=result_name,
            predictions_csv=predictions_csv,
            roi_table=roi_table,
            gallery_dir=gallery_dir,
            output_root=output_root,
        )
        results.append(summary)
        print(
            f"Rendered {summary['rendered_images']} images / {summary['rendered_rois']} ROIs "
            f"for {result_name}"
        )

    summary_file = output_root / "summary.json"
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "gallery_dir": str(gallery_dir),
            "roi_metadata_csv": str(roi_metadata_csv),
            "num_result_sets": len(results),
            "result_sets": results,
        },
        summary_file,
    )
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
