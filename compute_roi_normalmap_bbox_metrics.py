from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from component_memory_bank.data_io import load_run_samples


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_ROI_METADATA_CSV = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1"
    / "seed=0"
    / "roi_metadata.csv"
)
DEFAULT_LABELS_FILE = Path(r"C:\ai\AnomalyDINO\labeling_tables\dinov3_res688_roi_labels.xlsx")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute absolute normal-map bbox metrics for existing Hysteresis ROIs and render "
            "overlay images with the per-box values."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=0,
        help="Treat pixels with all RGB channels <= threshold as black background and ignore them.",
    )
    parser.add_argument(
        "--ring-inner-px",
        type=int,
        default=4,
        help="Inner gap in pixels between bbox edge and local reference ring.",
    )
    parser.add_argument(
        "--ring-outer-px",
        type=int,
        default=24,
        help="Outer radius in pixels for the local reference ring around the bbox.",
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


def load_table_file(table_file: Path) -> pd.DataFrame:
    suffix = table_file.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(table_file)
    if suffix in {".csv", ".txt", ".tsv"}:
        return pd.read_csv(table_file, sep=None, engine="python")
    raise ValueError(f"Unsupported table format: {table_file}")


def normalize_sample(sample: object) -> str:
    return str(sample).replace("\\", "/")


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
    return (experiment_dir / "roi_normalmap_bbox_metrics").resolve()


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


def class_color(label: str) -> Tuple[int, int, int]:
    upper = str(label).strip().upper()
    if upper == "2D":
        return (34, 139, 230)
    if upper == "3D":
        return (245, 140, 32)
    return (220, 220, 220)


def load_labels(labels_file: Path) -> pd.DataFrame:
    table = load_table_file(labels_file).copy()
    if not {"bildname", "roi_nummer", "label"}.issubset(table.columns):
        raise ValueError("Labels file must contain bildname, roi_nummer and label columns.")
    table["bildname"] = table["bildname"].map(normalize_bildname)
    table["roi_nummer"] = table["roi_nummer"].map(normalize_roi_nummer)
    table["label"] = table["label"].map(clean_label)
    if "Genaues Label" in table.columns:
        table["detailed_label"] = table["Genaues Label"].fillna("").astype(str).str.strip()
    else:
        table["detailed_label"] = ""
    if table.duplicated(["bildname", "roi_nummer"]).any():
        raise ValueError("Labels file contains duplicate bildname/roi_nummer entries.")
    return table[["bildname", "roi_nummer", "label", "detailed_label"]]


def load_roi_table(roi_metadata_csv: Path, labels_file: Path) -> pd.DataFrame:
    table = pd.read_csv(roi_metadata_csv).copy()
    table["sample"] = table["sample"].map(normalize_sample)
    table["bildname"] = table["sample"].map(normalize_bildname)
    table["roi_nummer"] = "roi" + table["roi_index"].astype(int).astype(str)
    labels = load_labels(labels_file)
    table = table.merge(labels, on=["bildname", "roi_nummer"], how="left")
    table["label"] = table["label"].fillna("").astype(str)
    table["detailed_label"] = table["detailed_label"].fillna("").astype(str)
    table["roi_uid"] = table["sample"] + "__roi_" + table["roi_index"].astype(int).map(lambda idx: f"{idx:03d}")
    return table


def decode_normal_map(image_rgb: np.ndarray) -> np.ndarray:
    normals = image_rgb.astype(np.float32) / 255.0
    normals = normals * 2.0 - 1.0
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return normals / norm


def valid_core_mask(valid_mask: np.ndarray) -> np.ndarray:
    padded = np.pad(valid_mask.astype(bool), 1, mode="constant", constant_values=False)
    neighbors = []
    for dy in range(3):
        for dx in range(3):
            neighbors.append(padded[dy : dy + valid_mask.shape[0], dx : dx + valid_mask.shape[1]])
    return np.logical_and.reduce(neighbors)


def compute_maps(normal_map_rgb: np.ndarray, black_threshold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normals = decode_normal_map(normal_map_rgb)
    valid_mask = np.any(normal_map_rgb > int(black_threshold), axis=2)
    valid_derivative_mask = valid_core_mask(valid_mask)

    inclination_deg = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0))).astype(np.float32)

    dnx_dx = np.gradient(normals[..., 0], axis=1)
    dny_dx = np.gradient(normals[..., 1], axis=1)
    dnz_dx = np.gradient(normals[..., 2], axis=1)
    dnx_dy = np.gradient(normals[..., 0], axis=0)
    dny_dy = np.gradient(normals[..., 1], axis=0)
    dnz_dy = np.gradient(normals[..., 2], axis=0)

    gradient_mag = np.sqrt(
        dnx_dx**2 + dny_dx**2 + dnz_dx**2 + dnx_dy**2 + dny_dy**2 + dnz_dy**2
    ).astype(np.float32)

    divergence = (dnx_dx + dny_dy).astype(np.float32)
    return (
        normals.astype(np.float32),
        inclination_deg,
        gradient_mag,
        divergence,
        valid_mask.astype(bool),
        valid_derivative_mask.astype(bool),
    )


def local_reference_normal(
    normals: np.ndarray,
    valid_mask: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    ring_inner_px: int,
    ring_outer_px: int,
) -> tuple[np.ndarray | None, int]:
    height, width = valid_mask.shape

    outer_x_min = max(0, x_min - ring_outer_px)
    outer_y_min = max(0, y_min - ring_outer_px)
    outer_x_max = min(width - 1, x_max + ring_outer_px)
    outer_y_max = min(height - 1, y_max + ring_outer_px)

    if outer_x_max < outer_x_min or outer_y_max < outer_y_min:
        return None, 0

    ring_mask = np.zeros_like(valid_mask, dtype=bool)
    ring_mask[outer_y_min : outer_y_max + 1, outer_x_min : outer_x_max + 1] = True

    inner_x_min = max(0, x_min - ring_inner_px)
    inner_y_min = max(0, y_min - ring_inner_px)
    inner_x_max = min(width - 1, x_max + ring_inner_px)
    inner_y_max = min(height - 1, y_max + ring_inner_px)
    ring_mask[inner_y_min : inner_y_max + 1, inner_x_min : inner_x_max + 1] = False

    valid_ring_mask = ring_mask & valid_mask
    valid_count = int(valid_ring_mask.sum())
    if valid_count == 0:
        return None, 0

    ring_normals = normals[valid_ring_mask]
    reference = np.median(ring_normals, axis=0).astype(np.float32)
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm <= 1e-8:
        return None, valid_count
    return reference / ref_norm, valid_count


def bbox_metrics(
    normals: np.ndarray,
    inclination_deg: np.ndarray,
    gradient_mag: np.ndarray,
    divergence: np.ndarray,
    valid_mask: np.ndarray,
    valid_derivative_mask: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    ring_inner_px: int,
    ring_outer_px: int,
) -> Dict[str, float | int]:
    x_min = max(0, min(int(x_min), inclination_deg.shape[1] - 1))
    x_max = max(0, min(int(x_max), inclination_deg.shape[1] - 1))
    y_min = max(0, min(int(y_min), inclination_deg.shape[0] - 1))
    y_max = max(0, min(int(y_max), inclination_deg.shape[0] - 1))
    if x_max < x_min or y_max < y_min:
        raise ValueError(f"Invalid bbox after clipping: {(x_min, y_min, x_max, y_max)}")

    incl_slice = inclination_deg[y_min : y_max + 1, x_min : x_max + 1]
    grad_slice = gradient_mag[y_min : y_max + 1, x_min : x_max + 1]
    div_slice = divergence[y_min : y_max + 1, x_min : x_max + 1]
    valid_slice = valid_mask[y_min : y_max + 1, x_min : x_max + 1]
    valid_deriv_slice = valid_derivative_mask[y_min : y_max + 1, x_min : x_max + 1]

    bbox_pixel_count = int(valid_slice.size)
    valid_pixel_count = int(valid_slice.sum())
    valid_derivative_pixel_count = int(valid_deriv_slice.sum())

    incl_values = incl_slice[valid_slice]
    grad_values = grad_slice[valid_deriv_slice]
    div_values = div_slice[valid_deriv_slice]

    reference_normal, ring_valid_pixel_count = local_reference_normal(
        normals=normals,
        valid_mask=valid_mask,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        ring_inner_px=int(ring_inner_px),
        ring_outer_px=int(ring_outer_px),
    )
    if reference_normal is not None and incl_values.size:
        bbox_normals = normals[y_min : y_max + 1, x_min : x_max + 1, :]
        valid_bbox_normals = bbox_normals[valid_slice]
        dots = np.clip(np.sum(valid_bbox_normals * reference_normal[None, :], axis=1), -1.0, 1.0)
        relative_incl_values = np.degrees(np.arccos(dots)).astype(np.float32)
        relative_inc99 = float(np.percentile(relative_incl_values, 99.0))
        ref_nx = float(reference_normal[0])
        ref_ny = float(reference_normal[1])
        ref_nz = float(reference_normal[2])
    else:
        relative_inc99 = float("nan")
        ref_nx = float("nan")
        ref_ny = float("nan")
        ref_nz = float("nan")

    return {
        "bbox_pixel_count": bbox_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "ignored_black_pixel_count": int(bbox_pixel_count - valid_pixel_count),
        "valid_derivative_pixel_count": valid_derivative_pixel_count,
        "ring_inner_px": int(ring_inner_px),
        "ring_outer_px": int(ring_outer_px),
        "ring_valid_pixel_count": int(ring_valid_pixel_count),
        "reference_nx": ref_nx,
        "reference_ny": ref_ny,
        "reference_nz": ref_nz,
        "peak_inclination_p99_deg": float(np.percentile(incl_values, 99.0)) if incl_values.size else float("nan"),
        "relative_peak_inclination_p99_deg": relative_inc99,
        "gradient_max": float(np.max(grad_values)) if grad_values.size else float("nan"),
        "divergence_min": float(np.min(div_values)) if div_values.size else float("nan"),
        "divergence_max": float(np.max(div_values)) if div_values.size else float("nan"),
    }


def fmt_metric(value: object, precision: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(numeric):
        return "n/a"
    return f"{numeric:.{precision}f}"


def draw_text_box(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: Tuple[int, int, int], background: Tuple[int, int, int, int], font) -> None:
    bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=1)
    pad_x = 3
    pad_y = 2
    rect = (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
    draw.rectangle(rect, fill=background)
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=1)


def render_sample_overlay(image_rgb: np.ndarray, sample_rows: pd.DataFrame) -> Image.Image:
    image = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    for _, row in sample_rows.iterrows():
        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])
        color = class_color(row["label"])
        outline = color + (255,)
        background = color + (190,)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=outline, width=3)

        label_text = str(row["label"]).strip()
        roi_text = f"roi{int(row['roi_index'])}"
        if label_text:
            head = f"{roi_text} {label_text}"
        else:
            head = roi_text
        text = (
            f"{head}\n"
            f"abs99={fmt_metric(row['peak_inclination_p99_deg'], 2)}\n"
            f"rel99={fmt_metric(row['relative_peak_inclination_p99_deg'], 2)}\n"
            f"gmax={fmt_metric(row['gradient_max'], 4)}\n"
            f"div-={fmt_metric(row['divergence_min'], 4)}\n"
            f"div+={fmt_metric(row['divergence_max'], 4)}"
        )

        text_y = y_min - 60
        if text_y < 0:
            text_y = y_min + 2
        draw_text_box(
            draw=draw,
            x=x_min + 2,
            y=text_y,
            text=text,
            fill=(255, 255, 255),
            background=background,
            font=font,
        )

    return image


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    output_dir = default_output_dir(experiment_dir, args.output_dir)
    overlay_root = output_dir / "overlay_images_with_metrics"
    ensure_dir(output_dir)
    ensure_dir(overlay_root)

    roi_table = load_roi_table(roi_metadata_csv, labels_file)
    samples = load_run_samples(experiment_dir, seed=args.seed)
    sample_map = {sample.sample: sample for sample in samples}

    image_cache: dict[str, np.ndarray] = {}
    map_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    metric_rows: list[dict[str, object]] = []
    sample_summary_rows: list[dict[str, object]] = []

    for sample_name, sample_rows in roi_table.groupby("sample", sort=True):
        if sample_name not in sample_map:
            raise KeyError(f"Sample {sample_name!r} from ROI metadata not found in run samples.")
        sample = sample_map[sample_name]

        if sample_name not in image_cache:
            image_rgb = np.array(Image.open(sample.image_path).convert("RGB"))
            image_cache[sample_name] = image_rgb
            map_cache[sample_name] = compute_maps(image_rgb, black_threshold=int(args.black_threshold))

        image_rgb = image_cache[sample_name]
        normals, inclination_deg, gradient_mag, divergence, valid_mask, valid_derivative_mask = map_cache[sample_name]

        sample_metric_rows: list[dict[str, object]] = []
        for _, row in sample_rows.iterrows():
            metrics_row = bbox_metrics(
                normals=normals,
                inclination_deg=inclination_deg,
                gradient_mag=gradient_mag,
                divergence=divergence,
                valid_mask=valid_mask,
                valid_derivative_mask=valid_derivative_mask,
                x_min=int(row["x_min"]),
                y_min=int(row["y_min"]),
                x_max=int(row["x_max"]),
                y_max=int(row["y_max"]),
                ring_inner_px=int(args.ring_inner_px),
                ring_outer_px=int(args.ring_outer_px),
            )
            full_row = {
                "roi_uid": row["roi_uid"],
                "sample": sample_name,
                "bildname": row["bildname"],
                "roi_index": int(row["roi_index"]),
                "roi_nummer": row["roi_nummer"],
                "label": row["label"],
                "detailed_label": row["detailed_label"],
                "x_min": int(row["x_min"]),
                "y_min": int(row["y_min"]),
                "x_max": int(row["x_max"]),
                "y_max": int(row["y_max"]),
                "region_row_min": int(row["region_row_min"]),
                "region_row_max": int(row["region_row_max"]),
                "region_col_min": int(row["region_col_min"]),
                "region_col_max": int(row["region_col_max"]),
                "region_patch_count": int(row["region_patch_count"]),
                "region_max_score": float(row["region_max_score"]),
                "primary_peak_score": float(row["primary_peak_score"]),
                "image_path": str(sample.image_path),
                **metrics_row,
            }
            metric_rows.append(full_row)
            sample_metric_rows.append(full_row)

        category, filename = category_and_filename_from_sample(sample_name)
        sample_overlay_dir = overlay_root / category
        ensure_dir(sample_overlay_dir)
        overlay_path = sample_overlay_dir / filename
        overlay_image = render_sample_overlay(image_rgb, pd.DataFrame(sample_metric_rows))
        overlay_image.save(overlay_path)

        sample_summary_rows.append(
            {
                "sample": sample_name,
                "category": category,
                "num_bboxes": int(len(sample_metric_rows)),
                "overlay_path": str(overlay_path),
                "image_path": str(sample.image_path),
                "max_peak_inclination_p99_deg": float(np.nanmax([row["peak_inclination_p99_deg"] for row in sample_metric_rows])),
                "max_relative_peak_inclination_p99_deg": float(np.nanmax([row["relative_peak_inclination_p99_deg"] for row in sample_metric_rows])),
                "max_gradient_max": float(np.nanmax([row["gradient_max"] for row in sample_metric_rows])),
                "min_divergence_min": float(np.nanmin([row["divergence_min"] for row in sample_metric_rows])),
                "max_divergence_max": float(np.nanmax([row["divergence_max"] for row in sample_metric_rows])),
            }
        )

    metrics_csv = output_dir / "roi_normalmap_bbox_metrics.csv"
    sample_summary_csv = output_dir / "sample_summary.csv"
    summary_json = output_dir / "summary.json"
    write_csv(metric_rows, metrics_csv)
    write_csv(sample_summary_rows, sample_summary_csv)
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "roi_metadata_csv": str(roi_metadata_csv),
            "labels_file": str(labels_file),
            "black_threshold": int(args.black_threshold),
            "ring_inner_px": int(args.ring_inner_px),
            "ring_outer_px": int(args.ring_outer_px),
            "num_samples": int(pd.Series(roi_table["sample"]).nunique()),
            "num_bboxes": int(len(metric_rows)),
            "metrics_csv": str(metrics_csv),
            "sample_summary_csv": str(sample_summary_csv),
            "overlay_root": str(overlay_root),
            "metric_definitions": {
                "peak_inclination_p99_deg": "99th percentile of the absolute inclination angle arccos(nz) in degrees inside the bbox.",
                "relative_peak_inclination_p99_deg": "99th percentile of the angle between bbox normals and the local ring-median reference normal.",
                "gradient_max": "Maximum magnitude of the normal-field gradient sqrt(sum((d/dx n)^2 + (d/dy n)^2)) inside the bbox.",
                "divergence_min": "Minimum divergence d(nx)/dx + d(ny)/dy inside the bbox.",
                "divergence_max": "Maximum divergence d(nx)/dx + d(ny)/dy inside the bbox.",
            },
        },
        summary_json,
    )

    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved sample summary: {sample_summary_csv}")
    print(f"Saved overlays: {overlay_root}")
    print(f"Saved summary: {summary_json}")
    print(f"Num bboxes: {len(metric_rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
