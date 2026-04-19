from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from component_memory_bank.data_io import load_run_samples


DEFAULT_MEMORY_BANK_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\component_memory_bank_backend\session_full\memory_bank_export"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render selected 16x16 memory-bank patches on the original part images."
    )
    parser.add_argument("--memory-bank-dir", type=Path, default=DEFAULT_MEMORY_BANK_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--font-size", type=int, default=18)
    return parser.parse_args()


def default_output_dir(memory_bank_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (memory_bank_dir / "selected_patch_overlays").resolve()


def sanitize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def label_color(label: str) -> tuple[int, int, int]:
    text = sanitize_text(label).upper()
    if text == "2D":
        return (40, 180, 40)
    if text == "3D":
        return (220, 60, 60)
    return (230, 180, 40)


def group_name_from_sample(sample: str) -> str:
    parts = Path(sample).parts
    if len(parts) >= 3 and parts[0] == "test" and parts[1] == "bad":
        return str(Path(parts[1]) / parts[2])
    if len(parts) >= 1:
        return parts[0]
    return "ungrouped"


def load_summary(memory_bank_dir: Path) -> dict:
    summary_file = memory_bank_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing memory-bank summary: {summary_file}")
    with summary_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_patch_meta(cache_file: Path) -> dict:
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


def patch_bounds_xyxy(row: int, col: int, meta: dict) -> Tuple[int, int, int, int]:
    x0_resized = col * meta["patch_size"]
    y0_resized = row * meta["patch_size"]
    x1_resized = min((col + 1) * meta["patch_size"], meta["cropped_w"])
    y1_resized = min((row + 1) * meta["patch_size"], meta["cropped_h"])

    x0 = max(0, min(meta["original_w"] - 1, int(np.floor(x0_resized * meta["original_w"] / meta["resized_w"]))))
    y0 = max(0, min(meta["original_h"] - 1, int(np.floor(y0_resized * meta["original_h"] / meta["resized_h"]))))
    x1 = max(x0 + 1, min(meta["original_w"], int(np.ceil(x1_resized * meta["original_w"] / meta["resized_w"]))))
    y1 = max(y0 + 1, min(meta["original_h"], int(np.ceil(y1_resized * meta["original_h"] / meta["resized_h"]))))
    return x0, y0, x1, y1


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    position_xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
    image_size: tuple[int, int],
) -> None:
    x, y = position_xy
    x = max(0, x)
    y = max(0, y)
    bbox = draw.textbbox((x, y), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    rect = [x, y, min(image_size[0] - 1, x + text_w + 8), min(image_size[1] - 1, y + text_h + 6)]
    draw.rectangle(rect, fill=fill)
    draw.text((x + 4, y + 2), text, fill=(255, 255, 255), font=font)


def render_sample(
    sample_rows: pd.DataFrame,
    sample_meta: dict,
    output_file: Path,
    line_width: int,
    font_size: int,
) -> dict:
    image_path = Path(str(sample_meta["image_path"]))
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    sample_rows = sample_rows.sort_values(["row", "col", "component_id", "rank_in_component"]).reset_index(drop=True)
    for _, patch_row in sample_rows.iterrows():
        row = int(patch_row["row"])
        col = int(patch_row["col"])
        bounds = patch_bounds_xyxy(row=row, col=col, meta=sample_meta["cache_meta"])
        x0, y0, x1, y1 = bounds
        label = sanitize_text(patch_row["component_label"]).upper()
        source_type = sanitize_text(patch_row.get("source_type"))
        rank_text = sanitize_text(patch_row.get("rank_in_component"))
        anomaly_score = float(patch_row.get("anomaly_score", 0.0))
        text = f"{label} r{rank_text} a={anomaly_score:.3f}"
        if source_type:
            text += f" {source_type}"
        color = label_color(label)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
        label_y = max(0, y0 - font_size - 8)
        draw_text_box(draw, (x0, label_y), text, color, font, image.size)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_file)
    return {
        "sample": sample_meta["sample"],
        "image_path": str(image_path),
        "output_path": str(output_file),
        "num_patches": int(sample_rows.shape[0]),
    }


def main() -> None:
    args = parse_args()
    memory_bank_dir = args.memory_bank_dir.resolve()
    output_dir = default_output_dir(memory_bank_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(memory_bank_dir)
    experiment_dir = Path(str(summary["experiment_dir"]))
    seed = int(summary.get("seed", 0))

    selected_patches_csv = memory_bank_dir / "selected_patches.csv"
    if not selected_patches_csv.exists():
        raise FileNotFoundError(f"Missing selected patches CSV: {selected_patches_csv}")
    patches_df = pd.read_csv(selected_patches_csv)
    if patches_df.empty:
        raise ValueError(f"No selected patches found in: {selected_patches_csv}")

    sample_map = {sample.sample: sample for sample in load_run_samples(experiment_dir, seed=seed)}
    sample_meta_map: Dict[str, dict] = {}
    for sample_name in patches_df["sample"].astype(str).unique():
        run_sample = sample_map.get(sample_name)
        if run_sample is None:
            raise KeyError(f"Sample {sample_name!r} not found in run samples.")
        sample_meta_map[sample_name] = {
            "sample": sample_name,
            "image_path": str(run_sample.image_path),
            "cache_meta": load_patch_meta(run_sample.feature_cache_path),
        }

    manifest_rows: list[dict] = []
    for sample_name, sample_rows in patches_df.groupby("sample", sort=True):
        sample_name = str(sample_name)
        group_name = group_name_from_sample(sample_name)
        filename = Path(sample_name).name
        output_file = output_dir / group_name / filename
        manifest_rows.append(
            render_sample(
                sample_rows=sample_rows,
                sample_meta=sample_meta_map[sample_name],
                output_file=output_file,
                line_width=args.line_width,
                font_size=args.font_size,
            )
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("sample").reset_index(drop=True)
    manifest_file = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_file, index=False)

    output_summary = {
        "memory_bank_dir": str(memory_bank_dir),
        "selected_patches_csv": str(selected_patches_csv),
        "output_dir": str(output_dir),
        "num_images": int(manifest_df.shape[0]),
        "num_patches": int(patches_df.shape[0]),
        "manifest_csv": str(manifest_file),
    }
    summary_file = output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(output_summary, handle, indent=2)

    print(f"Saved patch overlays: {output_dir}")
    print(f"Saved manifest: {manifest_file}")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
