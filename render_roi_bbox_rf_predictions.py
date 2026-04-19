from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    default_oof = Path(
        r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
        r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
        r"\roi_normalmap_bbox_metrics_local_ring_ignore_black\random_forest_groupcv_results\oof_predictions.csv"
    )
    default_output = default_oof.parent / "overlay_predictions"
    parser = argparse.ArgumentParser(
        description="Render ROI bounding-box overlays with Random Forest predictions."
    )
    parser.add_argument("--predictions-csv", type=Path, default=default_oof)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--font-scale", type=float, default=0.6)
    parser.add_argument("--thickness", type=int, default=2)
    return parser.parse_args()


def sanitize_label(value: str | float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def prediction_probability(row: pd.Series) -> float:
    pred = sanitize_label(row.get("predicted_label")).lower()
    if pred == "3d":
        return float(row.get("proba_3d", 0.0))
    return float(row.get("proba_2d", 0.0))


def box_color(correct: bool, predicted: str) -> tuple[int, int, int]:
    if correct:
        return (40, 180, 40)
    if predicted.lower() == "3d":
        return (0, 80, 220)
    return (0, 140, 255)


def group_name_from_sample(sample: str) -> str:
    sample_path = Path(sample)
    parts = sample_path.parts
    if len(parts) > 1:
        return parts[0]
    return "ungrouped"


def draw_label(
    image,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_x1 = max(0, x)
    box_y2 = max(text_h + baseline + 4, y)
    box_y1 = max(0, box_y2 - text_h - baseline - 8)
    box_x2 = min(image.shape[1] - 1, box_x1 + text_w + 8)
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


def render_sample(
    sample_df: pd.DataFrame,
    output_path: Path,
    font_scale: float,
    thickness: int,
) -> dict:
    image_path = Path(str(sample_df.iloc[0]["image_path"]))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    sample_df = sample_df.sort_values(["roi_index", "roi_nummer", "x_min", "y_min"]).reset_index(drop=True)
    rendered_rows = 0

    for _, row in sample_df.iterrows():
        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])
        gt = sanitize_label(row.get("label")).upper()
        pred = sanitize_label(row.get("predicted_label")).upper()
        roi_nummer = sanitize_label(row.get("roi_nummer")) or f"roi{int(row.get('roi_index', 0))}"
        prob = prediction_probability(row)
        correct = bool(int(row.get("correct", 0)))
        color = box_color(correct, pred)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        text = f"{roi_nummer} pred={pred} {prob * 100:.1f}% gt={gt}"
        label_y = max(18, y_min - 4)
        draw_label(
            image=image,
            text=text,
            x=x_min,
            y=label_y,
            color=color,
            font_scale=font_scale,
            thickness=max(1, thickness - 1),
        )
        rendered_rows += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return {
        "sample": sample_df.iloc[0]["sample"],
        "image_path": str(image_path),
        "output_path": str(output_path),
        "num_boxes": rendered_rows,
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions_csv)
    required_columns = {
        "sample",
        "image_path",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "predicted_label",
        "label",
        "correct",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for sample, sample_df in df.groupby("sample", sort=True):
        group_name = group_name_from_sample(str(sample))
        bildname = sanitize_label(sample_df.iloc[0].get("bildname")) or Path(str(sample)).name
        output_path = args.output_dir / group_name / bildname
        manifest_rows.append(
            render_sample(
                sample_df=sample_df,
                output_path=output_path,
                font_scale=args.font_scale,
                thickness=args.thickness,
            )
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("sample").reset_index(drop=True)
    manifest_path = args.output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "predictions_csv": str(args.predictions_csv),
        "output_dir": str(args.output_dir),
        "num_images": int(len(manifest_df)),
        "num_boxes": int(df.shape[0]),
        "manifest_csv": str(manifest_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved overlay predictions: {args.output_dir}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
