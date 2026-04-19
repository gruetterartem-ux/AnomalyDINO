from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_METRICS_CSV = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_normalmap_bbox_metrics_local_ring_ignore_black\bbox_normalmap_features12_anompatchmask"
    r"\roi_bbox_normalmap_features12_anompatchmask.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_normalmap_bbox_metrics_local_ring_ignore_black\bbox_normalmap_features12_anompatchmask"
    r"\svm_linear_final_all_boxes"
)
DEFAULT_FEATURE_COLUMNS = [
    "grad_p95",
    "grad_max",
    "dominant_angle_mean_deg",
    "dominant_angle_p95_deg",
    "delta_mean",
    "delta_p95",
    "grad_frac_gt_t1",
    "grad_largest_component_size_t1",
    "normal_total_variance",
    "nz_std",
    "directional_coherence",
    "delta_frac_gt_t2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit final ROI bbox SVM on labeled boxes, predict all boxes, and render overlays."
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--feature-columns", nargs="*", default=DEFAULT_FEATURE_COLUMNS)
    parser.add_argument("--threshold-3d", type=float, default=0.5)
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--font-scale", type=float, default=0.55)
    parser.add_argument("--thickness", type=int, default=2)
    return parser.parse_args()


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def sanitize_text(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def group_name_from_sample(sample: str) -> str:
    sample_path = Path(sample)
    parts = sample_path.parts
    if len(parts) >= 2:
        return parts[1]
    if len(parts) == 1:
        return parts[0]
    return "ungrouped"


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


def box_color(row: pd.Series) -> tuple[int, int, int]:
    used = bool(int(row["used_for_training"]))
    if not used:
        return (160, 160, 160)

    correct = bool(int(row["correct_labeled"]))
    pred = sanitize_text(row["predicted_label"]).lower()
    if correct:
        return (40, 180, 40)
    if pred == "3d":
        return (0, 80, 220)
    return (0, 140, 255)


def render_sample(
    sample_df: pd.DataFrame,
    output_path: Path,
    font_scale: float,
    thickness: int,
) -> dict[str, object]:
    image_path = Path(str(sample_df.iloc[0]["image_path"]))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    ordered = sample_df.sort_values(["roi_index", "roi_nummer", "x_min", "y_min"]).reset_index(drop=True)
    rendered_rows = 0

    for _, row in ordered.iterrows():
        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])
        roi_name = sanitize_text(row.get("roi_nummer"), f"roi{int(row.get('roi_index', 0))}")
        pred = sanitize_text(row.get("predicted_label"), "?").upper()
        prob = float(row.get("predicted_probability", 0.0))
        gt = sanitize_text(row.get("label_display"), "?").upper()
        used = "yes" if int(row.get("used_for_training", 0)) else "no"
        color = box_color(row)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        text = f"{roi_name} pred={pred} {prob*100:.1f}% gt={gt} used={used}"
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
        "num_used_labeled_boxes": int(sample_df["used_for_training"].sum()),
        "num_unlabeled_boxes": int((1 - sample_df["used_for_training"]).sum()),
    }


def main() -> None:
    args = parse_args()
    metrics_csv = args.metrics_csv.resolve()
    output_dir = args.output_dir.resolve()
    overlays_dir = output_dir / "overlay_images"
    ensure_dir(output_dir)
    ensure_dir(overlays_dir)

    table = pd.read_csv(metrics_csv).copy()
    if args.label_column not in table.columns:
        raise ValueError(f"Label column not found: {args.label_column}")

    feature_columns = list(args.feature_columns)
    missing_cols = [col for col in feature_columns if col not in table.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    table[args.label_column] = table[args.label_column].map(clean_label)
    labeled = table.loc[table[args.label_column].isin({"2d", "3d"})].copy()
    if labeled.empty:
        raise ValueError("No labeled 2D/3D rows found.")

    if table[feature_columns].isna().any().any():
        bad = table.index[table[feature_columns].isna().any(axis=1)].tolist()
        raise ValueError(f"NaN feature values found in rows: first rows {bad[:10]}")

    X_train = labeled[feature_columns].to_numpy(dtype=np.float32)
    y_train = np.array([0 if x == "2d" else 1 for x in labeled[args.label_column].to_numpy()], dtype=np.int32)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=args.kernel,
                    C=args.C,
                    gamma=args.gamma,
                    class_weight=None if args.class_weight == "none" else args.class_weight,
                    probability=True,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    X_all = table[feature_columns].to_numpy(dtype=np.float32)
    proba_all = model.predict_proba(X_all).astype(np.float32)
    pred_all = np.where(proba_all[:, 1] >= args.threshold_3d, "3d", "2d")
    pred_prob = np.where(pred_all == "3d", proba_all[:, 1], proba_all[:, 0])

    table = table.copy()
    table["used_for_training"] = table[args.label_column].isin({"2d", "3d"}).astype(int)
    table["predicted_label"] = pred_all
    table["proba_2d"] = proba_all[:, 0]
    table["proba_3d"] = proba_all[:, 1]
    table["predicted_probability"] = pred_prob
    table["label_display"] = table[args.label_column].replace("", "?")
    table["correct_labeled"] = np.where(
        table["used_for_training"].astype(bool),
        (table[args.label_column] == table["predicted_label"]).astype(int),
        -1,
    )

    manifest_rows: list[dict[str, object]] = []
    for sample, sample_df in table.groupby("sample", sort=True):
        group_name = group_name_from_sample(str(sample))
        bildname = sanitize_text(sample_df.iloc[0].get("bildname"), Path(str(sample)).name)
        output_path = overlays_dir / group_name / bildname
        manifest_rows.append(
            render_sample(
                sample_df=sample_df,
                output_path=output_path,
                font_scale=args.font_scale,
                thickness=args.thickness,
            )
        )

    all_predictions_csv = output_dir / "all_box_predictions.csv"
    manifest_csv = output_dir / "manifest.csv"
    summary_json = output_dir / "summary.json"
    model_info_json = output_dir / "model_info.json"

    table.to_csv(all_predictions_csv, index=False)
    manifest_df = pd.DataFrame(manifest_rows).sort_values("sample").reset_index(drop=True)
    manifest_df.to_csv(manifest_csv, index=False)

    summary = {
        "metrics_csv": str(metrics_csv),
        "output_dir": str(output_dir),
        "overlay_dir": str(overlays_dir),
        "threshold_3d": float(args.threshold_3d),
        "num_total_boxes": int(len(table)),
        "num_labeled_boxes_used_for_training": int(table["used_for_training"].sum()),
        "num_unlabeled_boxes_predicted_only": int((1 - table["used_for_training"]).sum()),
        "num_images": int(manifest_df.shape[0]),
        "all_box_predictions_csv": str(all_predictions_csv),
        "manifest_csv": str(manifest_csv),
    }
    model_info = {
        "classifier": "svm",
        "kernel": args.kernel,
        "label_column": args.label_column,
        "feature_columns": feature_columns,
        "C": float(args.C),
        "gamma": args.gamma,
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "random_state": int(args.random_state),
        "classes": ["2d", "3d"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model_info_json.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    print(f"Saved predictions CSV: {all_predictions_csv}")
    print(f"Saved overlays: {overlays_dir}")
    print(f"Saved manifest: {manifest_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
