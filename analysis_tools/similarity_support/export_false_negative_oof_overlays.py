from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from extract_labeled_roi_toppercent_multilayer_softmax_patch_features import (
    DEFAULT_EXPERIMENT_DIR,
    build_multilayer_run_context,
)
from analysis_tools.similarity_support.predict_and_render_roi_top10pct_multilayer_irelief_fixedk32_svm_all import (
    box_color,
    draw_label,
    group_name_from_sample,
    sanitize_text,
)


DEFAULT_EVAL_DIR = DEFAULT_EXPERIMENT_DIR / "nested_eval_current_best_expand1_rbf_fixedk32"
DEFAULT_OOF_CSV = DEFAULT_EVAL_DIR / "oof_predictions.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "false_negative_overlays"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render separate overlay exports for false negatives from OOF predictions. "
            "Creates one highlighted full-image overlay per false-negative ROI."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--oof-csv", type=Path, default=DEFAULT_OOF_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-subdir", type=str, default="patch_feature_cache_multilayer_l1to12")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--font-scale", type=float, default=0.55)
    parser.add_argument("--thickness", type=int, default=2)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_sample(sample: object) -> str:
    return str(sample).replace("\\", "/").strip()


def false_negative_dir_name(true_label: str, predicted_label: str) -> str:
    if true_label == "2d" and predicted_label == "3d":
        return "false_negative_2d"
    if true_label == "3d" and predicted_label == "2d":
        return "false_negative_3d"
    return "other_misclassified"


def render_focus_overlay(
    sample_df: pd.DataFrame,
    focus_roi_uid: str,
    output_path: Path,
    font_scale: float,
    thickness: int,
) -> dict[str, object]:
    image_path = Path(str(sample_df.iloc[0]["image_path"]))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    ordered = sample_df.sort_values(["roi_index", "roi_nummer", "x_min", "y_min"]).reset_index(drop=True)
    focus_color = (255, 255, 0)

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
        is_focus = str(row.get("roi_uid")) == focus_roi_uid

        color = box_color(row)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        if is_focus:
            cv2.rectangle(
                image,
                (max(0, x_min - 3), max(0, y_min - 3)),
                (min(image.shape[1] - 1, x_max + 3), min(image.shape[0] - 1, y_max + 3)),
                focus_color,
                thickness + 1,
            )
        text = f"{roi_name} pred={pred} {prob*100:.1f}% gt={gt} used={used}"
        if is_focus:
            text = f"{text} FN"
        label_y = max(18, y_min - 4)
        draw_label(
            image=image,
            text=text,
            x=x_min,
            y=label_y,
            color=focus_color if is_focus else color,
            font_scale=font_scale,
            thickness=max(1, thickness - 1),
        )

    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), image)
    return {
        "sample": sample_df.iloc[0]["sample"],
        "image_path": str(image_path),
        "output_path": str(output_path),
        "num_boxes": int(len(ordered)),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    oof_csv = args.oof_csv.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    oof = pd.read_csv(oof_csv)
    if oof.empty:
        raise ValueError(f"OOF CSV is empty: {oof_csv}")

    oof["sample"] = oof["sample"].map(normalize_sample)
    oof["true_label"] = oof["true_label"].astype(str).str.strip().str.lower()
    oof["predicted_label"] = oof["predicted_label"].astype(str).str.strip().str.lower()
    oof["correct_flag"] = oof["correct"].astype(int)
    oof["used_for_training"] = 1
    oof["correct_labeled"] = oof["correct_flag"]
    oof["label_display"] = oof["label"].astype(str)
    oof["predicted_probability"] = np.where(
        oof["predicted_label"].eq("3d"),
        oof["proba_3d"].astype(float),
        oof["proba_2d"].astype(float),
    )

    sample_map = build_multilayer_run_context(
        experiment_dir,
        seed=int(args.seed),
        cache_subdir=str(args.cache_subdir),
    )
    oof["image_path"] = oof["sample"].map(
        lambda sample: sample_map[str(sample)]["image_path"] if str(sample) in sample_map else ""
    )
    if (oof["image_path"] == "").any():
        missing = sorted(oof.loc[oof["image_path"] == "", "sample"].unique().tolist())
        raise KeyError(f"Missing image paths for samples: {missing[:5]}")

    fn = oof.loc[oof["correct_flag"] == 0].copy()
    if fn.empty:
        raise ValueError("No false negatives found in the provided OOF CSV.")

    sample_lookup = {sample: df.copy() for sample, df in oof.groupby("sample", sort=False)}
    manifest_rows: list[dict[str, object]] = []

    for row in fn.itertuples(index=False):
        sample = str(row.sample)
        roi_uid = str(row.roi_uid)
        roi_name = sanitize_text(getattr(row, "roi_nummer", ""), f"roi{int(getattr(row, 'roi_index', 0))}")
        true_label = str(row.true_label)
        predicted_label = str(row.predicted_label)
        group_dir = group_name_from_sample(sample)
        class_dir = false_negative_dir_name(true_label, predicted_label)

        sample_df = sample_lookup[sample]
        output_path = (
            output_dir
            / class_dir
            / group_dir
            / f"{Path(sample).stem}__{roi_name}.png"
        )

        render_info = render_focus_overlay(
            sample_df=sample_df,
            focus_roi_uid=roi_uid,
            output_path=output_path,
            font_scale=float(args.font_scale),
            thickness=int(args.thickness),
        )

        manifest_rows.append(
            {
                "roi_uid": roi_uid,
                "sample": sample,
                "roi_nummer": roi_name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "detailed_label": sanitize_text(getattr(row, "detailed_label", ""), ""),
                "proba_2d": float(getattr(row, "proba_2d")),
                "proba_3d": float(getattr(row, "proba_3d")),
                "fold": int(getattr(row, "fold", -1)),
                "class_dir": class_dir,
                **render_info,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "oof_csv": str(oof_csv),
        "output_dir": str(output_dir),
        "total_false_negatives": int(len(manifest_df)),
        "false_negative_2d": int((manifest_df["class_dir"] == "false_negative_2d").sum()),
        "false_negative_3d": int((manifest_df["class_dir"] == "false_negative_3d").sum()),
        "unique_samples": int(manifest_df["sample"].nunique()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
