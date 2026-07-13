from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalydino_app import app as app_mod  # noqa: E402


DEFAULT_TEST_DIR = Path(r"D:\Thesis\Thesis Bericht\bericht Medien\Test")
DEFAULT_IMAGE_DIR = DEFAULT_TEST_DIR / "verwendete_testbilder_normalmap_nio"
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "results_FINAL" / "normalmap_dinov3_vitb16_res688"
DEFAULT_CLASSIFIER_SUBDIR = "final_all_boxes_overthreshold_maxminmean_boruta_confirmed_extratrees_candidate"
DEFAULT_OUTPUT_DIR = DEFAULT_TEST_DIR / "extratrees_fixed_feature_candidates" / "boruta_extratrees" / "overlay_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ROI overlays for the labeled test images with the final Boruta + ExtraTrees classifier."
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--image-list-csv",
        type=Path,
        default=DEFAULT_TEST_DIR
        / "extratrees_fixed_feature_candidates"
        / "boruta_extratrees"
        / "roi_classifier_test_predictions_boruta_extratrees.csv",
    )
    parser.add_argument("--classifier-subdir", type=str, default=DEFAULT_CLASSIFIER_SUBDIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--object-name", type=str, default="normalmap")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_image_names(args: argparse.Namespace) -> list[str]:
    image_list_csv = args.image_list_csv.resolve()
    if image_list_csv.exists():
        table = pd.read_csv(image_list_csv)
        if "image_name" not in table.columns:
            raise ValueError(f"Missing image_name column in {image_list_csv}")
        return sorted({Path(str(value)).name for value in table["image_name"].dropna().tolist()})

    labels_file = args.test_dir.resolve() / "roi_classifier_test_labels_boruta.xlsx"
    if labels_file.exists():
        table = pd.read_excel(labels_file)
        if "image_name" not in table.columns:
            raise ValueError(f"Missing image_name column in {labels_file}")
        return sorted({Path(str(value)).name for value in table["image_name"].dropna().tolist()})

    return sorted(path.name for path in args.image_dir.resolve().glob("*.png"))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def existing_file_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        if not path.exists():
            continue
        stat = path.stat()
        signature.append((str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    experiment_dir_str = str(experiment_dir)
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    classifier_info = app_mod.load_component_roi_classifier_model(
        experiment_dir_str=experiment_dir_str,
        classifier_subdir=str(args.classifier_subdir),
        classifier_label="Boruta + ExtraTrees",
        cache_signature=existing_file_signature(
            [
                experiment_dir / str(args.classifier_subdir) / "selected_features.csv",
                experiment_dir / str(args.classifier_subdir) / "selected_feature_indices.npy",
                experiment_dir / str(args.classifier_subdir) / "classifier_pipeline.joblib",
                experiment_dir / str(args.classifier_subdir) / "model_info.json",
            ]
        ),
    )

    image_names = read_image_names(args)
    manifest_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    for index, image_name in enumerate(image_names, start=1):
        image_path = image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")
        print(f"[{index}/{len(image_names)}] Render Boruta+ExtraTrees overlay: {image_name}", flush=True)
        result = app_mod._run_component_test_for_external_image(
            experiment_dir_str=experiment_dir_str,
            seed=int(args.seed),
            object_name=str(args.object_name),
            sample_name=image_name,
            image_rgb=load_rgb(image_path),
            classifier_info=classifier_info,
            render_overlays=True,
            roi_logic_key=app_mod.DEFAULT_COMPONENT_TEST_ROI_LOGIC_KEY,
        )
        overlay_rgb = result.get("overlay_rgb")
        if overlay_rgb is None:
            continue
        output_path = output_dir / image_name
        Image.fromarray(np.asarray(overlay_rgb, dtype=np.uint8)).save(output_path)

        manifest_rows.append(
            {
                "image_name": image_name,
                "output_path": str(output_path),
                "image_score": float(result["image_score"]),
                "image_threshold": float(result["image_threshold"]),
                "io_nio": str(result["io_nio"]),
                "part_label": str(result["part_label"]),
                "num_rois": int(result["num_rois"]),
                "num_2d_rois": int(result["num_2d_rois"]),
                "num_3d_rois": int(result["num_3d_rois"]),
            }
        )
        for roi in result.get("roi_predictions", []):
            row = dict(roi)
            row["image_name"] = image_name
            roi_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    roi_predictions = pd.DataFrame(roi_rows)
    manifest.to_csv(output_dir.parent / "boruta_extratrees_overlay_manifest.csv", index=False, encoding="utf-8-sig")
    roi_predictions.to_csv(output_dir.parent / "boruta_extratrees_overlay_roi_predictions.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "classifier_subdir": str(args.classifier_subdir),
            "image_dir": str(image_dir),
            "output_dir": str(output_dir),
            "num_images": int(len(manifest_rows)),
            "num_roi_predictions": int(len(roi_rows)),
        },
        output_dir.parent / "boruta_extratrees_overlay_summary.json",
    )
    print(f"Saved overlays: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
