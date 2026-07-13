from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalydino_app import app as app_mod


TEST_DIR = Path(r"D:\Thesis\Thesis Bericht\bericht Medien\Test")
MODEL_RESULT_DIR = TEST_DIR / "albedo_dinov3_alllayers_boruta_k128_test"
PREDICTIONS_FILE = MODEL_RESULT_DIR / "roi_test_predictions.csv"
COORDINATES_FILE = (
    TEST_DIR
    / "extratrees_fixed_feature_candidates"
    / "boruta_extratrees"
    / "boruta_extratrees_overlay_roi_predictions.csv"
)
IMAGE_DIR = TEST_DIR / "verwendete_testbilder_normalmap_nio"
OUTPUT_DIR = MODEL_RESULT_DIR / "overlay_images"


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS_FILE)
    coordinates = pd.read_csv(COORDINATES_FILE)
    coordinate_lookup = {
        (str(row.image_name), str(row.roi_nummer)): row for row in coordinates.itertuples(index=False)
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for image_name, image_predictions in predictions.groupby("image_name", sort=True):
        image_path = IMAGE_DIR / str(image_name)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing test image: {image_path}")
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        roi_rows = []
        for row in image_predictions.itertuples(index=False):
            key = (str(image_name), str(row.roi_id))
            if key not in coordinate_lookup:
                raise KeyError(f"Missing ROI coordinates: {key}")
            coordinate = coordinate_lookup[key]
            predicted_label = str(row.predicted_label)
            probability = (
                float(row.probability_3d)
                if predicted_label == "3D"
                else float(row.probability_2d)
            )
            roi_rows.append(
                {
                    "roi_nummer": str(row.roi_id),
                    "x_min": int(coordinate.x_min),
                    "y_min": int(coordinate.y_min),
                    "x_max": int(coordinate.x_max),
                    "y_max": int(coordinate.y_max),
                    "predicted_label": predicted_label,
                    "predicted_probability": probability,
                }
            )
        part_label = "3D" if any(row["predicted_label"] == "3D" for row in roi_rows) else "2D"
        overlay = app_mod._render_component_test_overlay(
            sample_assets={"image_rgb": image_rgb, "grid_shape": (43, 56)},
            roi_prediction_rows=roi_rows,
            part_label=part_label,
        )
        output_path = OUTPUT_DIR / str(image_name)
        Image.fromarray(overlay).save(output_path)
        manifest_rows.append(
            {
                "image_name": str(image_name),
                "overlay_path": str(output_path),
                "predicted_component_label": part_label,
                "num_rois": len(roi_rows),
                "num_predicted_3d_rois": sum(
                    row["predicted_label"] == "3D" for row in roi_rows
                ),
            }
        )
    with (MODEL_RESULT_DIR / "overlay_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Rendered {len(manifest_rows)} overlays to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
