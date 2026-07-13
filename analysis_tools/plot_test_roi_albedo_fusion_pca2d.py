from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_tools.plot_test_roi_feature_pca2d import plot_model_pca


DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "results_FINAL"
    / "normalmap_dinov3_vitb16_res688"
    / "final_boruta152_normalmap_mrmr128_albedo_dinov3_rbf"
)
DEFAULT_TEST_RESULT_DIR = Path(
    r"D:\Thesis\Thesis Bericht\bericht Medien\Test\albedo_dinov3_alllayers_boruta_k128_test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PCA 2D scatter plot for the Normalmap-Albedo fusion classifier."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test-result-dir", type=Path, default=DEFAULT_TEST_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=None,
        help="Optional 3D probability threshold used to mark classification errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    test_result_dir = args.test_result_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else test_result_dir / "pca2d"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(test_result_dir / "roi_test_predictions.csv")
    normal_features = np.load(test_result_dir / "test_normal_features.npy").astype(np.float32)
    albedo_features = np.load(test_result_dir / "test_albedo_features.npy").astype(np.float32)
    normal_indices = np.load(model_dir / "selected_normal_feature_indices.npy").astype(np.int32)
    albedo_indices = np.load(model_dir / "selected_albedo_feature_indices.npy").astype(np.int32)
    classifier_pipeline = joblib.load(model_dir / "classifier_pipeline.joblib")

    if args.probability_threshold is not None and not 0.0 <= args.probability_threshold <= 1.0:
        raise ValueError("The probability threshold must be between 0 and 1.")

    if len(predictions) != len(normal_features) or len(predictions) != len(albedo_features):
        raise ValueError(
            "Test predictions and cached Normalmap/Albedo features have different row counts."
        )
    if not predictions["true_roi_label"].astype(str).isin(["2D", "3D"]).all():
        raise ValueError("The test prediction table contains invalid or missing true ROI labels.")

    combined_features = np.concatenate(
        [
            normal_features[:, normal_indices],
            albedo_features[:, albedo_indices],
        ],
        axis=1,
    ).astype(np.float32)
    if combined_features.shape[1] != 280:
        raise ValueError(f"Expected 280 selected fusion features, got {combined_features.shape[1]}.")

    if args.probability_threshold is None:
        reproduced_indices = classifier_pipeline.predict(combined_features)
        reproduced_labels = np.where(reproduced_indices == 1, "3D", "2D")
        stored_labels = predictions["predicted_label"].astype(str).to_numpy()
        mismatch_count = int(np.count_nonzero(reproduced_labels != stored_labels))
        if mismatch_count:
            raise RuntimeError(f"Could not reproduce {mismatch_count} stored test predictions.")
        model_name = "Fusion"
    else:
        classes = np.asarray(classifier_pipeline.classes_)
        class_3d_positions = np.flatnonzero(classes == 1)
        if class_3d_positions.size != 1:
            raise RuntimeError(f"Expected classifier classes [0, 1], got {classes.tolist()}.")
        probabilities_3d = classifier_pipeline.predict_proba(combined_features)[
            :, int(class_3d_positions[0])
        ]
        reproduced_labels = np.where(
            probabilities_3d >= args.probability_threshold,
            "3D",
            "2D",
        )
        model_name = "Fusion-PR-optimiert"

    label_table = predictions[
        ["image_name", "roi_id", "true_roi_label"]
    ].copy()
    label_table["pred_roi_label"] = reproduced_labels
    feature_lookup = {
        (str(row.image_name), str(row.roi_id)): combined_features[index]
        for index, row in enumerate(label_table.itertuples(index=False))
    }
    summary = plot_model_pca(
        model_name=model_name,
        label_table=label_table,
        feature_lookup=feature_lookup,
        classifier_info={"selected_indices": np.arange(280, dtype=np.int32)},
        output_dir=output_dir,
    )
    if int(summary["incorrect_count"]) != int((reproduced_labels != predictions["true_roi_label"]).sum()):
        raise RuntimeError("The PCA plot does not reproduce the stored classification errors.")
    if args.probability_threshold is not None:
        summary["probability_threshold_3d"] = float(args.probability_threshold)
        (output_dir / f"{model_name.lower()}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
