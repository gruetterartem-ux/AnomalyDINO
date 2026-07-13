from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_tools.plot_test_roi_feature_pca2d import plot_model_pca


FEATURES_DIR = (
    PROJECT_ROOT
    / "results_FINAL"
    / "normalmap_dinov3_vitb16_res688"
    / "roi_overthreshold_overlap_multilayer_l1to12_maxminmean_features_labeled"
)
FUSION_CV_DIR = (
    FEATURES_DIR
    / "experiment_albedo_dinov3_l1_2_3_4_5_6_7_8_9_10_11_12_maxminmean"
)
MODEL_DIR = (
    PROJECT_ROOT
    / "results_FINAL"
    / "normalmap_dinov3_vitb16_res688"
    / "final_boruta152_normalmap_mrmr128_albedo_dinov3_rbf"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Thesis\Thesis Bericht\bericht Medien\Test"
    r"\albedo_dinov3_alllayers_boruta_k128_test\pca2d"
)
DEFAULT_THRESHOLD = 0.6104560494422913


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a PCA 2D plot of the development ROIs in the final common fusion "
            "feature space and mark optimized OOF classification errors."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probability-threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.probability_threshold <= 1.0:
        raise ValueError("The probability threshold must be between 0 and 1.")

    output_dir = args.output_dir.resolve()
    normal_table = pd.read_csv(FEATURES_DIR / "roi_feature_table.csv")
    albedo_table = pd.read_csv(
        FUSION_CV_DIR
        / "albedo_dinov3_l1_2_3_4_5_6_7_8_9_10_11_12_metadata.csv"
    )
    oof_table = pd.read_csv(
        FUSION_CV_DIR
        / "pr_threshold_analysis_boruta_albedo_k128"
        / "oof_probabilities.csv"
    )

    normal_features = np.load(FEATURES_DIR / "roi_features_mean.npy").astype(np.float32)
    albedo_features = np.load(
        FUSION_CV_DIR
        / "albedo_dinov3_l1_2_3_4_5_6_7_8_9_10_11_12_maxminmean.npy"
    ).astype(np.float32)
    normal_indices = np.load(MODEL_DIR / "selected_normal_feature_indices.npy").astype(np.int32)
    albedo_indices = np.load(MODEL_DIR / "selected_albedo_feature_indices.npy").astype(np.int32)

    if len(normal_table) != len(normal_features) or len(albedo_table) != len(albedo_features):
        raise ValueError("Feature arrays and metadata tables have different row counts.")
    if not normal_table["roi_uid"].astype(str).equals(albedo_table["roi_uid"].astype(str)):
        raise ValueError("Normalmap and Albedo ROI rows are not aligned.")

    combined_features = np.concatenate(
        [
            normal_features[:, normal_indices],
            albedo_features[:, albedo_indices],
        ],
        axis=1,
    ).astype(np.float32)
    if combined_features.shape != (156, 280):
        raise ValueError(f"Expected fusion feature shape (156, 280), got {combined_features.shape}.")

    oof_table = oof_table.copy()
    oof_table["pred_roi_label"] = np.where(
        oof_table["probability_3d"].to_numpy(dtype=np.float64)
        >= args.probability_threshold,
        "3D",
        "2D",
    )
    label_table = oof_table[
        ["roi_uid", "bildname", "roi_nummer", "label_clean", "pred_roi_label"]
    ].rename(
        columns={
            "bildname": "image_name",
            "roi_nummer": "roi_id",
            "label_clean": "true_roi_label",
        }
    )

    feature_by_uid = {
        str(uid): combined_features[index]
        for index, uid in enumerate(normal_table["roi_uid"].astype(str))
    }
    missing = sorted(set(label_table["roi_uid"].astype(str)) - set(feature_by_uid))
    if missing:
        raise RuntimeError(f"Missing fusion features for {len(missing)} OOF ROIs.")
    feature_lookup = {
        (str(row.image_name), str(row.roi_id)): feature_by_uid[str(row.roi_uid)]
        for row in label_table.itertuples(index=False)
    }

    summary = plot_model_pca(
        model_name="Fusion-PR-optimiert",
        label_table=label_table,
        feature_lookup=feature_lookup,
        classifier_info={"selected_indices": np.arange(280, dtype=np.int32)},
        output_dir=output_dir,
        dataset_name="Kreuzvalidierungs-ROIs",
        file_prefix="fusion-pr-optimiert_cv_roi_pca2d",
    )
    expected_errors = int(
        np.count_nonzero(
            label_table["true_roi_label"].astype(str).to_numpy()
            != label_table["pred_roi_label"].astype(str).to_numpy()
        )
    )
    if int(summary["incorrect_count"]) != expected_errors:
        raise RuntimeError("The PCA plot does not reproduce the optimized OOF errors.")

    summary["probability_threshold_3d"] = float(args.probability_threshold)
    summary["prediction_source"] = "fold-safe OOF probabilities"
    summary["pca_feature_space"] = (
        "final common 280-dimensional fusion feature subset fitted on all development ROIs"
    )
    (output_dir / "fusion-pr-optimiert_cv_roi_pca2d_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
