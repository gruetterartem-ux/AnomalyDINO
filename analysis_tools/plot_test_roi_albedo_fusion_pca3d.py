from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "results_FINAL"
    / "normalmap_dinov3_vitb16_res688"
    / "final_boruta152_normalmap_mrmr128_albedo_dinov3_rbf"
)
DEFAULT_TEST_RESULT_DIR = Path(
    r"D:\Thesis\Thesis Bericht\bericht Medien\Test"
    r"\albedo_dinov3_alllayers_boruta_k128_test"
)
DEFAULT_THRESHOLD = 0.6104560494422913


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3D PCA plot for the Normalmap-Albedo fusion test ROIs."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test-result-dir", type=Path, default=DEFAULT_TEST_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--probability-threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.probability_threshold <= 1.0:
        raise ValueError("The probability threshold must be between 0 and 1.")

    model_dir = args.model_dir.resolve()
    test_result_dir = args.test_result_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else test_result_dir / "pca3d"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(test_result_dir / "roi_test_predictions.csv")
    normal_features = np.load(test_result_dir / "test_normal_features.npy").astype(np.float32)
    albedo_features = np.load(test_result_dir / "test_albedo_features.npy").astype(np.float32)
    normal_indices = np.load(model_dir / "selected_normal_feature_indices.npy").astype(np.int32)
    albedo_indices = np.load(model_dir / "selected_albedo_feature_indices.npy").astype(np.int32)
    classifier_pipeline = joblib.load(model_dir / "classifier_pipeline.joblib")

    if len(predictions) != len(normal_features) or len(predictions) != len(albedo_features):
        raise ValueError("Prediction table and feature arrays have different row counts.")
    true_labels = predictions["true_roi_label"].astype(str).to_numpy()
    if not np.isin(true_labels, ["2D", "3D"]).all():
        raise ValueError("The test prediction table contains invalid true ROI labels.")

    combined_features = np.concatenate(
        [
            normal_features[:, normal_indices],
            albedo_features[:, albedo_indices],
        ],
        axis=1,
    ).astype(np.float32)
    if combined_features.shape != (len(predictions), 280):
        raise ValueError(
            f"Expected fusion feature shape ({len(predictions)}, 280), got {combined_features.shape}."
        )

    classes = np.asarray(classifier_pipeline.classes_)
    class_3d_positions = np.flatnonzero(classes == 1)
    if class_3d_positions.size != 1:
        raise RuntimeError(f"Expected classifier classes [0, 1], got {classes.tolist()}.")
    probabilities_3d = classifier_pipeline.predict_proba(combined_features)[
        :, int(class_3d_positions[0])
    ]
    predicted_labels = np.where(
        probabilities_3d >= args.probability_threshold,
        "3D",
        "2D",
    )
    incorrect = predicted_labels != true_labels
    if int(incorrect.sum()) != 15:
        raise RuntimeError(
            f"Expected 15 optimized-threshold test errors, got {int(incorrect.sum())}."
        )

    scaled_features = StandardScaler().fit_transform(combined_features)
    pca = PCA(n_components=3, random_state=0)
    coordinates = pca.fit_transform(scaled_features).astype(np.float32)
    variance = pca.explained_variance_ratio_.astype(np.float64)

    result_table = predictions[["image_name", "roi_id", "true_roi_label"]].copy()
    result_table["pred_roi_label"] = predicted_labels
    result_table["probability_3d"] = probabilities_3d
    result_table["correct"] = ~incorrect
    result_table["pca1"] = coordinates[:, 0]
    result_table["pca2"] = coordinates[:, 1]
    result_table["pca3"] = coordinates[:, 2]

    output_png = output_dir / "fusion-pr-optimiert_test_roi_pca3d.png"
    output_csv = output_dir / "fusion-pr-optimiert_test_roi_pca3d_scores.csv"
    output_json = output_dir / "fusion-pr-optimiert_test_roi_pca3d_summary.json"

    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    colors = {"2D": "#2E7D32", "3D": "#C62828"}
    for label in ["2D", "3D"]:
        mask = true_labels == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            coordinates[mask, 2],
            s=30 if label == "2D" else 38,
            alpha=0.78,
            c=colors[label],
            label=f"{label} Ground Truth",
            depthshade=True,
            edgecolors="none",
        )
    if np.any(incorrect):
        ax.scatter(
            coordinates[incorrect, 0],
            coordinates[incorrect, 1],
            coordinates[incorrect, 2],
            s=105,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.4,
            label="falsch klassifiziert",
            depthshade=False,
        )

    ax.set_title("PCA-3D-Scatterplot der Test-ROIs (Feature-Level-Fusion)", pad=18)
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f} %)", labelpad=8)
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f} %)", labelpad=8)
    ax.set_zlabel(f"PC3 ({variance[2] * 100:.1f} %)", labelpad=8)
    ax.view_init(elev=22, azim=-58)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    result_table.to_csv(output_csv, index=False)
    summary = {
        "model": "Boruta-152 Normalmap + mRMR-128 Albedo + RBF-SVM",
        "fusion_type": "feature-level fusion",
        "dataset": "independent labeled test ROIs",
        "num_rois": int(len(result_table)),
        "class_counts": result_table["true_roi_label"].value_counts().to_dict(),
        "selected_feature_count": int(combined_features.shape[1]),
        "probability_threshold_3d": float(args.probability_threshold),
        "incorrect_count": int(incorrect.sum()),
        "explained_variance_ratio": [float(value) for value in variance],
        "explained_variance_ratio_sum": float(variance.sum()),
        "outputs": {
            "scatter_png": str(output_png),
            "scores_csv": str(output_csv),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
