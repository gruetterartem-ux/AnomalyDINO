from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_tools.plot_test_roi_feature_pca2d import load_label_table
from model_building.crossval_mrmr_maxminmean_rbf_svm import rank_mrmr_features
from model_building.rbf_svm_utils import build_classifier, compute_metrics


EXPERIMENT_DIR = PROJECT_ROOT / "results_FINAL" / "normalmap_dinov3_vitb16_res688"
FEATURES_DIR = (
    EXPERIMENT_DIR
    / "roi_overthreshold_overlap_multilayer_l1to12_maxminmean_features_labeled"
)
ALBEDO_TRAIN_DIR = (
    FEATURES_DIR / "experiment_albedo_dinov3_l1_2_3_4_5_6_7_8_9_10_11_12_maxminmean"
)
BORUTA_MODEL_DIR = (
    EXPERIMENT_DIR / "final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf"
)
TEST_DIR = Path(r"D:\Thesis\Thesis Bericht\bericht Medien\Test")
OUTPUT_DIR = TEST_DIR / "albedo_dinov3_alllayers_boruta_k128_test"
TEST_FEATURE_CACHE = OUTPUT_DIR
LAYERS = tuple(range(1, 13))
ALBEDO_K = 128


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    report = metrics["classification_report"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "3d_precision": float(report["3D"]["precision"]),
        "3d_recall": float(report["3D"]["recall"]),
        "3d_f1": float(report["3D"]["f1-score"]),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def component_metrics(roi_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for image_name, group in roi_table.groupby("image_name", sort=True):
        true_label = "3D" if (group["true_roi_label"] == "3D").any() else "2D"
        predicted_label = "3D" if (group["predicted_label"] == "3D").any() else "2D"
        rows.append(
            {
                "image_name": image_name,
                "true_component_label": true_label,
                "predicted_component_label": predicted_label,
                "num_rois": len(group),
                "correct": int(true_label == predicted_label),
            }
        )
    result = pd.DataFrame(rows)
    metrics = compute_metrics(
        result["true_component_label"].to_numpy(),
        result["predicted_component_label"].to_numpy(),
        ["2D", "3D"],
    )
    return result, metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_label_table(TEST_DIR / "roi_classifier_test_labels_boruta.xlsx")
    cache_labels = load_label_table(TEST_DIR / "roi_classifier_test_labels_mrmr.xlsx")
    identity_columns = ["image_name", "roi_id", "true_roi_label"]
    if not labels[identity_columns].equals(cache_labels[identity_columns]):
        raise ValueError("Boruta test labels do not align with the cached test feature order.")
    test_normal = np.load(TEST_FEATURE_CACHE / "test_normal_features.npy").astype(np.float32)
    test_albedo = np.load(TEST_FEATURE_CACHE / "test_albedo_features.npy").astype(np.float32)
    if len(labels) != len(test_normal) or len(labels) != len(test_albedo):
        raise ValueError("Cached test features do not align with the Boruta label table.")

    train_table = pd.read_csv(FEATURES_DIR / "roi_feature_table.csv").copy()
    train_normal = np.load(FEATURES_DIR / "roi_features_mean.npy").astype(np.float32)
    train_table["label_clean"] = train_table["label"].astype(str).str.strip().str.upper()
    valid = train_table["label_clean"].isin({"2D", "3D"})
    train_table = train_table.loc[valid].reset_index(drop=True)
    train_normal = train_normal[valid.to_numpy()]
    train_albedo = np.load(
        ALBEDO_TRAIN_DIR / "albedo_dinov3_l1_2_3_4_5_6_7_8_9_10_11_12_maxminmean.npy"
    ).astype(np.float32)
    y_train = (train_table["label_clean"].to_numpy() == "3D").astype(np.int32)
    y_test = labels["true_roi_label"].to_numpy()
    normal_indices = np.load(BORUTA_MODEL_DIR / "selected_feature_indices.npy").astype(np.int32)
    albedo_ranking, relevance = rank_mrmr_features(
        train_albedo,
        y_train,
        fixed_k=ALBEDO_K,
        prefilter_top=4096,
        random_state=0,
    )
    selected_rows = []
    for rank, feature_index in enumerate(albedo_ranking, start=1):
        layer_slot = int(feature_index) // 2304
        within_layer = int(feature_index) % 2304
        selected_rows.append(
            {
                "rank": rank,
                "feature_index": int(feature_index),
                "dinov3_layer": LAYERS[layer_slot],
                "aggregation": ("max", "min", "mean")[within_layer // 768],
                "channel": within_layer % 768,
                "mutual_information": float(relevance[feature_index]),
            }
        )
    pd.DataFrame(selected_rows).to_csv(OUTPUT_DIR / "selected_albedo_features.csv", index=False)

    baseline = joblib.load(BORUTA_MODEL_DIR / "classifier_pipeline.joblib")
    baseline_indices = baseline.predict(test_normal[:, normal_indices])
    baseline_labels = np.where(baseline_indices == 1, "3D", "2D")
    baseline_mismatches = int(
        np.count_nonzero(baseline_labels != labels["pred_roi_label"].to_numpy())
    )
    if baseline_mismatches:
        raise RuntimeError(f"Failed to reproduce {baseline_mismatches} Boruta baseline predictions")

    model = build_classifier(1.0, "scale", "balanced", 0)
    model.fit(
        np.concatenate(
            [train_normal[:, normal_indices], train_albedo[:, albedo_ranking]], axis=1
        ),
        y_train,
    )
    test_combined = np.concatenate(
        [test_normal[:, normal_indices], test_albedo[:, albedo_ranking]], axis=1
    )
    prediction_indices = model.predict(test_combined)
    probabilities = model.predict_proba(test_combined).astype(np.float32)
    predicted_labels = np.where(prediction_indices == 1, "3D", "2D")
    baseline_metrics = compute_metrics(y_test, baseline_labels, ["2D", "3D"])
    augmented_metrics = compute_metrics(y_test, predicted_labels, ["2D", "3D"])

    roi_predictions = labels.copy()
    roi_predictions["baseline_prediction"] = baseline_labels
    roi_predictions["predicted_label"] = predicted_labels
    roi_predictions["probability_2d"] = probabilities[:, 0]
    roi_predictions["probability_3d"] = probabilities[:, 1]
    roi_predictions["correct"] = predicted_labels == y_test
    roi_predictions.to_csv(OUTPUT_DIR / "roi_test_predictions.csv", index=False)
    baseline_component_input = roi_predictions.copy()
    baseline_component_input["predicted_label"] = baseline_labels
    _, baseline_component = component_metrics(baseline_component_input)
    component_predictions, augmented_component = component_metrics(roi_predictions)
    component_predictions.to_csv(OUTPUT_DIR / "component_test_predictions.csv", index=False)
    joblib.dump(model, OUTPUT_DIR / "classifier_pipeline.joblib")
    np.save(OUTPUT_DIR / "selected_normal_feature_indices.npy", normal_indices)
    np.save(OUTPUT_DIR / "selected_albedo_feature_indices.npy", albedo_ranking)

    summary = {
        "model": "Boruta-152 normalmap + mRMR-128 DINOv3 albedo + RBF-SVM",
        "training_rois": len(train_table),
        "test_rois": len(labels),
        "test_components": int(labels["image_name"].nunique()),
        "normal_features": len(normal_indices),
        "albedo_features": len(albedo_ranking),
        "combined_features": len(normal_indices) + len(albedo_ranking),
        "albedo_layers": list(LAYERS),
        "baseline_reproduction_mismatches": baseline_mismatches,
        "roi_level": {
            "baseline": compact_metrics(baseline_metrics),
            "augmented": compact_metrics(augmented_metrics),
        },
        "component_level": {
            "baseline": compact_metrics(baseline_component),
            "augmented": compact_metrics(augmented_component),
        },
    }
    write_json(summary, OUTPUT_DIR / "summary.json")
    write_json(
        {"status": "complete", "phase": "complete", "percent": 100.0},
        OUTPUT_DIR / "progress.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
