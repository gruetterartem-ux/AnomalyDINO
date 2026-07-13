from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

from model_building.rbf_svm_utils import build_classifier, compute_metrics


EXPERIMENT_DIR = (
    PROJECT_ROOT / "results_FINAL" / "normalmap_dinov3_vitb16_res688"
)
FEATURES_DIR = (
    EXPERIMENT_DIR
    / "roi_overthreshold_overlap_multilayer_l1to12_maxminmean_features_labeled"
)
FUSION_CV_DIR = (
    FEATURES_DIR
    / "experiment_albedo_dinov3_l1_2_3_4_5_6_7_8_9_10_11_12_maxminmean"
)
OUTPUT_DIR = FUSION_CV_DIR / "pr_threshold_analysis_boruta_albedo_k128"
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


def load_inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    table = pd.read_csv(FEATURES_DIR / "roi_feature_table.csv").copy()
    normal_features = np.load(FEATURES_DIR / "roi_features_mean.npy").astype(np.float32)
    albedo_features = np.load(
        FUSION_CV_DIR / "albedo_dinov3_l1_2_3_4_5_6_7_8_9_10_11_12_maxminmean.npy"
    ).astype(np.float32)
    albedo_metadata = pd.read_csv(
        FUSION_CV_DIR / "albedo_dinov3_l1_2_3_4_5_6_7_8_9_10_11_12_metadata.csv"
    )
    table["label_clean"] = table["label"].astype(str).str.strip().str.upper()
    valid = table["label_clean"].isin({"2D", "3D"})
    table = table.loc[valid].reset_index(drop=True)
    normal_features = normal_features[valid.to_numpy()]
    if len(table) != len(albedo_features) or len(table) != len(albedo_metadata):
        raise ValueError("Normalmap and Albedo feature rows have different lengths.")
    if table["roi_uid"].astype(str).tolist() != albedo_metadata["roi_uid"].astype(str).tolist():
        raise ValueError("Normalmap and Albedo ROI rows are not aligned by roi_uid.")
    return normal_features, albedo_features, table


def aggregate_components(
    table: pd.DataFrame,
    roi_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    work = table[["group_id", "label_clean"]].copy()
    work["predicted_label"] = roi_predictions
    rows: list[dict[str, str]] = []
    for group_id, group in work.groupby("group_id", sort=True):
        rows.append(
            {
                "group_id": str(group_id),
                "true_label": "3D" if (group["label_clean"] == "3D").any() else "2D",
                "predicted_label": "3D"
                if (group["predicted_label"] == "3D").any()
                else "2D",
            }
        )
    result = pd.DataFrame(rows)
    return result["true_label"].to_numpy(), result["predicted_label"].to_numpy()


def threshold_sweep(
    table: pd.DataFrame,
    probabilities_3d: np.ndarray,
    baseline_roi_metrics: dict[str, Any],
    baseline_component_metrics: dict[str, Any],
) -> pd.DataFrame:
    y_true = table["label_clean"].to_numpy()
    thresholds = np.unique(probabilities_3d.astype(np.float64))
    thresholds = np.concatenate(
        [
            [float(np.nextafter(thresholds.min(), -np.inf))],
            thresholds,
            [float(np.nextafter(thresholds.max(), np.inf))],
        ]
    )
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        roi_pred = np.where(probabilities_3d >= threshold, "3D", "2D")
        roi_metrics = compact_metrics(compute_metrics(y_true, roi_pred, ["2D", "3D"]))
        component_true, component_pred = aggregate_components(table, roi_pred)
        component_metrics = compact_metrics(
            compute_metrics(component_true, component_pred, ["2D", "3D"])
        )
        rows.append(
            {
                "threshold": float(threshold),
                **{f"roi_{key}": value for key, value in roi_metrics.items() if key != "confusion_matrix"},
                "roi_confusion_matrix": json.dumps(roi_metrics["confusion_matrix"]),
                **{
                    f"component_{key}": value
                    for key, value in component_metrics.items()
                    if key != "confusion_matrix"
                },
                "component_confusion_matrix": json.dumps(component_metrics["confusion_matrix"]),
                "roi_recall_not_lower": bool(
                    roi_metrics["3d_recall"] + 1e-12 >= baseline_roi_metrics["3d_recall"]
                ),
                "component_recall_not_lower": bool(
                    component_metrics["3d_recall"] + 1e-12
                    >= baseline_component_metrics["3d_recall"]
                ),
            }
        )
    return pd.DataFrame(rows)


def best_constrained_threshold(
    sweep: pd.DataFrame,
    level: str,
) -> dict[str, Any]:
    eligible = sweep.loc[sweep[f"{level}_recall_not_lower"]].copy()
    if eligible.empty:
        raise RuntimeError(f"No threshold preserves {level} 3D recall.")
    best = eligible.sort_values(
        [
            f"{level}_3d_precision",
            f"{level}_3d_f1",
            f"{level}_macro_f1",
            "threshold",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "accuracy": float(best[f"{level}_accuracy"]),
        "macro_precision": float(best[f"{level}_macro_precision"]),
        "macro_recall": float(best[f"{level}_macro_recall"]),
        "macro_f1": float(best[f"{level}_macro_f1"]),
        "3d_precision": float(best[f"{level}_3d_precision"]),
        "3d_recall": float(best[f"{level}_3d_recall"]),
        "3d_f1": float(best[f"{level}_3d_f1"]),
        "confusion_matrix": json.loads(best[f"{level}_confusion_matrix"]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    normal_features, albedo_features, table = load_inputs()
    y_labels = table["label_clean"].to_numpy()
    y = (y_labels == "3D").astype(np.int32)
    groups = table["group_id"].astype(str).to_numpy()

    boruta_table = pd.read_csv(
        FEATURES_DIR
        / "nested_eval_boruta_prefilter1000_relaxed_rbf"
        / "boruta_selected_by_fold.csv"
    )
    albedo_table = pd.read_csv(FUSION_CV_DIR / "selected_albedo_features_by_fold.csv")
    stored_oof = pd.read_csv(FUSION_CV_DIR / "oof_predictions_boruta.csv")
    stored_predictions = stored_oof["prediction_albedo_k128"].astype(str).to_numpy()

    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(
            normal_features, y, groups
        )
    )
    oof_predictions = np.empty(len(table), dtype=object)
    oof_probabilities_3d = np.zeros(len(table), dtype=np.float32)
    oof_decision_scores = np.zeros(len(table), dtype=np.float32)
    fold_assignments = np.zeros(len(table), dtype=np.int32)

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        normal_indices = boruta_table.loc[
            (boruta_table["outer_fold"] == fold)
            & (boruta_table["status"].astype(str).str.lower() == "confirmed"),
            "feature_index",
        ].to_numpy(dtype=np.int32)
        albedo_indices = (
            albedo_table.loc[albedo_table["fold"] == fold]
            .sort_values("rank")
            .head(ALBEDO_K)["albedo_feature_index"]
            .to_numpy(dtype=np.int32)
        )
        train_features = np.concatenate(
            [
                normal_features[train_idx][:, normal_indices],
                albedo_features[train_idx][:, albedo_indices],
            ],
            axis=1,
        )
        val_features = np.concatenate(
            [
                normal_features[val_idx][:, normal_indices],
                albedo_features[val_idx][:, albedo_indices],
            ],
            axis=1,
        )
        model = build_classifier(1.0, "scale", "balanced", fold)
        model.fit(train_features, y[train_idx])
        pred_indices = model.predict(val_features)
        oof_predictions[val_idx] = np.where(pred_indices == 1, "3D", "2D")
        oof_probabilities_3d[val_idx] = model.predict_proba(val_features)[:, 1]
        oof_decision_scores[val_idx] = model.decision_function(val_features)
        fold_assignments[val_idx] = fold
        print(f"[CV] fold {fold}/5 complete", flush=True)

    mismatch_count = int(np.count_nonzero(oof_predictions != stored_predictions))
    if mismatch_count:
        raise RuntimeError(f"Failed to reproduce {mismatch_count} stored OOF predictions.")

    baseline_roi_metrics = compact_metrics(
        compute_metrics(y_labels, oof_predictions, ["2D", "3D"])
    )
    baseline_component_true, baseline_component_pred = aggregate_components(
        table, oof_predictions
    )
    baseline_component_metrics = compact_metrics(
        compute_metrics(
            baseline_component_true,
            baseline_component_pred,
            ["2D", "3D"],
        )
    )
    sweep = threshold_sweep(
        table,
        oof_probabilities_3d,
        baseline_roi_metrics,
        baseline_component_metrics,
    )
    sweep.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)

    oof_output = table[
        ["roi_uid", "bildname", "roi_nummer", "group_id", "label_clean"]
    ].copy()
    oof_output["fold"] = fold_assignments
    oof_output["baseline_prediction"] = oof_predictions
    oof_output["probability_3d"] = oof_probabilities_3d
    oof_output["decision_score"] = oof_decision_scores
    oof_output.to_csv(OUTPUT_DIR / "oof_probabilities.csv", index=False)

    precision, recall, pr_thresholds = precision_recall_curve(y, oof_probabilities_3d)
    pd.DataFrame(
        {
            "threshold": np.append(pr_thresholds, np.nan),
            "precision_3d": precision,
            "recall_3d": recall,
        }
    ).to_csv(OUTPUT_DIR / "pr_curve.csv", index=False)

    best_roi = best_constrained_threshold(sweep, "roi")
    best_component = best_constrained_threshold(sweep, "component")

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=180)
    ax.plot(recall, precision, color="#0B6E4F", linewidth=2)
    ax.scatter(
        [baseline_roi_metrics["3d_recall"]],
        [baseline_roi_metrics["3d_precision"]],
        color="#C62828",
        s=60,
        label="Bisherige SVM-Entscheidung",
        zorder=3,
    )
    ax.scatter(
        [best_roi["3d_recall"]],
        [best_roi["3d_precision"]],
        color="#1565C0",
        marker="D",
        s=65,
        label=f"Gewaehlter Schwellenwert ({best_roi['threshold']:.5f})",
        zorder=4,
    )
    ax.set_xlabel("3D-Recall")
    ax.set_ylabel("3D-Precision")
    ax.set_title("PR-Kurve des Fusionsmodells (OOF-Validierungsdaten)")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pr_curve.png")
    plt.close(fig)

    summary = {
        "model": "Boruta Normalmap features + mRMR-128 Albedo features + RBF-SVM",
        "data": "5-fold stratified group OOF development predictions",
        "test_set_used": False,
        "oof_reproduction_mismatches": mismatch_count,
        "baseline_model_predict": {
            "roi": baseline_roi_metrics,
            "component": baseline_component_metrics,
        },
        "best_probability_threshold_with_non_decreasing_3d_recall": {
            "roi": best_roi,
            "component": best_component,
        },
        "note": (
            "Thresholds are selected on pooled OOF development predictions. Final unbiased "
            "performance must be measured once on untouched test data."
        ),
    }
    write_json(summary, OUTPUT_DIR / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
