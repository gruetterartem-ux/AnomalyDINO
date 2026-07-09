from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_tools.plot_test_roi_feature_pca2d import (  # noqa: E402
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_IMAGE_DIR,
    DEFAULT_TEST_DIR,
    collect_test_features,
    load_label_table,
    prepare_live_context,
)
from model_building.boruta_mrmr_prefilter_maxminmean import DEFAULT_FEATURES_DIR, load_inputs  # noqa: E402


DEFAULT_OUTPUT_DIR = DEFAULT_TEST_DIR / "rf_fixed_feature_candidates"


MODEL_CONFIGS = {
    "mrmr_rf": {
        "display_name": "mRMR features + RF",
        "selection_csv": DEFAULT_EXPERIMENT_DIR
        / "final_all_boxes_overthreshold_maxminmean_mrmr_fixedk384_rbf"
        / "selected_features.csv",
        "model_dir": DEFAULT_EXPERIMENT_DIR / "final_all_boxes_overthreshold_maxminmean_mrmr_fixedk384_rf_candidate",
    },
    "boruta_rf": {
        "display_name": "Boruta features + RF",
        "selection_csv": DEFAULT_EXPERIMENT_DIR
        / "final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf"
        / "selected_features.csv",
        "model_dir": DEFAULT_EXPERIMENT_DIR
        / "final_all_boxes_overthreshold_maxminmean_boruta_confirmed_rf_candidate",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fixed-feature Random Forest candidates and evaluate them on the labeled test dataset."
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--object-name", type=str, default="normalmap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classifier", type=str, choices=("rf", "extratrees"), default="rf")
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")
    parser.add_argument("--rf-class-weight", type=str, default="balanced_subsample")
    parser.add_argument("--rf-n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
    parser.add_argument("--ignore-labels", nargs="*", default=("skip", "unclear", "unknown"))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_binary_metrics(true_labels: list[str], pred_labels: list[str]) -> dict[str, Any]:
    labels = ["2D", "3D"]
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=labels).astype(int).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def load_selected_indices(selection_csv: Path) -> np.ndarray:
    table = pd.read_csv(selection_csv)
    if "feature_index" not in table.columns:
        raise ValueError(f"Missing feature_index column in {selection_csv}")
    if "status" in table.columns:
        confirmed = table.loc[table["status"].astype(str).str.lower() == "confirmed"].copy()
        if not confirmed.empty:
            table = confirmed
    return table["feature_index"].astype(np.int32).to_numpy()


def train_tree_model(args: argparse.Namespace, selected_indices: np.ndarray) -> RandomForestClassifier | ExtraTreesClassifier:
    X_train, _table, y_train, _y_labels, _class_names = load_inputs(
        features_dir=args.features_dir.resolve(),
        valid_labels=list(args.valid_labels),
        ignore_labels=list(args.ignore_labels),
    )
    estimator_cls = ExtraTreesClassifier if str(args.classifier) == "extratrees" else RandomForestClassifier
    model = estimator_cls(
        n_estimators=int(args.rf_n_estimators),
        min_samples_leaf=int(args.rf_min_samples_leaf),
        max_features=str(args.rf_max_features),
        class_weight=None if args.rf_class_weight == "none" else str(args.rf_class_weight),
        random_state=int(args.random_state),
        n_jobs=int(args.rf_n_jobs),
    )
    model.fit(X_train[:, selected_indices], y_train)
    return model


def save_model_artifacts(
    model: RandomForestClassifier | ExtraTreesClassifier,
    selected_indices: np.ndarray,
    model_dir: Path,
    model_info: dict[str, Any],
) -> None:
    ensure_dir(model_dir)
    joblib_dump(model, model_dir / "classifier_pipeline.joblib")
    np.save(model_dir / "selected_feature_indices.npy", selected_indices.astype(np.int32))
    pd.DataFrame({"feature_index": selected_indices.astype(int)}).to_csv(model_dir / "selected_features.csv", index=False)
    info = dict(model_info)
    info.update(
        {
            "classifier_joblib": str((model_dir / "classifier_pipeline.joblib").resolve()),
            "selected_feature_indices_npy": str((model_dir / "selected_feature_indices.npy").resolve()),
            "selected_features_csv": str((model_dir / "selected_features.csv").resolve()),
        }
    )
    write_json(info, model_dir / "model_info.json")


def evaluate_model(
    display_name: str,
    model: RandomForestClassifier | ExtraTreesClassifier,
    selected_indices: np.ndarray,
    test_table: pd.DataFrame,
    feature_lookup: dict[tuple[str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    missing: list[dict[str, str]] = []
    for row in test_table.itertuples(index=False):
        image_name = str(row.image_name)
        roi_id = str(row.roi_id)
        key = (image_name, roi_id)
        if key not in feature_lookup:
            missing.append({"image_name": image_name, "roi_id": roi_id})
            continue
        features.append(feature_lookup[key][selected_indices].astype(np.float32))
        rows.append(
            {
                "image_name": image_name,
                "roi_id": roi_id,
                "true_roi_label": str(row.true_roi_label),
            }
        )
    if missing:
        preview = ", ".join(f"{item['image_name']}:{item['roi_id']}" for item in missing[:10])
        raise RuntimeError(f"{display_name}: {len(missing)} test ROIs were not found after ROI extraction: {preview}")

    X_test = np.stack(features, axis=0).astype(np.float32)
    pred_indices = model.predict(X_test)
    pred_proba = model.predict_proba(X_test).astype(np.float32)
    for row_dict, pred_index, proba in zip(rows, pred_indices, pred_proba):
        pred_label = "3D" if int(pred_index) == 1 else "2D"
        row_dict["pred_roi_label"] = pred_label
        row_dict["pred_confidence_percent"] = round(float((proba[1] if pred_label == "3D" else proba[0]) * 100.0), 2)
        row_dict["pred_prob_2d"] = float(proba[0])
        row_dict["pred_prob_3d"] = float(proba[1])
        row_dict["correct"] = bool(row_dict["true_roi_label"] == pred_label)

    roi_table = pd.DataFrame(rows)
    roi_metrics = compute_binary_metrics(
        roi_table["true_roi_label"].astype(str).tolist(),
        roi_table["pred_roi_label"].astype(str).tolist(),
    )

    part_rows: list[dict[str, Any]] = []
    for image_name, group in roi_table.groupby("image_name", sort=True):
        true_part = "3D" if (group["true_roi_label"] == "3D").any() else "2D"
        pred_part = "3D" if (group["pred_roi_label"] == "3D").any() else "2D"
        part_rows.append(
            {
                "image_name": str(image_name),
                "true_part_label": true_part,
                "pred_part_label": pred_part,
                "num_rois": int(len(group)),
                "num_true_3d_rois": int((group["true_roi_label"] == "3D").sum()),
                "num_pred_3d_rois": int((group["pred_roi_label"] == "3D").sum()),
            }
        )
    part_table = pd.DataFrame(part_rows)
    part_metrics = compute_binary_metrics(
        part_table["true_part_label"].astype(str).tolist(),
        part_table["pred_part_label"].astype(str).tolist(),
    )

    metrics = {
        "classifier": display_name,
        "num_roi_rows": int(len(roi_table)),
        "num_parts": int(len(part_table)),
        "roi_metrics": roi_metrics,
        "part_metrics": part_metrics,
    }
    return roi_table, part_table, metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    labels_file = args.labels_file or (args.test_dir.resolve() / "roi_classifier_test_labels_mrmr.xlsx")
    test_table = load_label_table(labels_file).copy()
    context = prepare_live_context(args.experiment_dir.resolve(), int(args.seed), str(args.object_name))
    feature_lookup = collect_test_features(
        context=context,
        object_name=str(args.object_name),
        image_dir=args.image_dir.resolve(),
        label_tables=[test_table],
    )

    comparison_rows: list[dict[str, Any]] = []
    combined_summary: dict[str, Any] = {}
    for model_key, config in MODEL_CONFIGS.items():
        tree_label = "ExtraTrees" if str(args.classifier) == "extratrees" else "RF"
        display_name = str(config["display_name"]).replace("RF", tree_label)
        selected_indices = load_selected_indices(Path(config["selection_csv"]))
        model = train_tree_model(args, selected_indices)
        model_info = {
            "classifier": "extra_trees" if str(args.classifier) == "extratrees" else "random_forest",
            "source_feature_set": display_name,
            "selected_feature_count": int(selected_indices.size),
            "rf_n_estimators": int(args.rf_n_estimators),
            "rf_min_samples_leaf": int(args.rf_min_samples_leaf),
            "rf_max_features": str(args.rf_max_features),
            "rf_class_weight": str(args.rf_class_weight),
            "features_dir": str(args.features_dir.resolve()),
            "selection_csv": str(Path(config["selection_csv"]).resolve()),
        }
        model_dir = Path(config["model_dir"])
        if str(args.classifier) == "extratrees":
            model_dir = Path(str(model_dir).replace("_rf_candidate", "_extratrees_candidate"))
        save_model_artifacts(model, selected_indices, model_dir, model_info)
        roi_table, part_table, metrics = evaluate_model(display_name, model, selected_indices, test_table, feature_lookup)

        output_key = model_key if str(args.classifier) == "rf" else model_key.replace("_rf", "_extratrees")
        model_output_dir = output_dir / output_key
        ensure_dir(model_output_dir)
        roi_table.to_csv(model_output_dir / f"roi_classifier_test_predictions_{output_key}.csv", index=False, encoding="utf-8-sig")
        part_table.to_csv(model_output_dir / f"part_classifier_test_predictions_{output_key}.csv", index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(model_output_dir / f"roi_classifier_test_predictions_{output_key}.xlsx", engine="openpyxl") as writer:
            roi_table.to_excel(writer, sheet_name="ROI predictions", index=False)
            part_table.to_excel(writer, sheet_name="Part predictions", index=False)
        write_json(metrics, model_output_dir / f"roi_classifier_metrics_{output_key}.json")

        for level_name, metric_key in [("ROI", "roi_metrics"), ("Bauteil", "part_metrics")]:
            metric = metrics[metric_key]
            comparison_rows.append(
                {
                    "model": display_name,
                    "level": level_name,
                    "selected_feature_count": int(selected_indices.size),
                    "accuracy": float(metric["accuracy"]),
                    "macro_precision": float(metric["macro_precision"]),
                    "macro_recall": float(metric["macro_recall"]),
                    "macro_f1": float(metric["macro_f1"]),
                    "3d_precision": float(metric["per_class"]["3D"]["precision"]),
                    "3d_recall": float(metric["per_class"]["3D"]["recall"]),
                    "3d_f1": float(metric["per_class"]["3D"]["f1"]),
                    "confusion_matrix": json.dumps(metric["confusion_matrix"]),
                }
            )
        combined_summary[model_key] = metrics

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "rf_fixed_feature_test_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(combined_summary, output_dir / "summary.json")
    print(f"Saved comparison: {output_dir / 'rf_fixed_feature_test_comparison.csv'}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
