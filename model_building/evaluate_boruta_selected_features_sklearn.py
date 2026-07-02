from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_building.sklearn_eval_utils import (
    evaluate_subset,
    write_csv,
    write_json,
)


DEFAULT_FEATURES_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_overthreshold_overlap_multilayer_l1to12_maxminmean_features_labeled"
)

DEFAULT_SELECTED_CSV = (
    DEFAULT_FEATURES_DIR
    / "boruta_mrmr_prefilter1000_relaxed"
    / "boruta_confirmed_features.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed externally selected feature subset with StratifiedGroupKFold."
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--selected-features-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--classifier", type=str, default="svm_rbf", choices=("logreg", "svm_linear", "svm_rbf", "rf"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--rf-class-weight", type=str, default="balanced_subsample")
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None, selected_csv: Path) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / selected_csv.stem.replace(".csv", "") / "groupcv_eval").resolve()


def load_inputs(features_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    features = np.load(features_dir / "roi_features_mean.npy").astype(np.float32)
    table = pd.read_csv(features_dir / "roi_feature_table.csv").copy()
    table["label_clean"] = table["label"].astype(str).str.strip().str.lower()
    mask = table["label_clean"].isin({"2d", "3d"})
    table = table.loc[mask].reset_index(drop=True)
    features = features[mask.to_numpy()]
    y_labels = table["label_clean"].to_numpy()
    y = np.where(y_labels == "3d", 1, 0).astype(np.int32)
    class_names = ["2d", "3d"]
    return features, table, y, y_labels, class_names


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    selected_csv = args.selected_features_csv.resolve()
    output_dir = default_output_dir(features_dir, args.output_dir, selected_csv)
    ensure_dir(output_dir)

    X, table, y, y_labels, class_names = load_inputs(features_dir)
    groups = table["group_id"].astype(str).to_numpy()
    selected_table = pd.read_csv(selected_csv)
    if "feature_index" not in selected_table.columns:
        raise ValueError(f"selected-features csv has no feature_index column: {selected_csv}")
    selected_indices = selected_table["feature_index"].astype(int).to_numpy()
    X_subset = X[:, selected_indices]

    overall_metrics, fold_rows, oof_rows = evaluate_subset(
        X=X_subset,
        y=y,
        y_labels=y_labels,
        groups=groups,
        class_names=class_names,
        classifier=str(args.classifier),
        args=args,
    )

    merged_oof_rows = []
    for base_row, pred_row in zip(table.to_dict(orient="records"), oof_rows):
        merged = dict(base_row)
        merged.update(pred_row)
        merged_oof_rows.append(merged)

    write_csv(fold_rows, output_dir / "fold_metrics.csv")
    write_csv(merged_oof_rows, output_dir / "oof_predictions.csv")
    summary = {
        "features_dir": str(features_dir),
        "selected_features_csv": str(selected_csv),
        "classifier": str(args.classifier),
        "selected_feature_count": int(len(selected_indices)),
        "overall": overall_metrics,
        "folds": fold_rows,
    }
    write_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
