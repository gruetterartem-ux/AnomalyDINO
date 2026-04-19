from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_FEATURES_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_top10pct_centerinbox_pca2_softmax_patch_features_labeled"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a Random Forest on ROI features with StratifiedGroupKFold."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory containing roi_features_mean.npy and roi_feature_table.csv.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional CSV path. Defaults to <features-dir>/roi_feature_table.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <features-dir>/random_forest_groupcv_results.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--class-weight", type=str, default="balanced_subsample")
    parser.add_argument(
        "--ignore-labels",
        type=str,
        nargs="*",
        default=("skip", "unclear", "unknown"),
        help="Labels that should be ignored during training/evaluation.",
    )
    parser.add_argument(
        "--valid-labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit class labels, e.g. --valid-labels 2D 3D.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(rows: List[Dict[str, object]], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    if not rows:
        output_file.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_table_file(table_file: Path) -> pd.DataFrame:
    suffix = table_file.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(table_file)
    if suffix in {".csv", ".txt", ".tsv"}:
        return pd.read_csv(table_file, sep=None, engine="python")
    raise ValueError(f"Unsupported table format: {table_file}")


def normalize_roi_nummer(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("roi"):
        return text
    if text.isdigit():
        return f"roi{text}"
    return text


def normalize_bildname(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    return text.split("/")[-1]


def load_inputs(features_dir: Path, labels_file: Path) -> tuple[np.ndarray, pd.DataFrame]:
    features_file = features_dir / "roi_features_mean.npy"
    metadata_file = features_dir / "roi_feature_table.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Feature metadata file not found: {metadata_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels table not found: {labels_file}")

    features = np.load(features_file)
    metadata = pd.read_csv(metadata_file)
    labels = load_table_file(labels_file)
    if len(features) != len(metadata):
        raise ValueError(f"Length mismatch: {len(features)} features vs {len(metadata)} metadata rows")

    if labels_file.resolve() == metadata_file.resolve():
        table = metadata
    else:
        metadata = metadata.copy()
        metadata["bildname"] = metadata["image_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
        metadata["roi_nummer"] = metadata["roi_nummer"].map(normalize_roi_nummer)

        if "roi_uid" in labels.columns:
            keep_columns = [column for column in ("roi_uid", "label", "notes") if column in labels.columns]
            labels = labels[keep_columns].copy()
            if labels["roi_uid"].duplicated().any():
                raise ValueError("Custom labels file contains duplicated roi_uid values.")
            table = metadata.merge(labels, on="roi_uid", how="left", suffixes=("", "_custom"))
        elif {"bildname", "roi_nummer"}.issubset(labels.columns):
            keep_columns = [column for column in ("bildname", "roi_nummer", "label", "notes") if column in labels.columns]
            labels = labels[keep_columns].copy()
            labels["bildname"] = labels["bildname"].map(normalize_bildname)
            labels["roi_nummer"] = labels["roi_nummer"].map(normalize_roi_nummer)
            if labels.duplicated(["bildname", "roi_nummer"]).any():
                raise ValueError("Custom labels file contains duplicated bildname/roi_nummer values.")
            table = metadata.merge(labels, on=["bildname", "roi_nummer"], how="left", suffixes=("", "_custom"))
        else:
            raise ValueError(
                "Custom labels file must contain either 'roi_uid' or the pair 'bildname' + 'roi_nummer'."
            )
        if "label_custom" in table.columns:
            table["label"] = table["label_custom"]
            table = table.drop(columns=["label_custom"])
        if "notes_custom" in table.columns:
            table["notes"] = table["notes_custom"]
            table = table.drop(columns=["notes_custom"])
    return features, table


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / "random_forest_groupcv_results").resolve()


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    args = parse_args()
    features_dir = args.features_dir.resolve()
    labels_file = (args.labels_file or (features_dir / "roi_feature_table.csv")).resolve()
    output_dir = default_output_dir(features_dir, args.output_dir)
    ensure_dir(output_dir)

    features, table = load_inputs(features_dir, labels_file)
    table = table.copy()
    table["label"] = table["label"].map(clean_label)
    ignore_labels = {clean_label(label) for label in args.ignore_labels}
    valid_mask = table["label"].ne("") & ~table["label"].isin(ignore_labels)
    labeled_table = table.loc[valid_mask].copy()
    if labeled_table.empty:
        raise ValueError("No labeled ROIs found. Fill the 'label' column in the provided labels file first.")

    if args.valid_labels:
        valid_labels = [clean_label(label) for label in args.valid_labels]
        invalid_labels = sorted(set(labeled_table["label"]) - set(valid_labels))
        if invalid_labels:
            raise ValueError(
                f"Unsupported labels found: {invalid_labels}. Expected labels are {valid_labels}."
            )
    else:
        valid_labels = sorted(labeled_table["label"].unique().tolist())
        if len(valid_labels) < 2:
            raise ValueError("At least two distinct labels are required for classification.")

    labeled_features = features[labeled_table["feature_index"].to_numpy()]
    groups = labeled_table["group_id"].astype(str).to_numpy()
    class_names = valid_labels
    label_to_index = {label: index for index, label in enumerate(class_names)}
    y = np.array([label_to_index[label] for label in labeled_table["label"].to_numpy()], dtype=np.int32)

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    class_weight = None if args.class_weight == "none" else args.class_weight

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: List[Dict[str, object]] = []
    feature_importance_rows: List[Dict[str, object]] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(labeled_features, y, groups), start=1):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight=class_weight,
            random_state=args.random_state + fold_index,
            n_jobs=-1,
        )

        X_train = labeled_features[train_idx]
        X_val = labeled_features[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val).astype(np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = fold_metrics(
            np.array(class_names)[y_val],
            np.array(class_names)[y_pred],
            class_names,
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "num_train_rois": int(len(train_idx)),
                "num_val_rois": int(len(val_idx)),
                "num_train_groups": int(pd.Series(groups[train_idx]).nunique()),
                "num_val_groups": int(pd.Series(groups[val_idx]).nunique()),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )

        for feature_index, importance in enumerate(model.feature_importances_.tolist()):
            feature_importance_rows.append(
                {
                    "fold": fold_index,
                    "feature_index": int(feature_index),
                    "importance": float(importance),
                }
            )

    if np.any(oof_pred < 0):
        raise RuntimeError("Some out-of-fold predictions were not filled.")

    y_true_labels = np.array(class_names)[y]
    y_pred_labels = np.array(class_names)[oof_pred]
    overall_metrics = fold_metrics(y_true_labels, y_pred_labels, class_names)

    oof_rows: List[Dict[str, object]] = []
    labeled_table = labeled_table.reset_index(drop=True)
    for row_index, row in labeled_table.iterrows():
        output_row = row.to_dict()
        output_row["predicted_label"] = y_pred_labels[row_index]
        output_row["correct"] = int(y_true_labels[row_index] == y_pred_labels[row_index])
        for class_index, class_name in enumerate(class_names):
            output_row[f"proba_{class_name}"] = float(oof_proba[row_index, class_index])
        oof_rows.append(output_row)

    importances_df = pd.DataFrame(feature_importance_rows)
    mean_importances = (
        importances_df.groupby("feature_index", as_index=False)["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"})
        .sort_values("importance_mean", ascending=False)
    )
    mean_importance_rows = mean_importances.to_dict(orient="records")

    fold_metrics_file = output_dir / "fold_metrics.csv"
    oof_predictions_file = output_dir / "oof_predictions.csv"
    feature_importances_file = output_dir / "feature_importances_by_fold.csv"
    feature_importances_mean_file = output_dir / "feature_importances_mean.csv"
    summary_file = output_dir / "summary.json"

    write_csv(fold_rows, fold_metrics_file)
    write_csv(oof_rows, oof_predictions_file)
    write_csv(feature_importance_rows, feature_importances_file)
    write_csv(mean_importance_rows, feature_importances_mean_file)
    write_json(
        {
            "features_dir": str(features_dir),
            "labels_file": str(labels_file),
            "classifier": "random_forest",
            "num_labeled_rois": int(len(labeled_table)),
            "num_groups": int(labeled_table["group_id"].astype(str).nunique()),
            "class_names": class_names,
            "class_counts": {
                class_name: int((labeled_table["label"] == class_name).sum())
                for class_name in class_names
            },
            "cv_type": "StratifiedGroupKFold",
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "n_estimators": int(args.n_estimators),
            "max_depth": None if args.max_depth is None else int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": args.max_features,
            "class_weight": class_weight,
            "overall": overall_metrics,
            "folds": fold_rows,
            "fold_metrics_file": str(fold_metrics_file),
            "oof_predictions_file": str(oof_predictions_file),
            "feature_importances_file": str(feature_importances_file),
            "feature_importances_mean_file": str(feature_importances_mean_file),
        },
        summary_file,
    )

    print(f"Saved fold metrics: {fold_metrics_file}")
    print(f"Saved OOF predictions: {oof_predictions_file}")
    print(f"Saved feature importances: {feature_importances_mean_file}")
    print(f"Saved summary: {summary_file}")
    print(
        f"OOF macro F1: {overall_metrics['macro_f1']:.4f} | "
        f"Accuracy: {overall_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
