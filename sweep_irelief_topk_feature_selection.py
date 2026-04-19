from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


DEFAULT_FEATURES_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_top10pct_centerinbox_multilayer_l1to12_softmax_patch_features_labeled"
    r"\irelief_cosine_weighted_features"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep top-k I-Relief-ranked feature subsets and evaluate classifiers with "
            "StratifiedGroupKFold."
        )
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory containing roi_features_mean.npy, roi_feature_table.csv and irelief_feature_weights.npy.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional labels table. Defaults to <features-dir>/roi_feature_table.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <features-dir>/topk_sweep_<score-key>.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=("logreg", "svm_linear", "svm_rbf"),
        choices=("logreg", "svm_linear", "svm_rbf", "rf"),
        help="Classifiers to evaluate.",
    )
    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=None,
        help="Explicit top-k values. If omitted, a sensible default schedule is used.",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="macro_f1",
        choices=("macro_f1", "macro_recall", "accuracy", "3d_f1", "3d_recall"),
        help="Metric used to pick the best k.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of StratifiedGroupKFold splits.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state for fold shuffling.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=4000,
        help="Maximum iterations for linear models/SVMs.",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogReg/SVM.",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Gamma for RBF-SVM.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        help="Class weight for LogReg/SVM. Use 'balanced' or 'none'.",
    )
    parser.add_argument(
        "--rf-class-weight",
        type=str,
        default="balanced_subsample",
        help="Class weight for Random Forest. Use 'balanced_subsample', 'balanced' or 'none'.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=500,
        help="Number of trees for Random Forest.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Max depth for Random Forest.",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=2,
        help="Min samples leaf for Random Forest.",
    )
    parser.add_argument(
        "--rf-max-features",
        type=str,
        default="sqrt",
        help="Max features for Random Forest.",
    )
    parser.add_argument(
        "--valid-labels",
        nargs="*",
        default=("2D", "3D"),
        help="Class labels to keep.",
    )
    parser.add_argument(
        "--ignore-labels",
        nargs="*",
        default=("skip", "unclear", "unknown"),
        help="Labels to ignore.",
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


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def load_inputs(features_dir: Path, labels_file: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    features_file = features_dir / "roi_features_mean.npy"
    metadata_file = features_dir / "roi_feature_table.csv"
    weights_file = features_dir / "irelief_feature_weights.npy"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Feature metadata file not found: {metadata_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels table not found: {labels_file}")
    if not weights_file.exists():
        raise FileNotFoundError(f"I-Relief weights file not found: {weights_file}")

    features = np.load(features_file).astype(np.float32)
    metadata = pd.read_csv(metadata_file)
    labels = load_table_file(labels_file)
    weights = np.load(weights_file).astype(np.float32)

    if len(features) != len(metadata):
        raise ValueError(f"Length mismatch: {len(features)} features vs {len(metadata)} metadata rows")
    if weights.ndim != 1 or weights.shape[0] != features.shape[1]:
        raise ValueError(
            f"I-Relief weight shape mismatch: weights={weights.shape} vs feature_dim={features.shape[1]}"
        )

    if labels_file.resolve() == metadata_file.resolve():
        table = metadata
    else:
        metadata = metadata.copy()
        metadata["bildname"] = metadata["image_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
        metadata["roi_nummer"] = "roi" + metadata["roi_index"].astype(int).astype(str)

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

    return features, table, weights


def generate_default_k_values(feature_dim: int) -> list[int]:
    candidates = [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, feature_dim
    ]
    values = sorted({int(k) for k in candidates if 1 <= int(k) <= feature_dim})
    if values[-1] != feature_dim:
        values.append(feature_dim)
    return values


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None, score_key: str) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / f"topk_sweep_{score_key}").resolve()


def build_estimator(args: argparse.Namespace, classifier: str):
    class_weight = None if args.class_weight == "none" else args.class_weight
    if classifier == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=args.max_iter,
                        C=args.c_value,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
    if classifier == "svm_linear":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        C=args.c_value,
                        class_weight=class_weight,
                        probability=True,
                        max_iter=args.max_iter,
                    ),
                ),
            ]
        )
    if classifier == "svm_rbf":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=args.c_value,
                        gamma=args.gamma,
                        class_weight=class_weight,
                        probability=True,
                        max_iter=args.max_iter,
                    ),
                ),
            ]
        )
    if classifier == "rf":
        rf_class_weight = None if args.rf_class_weight == "none" else args.rf_class_weight
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=args.rf_max_features,
            class_weight=rf_class_weight,
            random_state=args.random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported classifier: {classifier}")


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }


def score_value(metrics: Dict[str, object], score_key: str) -> float:
    if score_key in ("macro_f1", "macro_recall", "accuracy"):
        return float(metrics[score_key])
    report = metrics["classification_report"]
    if score_key == "3d_f1":
        return float(report["3d"]["f1-score"])
    if score_key == "3d_recall":
        return float(report["3d"]["recall"])
    raise ValueError(f"Unsupported score key: {score_key}")


def evaluate_subset(
    X: np.ndarray,
    y: np.ndarray,
    y_labels: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
    classifier: str,
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[Dict[str, object]], list[Dict[str, object]]]:
    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    estimator = build_estimator(args, classifier)
    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[Dict[str, object]] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_val).astype(np.float32)
        else:
            y_proba = np.zeros((len(val_idx), len(class_names)), dtype=np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = fold_metrics(
            y_labels[val_idx],
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

    if np.any(oof_pred < 0):
        raise RuntimeError("Some out-of-fold predictions were not filled.")

    y_pred_labels = np.array(class_names)[oof_pred]
    overall_metrics = fold_metrics(y_labels, y_pred_labels, class_names)
    oof_rows: list[Dict[str, object]] = []
    for row_index in range(len(y)):
        row = {
            "row_index": int(row_index),
            "true_label": str(y_labels[row_index]),
            "predicted_label": str(y_pred_labels[row_index]),
            "correct": int(y_labels[row_index] == y_pred_labels[row_index]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"proba_{class_name}"] = float(oof_proba[row_index, class_index])
        oof_rows.append(row)

    return overall_metrics, fold_rows, oof_rows


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    labels_file = (args.labels_file or (features_dir / "roi_feature_table.csv")).resolve()
    output_dir = default_output_dir(features_dir, args.output_dir, args.score_key)
    ensure_dir(output_dir)

    features, table, weights = load_inputs(features_dir, labels_file)
    table = table.copy()
    table["label"] = table["label"].map(clean_label)
    ignore_labels = {clean_label(label) for label in args.ignore_labels}
    valid_labels = [clean_label(label) for label in args.valid_labels]
    valid_mask = table["label"].isin(valid_labels) & ~table["label"].isin(ignore_labels)
    labeled_table = table.loc[valid_mask].copy()
    if labeled_table.empty:
        raise ValueError("No labeled ROIs found after filtering labels.")

    labeled_features = features[labeled_table["feature_index"].to_numpy()].astype(np.float32)
    groups = labeled_table["group_id"].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_labels)
    y = label_encoder.transform(labeled_table["label"].to_numpy())
    y_labels = label_encoder.inverse_transform(y)
    class_names = list(label_encoder.classes_)

    if labeled_features.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Feature/weight mismatch: feature_dim={labeled_features.shape[1]} vs weights={weights.shape[0]}"
        )

    ranked_indices = np.argsort(-weights, kind="stable")
    k_values = args.k_values if args.k_values else generate_default_k_values(labeled_features.shape[1])
    k_values = sorted({int(k) for k in k_values if 1 <= int(k) <= labeled_features.shape[1]})
    if not k_values:
        raise ValueError("No valid k values to evaluate.")

    all_rows: list[Dict[str, object]] = []
    best_runs: dict[str, Dict[str, object]] = {}

    for classifier in args.classifiers:
        best_runs[classifier] = {
            "score": float("-inf"),
            "accuracy": float("-inf"),
            "k": None,
            "metrics": None,
            "fold_rows": None,
            "oof_rows": None,
            "selected_indices": None,
        }

        for k in k_values:
            selected_indices = ranked_indices[:k]
            X_subset = labeled_features[:, selected_indices]
            overall_metrics, fold_rows, oof_rows = evaluate_subset(
                X=X_subset,
                y=y,
                y_labels=y_labels,
                groups=groups,
                class_names=class_names,
                classifier=classifier,
                args=args,
            )

            selected_weights = weights[selected_indices]
            row = {
                "classifier": classifier,
                "k": int(k),
                "selected_weight_sum": float(selected_weights.sum()),
                "selected_weight_share": float(selected_weights.sum() / max(float(weights.sum()), 1e-8)),
                "selected_weight_min": float(selected_weights.min()),
                "selected_weight_max": float(selected_weights.max()),
                "accuracy": overall_metrics["accuracy"],
                "macro_precision": overall_metrics["macro_precision"],
                "macro_recall": overall_metrics["macro_recall"],
                "macro_f1": overall_metrics["macro_f1"],
                "2d_precision": float(overall_metrics["classification_report"]["2d"]["precision"]),
                "2d_recall": float(overall_metrics["classification_report"]["2d"]["recall"]),
                "2d_f1": float(overall_metrics["classification_report"]["2d"]["f1-score"]),
                "3d_precision": float(overall_metrics["classification_report"]["3d"]["precision"]),
                "3d_recall": float(overall_metrics["classification_report"]["3d"]["recall"]),
                "3d_f1": float(overall_metrics["classification_report"]["3d"]["f1-score"]),
            }
            all_rows.append(row)

            primary_score = score_value(overall_metrics, args.score_key)
            current_best = best_runs[classifier]
            if (
                primary_score > current_best["score"]
                or (
                    np.isclose(primary_score, current_best["score"])
                    and overall_metrics["accuracy"] > current_best["accuracy"]
                )
                or (
                    np.isclose(primary_score, current_best["score"])
                    and np.isclose(overall_metrics["accuracy"], current_best["accuracy"])
                    and (current_best["k"] is None or int(k) < int(current_best["k"]))
                )
            ):
                current_best.update(
                    {
                        "score": float(primary_score),
                        "accuracy": float(overall_metrics["accuracy"]),
                        "k": int(k),
                        "metrics": overall_metrics,
                        "fold_rows": fold_rows,
                        "oof_rows": oof_rows,
                        "selected_indices": selected_indices.copy(),
                    }
                )

    results_csv = output_dir / "results.csv"
    top_results_csv = output_dir / "top_results.csv"
    best_summary_json = output_dir / "best_by_classifier.json"
    config_json = output_dir / "config.json"

    sorted_rows = sorted(
        all_rows,
        key=lambda row: (float(row[args.score_key]), float(row["accuracy"]), -int(row["k"])),
        reverse=True,
    )
    write_csv(all_rows, results_csv)
    write_csv(sorted_rows, top_results_csv)

    best_summary: dict[str, object] = {
        "features_dir": str(features_dir),
        "labels_file": str(labels_file),
        "score_key": args.score_key,
        "classifiers": list(args.classifiers),
        "k_values": [int(k) for k in k_values],
        "num_labeled_rois": int(len(labeled_table)),
        "num_groups": int(pd.Series(groups).nunique()),
        "class_names": class_names,
        "class_counts": {
            class_name: int((labeled_table["label"] == class_name).sum())
            for class_name in class_names
        },
        "best": {},
    }

    for classifier, info in best_runs.items():
        if info["k"] is None:
            continue
        classifier_dir = output_dir / f"best_{classifier}"
        ensure_dir(classifier_dir)

        selected_indices = np.asarray(info["selected_indices"], dtype=np.int32)
        selected_weights = weights[selected_indices]
        feature_rows = [
            {
                "rank": int(rank + 1),
                "feature_index": int(feature_index),
                "irelief_weight": float(weights[feature_index]),
            }
            for rank, feature_index in enumerate(selected_indices)
        ]
        write_csv(feature_rows, classifier_dir / "selected_features.csv")
        np.save(classifier_dir / "selected_feature_indices.npy", selected_indices)
        write_csv(info["fold_rows"], classifier_dir / "fold_metrics.csv")

        oof_rows: list[Dict[str, object]] = []
        labeled_table_reset = labeled_table.reset_index(drop=True)
        for base_row, pred_row in zip(labeled_table_reset.to_dict(orient="records"), info["oof_rows"]):
            merged = dict(base_row)
            merged.update(pred_row)
            oof_rows.append(merged)
        write_csv(oof_rows, classifier_dir / "oof_predictions.csv")

        summary = {
            "classifier": classifier,
            "score_key": args.score_key,
            "best_k": int(info["k"]),
            "selected_weight_sum": float(selected_weights.sum()),
            "selected_weight_share": float(selected_weights.sum() / max(float(weights.sum()), 1e-8)),
            "overall": info["metrics"],
            "folds": info["fold_rows"],
        }
        write_json(summary, classifier_dir / "summary.json")
        best_summary["best"][classifier] = summary

    write_json(
        {
            "features_dir": str(features_dir),
            "labels_file": str(labels_file),
            "score_key": args.score_key,
            "classifiers": list(args.classifiers),
            "k_values": [int(k) for k in k_values],
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "c_value": float(args.c_value),
            "max_iter": int(args.max_iter),
            "gamma": args.gamma,
            "class_weight": None if args.class_weight == "none" else args.class_weight,
            "rf_class_weight": None if args.rf_class_weight == "none" else args.rf_class_weight,
        },
        config_json,
    )
    write_json(best_summary, best_summary_json)

    print(f"Saved results: {results_csv}")
    print(f"Saved top results: {top_results_csv}")
    print(f"Saved best summary: {best_summary_json}")
    for classifier, info in best_runs.items():
        if info["k"] is not None:
            print(
                f"{classifier}: best_k={info['k']} | {args.score_key}={info['score']:.4f} | "
                f"accuracy={info['accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()
