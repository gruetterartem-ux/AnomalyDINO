import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_EMBEDDINGS_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704\16-shot_preprocess=force_no_mask_no_rotation_all16_test_maxpatch_random\roi_crops_peak_seeds\seed=0\dinov2_vitb14_cls_embeddings"
)
VALID_CLASSES = ("contamination", "defect")
IMAGE_PRIORITY = {"contamination": 0, "defect": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a logistic regression on ROI embeddings with manual ROI annotations."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory that contains embeddings_cls.npy and embedding_metadata.csv.",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=None,
        help="CSV file with at least relative_path and roi_label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <embeddings-dir>/logreg_results.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of images used for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for the grouped split.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum solver iterations for logistic regression.",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_inputs(embeddings_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    embeddings_file = embeddings_dir / "embeddings_cls.npy"
    metadata_file = embeddings_dir / "embedding_metadata.csv"
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    embeddings = np.load(embeddings_file)
    metadata = pd.read_csv(metadata_file)
    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Embeddings and metadata length mismatch: {len(embeddings)} vs {len(metadata)}"
        )
    return embeddings, metadata


def normalize_relative_path(path_value: str) -> str:
    return str(Path(path_value).as_posix())


def load_annotations(annotations_file: Path) -> pd.DataFrame:
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    annotations = pd.read_csv(annotations_file)
    required_columns = {"relative_path", "roi_label"}
    missing_columns = required_columns.difference(annotations.columns)
    if missing_columns:
        raise ValueError(
            f"Annotations file is missing required columns: {sorted(missing_columns)}"
        )

    annotations = annotations.copy()
    annotations["relative_path"] = annotations["relative_path"].map(normalize_relative_path)
    annotations["roi_label"] = annotations["roi_label"].astype(str).str.strip().str.lower()

    invalid_labels = sorted(set(annotations["roi_label"]) - set(VALID_CLASSES))
    if invalid_labels:
        raise ValueError(
            f"Unsupported roi_label values found: {invalid_labels}. Expected only {list(VALID_CLASSES)}."
        )

    if annotations["relative_path"].duplicated().any():
        duplicated_paths = annotations.loc[annotations["relative_path"].duplicated(), "relative_path"].tolist()
        raise ValueError(f"Duplicate relative_path entries in annotations: {duplicated_paths[:10]}")

    return annotations


def attach_annotations(metadata: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    data = metadata.copy()
    data["relative_path"] = data["relative_path"].map(normalize_relative_path)
    merged = data.merge(annotations, on="relative_path", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No annotated ROIs matched embedding metadata.")
    return merged


def infer_image_id(row: pd.Series) -> str:
    if "roi_object" in row and "roi_sample" in row:
        return f"{row['roi_object']}::{row['roi_sample']}"
    return str(Path(row["relative_path"]).with_suffix("").parent.as_posix())


def split_grouped(data: pd.DataFrame, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    groups = data["image_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(data, groups=groups))
    return train_idx, test_idx


def build_pipeline(c_value: float, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=max_iter,
                    C=c_value,
                ),
            ),
        ]
    )


def summarize_split(name: str, data: pd.DataFrame) -> Dict[str, object]:
    return {
        "num_rois": int(len(data)),
        "num_images": int(data["image_id"].nunique()),
        "class_counts": {
            label: int(count)
            for label, count in data["roi_label"].value_counts().sort_index().items()
        },
    }


def image_level_label_from_rois(labels: List[str]) -> str:
    return max(labels, key=lambda label: IMAGE_PRIORITY[label])


def aggregate_image_predictions(probabilities: np.ndarray, class_names: List[str]) -> str:
    best_label = class_names[int(np.argmax(probabilities.mean(axis=0)))]
    if "defect" in class_names:
        defect_idx = class_names.index("defect")
        if np.any(np.argmax(probabilities, axis=1) == defect_idx):
            return "defect"
    if "contamination" in class_names:
        contamination_idx = class_names.index("contamination")
        if np.any(np.argmax(probabilities, axis=1) == contamination_idx):
            return "contamination"
    return best_label


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


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


def main():
    args = parse_args()
    embeddings_dir = args.embeddings_dir.resolve()
    annotations_file = (
        args.annotations_file.resolve()
        if args.annotations_file is not None
        else embeddings_dir / "roi_annotations.csv"
    )
    output_dir = (args.output_dir or (embeddings_dir / "logreg_results")).resolve()
    ensure_dir(output_dir)

    embeddings, metadata = load_inputs(embeddings_dir)
    annotations = load_annotations(annotations_file)
    data = attach_annotations(metadata, annotations)
    data["image_id"] = data.apply(infer_image_id, axis=1)

    embeddings_subset = embeddings[data["embedding_index"].to_numpy()]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["roi_label"])
    class_names = list(label_encoder.classes_)

    if sorted(class_names) != list(sorted(VALID_CLASSES)):
        print(f"Warning: observed classes are {class_names}; expected {list(VALID_CLASSES)}.")

    train_idx, test_idx = split_grouped(data, args.test_size, args.random_state)
    X_train = embeddings_subset[train_idx]
    X_test = embeddings_subset[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    pipeline = build_pipeline(args.c_value, args.max_iter)
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    y_prob_test = pipeline.predict_proba(X_test)

    roi_metrics = {
        "train": compute_metrics(train_data["roi_label"].to_numpy(), label_encoder.inverse_transform(y_pred_train), class_names),
        "test": compute_metrics(test_data["roi_label"].to_numpy(), label_encoder.inverse_transform(y_pred_test), class_names),
    }

    roi_prediction_rows: List[Dict[str, object]] = []
    for row_index, row in test_data.iterrows():
        pred_label = label_encoder.inverse_transform([y_pred_test[row_index]])[0]
        output_row: Dict[str, object] = {
            "split": "test",
            "embedding_index": int(row["embedding_index"]),
            "relative_path": row["relative_path"],
            "image_id": row["image_id"],
            "true_label": row["roi_label"],
            "pred_label": pred_label,
        }
        for class_index, class_name in enumerate(class_names):
            output_row[f"prob_{class_name}"] = float(y_prob_test[row_index, class_index])
        roi_prediction_rows.append(output_row)

    image_prediction_rows: List[Dict[str, object]] = []
    grouped = test_data.copy()
    grouped["pred_label"] = label_encoder.inverse_transform(y_pred_test)
    for image_id, image_group in grouped.groupby("image_id", sort=True):
        roi_indices = image_group.index.to_numpy()
        image_probabilities = y_prob_test[roi_indices]
        image_true = image_level_label_from_rois(image_group["roi_label"].tolist())
        image_pred = aggregate_image_predictions(image_probabilities, class_names)
        output_row: Dict[str, object] = {
            "split": "test",
            "image_id": image_id,
            "num_rois": int(len(image_group)),
            "true_label": image_true,
            "pred_label": image_pred,
        }
        for class_index, class_name in enumerate(class_names):
            output_row[f"mean_prob_{class_name}"] = float(image_probabilities[:, class_index].mean())
            output_row[f"max_prob_{class_name}"] = float(image_probabilities[:, class_index].max())
        image_prediction_rows.append(output_row)

    image_metrics = compute_metrics(
        np.array([row["true_label"] for row in image_prediction_rows]),
        np.array([row["pred_label"] for row in image_prediction_rows]),
        class_names,
    )

    summary = {
        "classes": class_names,
        "train_split": summarize_split("train", train_data),
        "test_split": summarize_split("test", test_data),
        "roi_metrics": roi_metrics,
        "image_metrics": image_metrics,
        "config": {
            "embeddings_dir": str(embeddings_dir),
            "annotations_file": str(annotations_file),
            "output_dir": str(output_dir),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "max_iter": args.max_iter,
            "c_value": args.c_value,
        },
    }

    roi_predictions_file = output_dir / "roi_predictions_test.csv"
    image_predictions_file = output_dir / "image_predictions_test.csv"
    summary_file = output_dir / "summary.json"
    model_file = output_dir / "logreg_model.joblib"

    write_csv(roi_prediction_rows, roi_predictions_file)
    write_csv(image_prediction_rows, image_predictions_file)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import joblib

        joblib.dump(
            {
                "pipeline": pipeline,
                "label_encoder": label_encoder,
                "class_names": class_names,
            },
            model_file,
        )
        saved_model = True
    except ModuleNotFoundError:
        saved_model = False

    print(f"Saved ROI predictions: {roi_predictions_file}")
    print(f"Saved image predictions: {image_predictions_file}")
    print(f"Saved summary: {summary_file}")
    if saved_model:
        print(f"Saved model: {model_file}")
    else:
        print("Model serialization skipped because joblib is not available.")

    print(
        "ROI test metrics:",
        f"accuracy={summary['roi_metrics']['test']['accuracy']:.4f}",
        f"macro_f1={summary['roi_metrics']['test']['macro_f1']:.4f}",
    )
    print(
        "Image test metrics:",
        f"accuracy={summary['image_metrics']['accuracy']:.4f}",
        f"macro_f1={summary['image_metrics']['macro_f1']:.4f}",
    )


if __name__ == "__main__":
    main()
