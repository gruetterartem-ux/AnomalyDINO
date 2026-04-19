from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from component_memory_bank.data_io import load_run_samples
from component_memory_bank.export import write_json


DEFAULT_AUGMENTED_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413\component_memory_bank_backend\session_full\memory_bank_export\aug64_t8_cls_logreg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine 64x64 DINOv3 CLS features with the original center patch feature "
            "and train a grouped logistic regression."
        )
    )
    parser.add_argument("--augmented-dir", type=Path, default=DEFAULT_AUGMENTED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="balanced")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))).tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, class_name in enumerate(class_names)
        },
    }


def load_augmented_context(augmented_dir: Path) -> tuple[pd.DataFrame, np.ndarray, Path, int]:
    feature_table_path = augmented_dir / "augmented_patch_cls_feature_table.csv"
    feature_array_path = augmented_dir / "augmented_patch_cls_features.npy"
    summary_path = augmented_dir / "summary.json"

    if not feature_table_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_table_path}")
    if not feature_array_path.exists():
        raise FileNotFoundError(f"Missing feature array: {feature_array_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    metadata_df = pd.read_csv(feature_table_path)
    cls_features = np.load(feature_array_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    experiment_dir = Path(summary["experiment_dir"]).resolve()
    seed = int(summary["seed"])
    return metadata_df, cls_features, experiment_dir, seed


def load_source_patch_feature_map(metadata_df: pd.DataFrame, experiment_dir: Path, seed: int) -> dict[str, np.ndarray]:
    sample_map = {sample.sample: sample for sample in load_run_samples(experiment_dir, seed=seed)}
    unique_sources = (
        metadata_df[["source_patch_uid", "sample", "patch_index"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    feature_cache: dict[str, np.ndarray] = {}
    source_feature_map: dict[str, np.ndarray] = {}

    for row in unique_sources.itertuples(index=False):
        sample_name = str(row.sample)
        if sample_name not in sample_map:
            raise KeyError(f"Sample not found in run samples: {sample_name}")
        if sample_name not in feature_cache:
            with np.load(sample_map[sample_name].feature_cache_path) as cache_data:
                feature_cache[sample_name] = cache_data["features"].astype(np.float32)
        patch_features = feature_cache[sample_name]
        patch_index = int(row.patch_index)
        if patch_index < 0 or patch_index >= patch_features.shape[0]:
            raise IndexError(f"Patch index {patch_index} out of range for sample {sample_name}")
        source_feature_map[str(row.source_patch_uid)] = patch_features[patch_index].astype(np.float32)

    return source_feature_map


def build_combined_features(
    metadata_df: pd.DataFrame,
    cls_features: np.ndarray,
    source_feature_map: dict[str, np.ndarray],
    output_dir: Path,
) -> np.ndarray:
    if len(metadata_df) != cls_features.shape[0]:
        raise ValueError("Metadata row count and CLS feature count do not match.")

    source_features: list[np.ndarray] = []
    for source_patch_uid in metadata_df["source_patch_uid"].astype(str).tolist():
        if source_patch_uid not in source_feature_map:
            raise KeyError(f"Missing source patch feature for {source_patch_uid}")
        source_features.append(source_feature_map[source_patch_uid])

    source_feature_array = np.stack(source_features, axis=0).astype(np.float32)
    combined = np.concatenate([cls_features.astype(np.float32), source_feature_array], axis=1).astype(np.float32)

    np.save(output_dir / "augmented_patch_cls_plus_source_features.npy", combined)
    feature_table = metadata_df.copy()
    feature_table["feature_type"] = "dinov3_cls_64x64_plus_source_patch_feature"
    feature_table["embedding_dim_cls"] = int(cls_features.shape[1])
    feature_table["embedding_dim_source_patch"] = int(source_feature_array.shape[1])
    feature_table["embedding_dim_total"] = int(combined.shape[1])
    feature_table.to_csv(output_dir / "augmented_patch_cls_plus_source_feature_table.csv", index=False)
    return combined


def train_grouped_logreg(
    features: np.ndarray,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    n_splits: int,
    random_state: int,
    c_value: float,
    max_iter: int,
    class_weight: str | None,
) -> None:
    y_labels = metadata_df["label"].astype(str).to_numpy()
    class_names = ["2D", "3D"]
    y = np.array([0 if label == "2D" else 1 for label in y_labels], dtype=np.int32)
    groups = metadata_df["group_id"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=max_iter,
                    C=c_value,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(features, y, groups), start=1):
        pipeline.fit(features[train_idx], y[train_idx])
        y_pred = pipeline.predict(features[val_idx])
        y_proba = pipeline.predict_proba(features[val_idx]).astype(np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = _compute_metrics(y[val_idx], y_pred, class_names)
        fold_rows.append(
            {
                "fold": fold_idx,
                "num_train_crops": int(len(train_idx)),
                "num_val_crops": int(len(val_idx)),
                "num_train_source_patches": int(pd.Series(groups[train_idx]).nunique()),
                "num_val_source_patches": int(pd.Series(groups[val_idx]).nunique()),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "f1_2d": metrics["per_class"]["2D"]["f1"],
                "f1_3d": metrics["per_class"]["3D"]["f1"],
            }
        )

    if np.any(oof_pred < 0):
        raise RuntimeError("Some OOF predictions are missing.")

    overall = _compute_metrics(y, oof_pred, class_names)
    oof_rows: list[dict[str, object]] = []
    for idx, row in metadata_df.reset_index(drop=True).iterrows():
        out = row.to_dict()
        out["true_label"] = y_labels[idx]
        out["predicted_label"] = class_names[oof_pred[idx]]
        out["correct"] = int(y_labels[idx] == class_names[oof_pred[idx]])
        out["proba_2D"] = float(oof_proba[idx, 0])
        out["proba_3D"] = float(oof_proba[idx, 1])
        oof_rows.append(out)

    write_csv(fold_rows, output_dir / "logreg_groupcv_fold_metrics.csv")
    write_csv(oof_rows, output_dir / "logreg_groupcv_oof_predictions.csv")
    write_json(
        {
            "classifier": "logreg",
            "feature_type": "cls_plus_source_patch",
            "cv_type": "StratifiedGroupKFold",
            "grouping": "source_patch_uid",
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "class_weight": class_weight,
            "c_value": float(c_value),
            "max_iter": int(max_iter),
            "num_augmented_crops_total": int(len(metadata_df)),
            "num_augmented_crops_2d": int((metadata_df["label"] == "2D").sum()),
            "num_augmented_crops_3d": int((metadata_df["label"] == "3D").sum()),
            "num_source_patches_total": int(metadata_df["group_id"].nunique()),
            "num_source_patches_2d": int(metadata_df.loc[metadata_df["label"] == "2D", "group_id"].nunique()),
            "num_source_patches_3d": int(metadata_df.loc[metadata_df["label"] == "3D", "group_id"].nunique()),
            "embedding_dim_cls": 768,
            "embedding_dim_source_patch": 768,
            "embedding_dim_total": 1536,
            "overall": overall,
            "folds": fold_rows,
        },
        output_dir / "logreg_groupcv_summary.json",
    )


def resolve_output_dir(augmented_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (augmented_dir.parent / "aug64_t8_cls_plus_source_logreg").resolve()


def main() -> int:
    args = parse_args()
    augmented_dir = args.augmented_dir.resolve()
    output_dir = resolve_output_dir(augmented_dir, args.output_dir)
    ensure_dir(output_dir)

    metadata_df, cls_features, experiment_dir, seed = load_augmented_context(augmented_dir)
    source_feature_map = load_source_patch_feature_map(metadata_df, experiment_dir, seed)
    combined_features = build_combined_features(metadata_df, cls_features, source_feature_map, output_dir)

    class_weight = None if args.class_weight == "none" else args.class_weight
    train_grouped_logreg(
        features=combined_features,
        metadata_df=metadata_df,
        output_dir=output_dir,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        c_value=float(args.c_value),
        max_iter=int(args.max_iter),
        class_weight=class_weight,
    )

    write_json(
        {
            "augmented_dir": str(augmented_dir),
            "experiment_dir": str(experiment_dir),
            "seed": int(seed),
            "output_dir": str(output_dir),
            "input_cls_feature_table_csv": str(augmented_dir / "augmented_patch_cls_feature_table.csv"),
            "input_cls_features_file": str(augmented_dir / "augmented_patch_cls_features.npy"),
            "combined_features_file": str(output_dir / "augmented_patch_cls_plus_source_features.npy"),
            "combined_feature_table_csv": str(output_dir / "augmented_patch_cls_plus_source_feature_table.csv"),
            "logreg_summary_json": str(output_dir / "logreg_groupcv_summary.json"),
        },
        output_dir / "summary.json",
    )

    print(f"CLS + source patch feature logreg complete. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
