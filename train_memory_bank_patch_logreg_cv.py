from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from component_memory_bank.export import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a patch-level logistic regression on the 2D/3D memory-bank features."
    )
    parser.add_argument("--memory-bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["logreg", "svm_linear", "svm_rbf"],
        default="logreg",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="balanced")
    return parser.parse_args()


def _align_metadata_to_feature_order(metadata: pd.DataFrame, n_2d: int, n_3d: int) -> pd.DataFrame:
    labels = metadata["component_label"].astype(str)
    meta_2d = metadata[labels == "2D"].copy()
    meta_3d = metadata[labels == "3D"].copy()
    if len(meta_2d) != n_2d or len(meta_3d) != n_3d:
        raise ValueError(
            f"Metadata size mismatch: expected {n_2d}x2D and {n_3d}x3D, "
            f"got {len(meta_2d)}x2D and {len(meta_3d)}x3D."
        )
    return pd.concat([meta_2d, meta_3d], axis=0, ignore_index=True)


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


def _build_pipeline(classifier: str, c_value: float, max_iter: int, class_weight: str | None) -> Pipeline:
    if classifier == "logreg":
        estimator = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            C=c_value,
            class_weight=class_weight,
        )
    elif classifier == "svm_linear":
        estimator = SVC(
            kernel="linear",
            C=c_value,
            class_weight=class_weight,
            probability=True,
            max_iter=max_iter,
        )
    elif classifier == "svm_rbf":
        estimator = SVC(
            kernel="rbf",
            C=c_value,
            class_weight=class_weight,
            probability=True,
            max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def main() -> int:
    args = parse_args()
    memory_bank_dir = args.memory_bank_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (memory_bank_dir / "patch_logreg_cv")
    output_dir.mkdir(parents=True, exist_ok=True)

    features_2d = np.load(memory_bank_dir / "2D-memory-bank.npy").astype(np.float32)
    features_3d = np.load(memory_bank_dir / "3D-memory-bank.npy").astype(np.float32)
    metadata = pd.read_csv(memory_bank_dir / "selected_patches.csv")
    metadata = _align_metadata_to_feature_order(metadata, len(features_2d), len(features_3d))

    X = np.vstack([features_2d, features_3d]).astype(np.float32, copy=False)
    y_labels = np.array(["2D"] * len(features_2d) + ["3D"] * len(features_3d))
    class_names = ["2D", "3D"]
    y = np.array([0 if label == "2D" else 1 for label in y_labels], dtype=np.int32)

    splitter = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    class_weight = None if args.class_weight == "none" else args.class_weight
    pipeline = _build_pipeline(
        classifier=args.classifier,
        c_value=args.c_value,
        max_iter=args.max_iter,
        class_weight=class_weight,
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[val_idx])
        y_proba = pipeline.predict_proba(X[val_idx]).astype(np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = _compute_metrics(y[val_idx], y_pred, class_names)
        fold_rows.append(
            {
                "fold": fold_idx,
                "num_train_patches": int(len(train_idx)),
                "num_val_patches": int(len(val_idx)),
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
    for idx, row in metadata.reset_index(drop=True).iterrows():
        out = row.to_dict()
        out["true_label"] = y_labels[idx]
        out["predicted_label"] = class_names[oof_pred[idx]]
        out["correct"] = int(y_labels[idx] == class_names[oof_pred[idx]])
        out["proba_2D"] = float(oof_proba[idx, 0])
        out["proba_3D"] = float(oof_proba[idx, 1])
        oof_rows.append(out)

    write_csv(fold_rows, output_dir / "fold_metrics.csv")
    write_csv(oof_rows, output_dir / "oof_predictions.csv")
    write_json(
        {
            "memory_bank_dir": str(memory_bank_dir),
            "output_dir": str(output_dir),
            "classifier": args.classifier,
            "cv_type": "StratifiedKFold",
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "class_weight": class_weight,
            "c_value": float(args.c_value),
            "max_iter": int(args.max_iter),
            "num_patches_total": int(len(y)),
            "num_patches_2d": int(len(features_2d)),
            "num_patches_3d": int(len(features_3d)),
            "overall": overall,
            "folds": fold_rows,
            "fold_metrics_file": str(output_dir / "fold_metrics.csv"),
            "oof_predictions_file": str(output_dir / "oof_predictions.csv"),
        },
        output_dir / "summary.json",
    )

    print(
        f"Patch-level logreg CV complete. Output: {output_dir}\n"
        f"Macro F1={overall['macro_f1']:.4f}, Accuracy={overall['accuracy']:.4f}, "
        f"F1_2D={overall['per_class']['2D']['f1']:.4f}, F1_3D={overall['per_class']['3D']['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
