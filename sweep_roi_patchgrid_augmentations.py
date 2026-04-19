from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from extract_labeled_roi_toppercent_multilayer_softmax_patch_features import (
    build_multilayer_run_context,
    concatenated_patch_features,
    load_multilayer_cache,
    load_patch_scores,
    load_labels_table,
    load_roi_table,
    prepare_labeled_roi_table,
    aggregate_selected_patch_features,
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LABELS_FILE,
    DEFAULT_ROI_METADATA_CSV,
)
from extract_labeled_roi_toppercent_pca_softmax_patch_features import (
    bbox_patch_window,
    select_roi_patches_center_in_box,
)


DEFAULT_IRELIEF_FEATURES_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_top10pct_centerinbox_multilayer_l1to12_softmax_patch_features_labeled"
    / "irelief_cosine_weighted_features"
)
DEFAULT_TOPK_SWEEP_DIR = DEFAULT_IRELIEF_FEATURES_DIR / "topk_sweep_macro_f1"
DEFAULT_MULTILAYER_CACHE_SUBDIR = "patch_feature_cache_multilayer_l1to12"


@dataclass(frozen=True)
class AugSpec:
    name: str
    dx: int
    dy: int
    grow: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test safe normalmap ROI augmentations by jittering the ROI patch window in patch space, "
            "then training classifiers on the current best I-Relief-weighted multi-layer feature setup."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--irelief-features-dir", type=Path, default=DEFAULT_IRELIEF_FEATURES_DIR)
    parser.add_argument("--topk-sweep-dir", type=Path, default=DEFAULT_TOPK_SWEEP_DIR)
    parser.add_argument("--multilayer-cache-subdir", type=str, default=DEFAULT_MULTILAYER_CACHE_SUBDIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-percent", type=float, default=0.10)
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--classifiers", nargs="*", default=("svm_rbf", "logreg"), choices=("svm_rbf", "logreg"))
    parser.add_argument("--score-key", type=str, default="macro_f1", choices=("macro_f1", "3d_recall", "3d_f1", "accuracy"))
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--logreg-max-iter", type=int, default=4000)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
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


def default_output_dir(irelief_features_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (irelief_features_dir / "augmentation_sweep_current_best").resolve()


def load_topk_config(topk_sweep_dir: Path, classifier: str) -> tuple[np.ndarray, dict]:
    indices_file = topk_sweep_dir / f"best_{classifier}" / "selected_feature_indices.npy"
    summary_file = topk_sweep_dir / f"best_{classifier}" / "summary.json"
    if not indices_file.exists():
        raise FileNotFoundError(f"Missing selected feature indices: {indices_file}")
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing best classifier summary: {summary_file}")
    indices = np.load(indices_file).astype(np.int32)
    with summary_file.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return indices, summary


def load_irelief_scale(irelief_features_dir: Path) -> np.ndarray:
    scale_path = irelief_features_dir / "irelief_feature_scale_sqrt.npy"
    if not scale_path.exists():
        raise FileNotFoundError(f"Missing I-Relief scale file: {scale_path}")
    scale = np.load(scale_path).astype(np.float32)
    if scale.ndim != 1:
        raise ValueError(f"I-Relief scale must be 1D, got {scale.shape}")
    return scale


def apply_irelief_scale(feature_matrix: np.ndarray, scale_sqrt: np.ndarray) -> np.ndarray:
    if feature_matrix.shape[1] != scale_sqrt.shape[0]:
        raise ValueError(
            f"Feature/scale mismatch: feature_dim={feature_matrix.shape[1]} vs scale_dim={scale_sqrt.shape[0]}"
        )
    weighted = feature_matrix * scale_sqrt[None, :]
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (weighted / norms).astype(np.float32)


def clip_patch_window(
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[int, int, int, int]:
    row_min = max(0, min(row_min, grid_rows - 1))
    col_min = max(0, min(col_min, grid_cols - 1))
    row_max = max(row_min + 1, min(row_max, grid_rows))
    col_max = max(col_min + 1, min(col_max, grid_cols))
    return row_min, row_max, col_min, col_max


def select_from_patch_window(
    score_grid: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    top_percent: float,
    min_patches: int,
) -> tuple[list[tuple[int, int]], int]:
    candidates: list[tuple[float, int, int]] = []
    for patch_row in range(row_min, row_max):
        for patch_col in range(col_min, col_max):
            candidates.append((float(score_grid[patch_row, patch_col]), patch_row, patch_col))
    candidates.sort(key=lambda item: item[0], reverse=True)
    num_candidates = len(candidates)
    top_k = max(int(min_patches), int(math.ceil(float(top_percent) * float(num_candidates))))
    top_k = min(top_k, num_candidates)
    selected = [(patch_row, patch_col) for _, patch_row, patch_col in candidates[:top_k]]
    return selected, int(num_candidates)


def augmentation_specs() -> dict[str, list[AugSpec]]:
    orthogonal = [
        AugSpec("dxm1_dy0", dx=-1, dy=0, grow=0),
        AugSpec("dxp1_dy0", dx=1, dy=0, grow=0),
        AugSpec("dx0_dym1", dx=0, dy=-1, grow=0),
        AugSpec("dx0_dyp1", dx=0, dy=1, grow=0),
    ]
    diagonal = [
        AugSpec("dxm1_dym1", dx=-1, dy=-1, grow=0),
        AugSpec("dxm1_dyp1", dx=-1, dy=1, grow=0),
        AugSpec("dxp1_dym1", dx=1, dy=-1, grow=0),
        AugSpec("dxp1_dyp1", dx=1, dy=1, grow=0),
    ]
    orthogonal_g1 = [AugSpec(spec.name + "_g1", dx=spec.dx, dy=spec.dy, grow=1) for spec in orthogonal]
    diagonal_g1 = [AugSpec(spec.name + "_g1", dx=spec.dx, dy=spec.dy, grow=1) for spec in diagonal]
    return {
        "none": [],
        "expand1": [AugSpec("expand1", dx=0, dy=0, grow=1)],
        "shift4": orthogonal,
        "shift8": orthogonal + diagonal,
        "shift4_expand1": orthogonal_g1,
        "shift8_expand1": orthogonal_g1 + diagonal_g1,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist(),
        "classification_report": report,
    }


def score_value(metrics: dict[str, object], score_key: str) -> float:
    if score_key in ("macro_f1", "accuracy"):
        return float(metrics[score_key])
    if score_key == "3d_recall":
        return float(metrics["classification_report"]["3d"]["recall"])
    if score_key == "3d_f1":
        return float(metrics["classification_report"]["3d"]["f1-score"])
    raise ValueError(f"Unsupported score key: {score_key}")


def build_classifier(classifier: str, args: argparse.Namespace):
    class_weight = None if args.class_weight == "none" else args.class_weight
    if classifier == "svm_rbf":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=args.svm_c,
                        gamma=args.svm_gamma,
                        class_weight=class_weight,
                        probability=True,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )
    if classifier == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        C=args.logreg_c,
                        max_iter=args.logreg_max_iter,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported classifier: {classifier}")


def build_policy_feature_sets(
    labeled_rois: pd.DataFrame,
    sample_map: dict[str, dict],
    top_percent: float,
    min_patches: int,
) -> dict[str, np.ndarray]:
    policy_specs = augmentation_specs()
    policy_features: dict[str, list[np.ndarray]] = {policy: [] for policy in policy_specs}
    cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], dict]] = {}

    for row in labeled_rois.itertuples(index=False):
        roi_row = pd.Series(row._asdict())
        sample_name = str(roi_row["sample"]).replace("\\", "/")
        if sample_name not in sample_map:
            raise KeyError(f"Sample missing from run context: {sample_name}")
        sample_info = sample_map[sample_name]

        if sample_name not in cache:
            features_layers, grid_shape, _, cache_meta = load_multilayer_cache(Path(sample_info["feature_cache_path"]))
            concat_features = concatenated_patch_features(features_layers)
            score_grid = load_patch_scores(sample_info["run_sample"])
            cache[sample_name] = (concat_features, score_grid, grid_shape, cache_meta)

        concat_features, score_grid, grid_shape, cache_meta = cache[sample_name]
        selected_base, _, _ = select_roi_patches_center_in_box(
            roi_row,
            cache_meta,
            score_grid,
            top_percent=float(top_percent),
            min_patches=int(min_patches),
        )
        base_feature, _, _ = aggregate_selected_patch_features(
            concat_features,
            score_grid,
            grid_shape,
            selected_base,
        )
        policy_features["none"].append(base_feature)

        base_row_min, base_row_max, base_col_min, base_col_max = bbox_patch_window(roi_row, cache_meta)
        for policy_name, specs in policy_specs.items():
            if policy_name == "none":
                continue
            aug_vectors: list[np.ndarray] = []
            seen_windows: set[tuple[int, int, int, int]] = set()
            for spec in specs:
                row_min, row_max, col_min, col_max = clip_patch_window(
                    base_row_min + spec.dy - spec.grow,
                    base_row_max + spec.dy + spec.grow,
                    base_col_min + spec.dx - spec.grow,
                    base_col_max + spec.dx + spec.grow,
                    cache_meta["grid_rows"],
                    cache_meta["grid_cols"],
                )
                window = (row_min, row_max, col_min, col_max)
                if window in seen_windows:
                    continue
                seen_windows.add(window)
                selected_aug, _ = select_from_patch_window(
                    score_grid,
                    row_min,
                    row_max,
                    col_min,
                    col_max,
                    top_percent=float(top_percent),
                    min_patches=int(min_patches),
                )
                aug_feature, _, _ = aggregate_selected_patch_features(
                    concat_features,
                    score_grid,
                    grid_shape,
                    selected_aug,
                )
                aug_vectors.append(aug_feature)
            if aug_vectors:
                policy_features[policy_name].append(np.stack(aug_vectors, axis=0))
            else:
                policy_features[policy_name].append(np.zeros((0, concat_features.shape[1]), dtype=np.float32))

    stacked: dict[str, np.ndarray] = {}
    for policy_name, per_roi_values in policy_features.items():
        if policy_name == "none":
            stacked[policy_name] = np.stack(per_roi_values, axis=0).astype(np.float32)
        else:
            stacked[policy_name] = np.array(per_roi_values, dtype=object)
    return stacked


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    irelief_features_dir = args.irelief_features_dir.resolve()
    topk_sweep_dir = args.topk_sweep_dir.resolve()
    output_dir = default_output_dir(irelief_features_dir, args.output_dir)
    ensure_dir(output_dir)

    roi_table = load_roi_table(roi_metadata_csv)
    labels_table = load_labels_table(labels_file, list(args.valid_labels) if args.valid_labels else None)
    labeled_rois = prepare_labeled_roi_table(roi_table, labels_table, limit=None).copy()
    labeled_rois["label"] = labeled_rois["label"].astype(str).str.strip()
    labeled_rois = labeled_rois[labeled_rois["label"].isin(list(args.valid_labels))].reset_index(drop=True)
    if labeled_rois.empty:
        raise ValueError("No labeled ROIs found after filtering valid labels.")

    sample_map = build_multilayer_run_context(
        experiment_dir,
        seed=int(args.seed),
        cache_subdir=str(args.multilayer_cache_subdir),
    )
    scale_sqrt = load_irelief_scale(irelief_features_dir)
    policy_feature_sets = build_policy_feature_sets(
        labeled_rois=labeled_rois,
        sample_map=sample_map,
        top_percent=float(args.top_percent),
        min_patches=int(args.min_patches),
    )

    label_encoder = LabelEncoder()
    label_encoder.fit([str(label).lower() for label in args.valid_labels])
    y_labels = labeled_rois["label"].astype(str).str.lower().to_numpy()
    y = label_encoder.transform(y_labels)
    class_names = list(label_encoder.classes_)
    groups = labeled_rois["sample"].astype(str).to_numpy()

    baseline_weighted = apply_irelief_scale(np.asarray(policy_feature_sets["none"], dtype=np.float32), scale_sqrt)

    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    splitter = StratifiedGroupKFold(
        n_splits=int(args.n_splits),
        shuffle=True,
        random_state=int(args.random_state),
    )

    for classifier in args.classifiers:
        selected_indices, selected_summary = load_topk_config(topk_sweep_dir, classifier)
        baseline_selected = baseline_weighted[:, selected_indices].astype(np.float32)
        best_score = float("-inf")
        best_accuracy = float("-inf")
        best_result: dict[str, object] | None = None

        for policy_name in augmentation_specs().keys():
            oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
            oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
            fold_rows: list[dict[str, object]] = []

            for fold_index, (train_idx, val_idx) in enumerate(splitter.split(baseline_selected, y, groups), start=1):
                X_train = baseline_selected[train_idx]
                y_train = y[train_idx]

                if policy_name != "none":
                    aug_objects = policy_feature_sets[policy_name][train_idx]
                    aug_rows: list[np.ndarray] = []
                    aug_labels: list[int] = []
                    for local_idx, aug_stack in enumerate(aug_objects):
                        aug_stack = np.asarray(aug_stack, dtype=np.float32)
                        if aug_stack.size == 0:
                            continue
                        aug_weighted = apply_irelief_scale(aug_stack, scale_sqrt)[:, selected_indices]
                        aug_rows.append(aug_weighted)
                        aug_labels.extend([int(y_train[local_idx])] * int(aug_weighted.shape[0]))
                    if aug_rows:
                        X_train = np.vstack([X_train, *aug_rows]).astype(np.float32)
                        y_train = np.concatenate([y_train, np.array(aug_labels, dtype=np.int32)], axis=0)

                model = build_classifier(classifier, args)
                model.fit(X_train, y_train)

                X_val = baseline_selected[val_idx]
                y_val = y[val_idx]
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val).astype(np.float32)
                oof_pred[val_idx] = y_pred
                oof_proba[val_idx] = y_proba

                fold_metrics = compute_metrics(
                    np.array(class_names)[y_val],
                    np.array(class_names)[y_pred],
                    class_names,
                )
                fold_rows.append(
                    {
                        "fold": fold_index,
                        "policy": policy_name,
                        "classifier": classifier,
                        "num_train_rois": int(len(train_idx)),
                        "num_val_rois": int(len(val_idx)),
                        "num_train_groups": int(pd.Series(groups[train_idx]).nunique()),
                        "num_val_groups": int(pd.Series(groups[val_idx]).nunique()),
                        "num_train_rows_after_aug": int(len(y_train)),
                        "accuracy": float(fold_metrics["accuracy"]),
                        "macro_precision": float(fold_metrics["macro_precision"]),
                        "macro_recall": float(fold_metrics["macro_recall"]),
                        "macro_f1": float(fold_metrics["macro_f1"]),
                    }
                )

            if np.any(oof_pred < 0):
                raise RuntimeError(f"OOF predictions incomplete for policy={policy_name}, classifier={classifier}")

            overall = compute_metrics(y_labels, np.array(class_names)[oof_pred], class_names)
            row = {
                "classifier": classifier,
                "policy": policy_name,
                "selected_topk": int(len(selected_indices)),
                "baseline_topk_classifier": classifier,
                "baseline_topk_score_key": str(selected_summary["score_key"]),
                "baseline_topk_best_k": int(selected_summary["best_k"]),
                "accuracy": float(overall["accuracy"]),
                "macro_precision": float(overall["macro_precision"]),
                "macro_recall": float(overall["macro_recall"]),
                "macro_f1": float(overall["macro_f1"]),
                "2d_precision": float(overall["classification_report"]["2d"]["precision"]),
                "2d_recall": float(overall["classification_report"]["2d"]["recall"]),
                "2d_f1": float(overall["classification_report"]["2d"]["f1-score"]),
                "3d_precision": float(overall["classification_report"]["3d"]["precision"]),
                "3d_recall": float(overall["classification_report"]["3d"]["recall"]),
                "3d_f1": float(overall["classification_report"]["3d"]["f1-score"]),
            }
            all_rows.append(row)

            primary_score = score_value(overall, args.score_key)
            if (
                primary_score > best_score
                or (np.isclose(primary_score, best_score) and float(overall["accuracy"]) > best_accuracy)
            ):
                best_score = float(primary_score)
                best_accuracy = float(overall["accuracy"])
                best_result = {
                    "classifier": classifier,
                    "policy": policy_name,
                    "score_key": args.score_key,
                    "selected_topk": int(len(selected_indices)),
                    "overall": overall,
                    "folds": fold_rows,
                }

        if best_result is not None:
            best_rows.append(
                {
                    "classifier": best_result["classifier"],
                    "policy": best_result["policy"],
                    "score_key": best_result["score_key"],
                    "selected_topk": best_result["selected_topk"],
                    "accuracy": float(best_result["overall"]["accuracy"]),
                    "macro_f1": float(best_result["overall"]["macro_f1"]),
                    "macro_recall": float(best_result["overall"]["macro_recall"]),
                    "3d_recall": float(best_result["overall"]["classification_report"]["3d"]["recall"]),
                    "3d_f1": float(best_result["overall"]["classification_report"]["3d"]["f1-score"]),
                }
            )
            classifier_dir = output_dir / f"best_{classifier}"
            write_json(best_result, classifier_dir / "summary.json")
            write_csv(best_result["folds"], classifier_dir / "fold_metrics.csv")

    results_csv = output_dir / "results.csv"
    best_csv = output_dir / "best_results.csv"
    summary_json = output_dir / "summary.json"
    write_csv(all_rows, results_csv)
    write_csv(best_rows, best_csv)
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "irelief_features_dir": str(irelief_features_dir),
            "topk_sweep_dir": str(topk_sweep_dir),
            "score_key": args.score_key,
            "classifiers": list(args.classifiers),
            "policies": list(augmentation_specs().keys()),
            "num_labeled_rois": int(len(labeled_rois)),
            "num_groups": int(pd.Series(groups).nunique()),
            "class_counts": {
                str(label): int((y_labels == str(label).lower()).sum())
                for label in class_names
            },
            "results_csv": str(results_csv),
            "best_results_csv": str(best_csv),
        },
        summary_json,
    )

    print(f"Saved results: {results_csv}")
    print(f"Saved best results: {best_csv}")
    for row in best_rows:
        print(
            f"{row['classifier']}: best policy={row['policy']} | {args.score_key}="
            f"{row[args.score_key] if args.score_key in row else row['macro_f1']:.4f} | "
            f"acc={row['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
