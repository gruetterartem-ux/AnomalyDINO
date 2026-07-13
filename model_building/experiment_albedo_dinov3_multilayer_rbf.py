from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalydino_app import app as app_mod
from extract_labeled_roi_overthreshold_multilayer_maxminmean_features import (
    aggregate_maxminmean_per_layer,
)
from model_building.crossval_mrmr_maxminmean_rbf_svm import rank_mrmr_features
from model_building.rbf_svm_utils import build_classifier, compute_metrics


EXPERIMENT_DIR = PROJECT_ROOT / "results_FINAL" / "normalmap_dinov3_vitb16_res688"
FEATURES_DIR = (
    EXPERIMENT_DIR
    / "roi_overthreshold_overlap_multilayer_l1to12_maxminmean_features_labeled"
)
ALBEDO_ROOT = Path(r"C:\anomalydino_data\albedo")
K_VALUES = (16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DINOv3 layer-12 max/min/mean features at the saved normal-map patch "
            "positions and evaluate fold-safe albedo feature augmentation."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--albedo-root", type=Path, default=ALBEDO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=(12,))
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--albedo-prefilter-top", type=int, default=1024)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_patch_positions(row: pd.Series) -> list[tuple[int, int]]:
    rows = [int(value) for value in str(row["selected_patch_rows"]).split(";") if value != ""]
    cols = [int(value) for value in str(row["selected_patch_cols"]).split(";") if value != ""]
    if len(rows) != len(cols) or not rows:
        raise ValueError(f"Invalid saved patch positions for {row['roi_uid']}")
    return list(zip(rows, cols))


def unique_image_lookup(root: Path) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {}
    for path in root.rglob("*.png"):
        grouped.setdefault(path.name, []).append(path.resolve())
    duplicates = [name for name, paths in grouped.items() if len(paths) > 1]
    if duplicates:
        raise ValueError(f"Ambiguous albedo image filenames: {duplicates[:5]}")
    return {name: paths[0] for name, paths in grouped.items()}


def extract_albedo_dino_features(
    experiment_dir: Path,
    albedo_root: Path,
    table: pd.DataFrame,
    output_dir: Path,
    layers: tuple[int, ...],
) -> np.ndarray:
    layer_key = "_".join(str(layer) for layer in layers)
    feature_dimension = len(layers) * 2304
    cache_file = output_dir / f"albedo_dinov3_l{layer_key}_maxminmean.npy"
    cache_metadata = output_dir / f"albedo_dinov3_l{layer_key}_metadata.csv"
    if cache_file.exists() and cache_metadata.exists():
        cached = np.load(cache_file).astype(np.float32)
        metadata = pd.read_csv(cache_metadata)
        if (
            cached.shape == (len(table), feature_dimension)
            and metadata["roi_uid"].astype(str).tolist() == table["roi_uid"].astype(str).tolist()
        ):
            print(f"[Albedo DINO] reuse cache: {cache_file}", flush=True)
            return cached

    backbone = app_mod.load_component_test_backbone(str(experiment_dir.resolve()))
    model = backbone["model"]
    image_lookup = unique_image_lookup(albedo_root)
    features = np.zeros((len(table), feature_dimension), dtype=np.float32)
    metadata_rows: list[dict[str, Any]] = [{} for _ in range(len(table))]
    grouped = list(table.groupby("bildname", sort=True))
    start = time.perf_counter()

    for image_index, (image_name, image_rows) in enumerate(grouped, start=1):
        if image_name not in image_lookup:
            raise FileNotFoundError(f"No albedo image found for {image_name}")
        image_path = image_lookup[image_name]
        with Image.open(image_path) as image:
            if image.mode != "L":
                raise ValueError(f"Expected grayscale albedo image, got {image.mode}: {image_path}")
            image_rgb = image.convert("RGB")
            image_tensor, grid_shape = model.prepare_image(image_rgb)
        layer_features, layer_indices = model.extract_multilayer_features(
            image_tensor, layer_indices=list(layers)
        )
        layer_features = np.asarray(layer_features, dtype=np.float32)
        if list(layer_indices) != list(layers) or layer_features.shape[1:] != (len(layers), 768):
            raise ValueError(
                f"Unexpected DINOv3 layer output for {image_name}: {layer_features.shape}, {layer_indices}"
            )

        for row_index, row in image_rows.iterrows():
            expected_grid = (int(row["grid_rows"]), int(row["grid_cols"]))
            if tuple(grid_shape) != expected_grid:
                raise ValueError(
                    f"Patch-grid mismatch for {row['roi_uid']}: albedo={grid_shape}, normal={expected_grid}"
                )
            selected_patches = parse_patch_positions(row)
            features[row_index] = aggregate_maxminmean_per_layer(
                layer_features, selected_patches, tuple(grid_shape)
            )
            metadata_rows[row_index] = {
                "roi_uid": str(row["roi_uid"]),
                "bildname": str(image_name),
                "albedo_path": str(image_path),
                "selected_patch_count": len(selected_patches),
                "grid_rows": int(grid_shape[0]),
                "grid_cols": int(grid_shape[1]),
                "dinov3_layers": ";".join(str(layer) for layer in layers),
                "aggregation": "max_min_mean",
                "feature_dimension": feature_dimension,
            }
        elapsed = time.perf_counter() - start
        eta = elapsed / image_index * (len(grouped) - image_index)
        write_json(
            {
                "status": "running",
                "phase": "extract_albedo_dinov3",
                "images_completed": image_index,
                "images_total": len(grouped),
                "percent": round(100.0 * image_index / len(grouped), 2),
                "eta_seconds": round(eta, 1),
            },
            output_dir / "progress.json",
        )
        print(f"[Albedo DINO] {image_index}/{len(grouped)} | ETA {eta:.1f}s", flush=True)

    np.save(cache_file, features)
    pd.DataFrame(metadata_rows).to_csv(cache_metadata, index=False)
    return features


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


def load_normal_selections(features_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    mrmr_table = pd.read_csv(
        features_dir / "nested_eval_mrmr_fixedk384_rbf" / "selected_topk_by_fold.csv"
    )
    boruta_table = pd.read_csv(
        features_dir
        / "nested_eval_boruta_prefilter1000_relaxed_rbf"
        / "boruta_selected_by_fold.csv"
    )
    boruta_table = boruta_table.loc[boruta_table["status"] == "confirmed"]
    return {
        "mrmr": {
            fold: mrmr_table.loc[mrmr_table["fold"] == fold, "feature_index"].to_numpy(
                dtype=np.int32
            )
            for fold in range(1, 6)
        },
        "boruta": {
            fold: boruta_table.loc[
                boruta_table["outer_fold"] == fold, "feature_index"
            ].to_numpy(dtype=np.int32)
            for fold in range(1, 6)
        },
    }


def evaluate_variants(
    normal_features: np.ndarray,
    albedo_features: np.ndarray,
    table: pd.DataFrame,
    features_dir: Path,
    output_dir: Path,
    albedo_prefilter_top: int,
    random_state: int,
    layers: tuple[int, ...],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    y_labels = table["label_clean"].to_numpy()
    y = (y_labels == "3D").astype(np.int32)
    groups = table["group_id"].astype(str).to_numpy()
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state).split(
            normal_features, y, groups
        )
    )
    normal_selections = load_normal_selections(features_dir)
    stored_baselines = {
        model_key: pd.read_csv(features_dir / directory / "oof_predictions.csv")[
            "predicted_label"
        ]
        .astype(str)
        .str.upper()
        .to_numpy()
        for model_key, directory in {
            "mrmr": "nested_eval_mrmr_fixedk384_rbf",
            "boruta": "nested_eval_boruta_prefilter1000_relaxed_rbf",
        }.items()
    }
    predictions = {
        model_key: {
            "baseline": np.empty(len(y), dtype=object),
            **{f"albedo_k{k}": np.empty(len(y), dtype=object) for k in k_values},
        }
        for model_key in normal_selections
    }
    selected_albedo_rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        albedo_ranking, relevance = rank_mrmr_features(
            X_train_raw=albedo_features[train_idx],
            y_train=y[train_idx],
            fixed_k=max(k_values),
            prefilter_top=int(albedo_prefilter_top),
            random_state=random_state + fold,
        )
        for rank, feature_index in enumerate(albedo_ranking, start=1):
            layer_slot = int(feature_index) // 2304
            within_layer = int(feature_index) % 2304
            selected_albedo_rows.append(
                {
                    "fold": fold,
                    "rank": rank,
                    "albedo_feature_index": int(feature_index),
                    "mutual_information": float(relevance[feature_index]),
                    "dinov3_layer": int(layers[layer_slot]),
                    "aggregation": ("max", "min", "mean")[within_layer // 768],
                    "channel": within_layer % 768,
                }
            )
        for model_key, selections in normal_selections.items():
            normal_indices = selections[fold]
            baseline = build_classifier(1.0, "scale", "balanced", random_state + fold)
            baseline.fit(normal_features[train_idx][:, normal_indices], y[train_idx])
            predictions[model_key]["baseline"][val_idx] = np.where(
                baseline.predict(normal_features[val_idx][:, normal_indices]) == 1, "3D", "2D"
            )
            for k in k_values:
                albedo_indices = albedo_ranking[:k]
                train_combined = np.concatenate(
                    [
                        normal_features[train_idx][:, normal_indices],
                        albedo_features[train_idx][:, albedo_indices],
                    ],
                    axis=1,
                )
                val_combined = np.concatenate(
                    [
                        normal_features[val_idx][:, normal_indices],
                        albedo_features[val_idx][:, albedo_indices],
                    ],
                    axis=1,
                )
                model = build_classifier(1.0, "scale", "balanced", random_state + fold)
                model.fit(train_combined, y[train_idx])
                predictions[model_key][f"albedo_k{k}"][val_idx] = np.where(
                    model.predict(val_combined) == 1, "3D", "2D"
                )
        elapsed = time.perf_counter() - start
        eta = elapsed / fold * (len(splits) - fold)
        write_json(
            {
                "status": "running",
                "phase": "cross_validation",
                "fold_completed": fold,
                "folds_total": len(splits),
                "percent": round(100.0 * fold / len(splits), 2),
                "eta_seconds": round(eta, 1),
            },
            output_dir / "progress.json",
        )
        print(f"[CV] {fold}/{len(splits)} | ETA {eta:.1f}s", flush=True)

    pd.DataFrame(selected_albedo_rows).to_csv(
        output_dir / "selected_albedo_features_by_fold.csv", index=False
    )
    summaries: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for model_key, variants in predictions.items():
        mismatches = int(
            np.count_nonzero(variants["baseline"].astype(str) != stored_baselines[model_key])
        )
        if mismatches:
            raise RuntimeError(f"{model_key}: failed to reproduce {mismatches} baseline predictions")
        summaries[model_key] = {"baseline_reproduction_mismatches": mismatches, "variants": {}}
        oof_table = table[["roi_uid", "bildname", "roi_nummer", "label_clean"]].copy()
        for variant_name, variant_predictions in variants.items():
            metrics = compact_metrics(
                compute_metrics(y_labels, variant_predictions.astype(str), ["2D", "3D"])
            )
            summaries[model_key]["variants"][variant_name] = metrics
            oof_table[f"prediction_{variant_name}"] = variant_predictions
            comparison_rows.append(
                {"normal_feature_model": model_key, "variant": variant_name, **metrics}
            )
        oof_table.to_csv(output_dir / f"oof_predictions_{model_key}.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison.csv", index=False)
    return summaries


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    layers = tuple(int(layer) for layer in args.layers)
    k_values = tuple(sorted({int(k) for k in args.k_values if int(k) > 0}))
    if not k_values:
        raise ValueError("At least one positive --k-values entry is required.")
    layer_key = "_".join(str(layer) for layer in layers)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else features_dir / f"experiment_albedo_dinov3_l{layer_key}_maxminmean"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(features_dir / "roi_feature_table.csv").copy()
    normal_features = np.load(features_dir / "roi_features_mean.npy").astype(np.float32)
    table["label_clean"] = table["label"].astype(str).str.strip().str.upper()
    valid = table["label_clean"].isin({"2D", "3D"})
    table = table.loc[valid].reset_index(drop=True)
    normal_features = normal_features[valid.to_numpy()]
    albedo_features = extract_albedo_dino_features(
        experiment_dir=args.experiment_dir.resolve(),
        albedo_root=args.albedo_root.resolve(),
        table=table,
        output_dir=output_dir,
        layers=layers,
    )
    summaries = evaluate_variants(
        normal_features=normal_features,
        albedo_features=albedo_features,
        table=table,
        features_dir=features_dir,
        output_dir=output_dir,
        albedo_prefilter_top=int(args.albedo_prefilter_top),
        random_state=int(args.random_state),
        layers=layers,
        k_values=k_values,
    )
    summary = {
        "experiment": "DINOv3 grayscale-albedo augmentation",
        "test_set_used": False,
        "num_rois": len(table),
        "num_groups": int(table["group_id"].nunique()),
        "normal_feature_dimension": int(normal_features.shape[1]),
        "raw_albedo_feature_dimension": int(albedo_features.shape[1]),
        "albedo_layers": list(layers),
        "albedo_aggregation": ["max", "min", "mean"],
        "albedo_k_values": list(k_values),
        "albedo_selector": "train-fold-only mRMR",
        "albedo_prefilter_top": int(args.albedo_prefilter_top),
        "models": summaries,
    }
    write_json(summary, output_dir / "summary.json")
    write_json(
        {"status": "complete", "phase": "complete", "percent": 100.0, "eta_seconds": 0},
        output_dir / "progress.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
