from __future__ import annotations

import argparse
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
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from anomalydino_similarity_app import app as app_mod
from extract_labeled_roi_overthreshold_multilayer_maxminmean_features import (
    aggregate_maxminmean_per_layer,
)
from extract_labeled_roi_overthreshold_multilayer_maxstd_features import (
    select_overlap_threshold_patches,
)
from src.post_eval import mean_top1p
from src.utils import dists2map


DEFAULT_TEST_DIR = Path(r"D:\Thesis\Thesis Bericht\bericht Medien\Test")
DEFAULT_IMAGE_DIR = DEFAULT_TEST_DIR / "verwendete_testbilder_normalmap_nio"
DEFAULT_OUTPUT_DIR = DEFAULT_TEST_DIR / "pca2d_test_roi_features"
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "results_FINAL" / "normalmap_dinov3_vitb16_res688"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCA 2D scatter plots for labeled test ROIs using final mRMR and Boruta feature subsets."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--object-name", type=str, default="normalmap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"2D", "2"}:
        return "2D"
    if text in {"3D", "3"}:
        return "3D"
    return ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_label_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label table not found: {path}")
    table = pd.read_excel(path)
    required = {"image_name", "roi_id", "true_roi_label", "pred_roi_label"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    table = table.copy()
    table["image_name"] = table["image_name"].astype(str).map(lambda value: Path(value).name)
    table["roi_id"] = table["roi_id"].astype(str)
    table["true_roi_label"] = table["true_roi_label"].map(normalize_label)
    table["pred_roi_label"] = table["pred_roi_label"].map(normalize_label)
    table = table.loc[table["true_roi_label"].isin(["2D", "3D"])].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"No valid labeled ROIs found in {path}")
    return table


def prepare_live_context(experiment_dir: Path, seed: int, object_name: str) -> dict[str, Any]:
    experiment_dir_str = str(experiment_dir.resolve())
    live_config = app_mod.load_component_test_live_config(experiment_dir_str, int(seed))
    backbone = app_mod.load_component_test_backbone(experiment_dir_str)
    confirmed_config = app_mod.load_anomaly_detection_confirmed_config(experiment_dir_str, object_name)
    if confirmed_config is None:
        raise FileNotFoundError(f"No confirmed anomaly detection config found for object {object_name!r}.")

    reference_paths = app_mod._resolve_confirmed_reference_paths(
        experiment_dir_str,
        object_name,
        confirmed_config,
    )
    reference_bank = app_mod.load_anomaly_detection_reference_bank(
        experiment_dir_str,
        int(seed),
        object_name,
        tuple(str(path.resolve()) for path in reference_paths),
        app_mod._file_signature_from_paths(reference_paths),
    )
    return {
        "experiment_dir_str": experiment_dir_str,
        "live_config": live_config,
        "model": backbone["model"],
        "confirmed_config": confirmed_config,
        "reference_bank": reference_bank,
        "image_threshold": float(confirmed_config["threshold"]),
    }


def compute_score_grid(
    context: dict[str, Any],
    object_name: str,
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, tuple[int, int], tuple[int, ...]]:
    model = context["model"]
    live_config = context["live_config"]
    reference_bank = context["reference_bank"]

    image_tensor, grid_shape = model.prepare_image(image_rgb)
    anomaly_features = np.asarray(model.extract_features(image_tensor), dtype=np.float32)
    features_layers, layer_indices = model.extract_multilayer_features(image_tensor, layer_indices=list(range(1, 13)))
    features_layers = np.asarray(features_layers, dtype=np.float32)

    masking_enabled = bool(live_config["masking_by_object"].get(object_name, False))
    valid_mask = np.asarray(
        model.compute_background_mask(anomaly_features, grid_shape, threshold=10, masking_type=masking_enabled),
        dtype=bool,
    )
    query_features = anomaly_features[valid_mask]
    patch_distances = app_mod._nearest_reference_distances(
        query_features,
        reference_bank["ref_features"],
        metric=str(reference_bank["knn_metric"]),
        k_neighbors=int(reference_bank["k_neighbors"]),
    )
    output_distances = np.zeros(anomaly_features.shape[0], dtype=np.float32)
    output_distances[valid_mask] = patch_distances
    score_grid = output_distances.reshape(tuple(int(v) for v in grid_shape)).astype(np.float32)

    aggregation_statistics = str(live_config["aggregation_statistics"])
    if aggregation_statistics == "meantop1p":
        image_score = float(mean_top1p(output_distances.flatten()))
    elif aggregation_statistics == "max_patch_distance":
        image_score = float(np.max(output_distances))
    else:
        image_score = float(np.max(dists2map(score_grid, image_rgb.shape)))

    return score_grid, image_score, features_layers, tuple(int(v) for v in grid_shape), tuple(int(v) for v in layer_indices)


def extract_full_roi_features_for_image(
    context: dict[str, Any],
    object_name: str,
    image_name: str,
    image_path: Path,
) -> dict[tuple[str, str], np.ndarray]:
    image_rgb = read_rgb(image_path)
    model = context["model"]
    image_threshold = float(context["image_threshold"])
    score_grid, image_score, features_layers, grid_shape, _layer_indices = compute_score_grid(
        context,
        object_name,
        image_rgb,
    )

    resized_image = model.resize_transform(Image.fromarray(image_rgb))
    resized_size = tuple(int(v) for v in resized_image.size)
    roi_rows = app_mod._extract_live_component_rois(
        sample_name=image_name,
        object_name=object_name,
        image_rgb=image_rgb,
        score_grid=score_grid,
        image_score=image_score,
        image_threshold=image_threshold,
        resized_size=resized_size,
        patch_multiple=int(model.patch_size),
        roi_logic_key=app_mod.DEFAULT_COMPONENT_TEST_ROI_LOGIC_KEY,
    )
    if roi_rows.empty:
        return {}

    cache_meta = app_mod._build_live_cache_meta(model, image_rgb, grid_shape)
    feature_lookup: dict[tuple[str, str], np.ndarray] = {}
    for roi_row in roi_rows.itertuples(index=False):
        row_series = pd.Series(roi_row._asdict())
        selected_patches, _selection_mode, _num_candidates = select_overlap_threshold_patches(
            row=row_series,
            meta=cache_meta,
            anomaly_grid=score_grid,
            image_threshold=image_threshold,
        )
        combined_feature = aggregate_maxminmean_per_layer(
            features_layers=features_layers,
            selected_patches=selected_patches,
            grid_shape=grid_shape,
        )
        feature_lookup[(image_name, str(row_series["roi_nummer"]))] = combined_feature.astype(np.float32)
    return feature_lookup


def collect_test_features(
    context: dict[str, Any],
    object_name: str,
    image_dir: Path,
    label_tables: list[pd.DataFrame],
) -> dict[tuple[str, str], np.ndarray]:
    wanted_images = sorted(
        {
            str(image_name)
            for table in label_tables
            for image_name in table["image_name"].astype(str).unique().tolist()
        }
    )
    feature_lookup: dict[tuple[str, str], np.ndarray] = {}
    for index, image_name in enumerate(wanted_images, start=1):
        image_path = image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")
        print(f"[{index}/{len(wanted_images)}] Extract ROI features: {image_name}", flush=True)
        feature_lookup.update(
            extract_full_roi_features_for_image(
                context=context,
                object_name=object_name,
                image_name=image_name,
                image_path=image_path,
            )
        )
    return feature_lookup


def plot_model_pca(
    model_name: str,
    label_table: pd.DataFrame,
    feature_lookup: dict[tuple[str, str], np.ndarray],
    classifier_info: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    selected_indices = np.asarray(classifier_info["selected_indices"], dtype=np.int32)
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    missing: list[dict[str, str]] = []
    for row in label_table.itertuples(index=False):
        image_name = str(row.image_name)
        roi_id = str(row.roi_id)
        key = (image_name, roi_id)
        if key not in feature_lookup:
            missing.append({"image_name": image_name, "roi_id": roi_id})
            continue
        full_feature = feature_lookup[key]
        features.append(full_feature[selected_indices].astype(np.float32))
        rows.append(
            {
                "image_name": image_name,
                "roi_id": roi_id,
                "true_roi_label": str(row.true_roi_label),
                "pred_roi_label": str(row.pred_roi_label),
                "correct": bool(str(row.true_roi_label) == str(row.pred_roi_label)),
            }
        )

    if missing:
        missing_preview = ", ".join(f"{item['image_name']}:{item['roi_id']}" for item in missing[:10])
        raise RuntimeError(f"{model_name}: {len(missing)} labeled ROIs were not found after ROI extraction: {missing_preview}")
    if not features:
        raise ValueError(f"{model_name}: no features available for PCA.")

    X = np.stack(features, axis=0).astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_scaled).astype(np.float32)

    result_table = pd.DataFrame(rows)
    result_table["pca1"] = coords[:, 0]
    result_table["pca2"] = coords[:, 1]

    ensure_dir(output_dir)
    output_png = output_dir / f"{model_name.lower()}_test_roi_pca2d.png"
    output_csv = output_dir / f"{model_name.lower()}_test_roi_pca2d_scores.csv"

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=180)
    colors = {"2D": "#2E7D32", "3D": "#C62828"}
    for label in ["2D", "3D"]:
        mask = result_table["true_roi_label"].to_numpy() == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=38 if label == "3D" else 32,
            alpha=0.8,
            c=colors[label],
            label=f"{label} Ground Truth",
            edgecolors="none",
        )

    wrong_mask = ~result_table["correct"].to_numpy(dtype=bool)
    if np.any(wrong_mask):
        ax.scatter(
            coords[wrong_mask, 0],
            coords[wrong_mask, 1],
            s=95,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.4,
            label="falsch klassifiziert",
        )

    ax.set_title(f"PCA 2D Scatterplot Test-ROIs ({model_name})")
    ax.set_xlabel("PCA-Komponente 1")
    ax.set_ylabel("PCA-Komponente 2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

    result_table.to_csv(output_csv, index=False)
    summary = {
        "model": model_name,
        "num_rois": int(len(result_table)),
        "selected_feature_count": int(selected_indices.size),
        "class_counts": result_table["true_roi_label"].value_counts().to_dict(),
        "incorrect_count": int((~result_table["correct"]).sum()),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_.tolist()],
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "outputs": {
            "scatter_png": str(output_png),
            "scores_csv": str(output_csv),
        },
    }
    (output_dir / f"{model_name.lower()}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {model_name} scatter: {output_png}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    test_dir = args.test_dir.resolve()
    image_dir = args.image_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    label_tables = {
        "mRMR": load_label_table(test_dir / "roi_classifier_test_labels_mrmr.xlsx"),
        "Boruta": load_label_table(test_dir / "roi_classifier_test_labels_boruta.xlsx"),
    }
    context = prepare_live_context(experiment_dir, args.seed, args.object_name)
    feature_lookup = collect_test_features(
        context=context,
        object_name=args.object_name,
        image_dir=image_dir,
        label_tables=list(label_tables.values()),
    )

    summaries = {}
    for model_name, label_table in label_tables.items():
        classifier_info = app_mod._load_selected_component_classifier(str(experiment_dir), model_name)
        summaries[model_name] = plot_model_pca(
            model_name=model_name,
            label_table=label_table,
            feature_lookup=feature_lookup,
            classifier_info=classifier_info,
            output_dir=output_dir,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment_dir": str(experiment_dir),
                "image_dir": str(image_dir),
                "test_dir": str(test_dir),
                "object_name": str(args.object_name),
                "roi_logic": str(app_mod.DEFAULT_COMPONENT_TEST_ROI_LOGIC_KEY),
                "model_summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved combined summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
