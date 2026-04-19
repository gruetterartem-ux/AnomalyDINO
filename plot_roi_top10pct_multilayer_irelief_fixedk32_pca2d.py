from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from extract_labeled_roi_toppercent_multilayer_softmax_patch_features import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LABELS_FILE,
    DEFAULT_ROI_METADATA_CSV,
    build_multilayer_run_context,
    load_labels_table,
    load_roi_table,
    prepare_labeled_roi_table,
)
from fit_roi_irelief_cosine import (
    build_weighted_feature_set,
    estimate_sigma,
    fit_irelief_cosine,
    l2_normalize_rows,
    pairwise_cosine_distance,
)
from predict_and_render_roi_top10pct_multilayer_irelief_fixedk32_svm_all import (
    build_base_and_expand1_features,
)


DEFAULT_MODEL_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "final_all_boxes_top10pct_multilayer_irelief_fixedk32_rbf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a PCA 2D scatter plot for the current top10pct multilayer "
            "I-Relief fixed-k=32 ROI feature path."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--multilayer-cache-subdir", type=str, default="patch_feature_cache_multilayer_l1to12")
    parser.add_argument("--top-percent", type=float, default=0.10)
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--selection-mode", type=str, default="center_in_box", choices=("center_in_box", "overlap"))
    parser.add_argument("--sigma-quantile", type=float, default=0.5)
    parser.add_argument("--min-sigma", type=float, default=1e-3)
    parser.add_argument("--irelief-max-iter", type=int, default=50)
    parser.add_argument("--irelief-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def default_output_dir(model_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (model_dir / "pca2d_plot").resolve()


def read_selected_indices(model_dir: Path) -> np.ndarray:
    selected_csv = model_dir / "selected_topk_features.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(f"selected_topk_features.csv not found: {selected_csv}")
    table = pd.read_csv(selected_csv)
    if "feature_index" not in table.columns:
        raise ValueError(f"feature_index column missing in {selected_csv}")
    return table["feature_index"].to_numpy(dtype=np.int32)


def plot_scatter(
    coords_2d: np.ndarray,
    labels_lower: np.ndarray,
    output_png: Path,
) -> None:
    ensure_dir(output_png.parent)
    plt.figure(figsize=(8, 7), dpi=160)
    mask_2d = labels_lower == "2d"
    mask_3d = labels_lower == "3d"
    plt.scatter(
        coords_2d[mask_2d, 0],
        coords_2d[mask_2d, 1],
        s=28,
        alpha=0.75,
        c="#2E7D32",
        label="2D",
        edgecolors="none",
    )
    plt.scatter(
        coords_2d[mask_3d, 0],
        coords_2d[mask_3d, 1],
        s=34,
        alpha=0.8,
        c="#C62828",
        label="3D",
        edgecolors="none",
    )
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("PCA 2D Scatter for 2D vs 3D ROIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = default_output_dir(model_dir, args.output_dir)
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
    X_base_raw, _ = build_base_and_expand1_features(
        roi_table=labeled_rois,
        sample_map=sample_map,
        top_percent=float(args.top_percent),
        min_patches=int(args.min_patches),
        selection_mode=str(args.selection_mode),
    )

    labels_lower = labeled_rois["label"].astype(str).str.lower().to_numpy()
    features_unit = l2_normalize_rows(X_base_raw)
    sigma = estimate_sigma(
        pairwise_cosine_distance(features_unit),
        quantile=float(args.sigma_quantile),
        min_sigma=float(args.min_sigma),
    )
    weights, trace = fit_irelief_cosine(
        features_unit=features_unit,
        labels=labels_lower,
        sigma=float(sigma),
        max_iter=int(args.irelief_max_iter),
        tol=float(args.irelief_tol),
    )
    X_weighted = build_weighted_feature_set(features_unit, weights)

    selected_indices = read_selected_indices(model_dir)
    X_selected = X_weighted[:, selected_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    pca = PCA(n_components=2, random_state=0)
    coords_2d = pca.fit_transform(X_scaled).astype(np.float32)

    result_table = labeled_rois.copy()
    result_table["label_lower"] = labels_lower
    result_table["pca1"] = coords_2d[:, 0]
    result_table["pca2"] = coords_2d[:, 1]

    mask_2d = labels_lower == "2d"
    mask_3d = labels_lower == "3d"
    centroids = {
        "2d": {
            "pca1_mean": float(coords_2d[mask_2d, 0].mean()),
            "pca2_mean": float(coords_2d[mask_2d, 1].mean()),
        },
        "3d": {
            "pca1_mean": float(coords_2d[mask_3d, 0].mean()),
            "pca2_mean": float(coords_2d[mask_3d, 1].mean()),
        },
    }

    output_png = output_dir / "pca2d_scatter.png"
    output_csv = output_dir / "pca2d_scores.csv"
    output_json = output_dir / "summary.json"
    plot_scatter(coords_2d=coords_2d, labels_lower=labels_lower, output_png=output_png)
    result_table.to_csv(output_csv, index=False)

    summary = {
        "experiment_dir": str(experiment_dir),
        "model_dir": str(model_dir),
        "selection_mode": str(args.selection_mode),
        "num_labeled_rois": int(len(labeled_rois)),
        "class_counts": {
            "2d": int(mask_2d.sum()),
            "3d": int(mask_3d.sum()),
        },
        "feature_dim_raw": int(X_base_raw.shape[1]),
        "selected_feature_count": int(len(selected_indices)),
        "sigma": float(sigma),
        "irelief_iterations": int(len(trace)),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "class_centroids": centroids,
        "outputs": {
            "scatter_png": str(output_png),
            "scores_csv": str(output_csv),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved scatter: {output_png}")
    print(f"Saved scores CSV: {output_csv}")
    print(f"Saved summary: {output_json}")
    print(
        f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.4f}, "
        f"sum={np.sum(pca.explained_variance_ratio_):.4f}"
    )


if __name__ == "__main__":
    main()
