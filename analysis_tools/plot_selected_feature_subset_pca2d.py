from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model_building.roi_sklearn_groupcv import clean_label, load_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PCA 2D scatter plot for a selected ROI feature subset."
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--selected-features-csv", type=Path, required=True)
    parser.add_argument("--feature-index-column", type=str, default="feature_index")
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--ignore-labels", type=str, nargs="*", default=("skip", "unclear", "unknown"))
    parser.add_argument("--valid-labels", type=str, nargs="*", default=("2D", "3D"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def default_output_dir(selected_features_csv: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (selected_features_csv.parent / "pca2d_plot").resolve()


def load_selected_indices(csv_path: Path, column_name: str) -> np.ndarray:
    table = pd.read_csv(csv_path)
    if column_name not in table.columns:
        raise ValueError(f"Column '{column_name}' not found in {csv_path}")
    indices = table[column_name].dropna().astype(int).to_numpy()
    if indices.size == 0:
        raise ValueError(f"No feature indices found in column '{column_name}' of {csv_path}")
    return np.unique(indices)


def plot_scatter(coords_2d: np.ndarray, labels_lower: np.ndarray, output_png: Path) -> None:
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
    plt.title("PCA 2D Scatter for Selected ROI Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    selected_features_csv = args.selected_features_csv.resolve()
    labels_file = (args.labels_file or (features_dir / "roi_feature_table.csv")).resolve()
    output_dir = default_output_dir(selected_features_csv, args.output_dir)
    ensure_dir(output_dir)

    features, table = load_inputs(features_dir, labels_file)
    selected_indices = load_selected_indices(selected_features_csv, args.feature_index_column)

    table = table.copy()
    table["label"] = table["label"].map(clean_label)
    ignore_labels = {clean_label(label) for label in args.ignore_labels}
    valid_labels = {clean_label(label) for label in args.valid_labels}
    valid_mask = table["label"].ne("") & ~table["label"].isin(ignore_labels) & table["label"].isin(valid_labels)
    labeled_table = table.loc[valid_mask].copy().reset_index(drop=True)
    if labeled_table.empty:
        raise ValueError("No labeled ROIs found after filtering.")

    X_full = features[labeled_table["feature_index"].to_numpy()]
    X = X_full[:, selected_indices]
    labels_lower = labeled_table["label"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    coords_2d = pca.fit_transform(X_scaled).astype(np.float32)

    result_table = labeled_table.copy()
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
    plot_scatter(coords_2d, labels_lower, output_png)
    result_table.to_csv(output_csv, index=False)

    summary = {
        "features_dir": str(features_dir),
        "selected_features_csv": str(selected_features_csv),
        "labels_file": str(labels_file),
        "num_labeled_rois": int(len(labeled_table)),
        "selected_feature_count": int(len(selected_indices)),
        "class_counts": {"2d": int(mask_2d.sum()), "3d": int(mask_3d.sum())},
        "feature_dim_after_selection": int(X.shape[1]),
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
        f"PC2={pca.explained_variance_ratio_[1]:.4f}, sum={np.sum(pca.explained_variance_ratio_):.4f}"
    )


if __name__ == "__main__":
    main()
