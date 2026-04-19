from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from component_memory_bank.export import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a 2D PCA view of the 2D/3D memory-bank features."
    )
    parser.add_argument("--memory-bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def _mean_distance_to_centroid(points: np.ndarray, centroid: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    return float(np.linalg.norm(points - centroid, axis=1).mean())


def main() -> int:
    args = parse_args()
    memory_bank_dir = args.memory_bank_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (memory_bank_dir / "pca_view")
    output_dir.mkdir(parents=True, exist_ok=True)

    bank_2d_path = memory_bank_dir / "2D-memory-bank.npy"
    bank_3d_path = memory_bank_dir / "3D-memory-bank.npy"
    metadata_path = memory_bank_dir / "selected_patches.csv"

    features_2d = np.load(bank_2d_path).astype(np.float32)
    features_3d = np.load(bank_3d_path).astype(np.float32)

    features = np.vstack([features_2d, features_3d]).astype(np.float32, copy=False)
    labels = np.array(["2D"] * len(features_2d) + ["3D"] * len(features_3d))

    if args.no_normalize:
        features_for_pca = features
        normalization_mode = "raw"
    else:
        features_for_pca = _normalize_rows(features)
        normalization_mode = "l2_normalized"

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(features_for_pca).astype(np.float32)

    coords_2d = coords[: len(features_2d)]
    coords_3d = coords[len(features_2d) :]
    centroid_2d = coords_2d.mean(axis=0)
    centroid_3d = coords_3d.mean(axis=0)

    metadata = None
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        label_column = metadata["component_label"].astype(str)
        metadata_2d = metadata[label_column == "2D"].copy()
        metadata_3d = metadata[label_column == "3D"].copy()
        metadata = pd.concat([metadata_2d, metadata_3d], axis=0, ignore_index=True)
        metadata = metadata.iloc[: len(labels)].copy()
    else:
        metadata = pd.DataFrame({"component_label": labels})

    coord_df = metadata.copy()
    coord_df["memory_bank_label"] = labels
    coord_df["pca_x"] = coords[:, 0]
    coord_df["pca_y"] = coords[:, 1]
    coord_df.to_csv(output_dir / "memory_bank_pca_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        s=28,
        c="#2E8B57",
        alpha=0.75,
        edgecolors="none",
        label=f"2D ({len(coords_2d)})",
    )
    ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        s=32,
        c="#C0392B",
        alpha=0.82,
        edgecolors="none",
        label=f"3D ({len(coords_3d)})",
    )
    ax.scatter(
        [centroid_2d[0], centroid_3d[0]],
        [centroid_2d[1], centroid_3d[1]],
        s=240,
        c=["#145A32", "#7B241C"],
        marker="X",
        edgecolors="white",
        linewidths=1.0,
        zorder=5,
        label="Klassenzentren",
    )
    ax.set_title(f"Memory-Bank PCA (2D) [{normalization_mode}]")
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% Varianz)"
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% Varianz)"
    )
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_bank_pca.png", dpi=180)
    plt.close(fig)

    summary = {
        "memory_bank_dir": str(memory_bank_dir),
        "output_dir": str(output_dir),
        "normalization_mode": normalization_mode,
        "num_2d_features": int(len(features_2d)),
        "num_3d_features": int(len(features_3d)),
        "feature_dim": int(features.shape[1]),
        "explained_variance_ratio": [
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        ],
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_[:2].sum()),
        "pca_centroid_distance": float(np.linalg.norm(centroid_2d - centroid_3d)),
        "pca_mean_distance_to_centroid_2d": _mean_distance_to_centroid(coords_2d, centroid_2d),
        "pca_mean_distance_to_centroid_3d": _mean_distance_to_centroid(coords_3d, centroid_3d),
        "silhouette_feature_space": float(silhouette_score(features_for_pca, labels)),
        "silhouette_pca_2d": float(silhouette_score(coords, labels)),
    }
    write_json(summary, output_dir / "summary.json")

    print(f"PCA plot written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
