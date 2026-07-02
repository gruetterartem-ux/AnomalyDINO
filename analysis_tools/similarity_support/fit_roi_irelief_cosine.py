from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURES_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
    r"\roi_top10pct_centerinbox_pca2_softmax_patch_features_labeled"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a cosine-based iterative Relief weighting on ROI feature vectors and export a "
            "weighted ROI feature set that can be reused by downstream classifiers."
        )
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory containing roi_features_mean.npy and roi_feature_table.csv.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional custom labels table. Defaults to <features-dir>/roi_feature_table.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <features-dir>/irelief_cosine_weighted_features.",
    )
    parser.add_argument(
        "--valid-labels",
        type=str,
        nargs="*",
        default=("2D", "3D"),
        help="Explicit labels to keep. Defaults to 2D and 3D.",
    )
    parser.add_argument(
        "--ignore-labels",
        type=str,
        nargs="*",
        default=("skip", "unclear", "unknown"),
        help="Labels that should be ignored.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum number of I-Relief iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance on the L1 weight change.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Kernel bandwidth for cosine distance. If omitted, it is estimated from the data.",
    )
    parser.add_argument(
        "--sigma-quantile",
        type=float,
        default=0.5,
        help="Quantile used to estimate sigma when --sigma is omitted.",
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=1e-3,
        help="Lower bound for the automatically estimated sigma.",
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
    return str(value).strip()


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / "irelief_cosine_weighted_features").resolve()


def load_inputs(features_dir: Path, labels_file: Path | None) -> Tuple[np.ndarray, pd.DataFrame]:
    features_file = features_dir / "roi_features_mean.npy"
    metadata_file = features_dir / "roi_feature_table.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Feature metadata file not found: {metadata_file}")

    features = np.load(features_file).astype(np.float32)
    metadata = pd.read_csv(metadata_file)
    if len(features) != len(metadata):
        raise ValueError(f"Length mismatch: {len(features)} features vs {len(metadata)} metadata rows")

    if labels_file is None:
        return features, metadata

    labels_file = labels_file.resolve()
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels table not found: {labels_file}")
    if labels_file == metadata_file.resolve():
        return features, metadata

    labels = load_table_file(labels_file).copy()
    metadata = metadata.copy()
    metadata["bildname"] = metadata["image_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
    metadata["roi_nummer"] = "roi" + metadata["roi_index"].astype(int).astype(str)

    if "roi_uid" in labels.columns:
        keep_columns = [column for column in ("roi_uid", "label", "notes", "detailed_label") if column in labels.columns]
        labels = labels[keep_columns].copy()
        if labels["roi_uid"].duplicated().any():
            raise ValueError("Custom labels file contains duplicated roi_uid values.")
        table = metadata.merge(labels, on="roi_uid", how="left", suffixes=("", "_custom"))
    elif {"bildname", "roi_nummer"}.issubset(labels.columns):
        keep_columns = [column for column in ("bildname", "roi_nummer", "label", "notes", "detailed_label") if column in labels.columns]
        labels = labels[keep_columns].copy()
        labels["bildname"] = labels["bildname"].map(normalize_bildname)
        labels["roi_nummer"] = labels["roi_nummer"].map(normalize_roi_nummer)
        if labels.duplicated(["bildname", "roi_nummer"]).any():
            raise ValueError("Custom labels file contains duplicated bildname/roi_nummer values.")
        table = metadata.merge(labels, on=["bildname", "roi_nummer"], how="left", suffixes=("", "_custom"))
    else:
        raise ValueError("Custom labels file must contain either roi_uid or bildname + roi_nummer.")

    for column in ("label", "notes", "detailed_label"):
        custom_name = f"{column}_custom"
        if custom_name in table.columns:
            table[column] = table[custom_name]
            table = table.drop(columns=[custom_name])
    return features, table


def filter_labeled_subset(
    features: np.ndarray,
    table: pd.DataFrame,
    valid_labels: List[str],
    ignore_labels: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    table = table.copy()
    if "label" not in table.columns:
        raise ValueError("Input metadata must contain a label column.")

    table["label_clean"] = table["label"].map(clean_label)
    valid_lookup = {label.lower(): label for label in valid_labels}
    ignore_set = {label.lower() for label in ignore_labels}

    mask_valid = table["label_clean"] != ""
    if ignore_set:
        mask_valid &= ~table["label_clean"].str.lower().isin(ignore_set)
    if valid_lookup:
        mask_valid &= table["label_clean"].str.lower().isin(valid_lookup.keys())
    subset = table.loc[mask_valid].copy().reset_index(drop=True)
    subset["label"] = subset["label_clean"].str.lower().map(valid_lookup)
    subset = subset.drop(columns=["label_clean"])
    subset_features = features[mask_valid.to_numpy()].astype(np.float32)
    if subset.empty:
        raise ValueError("No labeled rows remained after filtering.")
    return subset_features, subset


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (matrix / norms).astype(np.float32)


def pairwise_cosine_distance(matrix_unit: np.ndarray) -> np.ndarray:
    similarity = np.clip(matrix_unit @ matrix_unit.T, -1.0, 1.0)
    return (1.0 - similarity).astype(np.float32)


def estimate_sigma(distance_matrix: np.ndarray, quantile: float, min_sigma: float) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"sigma_quantile must be in (0, 1], got {quantile}")
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    values = distance_matrix[mask]
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size == 0:
        return float(min_sigma)
    sigma = float(np.quantile(values, quantile))
    return float(max(sigma, min_sigma))


def weighted_cosine_distance_matrix(features_unit: np.ndarray, scale_vector: np.ndarray) -> np.ndarray:
    weighted = features_unit * scale_vector[None, :]
    weighted = l2_normalize_rows(weighted)
    return pairwise_cosine_distance(weighted)


def build_neighbor_probabilities(
    distance_matrix: np.ndarray,
    same_mask: np.ndarray,
    diff_mask: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    kernel = np.exp(-distance_matrix / max(float(sigma), 1e-8)).astype(np.float32)
    np.fill_diagonal(kernel, 0.0)

    hit_kernel = kernel * same_mask.astype(np.float32)
    miss_kernel = kernel * diff_mask.astype(np.float32)

    hit_sum = hit_kernel.sum(axis=1, keepdims=True)
    miss_sum = miss_kernel.sum(axis=1, keepdims=True)
    hit_sum = np.maximum(hit_sum, 1e-8)
    miss_sum = np.maximum(miss_sum, 1e-8)
    hit_prob = (hit_kernel / hit_sum).astype(np.float32)
    miss_prob = (miss_kernel / miss_sum).astype(np.float32)
    return hit_prob, miss_prob


def fit_irelief_cosine(
    features_unit: np.ndarray,
    labels: np.ndarray,
    sigma: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    num_samples, feature_dim = features_unit.shape
    if num_samples < 3:
        raise ValueError("I-Relief requires at least 3 labeled samples.")

    same_mask = labels[:, None] == labels[None, :]
    diff_mask = ~same_mask
    np.fill_diagonal(same_mask, False)
    np.fill_diagonal(diff_mask, False)

    pairwise_diff = np.abs(features_unit[:, None, :] - features_unit[None, :, :]).astype(np.float32)
    weights = np.full((feature_dim,), 1.0 / feature_dim, dtype=np.float32)
    trace: List[Dict[str, float]] = []

    for iteration in range(1, max_iter + 1):
        scale_vector = np.sqrt(np.maximum(weights, 0.0)).astype(np.float32)
        distance_matrix = weighted_cosine_distance_matrix(features_unit, scale_vector)
        hit_prob, miss_prob = build_neighbor_probabilities(distance_matrix, same_mask, diff_mask, sigma)

        hit_expected = np.einsum("ij,ijd->id", hit_prob, pairwise_diff, optimize=True)
        miss_expected = np.einsum("ij,ijd->id", miss_prob, pairwise_diff, optimize=True)
        feature_scores = (miss_expected - hit_expected).mean(axis=0).astype(np.float32)
        feature_scores = np.maximum(feature_scores, 0.0)

        if float(feature_scores.sum()) <= 1e-12:
            feature_scores = np.full_like(weights, 1.0 / feature_dim)
        else:
            feature_scores = (feature_scores / feature_scores.sum()).astype(np.float32)

        delta_l1 = float(np.abs(feature_scores - weights).sum())
        trace.append(
            {
                "iteration": float(iteration),
                "delta_l1": delta_l1,
                "weight_max": float(feature_scores.max()),
                "weight_min": float(feature_scores.min()),
                "weight_entropy": float(-np.sum(feature_scores * np.log(np.maximum(feature_scores, 1e-12)))),
            }
        )
        weights = feature_scores
        if delta_l1 <= tol:
            break

    return weights.astype(np.float32), trace


def build_weighted_feature_set(features_unit: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scale_vector = np.sqrt(np.maximum(weights, 0.0)).astype(np.float32)
    weighted = features_unit * scale_vector[None, :]
    return l2_normalize_rows(weighted)


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    labels_file = args.labels_file.resolve() if args.labels_file is not None else None
    output_dir = default_output_dir(features_dir, args.output_dir)
    ensure_dir(output_dir)

    features, table = load_inputs(features_dir, labels_file)
    subset_features, subset_table = filter_labeled_subset(
        features,
        table,
        list(args.valid_labels),
        list(args.ignore_labels),
    )
    features_unit = l2_normalize_rows(subset_features)

    labels = subset_table["label"].astype(str).to_numpy()
    class_counts = {str(label): int(count) for label, count in subset_table["label"].value_counts().sort_index().items()}

    base_distance = pairwise_cosine_distance(features_unit)
    sigma = float(args.sigma) if args.sigma is not None else estimate_sigma(
        base_distance,
        quantile=float(args.sigma_quantile),
        min_sigma=float(args.min_sigma),
    )

    weights, trace = fit_irelief_cosine(
        features_unit=features_unit,
        labels=labels,
        sigma=sigma,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
    )
    weighted_features = build_weighted_feature_set(features_unit, weights)
    scale_vector = np.sqrt(np.maximum(weights, 0.0)).astype(np.float32)

    np.save(output_dir / "roi_features_mean.npy", weighted_features.astype(np.float32))
    subset_table.to_csv(output_dir / "roi_feature_table.csv", index=False)
    np.save(output_dir / "irelief_feature_weights.npy", weights.astype(np.float32))
    np.save(output_dir / "irelief_feature_scale_sqrt.npy", scale_vector.astype(np.float32))
    np.savez_compressed(
        output_dir / "irelief_model.npz",
        feature_weights=weights.astype(np.float32),
        feature_scale_sqrt=scale_vector.astype(np.float32),
        sigma=np.array([sigma], dtype=np.float32),
    )

    weight_rows: List[Dict[str, object]] = []
    order = np.argsort(-weights)
    for rank, feature_index in enumerate(order, start=1):
        weight_rows.append(
            {
                "rank": int(rank),
                "feature_index": int(feature_index),
                "weight": float(weights[feature_index]),
                "scale_sqrt_weight": float(scale_vector[feature_index]),
            }
        )
    write_csv(weight_rows, output_dir / "irelief_feature_weights.csv")
    write_csv(trace, output_dir / "irelief_iteration_trace.csv")

    layer_mass_rows: List[Dict[str, object]] = []
    num_layers = None
    layer_dim = None
    if "num_layers" in subset_table.columns and "layer_dim" in subset_table.columns:
        try:
            num_layers_values = subset_table["num_layers"].dropna().astype(int).unique().tolist()
            layer_dim_values = subset_table["layer_dim"].dropna().astype(int).unique().tolist()
            if len(num_layers_values) == 1 and len(layer_dim_values) == 1:
                num_layers = int(num_layers_values[0])
                layer_dim = int(layer_dim_values[0])
                if num_layers > 0 and layer_dim > 0 and num_layers * layer_dim == weights.shape[0]:
                    layer_indices_values = None
                    if "layer_indices" in subset_table.columns:
                        raw_value = str(subset_table["layer_indices"].dropna().iloc[0])
                        parsed = [int(item) for item in raw_value.split(";") if str(item).strip() != ""]
                        if len(parsed) == num_layers:
                            layer_indices_values = parsed
                    for layer_idx in range(num_layers):
                        start = layer_idx * layer_dim
                        end = start + layer_dim
                        layer_label = int(layer_indices_values[layer_idx]) if layer_indices_values is not None else int(layer_idx + 1)
                        layer_mass_rows.append(
                            {
                                "layer_order": int(layer_idx + 1),
                                "layer_index": int(layer_label),
                                "weight_mass": float(weights[start:end].sum()),
                                "mean_weight": float(weights[start:end].mean()),
                                "max_weight": float(weights[start:end].max()),
                                "num_nonzero_weights": int(np.count_nonzero(weights[start:end] > 0)),
                            }
                        )
        except Exception:
            layer_mass_rows = []

    if layer_mass_rows:
        write_csv(layer_mass_rows, output_dir / "irelief_layer_weight_mass.csv")

    summary = {
        "features_dir": str(features_dir),
        "labels_file": str(labels_file) if labels_file is not None else str(features_dir / "roi_feature_table.csv"),
        "num_samples": int(len(subset_table)),
        "feature_dim": int(features_unit.shape[1]),
        "class_counts": class_counts,
        "valid_labels": list(args.valid_labels),
        "ignore_labels": list(args.ignore_labels),
        "distance_metric": "cosine_distance",
        "feature_input_norm": "l2_row_normalized",
        "weight_application": "sqrt(weight) scaling followed by l2 row normalization",
        "sigma": float(sigma),
        "sigma_quantile": float(args.sigma_quantile) if args.sigma is None else None,
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "num_iterations_run": int(len(trace)),
        "converged": bool(trace[-1]["delta_l1"] <= float(args.tol)) if trace else False,
        "weight_sum": float(weights.sum()),
        "weight_max": float(weights.max()),
        "weight_min": float(weights.min()),
        "num_nonzero_weights": int(np.count_nonzero(weights > 0)),
        "top10_weight_mass": float(np.sort(weights)[::-1][:10].sum()),
        "weighted_feature_mean_l2_norm": float(np.linalg.norm(weighted_features, axis=1).mean()),
        "layer_mass_available": bool(layer_mass_rows),
    }
    if layer_mass_rows:
        summary["num_layers"] = int(num_layers)
        summary["layer_dim"] = int(layer_dim)
        summary["top_layer_by_mass"] = int(max(layer_mass_rows, key=lambda item: float(item["weight_mass"]))["layer_index"])
    write_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
