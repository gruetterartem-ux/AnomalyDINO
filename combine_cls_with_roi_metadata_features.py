import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_CLS_FEATURES_DIR = DEFAULT_EXPERIMENT_DIR / "cls_roi_features_labeled"
DEFAULT_METADATA_SOURCE = (
    DEFAULT_EXPERIMENT_DIR
    / "hq_sam_outputs_batch"
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1_seed=0_sam_hq_vit_tiny_maskpooled_features"
    / "roi_feature_table.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine raw ROI CLS-token features with anomaly-strength metadata features, "
            "without using the mask-pooled feature vector."
        )
    )
    parser.add_argument("--cls-features-dir", type=Path, default=DEFAULT_CLS_FEATURES_DIR)
    parser.add_argument("--metadata-source-csv", type=Path, default=DEFAULT_METADATA_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=None)
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


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def default_output_dir(cls_features_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (cls_features_dir.resolve().parent / "combined_cls_anomalyscore_features").resolve()


def load_feature_dir(features_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    features_file = features_dir / "roi_features_mean.npy"
    metadata_file = features_dir / "roi_feature_table.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    features = np.load(features_file).astype(np.float32)
    metadata = pd.read_csv(metadata_file)
    if len(features) != len(metadata):
        raise ValueError(f"Length mismatch in {features_dir}: {len(features)} vs {len(metadata)}")
    if "roi_uid" not in metadata.columns:
        raise ValueError(f"roi_feature_table.csv in {features_dir} must contain roi_uid.")
    if metadata["roi_uid"].duplicated().any():
        raise ValueError(f"Duplicate roi_uid values in {metadata_file}")
    return features, metadata


def load_metadata_source(metadata_source_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(metadata_source_csv)
    required = {
        "roi_uid",
        "region_max_score",
        "region_mass",
        "primary_peak_score",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Metadata source is missing required columns: {sorted(missing)}")
    if table["roi_uid"].duplicated().any():
        raise ValueError(f"Duplicate roi_uid values in metadata source: {metadata_source_csv}")
    return table


def derived_tabular_features(row: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
    feature_map = {
        "region_max_score": float(row["region_max_score"]),
        "region_mass": float(row["region_mass"]),
        "primary_peak_score": float(row["primary_peak_score"]),
    }
    return np.array(list(feature_map.values()), dtype=np.float32), feature_map


def build_combined_features(
    cls_features: np.ndarray,
    cls_metadata: pd.DataFrame,
    metadata_source: pd.DataFrame,
) -> Tuple[np.ndarray, List[Dict[str, object]], List[str]]:
    merged = cls_metadata.merge(
        metadata_source[
            [
                "roi_uid",
                "region_max_score",
                "region_mass",
                "primary_peak_score",
            ]
        ],
        on="roi_uid",
        how="left",
        suffixes=("", "_meta"),
    )
    if merged[["region_max_score_meta", "region_mass_meta", "primary_peak_score_meta"]].isna().any().any():
        missing = merged[merged["region_max_score_meta"].isna()]["roi_uid"].tolist()[:5]
        raise ValueError(f"Missing metadata matches for roi_uid entries: {missing}")

    tabular_rows: List[np.ndarray] = []
    output_rows: List[Dict[str, object]] = []
    feature_names: List[str] | None = None

    for feature_index, row in merged.iterrows():
        working_row = pd.Series(
            {
                "region_max_score": row["region_max_score_meta"] if "region_max_score_meta" in row.index else row["region_max_score"],
                "region_mass": row["region_mass_meta"] if "region_mass_meta" in row.index else row["region_mass"],
                "primary_peak_score": row["primary_peak_score_meta"] if "primary_peak_score_meta" in row.index else row["primary_peak_score"],
            }
        )
        tabular_vector, feature_map = derived_tabular_features(working_row)
        if feature_names is None:
            feature_names = list(feature_map.keys())
        tabular_rows.append(tabular_vector)

        output_row = cls_metadata.iloc[feature_index].to_dict()
        output_row["feature_index"] = feature_index
        output_row["feature_type"] = "dinov3_cls_plus_geomscore"
        output_row["cls_feature_dim"] = int(cls_features.shape[1])
        output_row["tabular_feature_dim"] = int(len(tabular_vector))
        output_row["embedding_dim"] = int(cls_features.shape[1] + len(tabular_vector))
        output_row["combined_feature_dim"] = int(cls_features.shape[1] + len(tabular_vector))
        for key, value in feature_map.items():
            output_row[key] = float(value)
        output_rows.append(output_row)

    tabular_array = np.vstack(tabular_rows).astype(np.float32)
    combined = np.concatenate([cls_features, tabular_array], axis=1).astype(np.float32)
    return combined, output_rows, feature_names or []


def main():
    args = parse_args()
    cls_features_dir = args.cls_features_dir.resolve()
    metadata_source_csv = args.metadata_source_csv.resolve()
    output_dir = default_output_dir(cls_features_dir, args.output_dir)
    ensure_dir(output_dir)

    cls_features, cls_metadata = load_feature_dir(cls_features_dir)
    metadata_source = load_metadata_source(metadata_source_csv)
    combined_features, output_rows, feature_names = build_combined_features(
        cls_features=cls_features,
        cls_metadata=cls_metadata,
        metadata_source=metadata_source,
    )

    features_file = output_dir / "roi_features_mean.npy"
    metadata_file = output_dir / "roi_feature_table.csv"
    summary_file = output_dir / "summary.json"

    np.save(features_file, combined_features)
    write_csv(output_rows, metadata_file)
    write_json(
        {
            "cls_features_dir": str(cls_features_dir),
            "metadata_source_csv": str(metadata_source_csv),
            "output_dir": str(output_dir),
            "features_file": str(features_file),
            "metadata_file": str(metadata_file),
            "feature_shape": list(combined_features.shape),
            "cls_feature_dim": int(cls_features.shape[1]),
            "tabular_feature_dim": int(len(feature_names)),
            "combined_feature_dim": int(combined_features.shape[1]),
            "tabular_feature_names": feature_names,
            "num_rois": int(len(output_rows)),
        },
        summary_file,
    )

    print(f"Saved combined features: {features_file}")
    print(f"Saved combined metadata: {metadata_file}")
    print(f"Saved summary: {summary_file}")
    print(f"Combined feature shape: {combined_features.shape}")


if __name__ == "__main__":
    main()
