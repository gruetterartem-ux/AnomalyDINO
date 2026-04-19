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
DEFAULT_LEFT_FEATURES_DIR = DEFAULT_EXPERIMENT_DIR / "cls_roi_features_labeled"
DEFAULT_RIGHT_FEATURES_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "hq_sam_outputs_batch"
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1_seed=0_sam_hq_vit_tiny_maskpooled_features"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine two ROI feature sets by roi_uid, e.g. raw ROI CLS-token embeddings plus "
            "mask-pooled SAM features, and write a single concatenated feature table."
        )
    )
    parser.add_argument(
        "--left-features-dir",
        type=Path,
        default=DEFAULT_LEFT_FEATURES_DIR,
        help="Feature directory for the first vector block, typically raw ROI CLS features.",
    )
    parser.add_argument(
        "--right-features-dir",
        type=Path,
        default=DEFAULT_RIGHT_FEATURES_DIR,
        help="Feature directory for the second vector block, typically mask-pooled features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <experiment-dir>/combined_cls_maskpooled_features.",
    )
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


def default_output_dir(left_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    experiment_dir = left_dir.resolve().parent
    return (experiment_dir / "combined_cls_maskpooled_features").resolve()


def load_feature_dir(features_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    features_file = features_dir / "roi_features_mean.npy"
    metadata_file = features_dir / "roi_feature_table.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    features = np.load(features_file)
    metadata = pd.read_csv(metadata_file)
    if len(features) != len(metadata):
        raise ValueError(f"Length mismatch in {features_dir}: {len(features)} features vs {len(metadata)} metadata rows")
    if "roi_uid" not in metadata.columns:
        raise ValueError(f"roi_feature_table.csv in {features_dir} must contain a roi_uid column.")
    if metadata["roi_uid"].duplicated().any():
        raise ValueError(f"Duplicate roi_uid values found in {metadata_file}")
    return features.astype(np.float32), metadata


def build_combined_table(
    left_features: np.ndarray,
    left_metadata: pd.DataFrame,
    right_features: np.ndarray,
    right_metadata: pd.DataFrame,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, int]]:
    left_lookup = left_metadata.reset_index().rename(columns={"index": "left_index"})
    right_lookup = right_metadata.reset_index().rename(columns={"index": "right_index"})

    merged = left_lookup.merge(
        right_lookup[["roi_uid", "right_index", "feature_dim", "selected_patch_count", "selected_patch_fraction"]],
        on="roi_uid",
        how="inner",
        suffixes=("", "_right"),
    )
    if merged.empty:
        raise ValueError("No shared roi_uid values between the two feature directories.")

    merged = merged.sort_values("left_index").reset_index(drop=True)
    left_block = left_features[merged["left_index"].to_numpy()]
    right_block = right_features[merged["right_index"].to_numpy()]
    combined = np.concatenate([left_block, right_block], axis=1).astype(np.float32)

    rows: List[Dict[str, object]] = []
    for output_index, row in merged.iterrows():
        metadata_row = left_metadata.iloc[int(row["left_index"])].to_dict()
        metadata_row["feature_index"] = output_index
        metadata_row["feature_type"] = "dinov3_cls_token_plus_mask_pooled"
        metadata_row["left_feature_dim"] = int(left_block.shape[1])
        metadata_row["right_feature_dim"] = int(right_block.shape[1])
        metadata_row["embedding_dim"] = int(combined.shape[1])
        metadata_row["combined_feature_dim"] = int(combined.shape[1])
        metadata_row["mask_selected_patch_count"] = int(row.get("selected_patch_count", 0)) if pd.notna(row.get("selected_patch_count", np.nan)) else 0
        metadata_row["mask_selected_patch_fraction"] = (
            float(row.get("selected_patch_fraction", 0.0)) if pd.notna(row.get("selected_patch_fraction", np.nan)) else 0.0
        )
        rows.append(metadata_row)

    stats = {
        "num_combined_rois": int(len(merged)),
        "num_left_only_rois": int(len(left_metadata) - len(merged)),
        "num_right_only_rois": int(len(right_metadata) - len(merged)),
    }
    return combined, rows, stats


def main():
    args = parse_args()
    left_dir = args.left_features_dir.resolve()
    right_dir = args.right_features_dir.resolve()
    output_dir = default_output_dir(left_dir, args.output_dir)
    ensure_dir(output_dir)

    left_features, left_metadata = load_feature_dir(left_dir)
    right_features, right_metadata = load_feature_dir(right_dir)
    combined_features, combined_rows, stats = build_combined_table(
        left_features=left_features,
        left_metadata=left_metadata,
        right_features=right_features,
        right_metadata=right_metadata,
    )

    features_file = output_dir / "roi_features_mean.npy"
    metadata_file = output_dir / "roi_feature_table.csv"
    summary_file = output_dir / "summary.json"

    np.save(features_file, combined_features)
    write_csv(combined_rows, metadata_file)
    write_json(
        {
            "left_features_dir": str(left_dir),
            "right_features_dir": str(right_dir),
            "output_dir": str(output_dir),
            "features_file": str(features_file),
            "metadata_file": str(metadata_file),
            "feature_shape": list(combined_features.shape),
            "left_feature_dim": int(left_features.shape[1]),
            "right_feature_dim": int(right_features.shape[1]),
            "combined_feature_dim": int(combined_features.shape[1]),
            **stats,
        },
        summary_file,
    )

    print(f"Saved combined features: {features_file}")
    print(f"Saved combined metadata: {metadata_file}")
    print(f"Saved summary: {summary_file}")
    print(f"Combined feature shape: {combined_features.shape}")


if __name__ == "__main__":
    main()
