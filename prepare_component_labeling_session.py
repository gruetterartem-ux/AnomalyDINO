import argparse
from pathlib import Path

from component_memory_bank.components import build_components
from component_memory_bank.data_io import load_patch_features, load_patch_scores, load_run_samples
from component_memory_bank.export import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a component-level labeling session from AnomalyDINO outputs."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=None,
        help="Patch anomaly threshold used for component gating. Defaults to the run threshold from measurements.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--include-only-substring",
        type=str,
        default=None,
        help="Optional substring filter on sample paths, e.g. 'test/bad/2D/'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_run_samples(args.experiment_dir, seed=args.seed)
    if args.include_only_substring:
        needle = args.include_only_substring.replace("\\", "/")
        samples = [sample for sample in samples if needle in sample.sample]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    if not samples:
        raise ValueError("No samples selected for component labeling session.")

    default_threshold = float(samples[0].image_threshold)
    anomaly_threshold = float(args.anomaly_threshold) if args.anomaly_threshold is not None else default_threshold

    inventory_rows = []
    sample_rows = []
    total_components = 0
    for sample in samples:
        _, grid_size = load_patch_features(sample)
        score_grid = load_patch_scores(sample)
        if tuple(score_grid.shape) != tuple(grid_size):
            raise ValueError(
                f"Grid mismatch for {sample.sample!r}: feature cache {grid_size}, anomaly grid {score_grid.shape}."
            )
        _, components = build_components(score_grid, anomaly_threshold)
        total_components += len(components)
        sample_rows.append(
            {
                "object_name": sample.object_name,
                "sample": sample.sample,
                "evaluation_group": sample.evaluation_group,
                "image_label": sample.image_label,
                "image_score": sample.image_score,
                "image_threshold": sample.image_threshold,
                "anomaly_threshold": anomaly_threshold,
                "image_path": str(sample.image_path),
                "anomaly_map_path": str(sample.anomaly_map_path),
                "feature_cache_path": str(sample.feature_cache_path),
                "grid_rows": score_grid.shape[0],
                "grid_cols": score_grid.shape[1],
                "num_components": len(components),
                "has_components": bool(components),
            }
        )
        for component in components:
            inventory_rows.append(
                {
                    "object_name": sample.object_name,
                    "sample": sample.sample,
                    "evaluation_group": sample.evaluation_group,
                    "image_label": sample.image_label,
                    "image_score": sample.image_score,
                    "image_threshold": sample.image_threshold,
                    "anomaly_threshold": anomaly_threshold,
                    "image_path": str(sample.image_path),
                    "anomaly_map_path": str(sample.anomaly_map_path),
                    "feature_cache_path": str(sample.feature_cache_path),
                    "grid_rows": score_grid.shape[0],
                    "grid_cols": score_grid.shape[1],
                    "component_id": component.component_id,
                    "component_size": component.size,
                    "component_bbox_row_min": component.bbox_row_min,
                    "component_bbox_row_max": component.bbox_row_max,
                    "component_bbox_col_min": component.bbox_col_min,
                    "component_bbox_col_max": component.bbox_col_max,
                    "component_max_score": component.max_score,
                    "component_mean_score": component.mean_score,
                    "top_k": min(5, component.size),
                    "label": "",
                    "notes": "",
                }
            )

    summary = {
        "experiment_dir": str(args.experiment_dir.resolve()),
        "seed": args.seed,
        "num_samples": len(samples),
        "anomaly_threshold": anomaly_threshold,
        "num_components": total_components,
        "output_dir": str(args.output_dir.resolve()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(inventory_rows, args.output_dir / "component_inventory.csv")
    write_csv(sample_rows, args.output_dir / "sample_inventory.csv")
    write_json(inventory_rows, args.output_dir / "component_inventory.json")
    write_json(summary, args.output_dir / "summary.json")
    print(f"Wrote {len(inventory_rows)} components from {len(samples)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
