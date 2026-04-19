import argparse
from pathlib import Path

import numpy as np

from component_memory_bank.data_io import load_patch_features, load_patch_scores, load_run_samples
from component_memory_bank.export import write_csv, write_json
from component_memory_bank.inference import (
    classify_components,
    classify_part,
    compute_patch_class_scores,
    summarize_components,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a 2D/3D patch memory bank to AnomalyDINO patch features and score components."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--memory-bank-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--anomaly-threshold", type=float, default=None)
    parser.add_argument("--tau-s", type=float, default=0.0)
    parser.add_argument("--tau-n", type=int, default=2)
    parser.add_argument("--tau-p", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-only-substring", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    bank_2d = np.load(args.memory_bank_dir / "2D-memory-bank.npy").astype(np.float32)
    bank_3d = np.load(args.memory_bank_dir / "3D-memory-bank.npy").astype(np.float32)
    samples = load_run_samples(args.experiment_dir, seed=args.seed)

    if args.include_only_substring:
        needle = args.include_only_substring.replace("\\", "/")
        samples = [sample for sample in samples if needle in sample.sample]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples selected for scoring.")

    default_threshold = float(samples[0].image_threshold)
    anomaly_threshold = float(args.anomaly_threshold) if args.anomaly_threshold is not None else default_threshold

    part_rows = []
    component_rows = []
    for sample in samples:
        features, grid_size = load_patch_features(sample)
        score_grid = load_patch_scores(sample)
        if tuple(score_grid.shape) != tuple(grid_size):
            raise ValueError(
                f"Grid mismatch for {sample.sample!r}: feature cache {grid_size}, anomaly grid {score_grid.shape}."
            )

        patch_scores = compute_patch_class_scores(
            patch_features=features,
            anomaly_scores=score_grid,
            bank_2d=bank_2d,
            bank_3d=bank_3d,
            k_neighbors=args.k_neighbors,
            anomaly_threshold=anomaly_threshold,
        )
        component_summaries = summarize_components(
            anomaly_scores=score_grid,
            active_mask=patch_scores["active_mask"],
            margin_c=patch_scores["margin_c"],
            weighted_margin_z=patch_scores["weighted_margin_z"],
        )
        component_decisions = classify_components(
            sample=sample.sample,
            component_summaries=component_summaries,
            tau_s=args.tau_s,
            tau_n=args.tau_n,
            tau_p=args.tau_p,
        )
        part_decision = classify_part(sample.sample, component_decisions)

        part_rows.append(
            {
                "object_name": sample.object_name,
                "sample": sample.sample,
                "evaluation_group": sample.evaluation_group,
                "image_label": sample.image_label,
                "predicted_label": part_decision.predicted_label,
                "has_3d_component": part_decision.has_3d_component,
                "num_components": part_decision.num_components,
                "num_3d_components": part_decision.num_3d_components,
            }
        )
        for decision in component_decisions:
            component_rows.append(
                {
                    "object_name": sample.object_name,
                    "sample": sample.sample,
                    "component_id": decision.component_id,
                    "score_s": decision.score_s,
                    "size_n": decision.size_n,
                    "peak_p": decision.peak_p,
                    "max_margin_c": decision.max_margin_c,
                    "predicted_label": "3D" if decision.is_3d else "2D",
                }
            )

    summary = {
        "experiment_dir": str(args.experiment_dir.resolve()),
        "memory_bank_dir": str(args.memory_bank_dir.resolve()),
        "seed": args.seed,
        "num_samples": len(part_rows),
        "k_neighbors": args.k_neighbors,
        "anomaly_threshold": anomaly_threshold,
        "tau_s": args.tau_s,
        "tau_n": args.tau_n,
        "tau_p": args.tau_p,
        "output_dir": str(args.output_dir.resolve()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(part_rows, args.output_dir / "part_predictions.csv")
    write_csv(component_rows, args.output_dir / "component_predictions.csv")
    write_json(summary, args.output_dir / "summary.json")
    print(f"Scored {len(part_rows)} samples with component memory-bank logic. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
