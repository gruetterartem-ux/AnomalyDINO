import argparse
from pathlib import Path

from component_memory_bank.data_io import load_run_samples
from component_memory_bank.export import export_memory_banks
from component_memory_bank.memory_bank import build_memory_banks, load_component_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build 2D/3D patch memory banks from labeled anomaly components."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_run_samples(args.experiment_dir, seed=args.seed)
    label_records = load_component_labels(args.labels_file)
    bundle = build_memory_banks(samples, label_records)

    summary = {
        "experiment_dir": str(args.experiment_dir.resolve()),
        "seed": args.seed,
        "labels_file": str(args.labels_file.resolve()),
        "num_labeled_components": sum(1 for record in label_records if record.label != "skip"),
        "num_selected_patches_2d": int(bundle.features_2d.shape[0]),
        "num_selected_patches_3d": int(bundle.features_3d.shape[0]),
        "feature_dim": int(bundle.features_2d.shape[1]),
        "output_dir": str(args.output_dir.resolve()),
    }

    export_memory_banks(
        output_dir=args.output_dir,
        features_2d=bundle.features_2d,
        features_3d=bundle.features_3d,
        patch_metadata_rows=bundle.metadata_rows,
        summary=summary,
    )
    print(
        f"Built memory banks at {args.output_dir} "
        f"(2D patches: {bundle.features_2d.shape[0]}, 3D patches: {bundle.features_3d.shape[0]})"
    )


if __name__ == "__main__":
    main()
