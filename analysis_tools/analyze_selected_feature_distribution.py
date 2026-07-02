from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_FINAL\normalmap_dinov3_vitb16_res688"
)
DEFAULT_SELECTIONS = {
    "boruta": DEFAULT_EXPERIMENT_DIR
    / "final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf"
    / "selected_features.csv",
    "mrmr": DEFAULT_EXPERIMENT_DIR
    / "final_all_boxes_overthreshold_maxminmean_mrmr_fixedk384_rbf"
    / "selected_features.csv",
}
AGGREGATION_COMPONENTS = ("max", "min", "mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze selected ROI feature distributions by DINO layer and aggregation component."
    )
    parser.add_argument(
        "--selection",
        action="append",
        nargs=2,
        metavar=("NAME", "SELECTED_FEATURES_CSV"),
        help=(
            "Selection CSV to analyze. Can be passed multiple times. "
            "If omitted, final Boruta and mRMR selections are used."
        ),
    )
    parser.add_argument("--layer-dim", type=int, default=768)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR / "selected_feature_distribution_analysis")
    return parser.parse_args()


def decode_feature_index(feature_index: int, layer_dim: int) -> dict[str, int | str]:
    per_layer = layer_dim * len(AGGREGATION_COMPONENTS)
    layer_zero = int(feature_index) // per_layer
    remainder = int(feature_index) % per_layer
    component_index = remainder // layer_dim
    within_component_dim = remainder % layer_dim
    if component_index < 0 or component_index >= len(AGGREGATION_COMPONENTS):
        raise ValueError(f"Invalid component index {component_index} for feature_index={feature_index}")
    return {
        "layer_index": int(layer_zero + 1),
        "aggregation_component": AGGREGATION_COMPONENTS[int(component_index)],
        "within_component_dim": int(within_component_dim),
    }


def load_selection(path: Path, layer_dim: int) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "feature_index" not in table.columns:
        raise ValueError(f"'feature_index' column missing in {path}")

    if "status" in table.columns:
        confirmed = table["status"].astype(str).str.lower().eq("confirmed")
        if confirmed.any():
            table = table.loc[confirmed].copy()

    table = table.copy().reset_index(drop=True)
    decoded = [decode_feature_index(int(index), layer_dim) for index in table["feature_index"]]
    decoded_table = pd.DataFrame(decoded)

    for column in ("layer_index", "aggregation_component", "within_component_dim"):
        if column in table.columns:
            table = table.drop(columns=[column])
    return pd.concat([table, decoded_table], axis=1)


def count_table(table: pd.DataFrame, group_columns: list[str], total_count: int) -> pd.DataFrame:
    counts = (
        table.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    counts["percent"] = (counts["count"].astype(float) / float(total_count) * 100.0).round(2)
    return counts


def analyze_selection(name: str, path: Path, output_dir: Path, layer_dim: int) -> dict[str, object]:
    selection_dir = output_dir / name
    selection_dir.mkdir(parents=True, exist_ok=True)

    table = load_selection(path, layer_dim)
    decoded_csv = selection_dir / "selected_features_decoded.csv"
    aggregation_csv = selection_dir / "aggregation_counts.csv"
    layer_csv = selection_dir / "layer_counts.csv"
    layer_by_aggregation_csv = selection_dir / "layer_by_aggregation_counts.csv"
    summary_json = selection_dir / "summary.json"

    table.to_csv(decoded_csv, index=False)
    total_count = int(len(table))
    aggregation_counts = count_table(table, ["aggregation_component"], total_count)
    layer_counts = count_table(table, ["layer_index"], total_count)
    layer_by_aggregation = count_table(table, ["layer_index", "aggregation_component"], total_count)

    aggregation_counts.to_csv(aggregation_csv, index=False)
    layer_counts.to_csv(layer_csv, index=False)
    layer_by_aggregation.to_csv(layer_by_aggregation_csv, index=False)

    summary = {
        "name": name,
        "selected_features_csv": str(path),
        "selected_feature_count": int(len(table)),
        "layer_dim": int(layer_dim),
        "aggregation_counts": {
            str(row["aggregation_component"]): int(row["count"])
            for _, row in aggregation_counts.iterrows()
        },
        "aggregation_percent": {
            str(row["aggregation_component"]): float(row["percent"])
            for _, row in aggregation_counts.iterrows()
        },
        "layer_counts": {
            str(int(row["layer_index"])): int(row["count"])
            for _, row in layer_counts.iterrows()
        },
        "layer_percent": {
            str(int(row["layer_index"])): float(row["percent"])
            for _, row in layer_counts.iterrows()
        },
        "outputs": {
            "selected_features_decoded_csv": str(decoded_csv),
            "aggregation_counts_csv": str(aggregation_csv),
            "layer_counts_csv": str(layer_csv),
            "layer_by_aggregation_counts_csv": str(layer_by_aggregation_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = (
        {name: Path(path) for name, path in args.selection}
        if args.selection
        else DEFAULT_SELECTIONS
    )

    summaries = []
    for name, path in selections.items():
        summary = analyze_selection(
            name=str(name),
            path=Path(path).resolve(),
            output_dir=output_dir,
            layer_dim=int(args.layer_dim),
        )
        summaries.append(summary)
        print(f"{summary['name']}: n={summary['selected_feature_count']}")
        print(f"  aggregation_counts={summary['aggregation_counts']}")
        print(f"  layer_counts={summary['layer_counts']}")

    combined_summary = {
        "output_dir": str(output_dir),
        "layer_dim": int(args.layer_dim),
        "selections": summaries,
    }
    combined_path = output_dir / "summary.json"
    combined_path.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    print(f"Saved combined summary: {combined_path}")


if __name__ == "__main__":
    main()
