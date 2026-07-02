from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.visualize import _save_patch_distance_grid_png


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688"
    r"\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all patch-grid anomaly maps (.npy) of a run as PNGs."
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir(experiment_dir: Path, seed: int, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (experiment_dir / "patchgrid_anomaly_map_pngs" / f"seed={seed}").resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    anomaly_root = experiment_dir / "anomaly_maps" / f"seed={args.seed}"
    output_root = default_output_dir(experiment_dir, args.seed, args.output_dir)
    ensure_dir(output_root)

    if not anomaly_root.exists():
        raise FileNotFoundError(f"Anomaly map directory not found: {anomaly_root}")

    npy_files = sorted(anomaly_root.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy anomaly maps found under: {anomaly_root}")

    exported = 0
    manifest_rows: list[dict[str, object]] = []
    for npy_path in npy_files:
        rel_path = npy_path.relative_to(anomaly_root)
        output_path = (output_root / rel_path).with_suffix(".png")
        ensure_dir(output_path.parent)
        dists = np.load(npy_path).astype(np.float32)
        _save_patch_distance_grid_png(dists, str(output_path))
        manifest_rows.append(
            {
                "npy_path": str(npy_path),
                "relative_path": str(rel_path).replace("\\", "/"),
                "png_path": str(output_path),
                "grid_h": int(dists.shape[0]),
                "grid_w": int(dists.shape[1]),
                "min_value": float(np.min(dists)),
                "max_value": float(np.max(dists)),
            }
        )
        exported += 1

    manifest_path = output_root / "manifest.json"
    summary_path = output_root / "summary.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    summary = {
        "experiment_dir": str(experiment_dir),
        "anomaly_root": str(anomaly_root),
        "output_root": str(output_root),
        "seed": int(args.seed),
        "num_pngs": int(exported),
        "manifest_json": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {exported} patch-grid PNG(s) to {output_root}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
