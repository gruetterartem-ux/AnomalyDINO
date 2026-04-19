import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml
from PIL import Image

from src.backbones import get_model


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a reusable per-image patch-feature cache for an existing AnomalyDINO run."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Run directory containing args.yaml and measurements_seed=*.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed index used for measurements_seed=<seed>.csv.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for backbone feature extraction.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
        help="Optional local backbone weights path.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional cache directory. Defaults to <experiment-dir>/patch_feature_cache/seed=<seed>.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_run_args(experiment_dir: Path) -> Dict[str, object]:
    args_path = experiment_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Run arguments not found: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_measurements(measurements_file: Path) -> List[Dict[str, str]]:
    if not measurements_file.exists():
        raise FileNotFoundError(f"Measurements file not found: {measurements_file}")
    with measurements_file.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique_samples(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: Dict[tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        deduped[(row["Object"], row["Sample"])] = row
    return [deduped[key] for key in sorted(deduped.keys())]


def resolve_image_path(data_root: Path, object_name: str, sample: str, inference_split: str, eval_remaining_train_good: bool) -> Path:
    sample_path = Path(sample)
    if eval_remaining_train_good:
        if sample.startswith("good_train_remaining/"):
            image_path = data_root / object_name / "train" / "good" / sample_path.relative_to("good_train_remaining")
        elif sample.startswith("good_test/"):
            image_path = data_root / object_name / "test" / "good" / sample_path.relative_to("good_test")
        elif sample.startswith("test/bad/"):
            image_path = data_root / object_name / "test" / "bad" / sample_path.relative_to("test/bad")
        elif sample.startswith("test/good/"):
            image_path = data_root / object_name / "test" / "good" / sample_path.relative_to("test/good")
        else:
            raise FileNotFoundError(f"Unsupported custom_eval sample: {sample}")
    else:
        image_path = data_root / object_name / inference_split / sample_path
    if not image_path.exists():
        raise FileNotFoundError(f"Original image not found: {image_path}")
    return image_path


def cache_file_for_sample(output_dir: Path, object_name: str, sample: str) -> Path:
    return output_dir / object_name / Path(sample).with_suffix(".npz")


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    run_args = load_run_args(experiment_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (experiment_dir / "patch_feature_cache" / f"seed={args.seed}").resolve()
    )
    ensure_dir(output_dir)

    measurements = unique_samples(load_measurements(experiment_dir / f"measurements_seed={args.seed}.csv"))
    if args.limit_samples is not None:
        measurements = measurements[: args.limit_samples]

    data_root = Path(str(run_args["data_root"])).resolve()
    model_name = str(run_args["model_name"])
    resolution = int(run_args["resolution"])
    inference_split = str(run_args.get("inference_split", "test"))
    eval_remaining_train_good = bool(run_args.get("eval_remaining_train_good", False))

    model = get_model(
        model_name,
        args.device,
        smaller_edge_size=resolution,
        weights_path=args.backbone_weights,
    )

    manifest_rows: List[Dict[str, object]] = []
    for index, row in enumerate(measurements, start=1):
        object_name = row["Object"]
        sample = row["Sample"].replace("\\", "/")
        image_path = resolve_image_path(
            data_root=data_root,
            object_name=object_name,
            sample=sample,
            inference_split=inference_split,
            eval_remaining_train_good=eval_remaining_train_good,
        )
        cache_file = cache_file_for_sample(output_dir, object_name, sample)
        ensure_dir(cache_file.parent)

        image_rgb = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image_tensor, grid_size = model.prepare_image(image_rgb)
        features = model.extract_features(image_tensor).astype(np.float16)
        if getattr(model, "resize_transform", None) is not None:
            resized_image = model.resize_transform(Image.fromarray(image_rgb))
            resized_width, resized_height = resized_image.size
        else:
            resized_height, resized_width = grid_size[0] * model.patch_size, grid_size[1] * model.patch_size

        np.savez_compressed(
            cache_file,
            object=object_name,
            sample=sample,
            image_path=str(image_path),
            features=features,
            grid_size=np.asarray(grid_size, dtype=np.int32),
            resized_size=np.asarray([resized_width, resized_height], dtype=np.int32),
            original_size=np.asarray([image_rgb.shape[1], image_rgb.shape[0]], dtype=np.int32),
            patch_size=np.asarray([int(model.patch_size)], dtype=np.int32),
        )
        manifest_rows.append(
            {
                "object": object_name,
                "sample": sample,
                "image_path": str(image_path),
                "cache_file": str(cache_file),
                "grid_h": int(grid_size[0]),
                "grid_w": int(grid_size[1]),
                "feature_dim": int(features.shape[1]),
                "patch_size": int(model.patch_size),
                "resized_width": int(resized_width),
                "resized_height": int(resized_height),
                "original_width": int(image_rgb.shape[1]),
                "original_height": int(image_rgb.shape[0]),
            }
        )
        print(f"[{index}/{len(measurements)}] Cached {object_name} :: {sample}")

    manifest_file = output_dir / "cache_manifest.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_file = output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_dir": str(experiment_dir),
                "seed": int(args.seed),
                "model_name": model_name,
                "resolution": resolution,
                "num_samples": int(len(manifest_rows)),
                "output_dir": str(output_dir),
                "manifest_file": str(manifest_file),
            },
            handle,
            indent=2,
        )

    print(f"Saved cache manifest: {manifest_file}")
    print(f"Saved cache summary: {summary_file}")


if __name__ == "__main__":
    main()
