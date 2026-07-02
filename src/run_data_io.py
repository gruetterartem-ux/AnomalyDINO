import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class RunSample:
    object_name: str
    sample: str
    evaluation_group: str
    image_label: int
    image_score: float
    image_threshold: float
    image_path: Path
    anomaly_map_path: Path
    feature_cache_path: Path


def load_run_args(experiment_dir: Path) -> Dict:
    args_path = experiment_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Could not find run arguments: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_feature_cache_manifest(cache_manifest_path: Path) -> Dict[str, Dict[str, str]]:
    if not cache_manifest_path.exists():
        raise FileNotFoundError(
            f"Feature cache manifest not found: {cache_manifest_path}. "
            "Build it first via run_anomalydino.py --save_patch_feature_cache or cache_run_patch_features.py."
        )

    manifest: Dict[str, Dict[str, str]] = {}
    with cache_manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = row["sample"].replace("\\", "/")
            manifest[sample] = row
    return manifest


def _load_measurements(measurements_file: Path) -> List[Dict[str, str]]:
    if not measurements_file.exists():
        raise FileNotFoundError(f"Measurements file not found: {measurements_file}")

    rows: List[Dict[str, str]] = []
    with measurements_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    return rows


def _resolve_anomaly_map_path(
    experiment_dir: Path,
    seed: int,
    sample: str,
    object_name: str,
    inference_split: str,
) -> Path:
    sample_path = Path(sample.replace("\\", "/"))
    stem = sample_path.with_suffix("")
    root = experiment_dir / "anomaly_maps" / f"seed={seed}" / object_name

    candidates = []
    sample_posix = sample_path.as_posix()
    if sample_posix.startswith("good_train_remaining/"):
        candidates.append(root / "custom_eval" / "good_train_remaining" / sample_path.relative_to("good_train_remaining"))
    elif sample_posix.startswith("good_test/"):
        candidates.append(root / "custom_eval" / "good_test" / sample_path.relative_to("good_test"))
    elif sample_posix.startswith("test/bad/"):
        candidates.append(root / "custom_eval" / "bad" / sample_path.relative_to("test/bad"))
    elif sample_posix.startswith("test/good/"):
        candidates.append(root / "custom_eval" / "good" / sample_path.relative_to("test/good"))
    else:
        candidates.append(root / inference_split / stem)

    for candidate in candidates:
        npy_path = candidate.with_suffix(".npy")
        if npy_path.exists():
            return npy_path

    checked = ", ".join(str(candidate.with_suffix(".npy")) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve anomaly map for sample {sample!r}. Checked: {checked}")


def load_run_samples(experiment_dir: Path, seed: int = 0) -> List[RunSample]:
    run_args = load_run_args(experiment_dir)
    inference_split = str(run_args.get("inference_split", "test"))
    measurements_file = experiment_dir / f"measurements_seed={seed}.csv"
    feature_cache_manifest = experiment_dir / "patch_feature_cache" / f"seed={seed}" / "cache_manifest.csv"

    measurements = _load_measurements(measurements_file)
    cache_rows = _load_feature_cache_manifest(feature_cache_manifest)

    samples: List[RunSample] = []
    for row in measurements:
        sample = row["Sample"].replace("\\", "/")
        if sample not in cache_rows:
            raise KeyError(
                f"Sample {sample!r} is in measurements but missing from feature cache manifest. "
                "Rebuild the patch feature cache for this run."
            )

        cache_row = cache_rows[sample]
        object_name = row["Object"]
        anomaly_map_path = _resolve_anomaly_map_path(experiment_dir, seed, sample, object_name, inference_split)
        samples.append(
            RunSample(
                object_name=object_name,
                sample=sample,
                evaluation_group=row["Evaluation_Group"],
                image_label=int(row["Label"]),
                image_score=float(row["Anomaly_Score"]),
                image_threshold=float(row["Threshold"]),
                image_path=Path(cache_row["image_path"]),
                anomaly_map_path=anomaly_map_path,
                feature_cache_path=Path(cache_row["cache_file"]),
            )
        )

    samples.sort(key=lambda sample: sample.sample)
    return samples


def load_patch_features(sample: RunSample) -> tuple[np.ndarray, tuple[int, int]]:
    if not sample.feature_cache_path.exists():
        raise FileNotFoundError(f"Feature cache file not found: {sample.feature_cache_path}")

    with np.load(sample.feature_cache_path) as data:
        features = data["features"].astype(np.float32)
        grid_size = tuple(int(v) for v in data["grid_size"].tolist())
    return features, grid_size


def load_patch_scores(sample: RunSample) -> np.ndarray:
    if not sample.anomaly_map_path.exists():
        raise FileNotFoundError(f"Patch anomaly map not found: {sample.anomaly_map_path}")
    return np.load(sample.anomaly_map_path).astype(np.float32)


def load_image_bgr(sample: RunSample) -> np.ndarray:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {sample.image_path}")
    return image


def load_image_rgb(sample: RunSample) -> np.ndarray:
    return cv2.cvtColor(load_image_bgr(sample), cv2.COLOR_BGR2RGB)
