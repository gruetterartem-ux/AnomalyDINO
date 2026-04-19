import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .components import build_components
from .data_io import RunSample, load_patch_features, load_patch_scores
from .selection import select_top_k_patches


VALID_COMPONENT_LABELS = {"2D", "3D", "skip"}
VALID_MANUAL_PATCH_LABELS = {"2D", "3D"}


@dataclass(frozen=True)
class ComponentLabelRecord:
    object_name: str
    sample: str
    component_id: int
    anomaly_threshold: float
    top_k: int
    label: str


@dataclass(frozen=True)
class ManualPatchLabelRecord:
    object_name: str
    sample: str
    row: int
    col: int
    patch_index: int
    anomaly_score: float
    label: str


@dataclass
class MemoryBankBundle:
    features_2d: np.ndarray
    features_3d: np.ndarray
    metadata_rows: List[Dict[str, object]]


def load_component_labels(labels_file: Path) -> List[ComponentLabelRecord]:
    if not labels_file.exists():
        raise FileNotFoundError(f"Component labels file not found: {labels_file}")

    if labels_file.suffix.lower() == ".json":
        with labels_file.open("r", encoding="utf-8") as handle:
            raw_rows = json.load(handle)
    else:
        with labels_file.open("r", newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))

    records: List[ComponentLabelRecord] = []
    for row in raw_rows:
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        if label not in VALID_COMPONENT_LABELS:
            raise ValueError(f"Unsupported component label {label!r}. Supported: {sorted(VALID_COMPONENT_LABELS)}")

        records.append(
            ComponentLabelRecord(
                object_name=str(row["object_name"]),
                sample=str(row["sample"]).replace("\\", "/"),
                component_id=int(row["component_id"]),
                anomaly_threshold=float(row["anomaly_threshold"]),
                top_k=int(row.get("top_k", 5)),
                label=label,
            )
        )
    return records


def load_manual_patch_labels(labels_file: Path) -> List[ManualPatchLabelRecord]:
    if not labels_file.exists():
        return []

    with labels_file.open("r", newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    records: List[ManualPatchLabelRecord] = []
    for row in raw_rows:
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        if label not in VALID_MANUAL_PATCH_LABELS:
            raise ValueError(
                f"Unsupported manual patch label {label!r}. Supported: {sorted(VALID_MANUAL_PATCH_LABELS)}"
            )

        records.append(
            ManualPatchLabelRecord(
                object_name=str(row["object_name"]),
                sample=str(row["sample"]).replace("\\", "/"),
                row=int(row["row"]),
                col=int(row["col"]),
                patch_index=int(row["patch_index"]),
                anomaly_score=float(row["anomaly_score"]),
                label=label,
            )
        )
    return records


def build_memory_banks(
    samples: Iterable[RunSample],
    label_records: Iterable[ComponentLabelRecord],
    manual_patch_records: Iterable[ManualPatchLabelRecord] | None = None,
) -> MemoryBankBundle:
    sample_map = {sample.sample: sample for sample in samples}

    feature_cache: Dict[str, np.ndarray] = {}
    score_cache: Dict[str, np.ndarray] = {}
    component_cache: Dict[tuple[str, float], Dict[int, object]] = {}
    selected_patch_map: Dict[tuple[str, int], tuple[str, np.ndarray, Dict[str, object]]] = {}

    for record in label_records:
        if record.label == "skip":
            continue
        if record.sample not in sample_map:
            raise KeyError(f"Labeled component refers to unknown sample {record.sample!r}.")

        sample = sample_map[record.sample]
        if record.sample not in feature_cache:
            feature_cache[record.sample], grid_size = load_patch_features(sample)
            score_cache[record.sample] = load_patch_scores(sample)
            if tuple(score_cache[record.sample].shape) != tuple(grid_size):
                raise ValueError(
                    f"Feature grid {grid_size} and anomaly grid {score_cache[record.sample].shape} do not match for sample {record.sample!r}."
                )

        cache_key = (record.sample, round(record.anomaly_threshold, 8))
        if cache_key not in component_cache:
            _, components = build_components(score_cache[record.sample], record.anomaly_threshold)
            component_cache[cache_key] = {component.component_id: component for component in components}

        component = component_cache[cache_key].get(record.component_id)
        if component is None:
            raise KeyError(
                f"Component {record.component_id} not found in sample {record.sample!r} for threshold {record.anomaly_threshold:.6f}."
            )

        selected_patches = select_top_k_patches(component, score_cache[record.sample], record.top_k)
        for selected_patch in selected_patches:
            feature_vector = feature_cache[record.sample][selected_patch.patch_index].astype(np.float32, copy=False)
            selected_patch_map[(sample.sample, selected_patch.patch_index)] = (
                record.label,
                feature_vector,
                {
                    "object_name": sample.object_name,
                    "sample": sample.sample,
                    "component_id": record.component_id,
                    "component_label": record.label,
                    "top_k": record.top_k,
                    "rank_in_component": selected_patch.rank_in_component,
                    "patch_index": selected_patch.patch_index,
                    "row": selected_patch.row,
                    "col": selected_patch.col,
                    "anomaly_score": selected_patch.anomaly_score,
                    "anomaly_threshold": record.anomaly_threshold,
                    "source_type": "component",
                },
            )

    manual_patch_records = list(manual_patch_records or [])
    for record in manual_patch_records:
        if record.sample not in sample_map:
            raise KeyError(f"Manually labeled patch refers to unknown sample {record.sample!r}.")

        sample = sample_map[record.sample]
        if record.sample not in feature_cache:
            feature_cache[record.sample], grid_size = load_patch_features(sample)
            score_cache[record.sample] = load_patch_scores(sample)
            if tuple(score_cache[record.sample].shape) != tuple(grid_size):
                raise ValueError(
                    f"Feature grid {grid_size} and anomaly grid {score_cache[record.sample].shape} do not match for sample {record.sample!r}."
                )

        if not (0 <= record.patch_index < feature_cache[record.sample].shape[0]):
            raise IndexError(
                f"Patch index {record.patch_index} out of bounds for sample {record.sample!r}."
            )

        feature_vector = feature_cache[record.sample][record.patch_index].astype(np.float32, copy=False)
        selected_patch_map[(record.sample, record.patch_index)] = (
            record.label,
            feature_vector,
            {
                "object_name": sample.object_name,
                "sample": sample.sample,
                "component_id": "",
                "component_label": record.label,
                "top_k": "",
                "rank_in_component": "",
                "patch_index": record.patch_index,
                "row": record.row,
                "col": record.col,
                "anomaly_score": record.anomaly_score,
                "anomaly_threshold": "",
                "source_type": "manual",
            },
        )

    features_2d: List[np.ndarray] = []
    features_3d: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []
    for _, (label, feature_vector, metadata) in sorted(
        selected_patch_map.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        if label == "2D":
            features_2d.append(feature_vector)
        else:
            features_3d.append(feature_vector)
        metadata_rows.append(metadata)

    if not features_2d:
        raise ValueError("No 2D patches were selected. Label at least one component as 2D.")
    if not features_3d:
        raise ValueError("No 3D patches were selected. Label at least one component as 3D.")

    return MemoryBankBundle(
        features_2d=np.asarray(features_2d, dtype=np.float32),
        features_3d=np.asarray(features_3d, dtype=np.float32),
        metadata_rows=metadata_rows,
    )
