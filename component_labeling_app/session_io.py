from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from component_memory_bank.data_io import load_run_args, load_run_samples
from component_memory_bank.export import export_memory_banks, write_json
from component_memory_bank.memory_bank import (
    build_memory_banks,
    load_component_labels,
    load_manual_patch_labels,
)


COMPONENT_ANNOTATIONS_FILE = "component_annotations.csv"
PART_LABELS_FILE = "part_labels.csv"
PART_LABELS_JSON_FILE = "part_labels.json"
MANUAL_PATCH_ANNOTATIONS_FILE = "manual_patch_annotations.csv"
MEMORY_BANK_EXPORT_DIR = "memory_bank_export"


@dataclass(frozen=True)
class SessionContext:
    session_dir: Path
    experiment_dir: Path
    seed: int
    anomaly_threshold: float
    component_annotations_path: Path
    part_labels_path: Path
    manual_patch_annotations_path: Path


def _normalize_path_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ("sample", "image_path", "anomaly_map_path", "feature_cache_path"):
        if column in out.columns:
            out[column] = out[column].astype(str).str.replace("\\", "/", regex=False)
    return out


def _load_session_summary(session_dir: Path) -> dict:
    summary_path = session_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Session summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_inventory(session_dir: Path) -> pd.DataFrame:
    inventory_path = session_dir / "component_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Component inventory not found: {inventory_path}")

    inventory = pd.read_csv(inventory_path)
    inventory = _normalize_path_columns(inventory)
    inventory["component_id"] = inventory["component_id"].astype(int)
    inventory["top_k"] = inventory["top_k"].fillna(5).astype(int)
    inventory["label"] = inventory["label"].fillna("").astype(str)
    inventory["notes"] = inventory["notes"].fillna("").astype(str)
    inventory["anomaly_threshold"] = inventory["anomaly_threshold"].astype(float)
    inventory.sort_values(["sample", "component_id"], inplace=True)
    inventory.reset_index(drop=True, inplace=True)
    return inventory


def _load_sample_inventory(session_dir: Path, inventory: pd.DataFrame) -> pd.DataFrame:
    sample_inventory_path = session_dir / "sample_inventory.csv"
    if sample_inventory_path.exists():
        sample_inventory = pd.read_csv(sample_inventory_path)
        sample_inventory = _normalize_path_columns(sample_inventory)
        sample_inventory["num_components"] = sample_inventory["num_components"].fillna(0).astype(int)
        sample_inventory["has_components"] = sample_inventory["has_components"].fillna(False).astype(bool)
        sample_inventory.sort_values(["sample"], inplace=True)
        sample_inventory.reset_index(drop=True, inplace=True)
        return sample_inventory

    sample_rows = (
        inventory.sort_values(["sample", "component_id"])
        .groupby("sample", as_index=False)
        .first()[
            [
                "object_name",
                "sample",
                "evaluation_group",
                "image_score",
                "image_threshold",
                "anomaly_threshold",
                "image_path",
                "anomaly_map_path",
                "feature_cache_path",
                "grid_rows",
                "grid_cols",
            ]
        ]
        .copy()
    )
    sample_rows["num_components"] = inventory.groupby("sample")["component_id"].count().reindex(sample_rows["sample"]).to_numpy()
    sample_rows["has_components"] = sample_rows["num_components"] > 0
    return sample_rows


def _init_component_annotations(session_dir: Path, inventory: pd.DataFrame) -> pd.DataFrame:
    annotations_path = session_dir / COMPONENT_ANNOTATIONS_FILE
    if annotations_path.exists():
        annotations = pd.read_csv(annotations_path)
        annotations = _normalize_path_columns(annotations)
        annotations["component_id"] = annotations["component_id"].astype(int)
        annotations["top_k"] = annotations["top_k"].fillna(5).astype(int)
        annotations["label"] = annotations["label"].fillna("").astype(str)
        annotations["notes"] = annotations["notes"].fillna("").astype(str)

        keep_cols = ["sample", "component_id", "label", "top_k", "notes"]
        merged = inventory.drop(columns=["label", "top_k", "notes"]).merge(
            annotations[keep_cols],
            how="left",
            on=["sample", "component_id"],
        )
        merged["label"] = merged["label"].fillna("").astype(str)
        merged["top_k"] = merged["top_k"].fillna(inventory["top_k"]).astype(int)
        merged["notes"] = merged["notes"].fillna("").astype(str)
        return merged

    inventory.to_csv(annotations_path, index=False)
    return inventory.copy()


def _init_part_labels(session_dir: Path, sample_inventory: pd.DataFrame) -> pd.DataFrame:
    part_labels_path = session_dir / PART_LABELS_FILE
    sample_rows = sample_inventory[
        [
            "object_name",
            "sample",
            "evaluation_group",
            "image_path",
            "anomaly_map_path",
            "feature_cache_path",
            "grid_rows",
            "grid_cols",
            "num_components",
            "has_components",
        ]
    ].copy()
    sample_rows["part_label"] = ""
    sample_rows["part_notes"] = ""

    if part_labels_path.exists():
        part_labels = pd.read_csv(part_labels_path)
        part_labels = _normalize_path_columns(part_labels)
        merged = sample_rows.drop(columns=["part_label", "part_notes"]).merge(
            part_labels[["sample", "part_label", "part_notes"]],
            how="left",
            on="sample",
        )
        merged["part_label"] = merged["part_label"].fillna("").astype(str)
        merged["part_notes"] = merged["part_notes"].fillna("").astype(str)
        return merged

    sample_rows.to_csv(part_labels_path, index=False)
    write_json(sample_rows.to_dict(orient="records"), session_dir / PART_LABELS_JSON_FILE)
    return sample_rows


def _empty_manual_patch_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "object_name",
            "sample",
            "evaluation_group",
            "image_path",
            "anomaly_map_path",
            "feature_cache_path",
            "grid_rows",
            "grid_cols",
            "component_id",
            "row",
            "col",
            "patch_index",
            "anomaly_score",
            "label",
        ]
    )


def _init_manual_patch_annotations(session_dir: Path) -> pd.DataFrame:
    manual_path = session_dir / MANUAL_PATCH_ANNOTATIONS_FILE
    if manual_path.exists():
        manual = pd.read_csv(manual_path)
        manual = _normalize_path_columns(manual)
        if manual.empty:
            return _empty_manual_patch_annotations()
        manual["row"] = manual["row"].astype(int)
        manual["col"] = manual["col"].astype(int)
        manual["patch_index"] = manual["patch_index"].astype(int)
        manual["grid_rows"] = manual["grid_rows"].astype(int)
        manual["grid_cols"] = manual["grid_cols"].astype(int)
        manual["anomaly_score"] = manual["anomaly_score"].astype(float)
        manual["label"] = manual["label"].fillna("").astype(str)
        manual["component_id"] = manual["component_id"].fillna("").astype(str)
        manual.sort_values(["sample", "row", "col"], inplace=True)
        manual.reset_index(drop=True, inplace=True)
        return manual

    manual = _empty_manual_patch_annotations()
    manual.to_csv(manual_path, index=False)
    return manual


def load_session(
    session_dir: Path,
) -> tuple[SessionContext, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    session_dir = session_dir.resolve()
    summary = _load_session_summary(session_dir)
    inventory = _load_inventory(session_dir)
    sample_inventory = _load_sample_inventory(session_dir, inventory)
    component_annotations = _init_component_annotations(session_dir, inventory)
    part_labels = _init_part_labels(session_dir, sample_inventory)
    manual_patch_annotations = _init_manual_patch_annotations(session_dir)

    context = SessionContext(
        session_dir=session_dir,
        experiment_dir=Path(summary["experiment_dir"]).resolve(),
        seed=int(summary["seed"]),
        anomaly_threshold=float(summary["anomaly_threshold"]),
        component_annotations_path=session_dir / COMPONENT_ANNOTATIONS_FILE,
        part_labels_path=session_dir / PART_LABELS_FILE,
        manual_patch_annotations_path=session_dir / MANUAL_PATCH_ANNOTATIONS_FILE,
    )
    run_args = load_run_args(context.experiment_dir)
    return context, component_annotations, part_labels, manual_patch_annotations, run_args, sample_inventory


def save_component_annotations(context: SessionContext, component_annotations: pd.DataFrame) -> None:
    out = component_annotations.copy()
    out.sort_values(["sample", "component_id"], inplace=True)
    out.to_csv(context.component_annotations_path, index=False)


def save_part_labels(context: SessionContext, part_labels: pd.DataFrame) -> None:
    out = part_labels.copy()
    out.sort_values(["sample"], inplace=True)
    out.to_csv(context.part_labels_path, index=False)
    write_json(out.to_dict(orient="records"), context.session_dir / PART_LABELS_JSON_FILE)


def save_manual_patch_annotations(context: SessionContext, manual_patch_annotations: pd.DataFrame) -> None:
    out = manual_patch_annotations.copy()
    out.sort_values(["sample", "row", "col"], inplace=True)
    out.to_csv(context.manual_patch_annotations_path, index=False)


def build_memory_bank_export(
    context: SessionContext,
    component_annotations: pd.DataFrame | None = None,
    manual_patch_annotations: pd.DataFrame | None = None,
) -> Path:
    samples = load_run_samples(context.experiment_dir, seed=context.seed)
    if component_annotations is not None:
        tmp_component_labels = context.session_dir / "_tmp_component_annotations_export.csv"
        component_annotations.to_csv(tmp_component_labels, index=False)
        label_records = load_component_labels(tmp_component_labels)
        tmp_component_labels.unlink(missing_ok=True)
    else:
        label_records = load_component_labels(context.component_annotations_path)

    if manual_patch_annotations is not None:
        tmp_manual_labels = context.session_dir / "_tmp_manual_patch_annotations_export.csv"
        manual_patch_annotations.to_csv(tmp_manual_labels, index=False)
        manual_patch_records = load_manual_patch_labels(tmp_manual_labels)
        tmp_manual_labels.unlink(missing_ok=True)
    else:
        manual_patch_records = load_manual_patch_labels(context.manual_patch_annotations_path)

    bundle = build_memory_banks(samples, label_records, manual_patch_records)

    output_dir = context.session_dir / MEMORY_BANK_EXPORT_DIR
    num_manual_2d = int(sum(1 for record in manual_patch_records if record.label == "2D"))
    num_manual_3d = int(sum(1 for record in manual_patch_records if record.label == "3D"))
    summary = {
        "experiment_dir": str(context.experiment_dir),
        "seed": context.seed,
        "labels_file": str(context.component_annotations_path),
        "manual_patch_labels_file": str(context.manual_patch_annotations_path),
        "num_labeled_components": sum(1 for record in label_records if record.label != "skip"),
        "num_manual_patches_2d": num_manual_2d,
        "num_manual_patches_3d": num_manual_3d,
        "num_selected_patches_2d": int(bundle.features_2d.shape[0]),
        "num_selected_patches_3d": int(bundle.features_3d.shape[0]),
        "feature_dim": int(bundle.features_2d.shape[1]),
        "output_dir": str(output_dir),
    }
    export_memory_banks(
        output_dir=output_dir,
        features_2d=bundle.features_2d,
        features_3d=bundle.features_3d,
        patch_metadata_rows=bundle.metadata_rows,
        summary=summary,
    )
    return output_dir


def component_progress(component_annotations: pd.DataFrame) -> dict:
    label_series = component_annotations["label"].fillna("").astype(str)
    labeled_mask = label_series != ""
    return {
        "total": int(component_annotations.shape[0]),
        "labeled": int(labeled_mask.sum()),
        "unlabeled": int((~labeled_mask).sum()),
        "2D": int((label_series == "2D").sum()),
        "3D": int((label_series == "3D").sum()),
        "skip": int((label_series == "skip").sum()),
    }


def part_progress(part_labels: pd.DataFrame) -> dict:
    label_series = part_labels["part_label"].fillna("").astype(str)
    labeled_mask = label_series != ""
    return {
        "total": int(part_labels.shape[0]),
        "labeled": int(labeled_mask.sum()),
        "unlabeled": int((~labeled_mask).sum()),
        "2D": int((label_series == "2D").sum()),
        "3D": int((label_series == "3D").sum()),
        "skip": int((label_series == "skip").sum()),
    }


def manual_patch_progress(manual_patch_annotations: pd.DataFrame) -> dict:
    if manual_patch_annotations.empty:
        return {"total": 0, "2D": 0, "3D": 0}
    label_series = manual_patch_annotations["label"].fillna("").astype(str)
    return {
        "total": int(manual_patch_annotations.shape[0]),
        "2D": int((label_series == "2D").sum()),
        "3D": int((label_series == "3D").sum()),
    }


def memory_bank_patch_progress(
    component_annotations: pd.DataFrame,
    manual_patch_annotations: pd.DataFrame,
) -> dict:
    component_labels = component_annotations["label"].fillna("").astype(str)
    top_k_series = component_annotations["top_k"].fillna(0).astype(int)

    component_patch_2d = int(top_k_series[component_labels == "2D"].sum())
    component_patch_3d = int(top_k_series[component_labels == "3D"].sum())

    manual_labels = manual_patch_annotations["label"].fillna("").astype(str) if not manual_patch_annotations.empty else pd.Series(dtype=str)
    manual_patch_2d = int((manual_labels == "2D").sum())
    manual_patch_3d = int((manual_labels == "3D").sum())

    return {
        "component_patch_2D": component_patch_2d,
        "component_patch_3D": component_patch_3d,
        "manual_patch_2D": manual_patch_2d,
        "manual_patch_3D": manual_patch_3d,
        "total_memory_bank_patches_2D": component_patch_2d + manual_patch_2d,
        "total_memory_bank_patches_3D": component_patch_3d + manual_patch_3d,
    }
