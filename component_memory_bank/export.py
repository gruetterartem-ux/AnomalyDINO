import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def write_csv(rows: Iterable[Mapping[str, object]], output_file: Path) -> None:
    rows = list(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_file.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: object, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def export_memory_banks(
    output_dir: Path,
    features_2d: np.ndarray,
    features_3d: np.ndarray,
    patch_metadata_rows: Iterable[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "2D-memory-bank.npy", features_2d.astype(np.float32))
    np.save(output_dir / "3D-memory-bank.npy", features_3d.astype(np.float32))
    write_csv(patch_metadata_rows, output_dir / "selected_patches.csv")
    write_json(summary, output_dir / "summary.json")
