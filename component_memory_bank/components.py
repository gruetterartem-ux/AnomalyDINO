from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(frozen=True)
class PatchComponent:
    component_id: int
    rows: np.ndarray
    cols: np.ndarray
    patch_indices: np.ndarray
    size: int
    bbox_row_min: int
    bbox_row_max: int
    bbox_col_min: int
    bbox_col_max: int
    max_score: float
    mean_score: float


def build_components_from_mask(active_mask: np.ndarray, score_grid: np.ndarray) -> List[PatchComponent]:
    if active_mask.shape != score_grid.shape:
        raise ValueError(
            f"Mask shape {active_mask.shape} does not match score grid shape {score_grid.shape}."
        )

    active_uint8 = active_mask.astype(np.uint8)
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(active_uint8, connectivity=8)

    components: List[PatchComponent] = []
    grid_w = score_grid.shape[1]
    for component_id in range(1, num_components):
        rows, cols = np.where(labels == component_id)
        if rows.size == 0:
            continue

        scores = score_grid[rows, cols]
        x, y, w, h, area = stats[component_id]
        patch_indices = rows * grid_w + cols
        components.append(
            PatchComponent(
                component_id=component_id,
                rows=rows.astype(np.int32),
                cols=cols.astype(np.int32),
                patch_indices=patch_indices.astype(np.int32),
                size=int(area),
                bbox_row_min=int(y),
                bbox_row_max=int(y + h - 1),
                bbox_col_min=int(x),
                bbox_col_max=int(x + w - 1),
                max_score=float(scores.max()),
                mean_score=float(scores.mean()),
            )
        )

    components.sort(key=lambda component: component.component_id)
    return components


def build_components(score_grid: np.ndarray, anomaly_threshold: float) -> tuple[np.ndarray, List[PatchComponent]]:
    active_mask = score_grid > float(anomaly_threshold)
    components = build_components_from_mask(active_mask, score_grid)
    return active_mask, components
