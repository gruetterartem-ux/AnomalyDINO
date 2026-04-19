from dataclasses import dataclass
from typing import List

import numpy as np

from .components import PatchComponent


@dataclass(frozen=True)
class SelectedPatch:
    component_id: int
    rank_in_component: int
    patch_index: int
    row: int
    col: int
    anomaly_score: float


def select_top_k_patches(component: PatchComponent, score_grid: np.ndarray, top_k: int) -> List[SelectedPatch]:
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    scores = score_grid[component.rows, component.cols]
    order = np.lexsort((component.cols, component.rows, -scores))
    k = min(int(top_k), component.size)

    selected: List[SelectedPatch] = []
    for rank, idx in enumerate(order[:k], start=1):
        row = int(component.rows[idx])
        col = int(component.cols[idx])
        selected.append(
            SelectedPatch(
                component_id=component.component_id,
                rank_in_component=rank,
                patch_index=int(component.patch_indices[idx]),
                row=row,
                col=col,
                anomaly_score=float(score_grid[row, col]),
            )
        )
    return selected
