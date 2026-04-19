from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from component_memory_bank.components import PatchComponent
from component_memory_bank.selection import select_top_k_patches


PALETTE = [
    (255, 99, 71),
    (60, 179, 113),
    (65, 105, 225),
    (255, 165, 0),
    (138, 43, 226),
    (0, 206, 209),
    (255, 20, 147),
    (154, 205, 50),
    (255, 215, 0),
    (70, 130, 180),
]


def prepare_display_image(image_path: Path, smaller_edge_size: int, patch_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    resize_transform = transforms.Resize(
        size=smaller_edge_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    image = resize_transform(image)
    image_np = np.array(image)
    cropped_h = image_np.shape[0] - image_np.shape[0] % patch_size
    cropped_w = image_np.shape[1] - image_np.shape[1] % patch_size
    return image_np[:cropped_h, :cropped_w].copy()


def grid_edges(image_shape: tuple[int, int, int], grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    rows, cols = grid_shape
    row_edges = np.linspace(0, h, rows + 1).round().astype(int)
    col_edges = np.linspace(0, w, cols + 1).round().astype(int)
    return row_edges, col_edges


def blend(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(base_rgb, 1.0 - alpha, overlay_rgb, alpha, 0.0)


def draw_patch_grid(image_rgb: np.ndarray, grid_shape: tuple[int, int], color=(200, 200, 200)) -> np.ndarray:
    canvas = image_rgb.copy()
    row_edges, col_edges = grid_edges(canvas.shape, grid_shape)
    for y in row_edges:
        cv2.line(canvas, (0, int(y)), (canvas.shape[1] - 1, int(y)), color, 1, lineType=cv2.LINE_AA)
    for x in col_edges:
        cv2.line(canvas, (int(x), 0), (int(x), canvas.shape[0] - 1), color, 1, lineType=cv2.LINE_AA)
    return canvas


def _resize_grid(grid: np.ndarray, image_shape: tuple[int, int, int], interpolation: int) -> np.ndarray:
    h, w = image_shape[:2]
    return cv2.resize(grid.astype(np.float32), (w, h), interpolation=interpolation)


def render_heatmap_overlay(image_rgb: np.ndarray, score_grid: np.ndarray) -> np.ndarray:
    resized = _resize_grid(score_grid, image_rgb.shape, cv2.INTER_LINEAR)
    resized = resized - resized.min()
    denom = float(resized.max())
    if denom > 0:
        resized = resized / denom
    heat = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return blend(image_rgb, heat, 0.45)


def render_binary_mask_overlay(image_rgb: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    resized = _resize_grid(active_mask.astype(np.float32), image_rgb.shape, cv2.INTER_NEAREST)
    overlay = image_rgb.copy()
    green = np.zeros_like(image_rgb)
    green[..., 1] = 255
    mask = resized > 0.5
    overlay[mask] = blend(image_rgb, green, 0.5)[mask]
    return overlay


def render_components_overlay(
    image_rgb: np.ndarray,
    grid_shape: tuple[int, int],
    components: List[PatchComponent],
    selected_component_id: int | None = None,
    component_labels: dict[int, str] | None = None,
) -> np.ndarray:
    canvas = image_rgb.copy()
    row_edges, col_edges = grid_edges(canvas.shape, grid_shape)

    for idx, component in enumerate(components):
        color = PALETTE[idx % len(PALETTE)]
        overlay = canvas.copy()
        for row, col in zip(component.rows, component.cols):
            y0, y1 = row_edges[int(row)], row_edges[int(row) + 1]
            x0, x1 = col_edges[int(col)], col_edges[int(col) + 1]
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, -1)
        canvas = blend(canvas, overlay, 0.35)

        cy = int((row_edges[component.bbox_row_min] + row_edges[component.bbox_row_max + 1]) / 2)
        cx = int((col_edges[component.bbox_col_min] + col_edges[component.bbox_col_max + 1]) / 2)
        cv2.putText(
            canvas,
            str(component_labels.get(component.component_id, component.component_id))
            if component_labels is not None
            else str(component.component_id),
            (max(0, cx - 10), max(15, cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        if selected_component_id is not None and component.component_id == selected_component_id:
            y0, y1 = row_edges[component.bbox_row_min], row_edges[component.bbox_row_max + 1]
            x0, x1 = col_edges[component.bbox_col_min], col_edges[component.bbox_col_max + 1]
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 3)

    return draw_patch_grid(canvas, grid_shape)


def render_selected_component_overlay(
    image_rgb: np.ndarray,
    grid_shape: tuple[int, int],
    component: PatchComponent,
    score_grid: np.ndarray,
    top_k: int,
) -> np.ndarray:
    canvas = image_rgb.copy()
    row_edges, col_edges = grid_edges(canvas.shape, grid_shape)

    overlay = canvas.copy()
    for row, col in zip(component.rows, component.cols):
        y0, y1 = row_edges[int(row)], row_edges[int(row) + 1]
        x0, x1 = col_edges[int(col)], col_edges[int(col) + 1]
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), -1)
    canvas = blend(canvas, overlay, 0.35)

    for selected_patch in select_top_k_patches(component, score_grid, top_k):
        y0, y1 = row_edges[selected_patch.row], row_edges[selected_patch.row + 1]
        x0, x1 = col_edges[selected_patch.col], col_edges[selected_patch.col + 1]
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 3)
        cv2.putText(
            canvas,
            str(selected_patch.rank_in_component),
            (x0 + 4, min(y1 - 6, y0 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    y0, y1 = row_edges[component.bbox_row_min], row_edges[component.bbox_row_max + 1]
    x0, x1 = col_edges[component.bbox_col_min], col_edges[component.bbox_col_max + 1]
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 3)
    return draw_patch_grid(canvas, grid_shape)


def render_manual_patch_overlay(
    image_rgb: np.ndarray,
    grid_shape: tuple[int, int],
    manual_patches: list[dict[str, object]] | None = None,
    selected_patches: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    canvas = image_rgb.copy()
    row_edges, col_edges = grid_edges(canvas.shape, grid_shape)

    for patch in manual_patches or []:
        row = int(patch["row"])
        col = int(patch["col"])
        label = str(patch["label"])
        color = (0, 200, 0) if label == "2D" else (220, 40, 40)
        y0, y1 = row_edges[row], row_edges[row + 1]
        x0, x1 = col_edges[col], col_edges[col + 1]
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, 3)
        cv2.putText(
            canvas,
            label,
            (x0 + 4, min(y1 - 6, y0 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    for selected_patch in selected_patches or []:
        row, col = int(selected_patch[0]), int(selected_patch[1])
        y0, y1 = row_edges[row], row_edges[row + 1]
        x0, x1 = col_edges[col], col_edges[col + 1]
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 4)

    return draw_patch_grid(canvas, grid_shape)
