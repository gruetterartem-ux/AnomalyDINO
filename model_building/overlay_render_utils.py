from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sanitize_text(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def group_name_from_sample(sample: str) -> str:
    sample_path = Path(sample)
    parts = sample_path.parts
    if len(parts) >= 2:
        return parts[-2]
    if len(parts) == 1:
        return parts[0]
    return "ungrouped"


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_x1 = max(0, x)
    box_y2 = max(text_h + baseline + 4, y)
    box_y1 = max(0, box_y2 - text_h - baseline - 8)
    box_x2 = min(image.shape[1] - 1, box_x1 + text_w + 8)
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(
        image,
        text,
        (box_x1 + 4, box_y2 - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
