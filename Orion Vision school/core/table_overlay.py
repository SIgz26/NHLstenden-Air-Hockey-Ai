"""OpenCV drawing helpers for calibrated air-hockey table overlays."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

CornerPoints = Sequence[tuple[float, float]]


def draw_table_overlay(
    frame: np.ndarray,
    corners: CornerPoints | None,
    enabled: bool = True,
) -> np.ndarray:
    """Draw the calibrated table outline, corner dots, and corner labels.

    The frame is modified in place and returned. Corner points must be in
    source-frame pixel coordinates ordered ``TL, TR, BR, BL``. One to four
    points are accepted for live manual-calibration previews.
    """
    if not enabled or corners is None or not 1 <= len(corners) <= 4:
        return frame

    points = np.asarray(
        [[round(x), round(y)] for x, y in corners], dtype=np.int32
    ).reshape((-1, 1, 2))
    cv2.polylines(
        frame,
        [points],
        isClosed=len(corners) == 4,
        color=(255, 255, 0),
        thickness=2,
    )

    labels = ("TL", "TR", "BR", "BL")
    height, width = frame.shape[:2]
    for point, label in zip(points.reshape(4, 2), labels):
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 6, (0, 255, 255), thickness=-1)
        cv2.circle(frame, (x, y), 8, (20, 20, 20), thickness=1)
        text_x = min(max(x + 8, 0), max(0, width - 30))
        text_y = min(max(y - 8, 14), max(14, height - 2))
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return frame
