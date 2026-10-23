"""Helpers for measuring and drawing camera input FPS."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class FpsOverlay:
    """Measure frame arrival rate and draw it compactly in the top-right corner."""

    def __init__(self, smoothing: float = 0.2, label: str = "Input FPS", top: int = 6) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in the interval (0, 1]")
        self.smoothing = smoothing
        self.label = label
        self.top = top
        self.fps: float | None = None
        self._last_frame_time: float | None = None
        self._lock = threading.Lock()

    def set_fps(self, fps: float | None) -> None:
        """Set an externally measured FPS value for display."""
        with self._lock:
            self.fps = fps
    def record_frame(self, now: float | None = None) -> None:
        """Update the camera input rate when a frame arrives, before processing."""
        frame_time = time.monotonic() if now is None else now
        with self._lock:
            if self._last_frame_time is not None:
                elapsed = frame_time - self._last_frame_time
                if elapsed > 0.0:
                    measured_fps = 1.0 / elapsed
                    self.fps = (
                        measured_fps
                        if self.fps is None
                        else self.smoothing * measured_fps
                        + (1.0 - self.smoothing) * self.fps
                    )
            self._last_frame_time = frame_time

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the latest pre-processing frame-arrival rate."""
        with self._lock:
            fps = self.fps
        text = f"{self.label}: --" if fps is None else f"{self.label}: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        height, width = frame.shape[:2]
        right = max(0, width - 6)
        top = self.top
        left = max(0, right - text_width - 8)
        bottom = min(height - 1, top + text_height + baseline + 6)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            frame,
            text,
            (left + 4, min(height - 1, top + text_height + 3)),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
        return frame
