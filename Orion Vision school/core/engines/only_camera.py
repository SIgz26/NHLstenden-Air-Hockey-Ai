"""Raw camera preview engine.

This engine deliberately does not apply a filter or object detection.  It is
useful when the camera feed itself needs to be checked without the vision
processing overlays.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from core.engines.base_engine import BaseEngine


class OnlyCameraEngine(BaseEngine):
    """Pass camera frames through unchanged and expose an empty mask."""

    name = "only_camera"

    def process_frame(self, frame: np.ndarray):
        """Return the untouched camera frame and an empty secondary image."""
        if frame is None or frame.size == 0:
            empty = np.zeros((0, 0), dtype=np.uint8)
            self.last_result = self.standardize_result(
                frame, 0.0, 0.0, 0.0, 0.0, empty, empty,
                {"status": "EMPTY", "in_range": False},
            )
            return frame, empty

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.last_result = self.standardize_result(
            frame, 0.0, 0.0, 0.0, 0.0, mask, mask,
            {"status": "CAMERA_ONLY", "in_range": False},
        )
        return frame, mask


def open_camera(camera_index: int = 0) -> None:
    """Open a camera directly, without starting the PyQt application UI."""
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)

    if not capture.isOpened():
        raise RuntimeError(f"Kan camera {camera_index} niet openen.")

    window_name = "Only Camera - druk op Q om te sluiten"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open alleen de camera zonder Orion UI.")
    parser.add_argument("--camera", type=int, default=0, help="Camera-index (standaard: 0).")
    args = parser.parse_args()
    open_camera(args.camera)


if __name__ == "__main__":
    main()
