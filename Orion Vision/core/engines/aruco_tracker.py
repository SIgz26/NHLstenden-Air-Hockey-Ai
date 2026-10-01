"""Track robot and opponent mallets using uniquely identified ArUco markers."""

from __future__ import annotations

from typing import TypeAlias

import cv2
import numpy as np

MalletPosition: TypeAlias = tuple[float, float] | None


class ArUcoMalletTracker:
    """Detect robot marker ID 0 and opponent marker ID 1.

    Marker centers are returned in input-frame pixel coordinates. Unseen
    markers return ``None`` for that position; callers may retain their last
    known coordinates if desired.
    """

    ROBOT_MARKER_ID = 0
    OPPONENT_MARKER_ID = 1

    def __init__(self) -> None:
        """Create an OpenCV ArUco detector using the 4x4/50 dictionary."""
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
            raise RuntimeError(
                "ArUcoDetector is unavailable; install an OpenCV build with the aruco module."
            )
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect_mallets(self, frame: np.ndarray) -> tuple[MalletPosition, MalletPosition]:
        """Return ``(robot_position, opponent_position)`` for one image.

        Each detected position is the mean of the marker's four corner points.
        The method accepts grayscale, BGR, or BGRA frames.
        """
        if frame is None or frame.size == 0:
            return None, None
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("frame must be grayscale, BGR, or BGRA")

        corners, ids, _rejected = self.detector.detectMarkers(gray)
        if ids is None or not corners:
            return None, None

        found: dict[int, tuple[float, float]] = {}
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            marker_id_int = int(marker_id)
            if marker_id_int not in (self.ROBOT_MARKER_ID, self.OPPONENT_MARKER_ID):
                continue
            center = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2).mean(axis=0)
            found[marker_id_int] = (float(center[0]), float(center[1]))

        return (
            found.get(self.ROBOT_MARKER_ID),
            found.get(self.OPPONENT_MARKER_ID),
        )
