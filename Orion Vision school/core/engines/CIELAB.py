from __future__ import annotations

import cv2
import numpy as np

from core.engines.base_engine import BaseEngine


class CIELABEngine(BaseEngine):
    """Puck detector using a configurable CIE L*a*b* color mask and Kalman tracking."""

    name = "cielab"
    parameter_schema = (
        {"name": "l_min", "label": "L Min", "type": "int", "min": 0, "max": 255, "default": 0},
        {"name": "l_max", "label": "L Max", "type": "int", "min": 0, "max": 255, "default": 255},
        {"name": "a_min", "label": "A Min", "type": "int", "min": 0, "max": 255, "default": 140},
        {"name": "a_max", "label": "A Max", "type": "int", "min": 0, "max": 255, "default": 255},
        {"name": "b_min", "label": "B Min", "type": "int", "min": 0, "max": 255, "default": 0},
        {"name": "b_max", "label": "B Max", "type": "int", "min": 0, "max": 255, "default": 255},
        {"name": "min_area", "label": "Min Area", "type": "int", "min": 1, "max": 5000, "default": 10},
        {"name": "max_area", "label": "Max Area", "type": "int", "min": 10, "max": 20000, "default": 2500},
        {"name": "morph_iterations", "label": "Morphology", "type": "int", "min": 0, "max": 3, "default": 1},
        {"name": "max_lost_frames", "label": "Max Lost Frames", "type": "int", "min": 0, "max": 60, "default": 12},
    )

    def __init__(self) -> None:
        super().__init__()
        self.l_min, self.l_max = 0, 255
        self.a_min, self.a_max = 140, 255
        self.b_min, self.b_max = 0, 255
        self.min_area, self.max_area = 10, 2500
        self.morph_iterations = 1
        self.max_lost_frames = 12
        self.initialized = False
        self.frames_without_detection = 0
        self._current_mask = np.zeros((0, 0), dtype=np.uint8)
        self.kf = self._create_kalman_filter()

    @staticmethod
    def _create_kalman_filter() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        return kf

    def update_settings(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, int(value))

    def _detect_position(self, frame: np.ndarray) -> tuple[int, int] | None:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lower = np.array([self.l_min, self.a_min, self.b_min], dtype=np.uint8)
        upper = np.array([self.l_max, self.a_max, self.b_max], dtype=np.uint8)
        mask = cv2.inRange(lab, lower, upper)

        if self.morph_iterations > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations
            )

        self._current_mask = mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = [
            contour
            for contour in contours
            if self.min_area <= cv2.contourArea(contour) <= self.max_area
        ]
        if not candidates:
            return None

        moments = cv2.moments(max(candidates, key=cv2.contourArea))
        if moments["m00"] <= 0:
            return None
        return (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame is None or frame.size == 0:
            empty = np.zeros((0, 0), dtype=np.uint8)
            self.last_result = self.standardize_result(
                frame, 0, 0, 0, 0, empty, empty,
                {"best": None, "in_range": False, "status": "SEARCHING"},
            )
            return frame, empty

        position = self._detect_position(frame)
        result = frame.copy()
        x = y = vx = vy = 0.0

        if position is not None:
            self.frames_without_detection = 0
            measurement = np.array(
                [[np.float32(position[0])], [np.float32(position[1])]],
                dtype=np.float32,
            )
            if not self.initialized:
                self.kf.statePost = np.array(
                    [[position[0]], [position[1]], [0.0], [0.0]],
                    dtype=np.float32,
                )
                self.initialized = True
            else:
                self.kf.correct(measurement)

            prediction = self.kf.predict()
            x, y, vx, vy = (float(value) for value in prediction.flatten())
            status, color = "TRACKING", (0, 255, 0)
            cv2.circle(result, position, 6, (0, 0, 255), 2)
        elif self.initialized and self.frames_without_detection < self.max_lost_frames:
            self.frames_without_detection += 1
            prediction = self.kf.predict()
            x, y, vx, vy = (float(value) for value in prediction.flatten())
            status, color = "PREDICTING", (0, 165, 255)
        else:
            self.initialized = False
            status, color = "SEARCHING", (0, 0, 255)

        if self.initialized:
            cv2.circle(result, (int(x), int(y)), 6, color, 2)
            cv2.arrowedLine(
                result, (int(x), int(y)),
                (int(x + vx * 5), int(y + vy * 5)),
                (0, 255, 255), 2, tipLength=0.3,
            )

        cv2.putText(
            result, f"CIELAB | {status}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        mask_bgr = cv2.cvtColor(self._current_mask, cv2.COLOR_GRAY2BGR)
        self.last_result = self.standardize_result(
            result, x, y, vx, vy,
            self._current_mask, mask_bgr,
            {
                "best": position,
                "in_range": position is not None,
                "status": status,
            },
        )
        return result, mask_bgr
