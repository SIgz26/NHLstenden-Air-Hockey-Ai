"""Fast contour based puck tracker with Kalman prediction and wall bounces.

The tracker is deliberately independent from colour segmentation.  Callers
that already have contours can use :meth:`track_contours`; ``process_frame``
only supplies a small HSV fallback for the engine interface.
"""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

from core.engines.base_engine import BaseEngine


class AdvancedPuckTracker(BaseEngine):
    """Track one puck using contour quality, distance and Kalman prediction."""

    name = "advanced_puck_tracker"
    parameter_schema = (
        {"name": "slider_min_area", "label": "Min area (px)", "type": "int", "min": 0, "max": 20000, "default": 20},
        {"name": "slider_max_area", "label": "Max area (px)", "type": "int", "min": 1, "max": 50000, "default": 2500},
        {"name": "slider_min_circularity", "label": "Min circularity (x100)", "type": "int", "min": 0, "max": 100, "default": 55},
        {"name": "slider_pixels_per_cm", "label": "Pixels per cm", "type": "int", "min": 1, "max": 500, "default": 10},
        {"name": "slider_max_dist_cm", "label": "Max distance (cm)", "type": "int", "min": 1, "max": 100, "default": 5},
        {"name": "slider_max_lost_frames", "label": "Max lost frames", "type": "int", "min": 0, "max": 120, "default": 12},
    )

    def __init__(self) -> None:
        super().__init__()
        self.slider_min_area = 20
        self.slider_max_area = 2500
        self.slider_min_circularity = 55
        self.slider_pixels_per_cm = 10
        self.slider_max_dist_cm = 5
        self.slider_max_lost_frames = 12

        # Image-space limits.  None means the current frame extent.
        self.x_min: float | None = None
        self.x_max: float | None = None
        self.y_min: float | None = None
        self.y_max: float | None = None

        self.red_h_min, self.red_h_max = 0, 10
        self.red_h2_min, self.red_h2_max = 170, 179
        self.red_s_min, self.red_v_min = 80, 40
        self.initialized = False
        self.status = "UNINITIALIZED"
        self.frames_without_detection = 0
        self._last_shape: tuple[int, int] | None = None
        self._mask = np.zeros((0, 0), dtype=np.uint8)
        self._table_corners: tuple[tuple[float, float], ...] | None = None
        self._image_to_table: np.ndarray | None = None
        self._table_to_image: np.ndarray | None = None
        self._table_size: tuple[int, int] | None = None
        self.kalman = self._new_kalman()

    @staticmethod
    def _new_kalman() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        return kf

    def reset(self) -> None:
        self.kalman = self._new_kalman()
        self.initialized = False
        self.frames_without_detection = 0
        self.status = "UNINITIALIZED"

    def _bounds(self, shape: tuple[int, int]) -> tuple[float, float, float, float]:
        height, width = shape
        return (
            0.0 if self.x_min is None else float(self.x_min),
            float(width - 1) if self.x_max is None else float(self.x_max),
            0.0 if self.y_min is None else float(self.y_min),
            float(height - 1) if self.y_max is None else float(self.y_max),
        )

    def set_table_corners(self, corners: tuple[tuple[float, float], ...] | None) -> None:
        """Set table corners in TL, TR, BR, BL order for path prediction."""
        if corners is None or len(corners) == 0:
            self._table_corners = None
            self._image_to_table = None
            self._table_to_image = None
            self._table_size = None
            return

        normalized = tuple((float(point[0]), float(point[1])) for point in corners)
        if normalized == self._table_corners:
            return
        source = np.asarray(normalized, dtype=np.float32)
        if (
            source.shape != (4, 2)
            or not np.isfinite(source).all()
            or abs(float(cv2.contourArea(source))) < 1.0
            or not cv2.isContourConvex(source.reshape((-1, 1, 2)))
        ):
            raise ValueError("table corners must be a convex TL, TR, BR, BL quadrilateral")

        width = max(2, round((np.linalg.norm(source[1] - source[0]) + np.linalg.norm(source[2] - source[3])) / 2.0))
        height = max(2, round((np.linalg.norm(source[2] - source[1]) + np.linalg.norm(source[3] - source[0])) / 2.0))
        destination = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        self._image_to_table = cv2.getPerspectiveTransform(source, destination)
        self._table_to_image = cv2.getPerspectiveTransform(destination, source)
        self._table_corners = normalized
        self._table_size = (width, height)

    @staticmethod
    def _transform_point(point: tuple[float, float], transform: np.ndarray) -> tuple[float, float]:
        source = np.asarray([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(source, transform)[0, 0]
        return float(transformed[0]), float(transformed[1])

    def _predict_with_bounce(self, shape: tuple[int, int]) -> np.ndarray:
        prediction = self.kalman.predict()
        x, y, vx, vy = (float(value) for value in prediction[:, 0])
        x_min, x_max, y_min, y_max = self._bounds(shape)

        # Reflect once per wall crossing and clamp, keeping the corrected
        # state in both OpenCV buffers so the next correct() uses it.
        if x < x_min or x > x_max:
            x = min(max(x, x_min), x_max)
            vx = -vx
        if y < y_min or y > y_max:
            y = min(max(y, y_min), y_max)
            vy = -vy
        state = np.array([[x], [y], [vx], [vy]], dtype=np.float32)
        self.kalman.statePre = state.copy()
        self.kalman.statePost = state.copy()
        return state

    def _shape_candidates(self, contours: Iterable[np.ndarray]) -> tuple[list[dict], list[np.ndarray]]:
        """Apply only area/circularity filtering; no motion gate yet."""
        candidates, rejected = [], []
        limit = float(self.slider_min_circularity)
        limit = limit / 100.0 if limit > 1.0 else limit
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 0.0 else 4.0 * math.pi * area / (perimeter * perimeter)
            moments = cv2.moments(contour)
            if (area < float(self.slider_min_area) or area > float(self.slider_max_area)
                    or circularity < limit or moments["m00"] <= 0.0):
                rejected.append(contour)
                continue
            candidates.append({
                "contour": contour,
                "x": float(moments["m10"] / moments["m00"]),
                "y": float(moments["m01"] / moments["m00"]),
                "area": area,
                "circularity": circularity,
            })
        return candidates, rejected

    def _select_candidate(self, candidates: list[dict], prediction: np.ndarray, allow_any_distance: bool):
        """Apply the distance gate only after shape candidates exist."""
        max_dist_px = float(self.slider_max_dist_cm) * max(1.0, float(self.slider_pixels_per_cm))
        best, rejected, best_score = None, [], -math.inf
        for candidate in candidates:
            distance = math.hypot(candidate["x"] - float(prediction[0, 0]), candidate["y"] - float(prediction[1, 0]))
            if not allow_any_distance and distance > max_dist_px:
                rejected.append(candidate["contour"])
                continue
            score = candidate["circularity"] - (0.001 * distance if not allow_any_distance else 0.0)
            if score > best_score:
                if best is not None:
                    rejected.append(best["contour"])
                best, best_score = candidate, score
            else:
                rejected.append(candidate["contour"])
        return best, rejected

    def predict_path(
        self,
        start_pos: tuple[float, float],
        velocity: tuple[float, float],
        steps: int = 90,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> list[tuple[int, int]]:
        """Same forward path projection used by BlobKalmanEngine."""
        x_min, x_max, y_min, y_max = bounds or self._bounds(self._last_shape or (1, 1))
        x, y = start_pos
        vx, vy = velocity
        path: list[tuple[int, int]] = []
        for _ in range(steps):
            x += vx
            y += vy
            if x <= x_min:
                x, vx = x_min, -vx * 0.9
            elif x >= x_max:
                x, vx = x_max, -vx * 0.9
            if y <= y_min:
                y, vy = y_min, -vy * 0.9
            elif y >= y_max:
                y, vy = y_max, -vy * 0.9
            path.append((int(x), int(y)))
        return path

    def _predict_display_path(
        self,
        start_pos: tuple[float, float],
        velocity: tuple[float, float],
        steps: int = 90,
    ) -> list[tuple[int, int]]:
        """Predict in table coordinates when calibrated, then map to image."""
        if self._image_to_table is None or self._table_to_image is None or self._table_size is None:
            return self.predict_path(start_pos, velocity, steps)

        table_start = self._transform_point(start_pos, self._image_to_table)
        table_next = self._transform_point(
            (start_pos[0] + velocity[0], start_pos[1] + velocity[1]),
            self._image_to_table,
        )
        width, height = self._table_size
        table_path = self.predict_path(
            table_start,
            (table_next[0] - table_start[0], table_next[1] - table_start[1]),
            steps,
            bounds=(0.0, float(width - 1), 0.0, float(height - 1)),
        )
        return [
            tuple(round(value) for value in self._transform_point(point, self._table_to_image))
            for point in table_path
        ]

    def track_contours(
        self,
        contours: Iterable[np.ndarray],
        frame_shape: tuple[int, ...],
        result: np.ndarray | None = None,
    ) -> dict:
        """Consume contours produced elsewhere and return tracker telemetry."""
        shape = (int(frame_shape[0]), int(frame_shape[1]))
        self._last_shape = shape
        # Exactly one prediction per frame.  LOST/UNINITIALIZED still run
        # predict, but their prediction is not used as a search restriction.
        prediction = self._predict_with_bounce(shape)
        shape_candidates, rejected = self._shape_candidates(contours)
        candidate, distance_rejected = self._select_candidate(
            shape_candidates,
            prediction,
            allow_any_distance=self.status in {"LOST", "UNINITIALIZED"},
        )
        rejected.extend(distance_rejected)
        if candidate is not None:
            x = candidate["x"]
            y = candidate["y"]
            if not self.initialized:
                self.kalman.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
                self.kalman.statePre = self.kalman.statePost.copy()
            measurement = np.array([[x], [y]], dtype=np.float32)
            self.kalman.correct(measurement)
            # Match BlobKalmanEngine: correct with the measurement, then
            # predict the next position and expose that prediction.
            state = self._predict_with_bounce(shape)
            self.initialized = True
            self.frames_without_detection = 0
            status = "TRACKING"
        elif self.initialized:
            self.frames_without_detection += 1
            state = prediction
            status = "PREDICTING"
            if self.frames_without_detection > self.slider_max_lost_frames:
                self.initialized = False
                self.status = "LOST"
                status = "LOST"
        else:
            self.frames_without_detection += 1
            state = prediction
            status = self.status
        self.status = status

        return {
            "x": float(state[0, 0]), "y": float(state[1, 0]),
            "vx": float(state[2, 0]), "vy": float(state[3, 0]),
            "candidate": candidate, "prediction": (float(state[0, 0]), float(state[1, 0])),
            "rejected_contours": rejected, "status": status,
            "detected": candidate is not None, "lost_frames": self.frames_without_detection,
            "candidate_count": len(shape_candidates), "contour_count": len(contours),
        }

    @staticmethod
    def _draw_debug(result: np.ndarray, telemetry: dict) -> None:
        for contour in telemetry["rejected_contours"]:
            cv2.drawContours(result, [contour], -1, (0, 0, 255), 1, cv2.LINE_AA)
        px, py = (round(value) for value in telemetry["prediction"])
        cv2.drawMarker(result, (px, py), (255, 0, 0), cv2.MARKER_CROSS, 14, 2)
        future_path = telemetry.get("future_path", [])
        for start, end in zip(future_path, future_path[1:]):
            cv2.line(result, start, end, (0, 255, 255), 1, cv2.LINE_AA)
        candidate = telemetry["candidate"]
        if candidate is not None:
            (cx, cy), radius = cv2.minEnclosingCircle(candidate["contour"])
            center = (round(cx), round(cy))
            cv2.circle(result, center, max(4, round(radius)), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(result, center, 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            result,
            f"Status: {telemetry['status']} | Lost Frames: {telemetry['lost_frames']} | Candidates: {telemetry['candidate_count']}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def process_mask(self, mask: np.ndarray, result: np.ndarray | None = None) -> dict:
        """Apply the required 3x3 opening and track its external contours."""
        self._mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(self._mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        telemetry = self.track_contours(contours, self._mask.shape, result)
        telemetry["future_path"] = self._predict_display_path(
            (telemetry["x"], telemetry["y"]),
            (telemetry["vx"], telemetry["vy"]),
        ) if telemetry["status"] in {"TRACKING", "PREDICTING"} else []
        if result is not None:
            self._draw_debug(result, telemetry)
        return telemetry

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame is None or frame.size == 0:
            empty = np.zeros((0, 0), dtype=np.uint8)
            self.last_result = self.standardize_result(frame, 0, 0, 0, 0, empty, empty, {"status": "SEARCHING"})
            return frame, empty
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([self.red_h_min, self.red_s_min, self.red_v_min], np.uint8)
        upper1 = np.array([self.red_h_max, 255, 255], np.uint8)
        lower2 = np.array([self.red_h2_min, self.red_s_min, self.red_v_min], np.uint8)
        upper2 = np.array([self.red_h2_max, 255, 255], np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        telemetry = self.process_mask(mask, frame)
        output = frame.copy()
        self._draw_debug(output, telemetry)
        mask_bgr = cv2.cvtColor(self._mask, cv2.COLOR_GRAY2BGR)
        self.last_result = self.standardize_result(output, telemetry["x"], telemetry["y"], telemetry["vx"], telemetry["vy"], self._mask, mask_bgr, telemetry)
        return output, mask_bgr


# Convenient names for callers that refer to the file as the previous tracker.
PrevEngine = AdvancedPuckTracker
PuckTracker = AdvancedPuckTracker
