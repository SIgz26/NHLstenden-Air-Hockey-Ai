import cv2
import numpy as np

from core.engines.base_engine import BaseEngine


class BlobKalmanEngine(BaseEngine):
    """Detect a red puck blob, smooth its position, and predict its path."""

    name = "blob_kalman"
    parameter_schema = (
        {"name": "red_h_min", "label": "Red Hue Min", "type": "int", "min": 0, "max": 179, "default": 0},
        {"name": "red_h_max", "label": "Red Hue Max", "type": "int", "min": 0, "max": 179, "default": 10},
        {"name": "red_s_min", "label": "Red Sat Min", "type": "int", "min": 0, "max": 255, "default": 120},
        {"name": "red_s_max", "label": "Red Sat Max", "type": "int", "min": 0, "max": 255, "default": 255},
        {"name": "red_v_min", "label": "Red Value Min", "type": "int", "min": 0, "max": 255, "default": 70},
        {"name": "red_v_max", "label": "Red Value Max", "type": "int", "min": 0, "max": 255, "default": 255},
        {"name": "min_area", "label": "Min Blob Area", "type": "int", "min": 0, "max": 5000, "default": 1},
        {"name": "max_area", "label": "Max Blob Area", "type": "int", "min": 1, "max": 20000, "default": 2500},
        {"name": "max_lost_frames", "label": "Max Lost Frames", "type": "int", "min": 0, "max": 60, "default": 12},
    )

    _REFERENCE_WIDTH = 640
    _REFERENCE_HEIGHT = 480

    def __init__(self) -> None:
        super().__init__()
        self.red_h_min = 0
        self.red_h_max = 10
        self.red_s_min = 120
        self.red_s_max = 255
        self.red_v_min = 70
        self.red_v_max = 255
        self.min_area = 1
        self.max_area = 2500
        self.max_lost_frames = 12
        self.table_bounds = {
            "left": 50,
            "right": 590,
            "top": 50,
            "bottom": 430,
        }
        self.puck_radius = 3
        self.initialized = False
        self.frames_without_detection = 0
        self._current_mask = np.zeros((0, 0), dtype=np.uint8)
        self._table_corners: tuple[tuple[float, float], ...] | None = None
        self._image_to_table: np.ndarray | None = None
        self._table_to_image: np.ndarray | None = None
        self._calibrated_table_size: tuple[int, int] | None = None
        self.kf = self._create_kalman_filter()

    @staticmethod
    def _create_kalman_filter() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        return kf

    def detect_puck(self, frame: np.ndarray) -> tuple[int, int] | None:
        """Find the largest contour within the configured red HSV and area range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array(
            [self.red_h_min, self.red_s_min, self.red_v_min], dtype=np.uint8
        )
        upper_red = np.array(
            [self.red_h_max, self.red_s_max, self.red_v_max], dtype=np.uint8
        )
        mask = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
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

    def set_table_corners(
        self,
        corners: tuple[tuple[float, float], ...] | None,
    ) -> None:
        """Set the calibrated image-space table corners in TL, TR, BR, BL order."""
        if corners is None:
            self._table_corners = None
            self._image_to_table = None
            self._table_to_image = None
            self._calibrated_table_size = None
            return

        normalized_corners = tuple(
            (float(point[0]), float(point[1])) for point in corners
        )
        if normalized_corners == self._table_corners:
            return

        source = np.asarray(normalized_corners, dtype=np.float32)
        if (
            source.shape != (4, 2)
            or not np.isfinite(source).all()
            or abs(float(cv2.contourArea(source))) < 1.0
            or not cv2.isContourConvex(source.reshape((-1, 1, 2)))
        ):
            raise ValueError(
                "calibrated table corners must form a convex TL, TR, BR, BL quadrilateral"
            )

        top_width = float(np.linalg.norm(source[1] - source[0]))
        bottom_width = float(np.linalg.norm(source[2] - source[3]))
        right_height = float(np.linalg.norm(source[2] - source[1]))
        left_height = float(np.linalg.norm(source[3] - source[0]))
        table_width = max(2, round((top_width + bottom_width) / 2.0))
        table_height = max(2, round((right_height + left_height) / 2.0))
        destination = np.asarray(
            [
                [0, 0],
                [table_width - 1, 0],
                [table_width - 1, table_height - 1],
                [0, table_height - 1],
            ],
            dtype=np.float32,
        )

        self._image_to_table = cv2.getPerspectiveTransform(source, destination)
        self._table_to_image = cv2.getPerspectiveTransform(destination, source)
        self._table_corners = normalized_corners
        self._calibrated_table_size = (table_width, table_height)

    def predict_path(
        self,
        start_pos: tuple[float, float],
        velocity: tuple[float, float],
        steps: int = 60,
        bounds: dict[str, int] | None = None,
    ) -> list[tuple[int, int]]:
        """Predict a puck path and reflect velocity at each table boundary."""
        table = bounds or self.table_bounds
        curr_x, curr_y = start_pos
        vx, vy = velocity
        path = []

        for _ in range(steps):
            curr_x += vx
            curr_y += vy

            if curr_x <= table["left"]:
                vx *= -0.9
                curr_x = table["left"]
            elif curr_x >= table["right"]:
                vx *= -0.9
                curr_x = table["right"]

            if curr_y <= table["top"]:
                vy *= -0.9
                curr_y = table["top"]
            elif curr_y >= table["bottom"]:
                vy *= -0.9
                curr_y = table["bottom"]

            path.append((int(curr_x), int(curr_y)))

        return path

    @staticmethod
    def _transform_point(
        point: tuple[float, float],
        transform: np.ndarray,
    ) -> tuple[float, float]:
        source = np.asarray([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(source, transform)[0, 0]
        return float(transformed[0]), float(transformed[1])

    def _predict_calibrated_path(
        self,
        start_pos: tuple[float, float],
        velocity: tuple[float, float],
        steps: int,
    ) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
        if self._image_to_table is None or self._table_to_image is None:
            raise RuntimeError("table corners must be calibrated before path prediction")
        if self._calibrated_table_size is None:
            raise RuntimeError("calibrated table dimensions are unavailable")

        table_x, table_y = self._transform_point(start_pos, self._image_to_table)
        next_x, next_y = self._transform_point(
            (start_pos[0] + velocity[0], start_pos[1] + velocity[1]),
            self._image_to_table,
        )
        table_width, table_height = self._calibrated_table_size
        table_bounds = {
            "left": 0,
            "right": table_width - 1,
            "top": 0,
            "bottom": table_height - 1,
        }
        table_path = self.predict_path(
            (table_x, table_y),
            (next_x - table_x, next_y - table_y),
            steps=steps,
            bounds=table_bounds,
        )
        image_path = [
            tuple(
                round(coordinate)
                for coordinate in self._transform_point(point, self._table_to_image)
            )
            for point in table_path
        ]

        table_intercept = next(
            (point for point in table_path if point[0] >= table_bounds["right"]),
            None,
        )
        image_intercept = None
        if table_intercept is not None:
            transformed_intercept = self._transform_point(
                table_intercept,
                self._table_to_image,
            )
            image_intercept = tuple(round(value) for value in transformed_intercept)

        return image_path, image_intercept

    def _frame_table_bounds(self, width: int, height: int) -> dict[str, int]:
        x_scale = width / self._REFERENCE_WIDTH
        y_scale = height / self._REFERENCE_HEIGHT
        return {
            "left": max(0, min(width - 2, round(self.table_bounds["left"] * x_scale))),
            "right": max(1, min(width - 1, round(self.table_bounds["right"] * x_scale))),
            "top": max(0, min(height - 2, round(self.table_bounds["top"] * y_scale))),
            "bottom": max(1, min(height - 1, round(self.table_bounds["bottom"] * y_scale))),
        }

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame is None or frame.size == 0:
            empty = np.zeros((0, 0), dtype=np.uint8)
            self.last_result = self.standardize_result(
                frame,
                0.0,
                0.0,
                0.0,
                0.0,
                mask=empty,
                secondary=empty,
                metadata={"best": None, "in_range": False, "status": "SEARCHING"},
            )
            return frame, empty

        height, width = frame.shape[:2]
        bounds = self._frame_table_bounds(width, height)
        position = self.detect_puck(frame)
        mask = self._current_mask
        result = frame.copy()
        x = y = vx = vy = 0.0
        best = position

        if position is not None:
            self.frames_without_detection = 0
            if not self.initialized:
                self.kf.statePost = np.array(
                    [[position[0]], [position[1]], [0.0], [0.0]],
                    dtype=np.float32,
                )
                self.kf.statePre = self.kf.statePost.copy()
                self.initialized = True
            else:
                measurement = np.array(
                    [[np.float32(position[0])], [np.float32(position[1])]],
                    dtype=np.float32,
                )
                self.kf.correct(measurement)
            prediction = self.kf.predict()
            x, y, vx, vy = (float(value) for value in prediction.flatten())
            status = "TRACKING"
            color = (0, 255, 0)
            cv2.circle(result, position, self.puck_radius * 2, (0, 0, 255), 2)
        elif self.initialized:
            self.frames_without_detection += 1
            if self.frames_without_detection <= self.max_lost_frames:
                prediction = self.kf.predict()
                x, y, vx, vy = (float(value) for value in prediction.flatten())
                status = "PREDICTING (OCCLUSION)"
                color = (0, 165, 255)
            else:
                self.initialized = False
                status = "LOST"
                color = (0, 0, 255)
        else:
            status = "SEARCHING"
            color = (0, 0, 255)

        future_path = []
        intercept = None
        if self.initialized:
            if self._table_corners is not None:
                future_path, intercept = self._predict_calibrated_path(
                    (x, y),
                    (vx, vy),
                    steps=90,
                )
            else:
                future_path = self.predict_path(
                    (x, y), (vx, vy), steps=90, bounds=bounds
                )
                intercept = next(
                    (point for point in future_path if point[0] >= bounds["right"]),
                    None,
                )
            for start, end in zip(future_path, future_path[1:]):
                cv2.line(result, start, end, (0, 255, 255), 1, cv2.LINE_AA)

            if intercept is not None:
                cv2.drawMarker(
                    result,
                    (bounds["right"], intercept[1]),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )
                cv2.putText(
                    result,
                    "INTERCEPT",
                    (max(0, bounds["right"] - 70), max(15, intercept[1] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
            cv2.circle(result, (int(x), int(y)), 6, color, 2)
            cv2.arrowedLine(
                result,
                (int(x), int(y)),
                (int(x + vx * 5), int(y + vy * 5)),
                (0, 255, 255),
                2,
                tipLength=0.3,
            )

        cv2.putText(
            result,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        if self._table_corners is None:
            cv2.rectangle(
                result,
                (bounds["left"], bounds["top"]),
                (bounds["right"], bounds["bottom"]),
                (255, 255, 255),
                1,
            )
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.last_result = self.standardize_result(
            result,
            x,
            y,
            vx,
            vy,
            mask=mask,
            secondary=mask_bgr,
            metadata={
                "best": best,
                "in_range": position is not None,
                "status": status,
                "future_path": future_path,
                "intercept": intercept,
            },
        )
        return result, mask_bgr
