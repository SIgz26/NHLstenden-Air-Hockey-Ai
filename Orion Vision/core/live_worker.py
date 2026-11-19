import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QReadWriteLock
from PyQt5.QtGui import QImage, QPixmap

from core.engines.hsv_engine import HsvEngine
from core.engines.hough_circle_engine import HoughCircleEngine
from core.engines.bg_subtraction_engine import BackgroundSubtractionEngine
from core.engines.circle_contour_engine import CircleContourEngine
from core.engines.hybrid_engine import HybridEngine

from core.engines.aruco_tracker import ArUcoMalletTracker
from core.sim_bridge import SimBridge
from core.table_overlay import draw_table_overlay


class LiveCameraWorker(QThread):
    """
    Background thread that captures camera frames, applies HSV-based
    blob detection with Kalman filtering, and emits throttled preview frames.
    """

    frame_processed = pyqtSignal(object, object)  # BGR result and mask arrays for UI conversion
    status_signal = pyqtSignal(str)
    cycle_metrics_updated = pyqtSignal(float, float, float)

    # Tuning constants – centralised so they're easy to adjust
    MAX_CAMERA_FPS = 90.0
    MAX_PREVIEW_FPS = 30.0
    CALIBRATION_MARKER_ID = 0
    CALIBRATION_MARKER_INTERVAL_S = 0.1
    CALIBRATION_MARKER_MAX_AGE_S = 0.15
    CALIBRATION_MAX_ROBOT_SPEED_MM_S = 30.0
    # V4L2 exposure units are 100 us; DirectShow uses log2 seconds.
    SHORT_EXPOSURE_VALUE = 80.0 if sys.platform.startswith("linux") else -7.0
    PAUSE_SLEEP_MS = 50
    SCORE_DISTANCE_WEIGHT = 0.35
    SCORE_CIRCULARITY_WEIGHT = 120
    MIN_CIRCULARITY = 0.55
    KALMAN_PROCESS_NOISE = 0.08
    KALMAN_MEASUREMENT_NOISE = 1.0

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.paused = False
        self.recording = False
        self.record_output_dir = None
        self.record_path = None
        self.record_writer = None
        self.target_fps = self.MAX_CAMERA_FPS
        self.record_fps = 20.0
        self._fps = 0.0
        self._frame_counter = 0
        self._fps_last_update = time.monotonic()
        self._last_capture_time = time.monotonic()
        self._last_preview_time = 0.0
        self._frame_timestamps: deque[float] = deque(maxlen=30)
        self._capture_times_ms: deque[float] = deque(maxlen=60)
        self._vision_times_ms: deque[float] = deque(maxlen=60)
        self._capture_ms_avg = 0.0
        self._vision_ms_avg = 0.0
        self._capture_fps = 0.0

        # Separate lock for the HSV frame so reads in the GUI thread
        # do not block camera captures longer than necessary.
        self._hsv_frame_lock = QReadWriteLock()
        self._current_hsv_frame = None

        # General mutex for settings & state flags
        self._mutex = QMutex()

        # HSV parameters (default: orange puck-like values)
        self.h_min, self.h_max = 35, 95
        self.s_min, self.s_max = 20, 255
        self.v_min, self.v_max = 40, 255
        self.min_area = 40
        self.max_area = 2500

        self.engine_name = "hsv"
        self.engine = self._build_engine(self.engine_name)
        self.sim_bridge = SimBridge()
        self.aruco_tracker = ArUcoMalletTracker()
        self._aruco_dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        if hasattr(cv2.aruco, "ArucoDetector"):
            parameters = cv2.aruco.DetectorParameters()
            self._calibration_marker_detector = cv2.aruco.ArucoDetector(
                self._aruco_dictionary, parameters
            )
            self._calibration_marker_parameters = None
        else:
            self._calibration_marker_detector = None
            if hasattr(cv2.aruco, "DetectorParameters"):
                self._calibration_marker_parameters = (
                    cv2.aruco.DetectorParameters()
                )
            else:
                self._calibration_marker_parameters = (
                    cv2.aruco.DetectorParameters_create()
                )
        self._last_calibration_marker_scan = 0.0
        self._last_calibration_marker_corners: np.ndarray | None = None
        self._last_calibration_marker_time = 0.0
        self.show_table_overlay = True
        self._table_calibration_preview: tuple[tuple[float, float], ...] = ()

        # Fisheye correction: applied before HSV detection so the live feed
        # and the processed result are both corrected in one pipeline step.
        self.fisheye_enabled = True
        self.fisheye_k1 = -0.40
        self.fisheye_k2 = 0.05
        self._fisheye_map1 = None
        self._fisheye_map2 = None
        self._fisheye_size = None

        self._init_kalman()
        self.initialized = False
        self.last_px: int | None = None
        self.last_py: int | None = None
        self.frames_without_detection = 0
        self.MAX_LOST_FRAMES = 12

    # ------------------------------------------------------------------
    # Kalman filter
    # ------------------------------------------------------------------

    def _init_kalman(self) -> None:
        """Create and configure a 4-state / 2-measurement Kalman filter."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * self.KALMAN_PROCESS_NOISE
        )
        kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * self.KALMAN_MEASUREMENT_NOISE
        )
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kalman = kf

    # ------------------------------------------------------------------
    # Public interface (called from the GUI thread)
    # ------------------------------------------------------------------

    @property
    def current_hsv_frame(self) -> np.ndarray | None:
        """Thread-safe read of the latest HSV frame."""
        self._hsv_frame_lock.lockForRead()
        try:
            frame = self._current_hsv_frame
            return frame.copy() if frame is not None else None
        finally:
            self._hsv_frame_lock.unlock()

    @staticmethod
    def get_available_cameras(max_checks: int = 5) -> list[int]:
        """Return indices of cameras that can deliver at least one frame."""
        available: list[int] = []
        if sys.platform.startswith("linux"):
            camera_indices = sorted(
                {
                    int(device.name.removeprefix("video"))
                    for device in Path("/dev").glob("video[0-9]*")
                    if device.name.removeprefix("video").isdigit()
                }
            )[:max_checks]
        else:
            camera_indices = list(range(max_checks))

        # Suppress noisy OpenCV backend messages during probing
        old_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)

        backend = LiveCameraWorker._camera_backend()
        for i in camera_indices:
            cap = cv2.VideoCapture(i, backend)
            try:
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
            finally:
                cap.release()

        if hasattr(cv2, "setLogLevel") and old_level is not None:
            cv2.setLogLevel(old_level)

        return available

    @staticmethod
    def _camera_backend() -> int:
        if os.name == "nt":
            return cv2.CAP_DSHOW
        if sys.platform.startswith("linux"):
            return cv2.CAP_V4L2
        return cv2.CAP_ANY

    def _normalize_engine_name(self, engine_name: str) -> str:
        raw = str(engine_name).strip().lower()
        normalized = raw.replace("-", "_").replace(" ", "_")
        aliases = {
            "hsv": "hsv",
            "hsv_detection": "hsv",
            "hough_circle": "hough_circle",
            "hough_circle_detection": "hough_circle",
            "hough": "hough_circle",
            "background_subtraction": "bg_subtraction",
            "background_subtraction_detection": "bg_subtraction",
            "circle_contour": "circle_contour",
            "circle_contour_detection": "circle_contour",
            "hybrid": "hybrid",
            "hybrid_detection": "hybrid",
        }
        return aliases.get(normalized, normalized)

    def _build_engine(self, engine_name: str):
        normalized = self._normalize_engine_name(engine_name)
        mapping = {
            "hsv": HsvEngine,
            "hough_circle": HoughCircleEngine,
            "bg_subtraction": BackgroundSubtractionEngine,
            "circle_contour": CircleContourEngine,
            "hybrid": HybridEngine,
        }
        engine_cls = mapping.get(normalized, HsvEngine)
        return engine_cls()

    def set_engine(self, engine_name: str) -> None:
        normalized = self._normalize_engine_name(engine_name)
        self.engine_name = normalized
        self.engine = self._build_engine(normalized)
        self.engine.update_settings(
            h_min=self.h_min,
            h_max=self.h_max,
            s_min=self.s_min,
            s_max=self.s_max,
            v_min=self.v_min,
            v_max=self.v_max,
        )

    def set_fisheye_correction(self, enabled: bool, k1: float | None = None, k2: float | None = None) -> None:
        with QMutexLocker(self._mutex):
            self.fisheye_enabled = bool(enabled)
            if k1 is not None:
                self.fisheye_k1 = float(k1)
            if k2 is not None:
                self.fisheye_k2 = float(k2)

    def set_show_table_overlay(self, enabled: bool) -> None:
        """Enable or disable calibrated table corners on the displayed frame."""
        self.show_table_overlay = bool(enabled)

    set_table_overlay_enabled = set_show_table_overlay

    def set_table_calibration_preview(
        self, corners: tuple[tuple[float, float], ...]
    ) -> None:
        """Update partial manual corner points displayed over the camera feed."""
        self._table_calibration_preview = tuple(corners[:4])

    def _refresh_fisheye_maps(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._fisheye_size == (w, h) and self._fisheye_map1 is not None and self._fisheye_map2 is not None:
            return

        fx = fy = w * 0.9
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.array([self.fisheye_k1, self.fisheye_k2, 0.0, 0.0, 0.0], dtype=np.float64)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        self._fisheye_map1, self._fisheye_map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )
        self._fisheye_size = (w, h)

    def _apply_fisheye_correction(self, frame: np.ndarray) -> np.ndarray:
        if not self.fisheye_enabled:
            return frame

        self._refresh_fisheye_maps(frame)
        return cv2.remap(frame, self._fisheye_map1, self._fisheye_map2, cv2.INTER_LINEAR)

    def set_camera_index(self, index: int) -> None:
        with QMutexLocker(self._mutex):
            self.camera_index = index

    def update_hsv(
        self,
        h_min: int, h_max: int,
        s_min: int, s_max: int,
        v_min: int = 20, v_max: int = 255,
    ) -> None:
        with QMutexLocker(self._mutex):
            self.h_min, self.h_max = h_min, h_max
            self.s_min, self.s_max = s_min, s_max
            self.v_min, self.v_max = v_min, v_max

    def toggle_pause(self) -> bool:
        with QMutexLocker(self._mutex):
            self.paused = not self.paused
            return self.paused

    def start_recording(self, output_dir: str) -> str | None:
        if not output_dir:
            return None

        os.makedirs(output_dir, exist_ok=True)
        self.record_output_dir = output_dir
        self.recording = True
        self.record_path = None
        self.record_writer = None
        return output_dir

    def stop_recording(self) -> str | None:
        if self.record_writer is not None:
            self.record_writer.release()
            self.record_writer = None

        path = self.record_path
        self.recording = False
        self.record_path = None
        self.record_output_dir = None

        if path:
            self.status_signal.emit(f"Opname opgeslagen: {path}")
        return path

    def set_target_fps(self, fps: float | int) -> None:
        with QMutexLocker(self._mutex):
            self.target_fps = max(1.0, min(float(fps), self.MAX_CAMERA_FPS))

    def set_record_fps(self, fps: float | int) -> None:
        with QMutexLocker(self._mutex):
            self.record_fps = max(1.0, min(float(fps), 60.0))

    def stop(self) -> None:
        self.running = False
        self.stop_recording()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, self._camera_backend())

        if not cap.isOpened():
            self.status_signal.emit(
                f"Error: Cannot open camera {self.camera_index}."
            )
            return

        self.status_signal.emit(
            f"Camera {self.camera_index} connected & active"
        )

        with QMutexLocker(self._mutex):
            target_fps = self.target_fps
        fps_supported = cap.set(cv2.CAP_PROP_FPS, target_fps)
        exposure_supported = False
        manual_exposure = True
        if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
            manual_exposure = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if manual_exposure and hasattr(cv2, "CAP_PROP_EXPOSURE"):
            exposure_supported = cap.set(
                cv2.CAP_PROP_EXPOSURE, self.SHORT_EXPOSURE_VALUE
            )
        self.status_signal.emit(
            f"Camera target {target_fps:.0f} FPS "
            f"({'accepted' if fps_supported else 'not confirmed'}), "
            f"short exposure "
            f"{'accepted' if exposure_supported else 'unsupported/not confirmed'}"
        )

        while self.running:
            # --- Pause guard ---
            with QMutexLocker(self._mutex):
                is_paused = self.paused

            if is_paused:
                self.msleep(self.PAUSE_SLEEP_MS)
                continue

            with QMutexLocker(self._mutex):
                target_fps = self.target_fps
            target_interval = 1.0 / target_fps if target_fps > 0 else 0.0
            if target_interval > 0:
                now = time.monotonic()
                elapsed = now - self._last_capture_time
                if elapsed < target_interval:
                    self.usleep(max(1, int((target_interval - elapsed) * 1_000_000)))
                    continue
                self._last_capture_time = now

            # --- Capture ---
            capture_start = time.perf_counter()
            ret, frame = cap.read()
            capture_elapsed_ms = (time.perf_counter() - capture_start) * 1000.0
            if not ret:
                self.status_signal.emit("Error: No frame received from camera")
                self.msleep(100)
                continue

            captured_at = time.monotonic()
            self._frame_timestamps.append(captured_at)
            if len(self._frame_timestamps) > 1:
                capture_span = self._frame_timestamps[-1] - self._frame_timestamps[0]
                if capture_span > 0.0:
                    self._capture_fps = (
                        len(self._frame_timestamps) - 1
                    ) / capture_span

            vision_start = time.perf_counter()
            # --- Pre-processing ---
            frame = self._apply_fisheye_correction(frame)
            robot_position, opponent_position = self.aruco_tracker.detect_mallets(frame)
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Store HSV frame for the colour-picker (write lock)
            self._hsv_frame_lock.lockForWrite()
            self._current_hsv_frame = hsv
            self._hsv_frame_lock.unlock()

            # Read HSV bounds under the general mutex
            with QMutexLocker(self._mutex):
                lower = np.array([self.h_min, self.s_min, self.v_min])
                upper = np.array([self.h_max, self.s_max, self.v_max])

            # --- Mask ---
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # --- Detection ---
            result = frame.copy()
            engine_result = None
            if hasattr(self, "engine") and self.engine is not None:
                try:
                    result, mask = self.engine.process_frame(frame)
                    engine_result = self.engine.last_result
                except Exception:
                    result = frame.copy()
                    self._detect_and_draw(mask, result)
            else:
                self._detect_and_draw(mask, result)

            self._update_calibration_marker(frame, result)
            if self.engine is not None:
                self.sim_bridge.update_from_engine(
                    engine_result,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    frame_rate=self._capture_fps or target_fps,
                )
            self.sim_bridge.update_telemetry(
                robot_position=robot_position,
                opponent_position=opponent_position,
            )
            draw_table_overlay(
                result,
                self._table_calibration_preview or self.sim_bridge.get_table_corners(),
                enabled=self.show_table_overlay or bool(self._table_calibration_preview),
            )
            vision_elapsed_ms = (time.perf_counter() - vision_start) * 1000.0
            self._capture_times_ms.append(capture_elapsed_ms)
            self._vision_times_ms.append(vision_elapsed_ms)
            self._capture_ms_avg = float(np.mean(self._capture_times_ms))
            self._vision_ms_avg = float(np.mean(self._vision_times_ms))

            if self.recording:
                if self.record_writer is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.record_path = os.path.join(
                        self.record_output_dir,
                        f"orion_live_{timestamp}.mp4",
                    )
                    self.record_writer = cv2.VideoWriter(
                        self.record_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.record_fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not self.record_writer.isOpened():
                        self.status_signal.emit(
                            "Fout: opname kon niet worden gestart. Controleer de map.")
                        self.record_writer = None
                        self.record_path = None
                        self.recording = False

                if self.record_writer is not None:
                    self.record_writer.write(frame)

            self._frame_counter += 1
            now = time.monotonic()
            if now - self._fps_last_update >= 0.5:
                elapsed = max(now - self._fps_last_update, 0.001)
                self._fps = self._frame_counter / elapsed
                self._frame_counter = 0
                self._fps_last_update = now
                self.status_signal.emit(
                    f"Vision {self._fps:.1f} FPS | capture avg "
                    f"{self._capture_ms_avg:.2f} ms | processing avg "
                    f"{self._vision_ms_avg:.2f} ms"
                )
                self.cycle_metrics_updated.emit(
                    self._capture_ms_avg, self._vision_ms_avg, self._fps
                )

            preview_interval = 1.0 / self.MAX_PREVIEW_FPS
            if now - self._last_preview_time >= preview_interval:
                if len(mask.shape) == 2 or mask.shape[2] == 1:
                    mask_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                else:
                    mask_preview = mask
                self.frame_processed.emit(result.copy(), mask_preview.copy())
                self._last_preview_time = now

        cap.release()
        self.status_signal.emit("Camera stopped")

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _update_calibration_marker(
        self, frame: np.ndarray, result: np.ndarray
    ) -> None:
        """Detect and display robot marker ID 0 at no more than 10 Hz."""
        now = time.monotonic()
        robot_speed = self.sim_bridge.plc_robot_speed_mm_s
        can_scan = (
            robot_speed is not None
            and robot_speed < self.CALIBRATION_MAX_ROBOT_SPEED_MM_S
            and now - self._last_calibration_marker_scan
            >= self.CALIBRATION_MARKER_INTERVAL_S
        )
        if can_scan:
            self._last_calibration_marker_scan = now
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._calibration_marker_detector is not None:
                corners, ids, _ = (
                    self._calibration_marker_detector.detectMarkers(gray)
                )
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    self._aruco_dictionary,
                    parameters=self._calibration_marker_parameters,
                )

            marker_corners = None
            if ids is not None:
                for detected_corners, marker_id in zip(
                    corners, ids.reshape(-1)
                ):
                    if int(marker_id) == self.CALIBRATION_MARKER_ID:
                        marker_corners = np.asarray(
                            detected_corners, dtype=np.float32
                        ).reshape(4, 2)
                        break

            self._last_calibration_marker_corners = marker_corners
            if marker_corners is not None:
                self._last_calibration_marker_time = now
                center = marker_corners.mean(axis=0)
                self.sim_bridge.update_mallet_bias(
                    float(center[0]),
                    float(center[1]),
                    max_speed_mm_s=self.CALIBRATION_MAX_ROBOT_SPEED_MM_S,
                )

        marker_corners = self._last_calibration_marker_corners
        if (
            marker_corners is None
            or now - self._last_calibration_marker_time
            > self.CALIBRATION_MARKER_MAX_AGE_S
        ):
            return
        polygon = np.rint(marker_corners).astype(np.int32).reshape(-1, 1, 2)
        center = tuple(np.rint(marker_corners.mean(axis=0)).astype(int))
        cv2.polylines(result, [polygon], True, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(result, center, 3, (0, 220, 220), -1, cv2.LINE_AA)

    def _best_contour_candidate(
        self, contours: list
    ) -> tuple | None:
        """
        Score every valid contour and return the best one.

        Scoring rewards high circularity and penalises distance from the
        last known position, giving temporal continuity without requiring
        the Kalman prediction to land exactly on the blob.
        """
        best_candidate = None
        best_score = float("-inf")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.MIN_CIRCULARITY:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx), int(cy)

            score = circularity * self.SCORE_CIRCULARITY_WEIGHT
            if self.last_px is not None:
                dist = np.hypot(cx - self.last_px, cy - self.last_py)
                score -= dist * self.SCORE_DISTANCE_WEIGHT

            if score > best_score:
                best_score = score
                best_candidate = (cx, cy, radius, area, circularity)

        return best_candidate

    def _detect_and_draw(self, mask: np.ndarray, result: np.ndarray) -> None:
        """Run contour detection, update the Kalman filter, and annotate *result*."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidate = self._best_contour_candidate(contours)

        if candidate is not None:
            x, y, radius, area, circ = candidate
            self.frames_without_detection = 0

            measurement = np.array([[np.float32(x)], [np.float32(y)]])

            if not self.initialized:
                # Seed the filter AND correct on the very first detection
                self.kalman.statePost = np.array(
                    [[x], [y], [0.0], [0.0]], dtype=np.float32
                )
                self.initialized = True

            # Always correct when we have a real measurement
            self.kalman.correct(measurement)
            prediction = self.kalman.predict()

            px = int(prediction[0, 0])
            py = int(prediction[1, 0])
            vx = float(prediction[2, 0])
            vy = float(prediction[3, 0])
            self.last_px, self.last_py = px, py

            # Draw detected blob
            cv2.circle(result, (x, y), int(radius), (0, 0, 255), 2)
            # Draw Kalman prediction
            cv2.circle(result, (px, py), 8, (0, 255, 255), 2)
            cv2.arrowedLine(
                result,
                (px, py),
                (int(px + vx * 5), int(py + vy * 5)),
                (0, 255, 255), 2, tipLength=0.3,
            )
            cv2.putText(
                result,
                f"Puck  circ:{circ:.2f}  area:{int(area)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            self.status_signal.emit(f"LIVE | Pos: {x},{y} | Pred: {px},{py}")

        else:
            self.frames_without_detection += 1

            if self.frames_without_detection > self.MAX_LOST_FRAMES:
                self.initialized = False
                self.last_px, self.last_py = None, None
                self.status_signal.emit("LIVE | Status: SEARCHING...")

            elif self.initialized:
                # Keep predicting while the tracker is still warm
                prediction = self.kalman.predict()
                px = int(prediction[0, 0])
                py = int(prediction[1, 0])
                cv2.circle(result, (px, py), 8, (0, 165, 255), 2)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _mat_to_pixmap(mat_img: np.ndarray) -> QPixmap:
        """Convert a BGR OpenCV image to a QPixmap."""
        rgb = cv2.cvtColor(mat_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    # Keep the old name as an alias so existing call-sites don't break
    mat_to_pixmap = _mat_to_pixmap