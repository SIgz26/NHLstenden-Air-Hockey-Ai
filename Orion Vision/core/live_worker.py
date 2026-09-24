import os
import time
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QReadWriteLock
from PyQt5.QtGui import QImage, QPixmap

from core.engines.hsv_engine import HsvEngine
from core.engines.hough_circle_engine import HoughCircleEngine
from core.engines.bg_subtraction_engine import BackgroundSubtractionEngine
from core.engines.circle_contour_engine import CircleContourEngine
from core.engines.hybrid_engine import HybridEngine


class LiveCameraWorker(QThread):
    """
    Background thread that captures camera frames, applies HSV-based
    blob detection with Kalman filtering, and emits processed pixmaps.
    """

    frame_processed = pyqtSignal(QPixmap, QPixmap)  # (Result, Mask)
    status_signal = pyqtSignal(str)

    # Tuning constants – centralised so they're easy to adjust
    FRAME_SLEEP_MS = 15
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
        self.target_fps = 20.0
        self.record_fps = 20.0
        self._fps = 0.0
        self._frame_counter = 0
        self._fps_last_update = time.monotonic()
        self._last_capture_time = time.monotonic()

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

        # Suppress noisy OpenCV backend messages during probing
        old_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        for i in range(max_checks):
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
            self.target_fps = max(1.0, min(float(fps), 60.0))

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
        # Use platform-native backend to avoid backend warnings on Windows
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_index, backend)

        if not cap.isOpened():
            self.status_signal.emit(
                f"Error: Cannot open camera {self.camera_index}."
            )
            return

        self.status_signal.emit(
            f"Camera {self.camera_index} connected & active"
        )

        while self.running:
            # --- Pause guard ---
            with QMutexLocker(self._mutex):
                is_paused = self.paused

            if is_paused:
                self.msleep(self.PAUSE_SLEEP_MS)
                continue

            target_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
            if target_interval > 0:
                now = time.monotonic()
                elapsed = now - self._last_capture_time
                if elapsed < target_interval:
                    self.msleep(max(1, int((target_interval - elapsed) * 1000)))
                    continue
                self._last_capture_time = now

            # --- Capture ---
            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("Error: No frame received from camera")
                self.msleep(100)
                continue

            # --- Pre-processing ---
            frame = self._apply_fisheye_correction(frame)
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
            if hasattr(self, "engine") and self.engine is not None:
                try:
                    result, mask = self.engine.process_frame(frame)
                except Exception:
                    result = frame.copy()
                    self._detect_and_draw(mask, result)
            else:
                self._detect_and_draw(mask, result)

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
                self.status_signal.emit(f"Live FPS: {self._fps:.1f}")

            # --- Emit ---
            pixmap_result = self._mat_to_pixmap(result)
            
            # Veilige conversie voor het masker/secondary beeld naar BGR voor de QPixmap
            if len(mask.shape) == 2:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            elif mask.shape[2] == 1:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_bgr = mask

            pixmap_mask = self._mat_to_pixmap(mask_bgr)
            self.frame_processed.emit(pixmap_result, pixmap_mask)
            self.msleep(self.FRAME_SLEEP_MS)

        cap.release()
        self.status_signal.emit("Camera stopped")

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

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