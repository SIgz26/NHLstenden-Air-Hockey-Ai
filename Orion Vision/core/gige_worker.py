import numpy as np
import cv2

from queue import Queue, Empty, Full

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

try:
    import vmbpy
    VMBPY_AVAILABLE = True
except ImportError:
    VMBPY_AVAILABLE = False


# ── VmbPy Frame Handler (identiek aan werkend script) ─────────────────────────

class _FrameHandler:
    """VmbPy callback die frames veilig omzet naar OpenCV en in een queue zet.

    Dit volgt het werkende voorbeeld uit v2_maxfps_ROI.py: de callback doet geen
    GUI-werk, alleen frame-conversie en queueing. De worker-thread leest daarna
    de queue en verwerkt de beelden in de Qt-thread.
    """

    def __init__(self, queue_size: int = 10) -> None:
        self.queue = Queue(queue_size)
        self.frame_count = 0
        self.incomplete = 0

    @staticmethod
    def _to_bgr(frame: "vmbpy.Frame") -> np.ndarray | None:
        try:
            if frame.get_pixel_format() == vmbpy.PixelFormat.Bgr8:
                return frame.as_opencv_image()

            converted = frame.convert_pixel_format(vmbpy.PixelFormat.Bgr8)
            return converted.as_opencv_image()
        except Exception:
            try:
                mono = frame.convert_pixel_format(vmbpy.PixelFormat.Mono8)
                img = mono.as_opencv_image()
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass
            return None

    def __call__(
        self,
        cam: "vmbpy.Camera",
        stream: "vmbpy.Stream",
        frame: "vmbpy.Frame",
    ) -> None:
        status = frame.get_status()

        if status != vmbpy.FrameStatus.Complete:
            self.incomplete += 1
            cam.queue_frame(frame)
            return

        self.frame_count += 1

        try:
            image = self._to_bgr(frame)
            if image is None:
                self.incomplete += 1
                cam.queue_frame(frame)
                return

            try:
                self.queue.put_nowait(image)
            except Full:
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.queue.put_nowait(image)
                except Full:
                    pass
        finally:
            cam.queue_frame(frame)


# ── GigE Worker ───────────────────────────────────────────────────────────────

class GigECameraWorker(QThread):
    frame_processed = pyqtSignal(QPixmap, QPixmap)
    status_signal   = pyqtSignal(str)
    debug_signal    = pyqtSignal(str)

    PAUSE_SLEEP_MS           = 50
    FRAME_SLEEP_MS           = 5     # Lager dan sync versie: async heeft minder delay nodig
    SCORE_DISTANCE_WEIGHT    = 0.35
    SCORE_CIRCULARITY_WEIGHT = 120
    MIN_CIRCULARITY          = 0.55
    KALMAN_PROCESS_NOISE     = 0.08
    KALMAN_MEASUREMENT_NOISE = 1.0

    def __init__(self, camera_id: str) -> None:
        super().__init__()
        if not VMBPY_AVAILABLE:
            raise RuntimeError("VmbPy niet geïnstalleerd.")

        self.camera_id = camera_id
        self.running   = True
        self.paused    = False

        self._hsv_frame_lock                       = QReadWriteLock()
        self._current_hsv_frame: np.ndarray | None = None
        self._mutex                                = QMutex()

        self.h_min, self.h_max = 35, 95
        self.s_min, self.s_max = 20, 255
        self.v_min, self.v_max = 40, 255
        self.min_area  = 40
        self.max_area  = 2500

        self.engine_name = "hsv"
        self.engine = self._build_engine(self.engine_name)
        self.sim_bridge = SimBridge()
        self.aruco_tracker = ArUcoMalletTracker()
        self.show_table_overlay = True
        self._table_calibration_preview: tuple[tuple[float, float], ...] = ()

        # Fisheye correction: applied before detection so the live feed is
        # corrected immediately when the camera starts streaming.
        self.fisheye_enabled = True
        self.fisheye_k1 = -0.40
        self.fisheye_k2 = 0.05
        self._fisheye_map1 = None
        self._fisheye_map2 = None
        self._fisheye_size = None

        self._init_kalman()
        self.initialized              = False
        self.last_px: int | None      = None
        self.last_py: int | None      = None
        self.frames_without_detection = 0
        self.MAX_LOST_FRAMES          = 12

    # ── Kalman ────────────────────────────────────────────────────────

    def _init_kalman(self) -> None:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32
        )
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        )
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * self.KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.KALMAN_MEASUREMENT_NOISE
        kf.statePost           = np.zeros((4, 1), dtype=np.float32)
        self.kalman = kf

    # ── Publieke interface ─────────────────────────────────────────────

    @property
    def current_hsv_frame(self) -> np.ndarray | None:
        self._hsv_frame_lock.lockForRead()
        try:
            f = self._current_hsv_frame
            return f.copy() if f is not None else None
        finally:
            self._hsv_frame_lock.unlock()

    @staticmethod
    def get_available_gige_cameras() -> list[dict]:
        if not VMBPY_AVAILABLE:
            return []
        result = []
        try:
            with vmbpy.VmbSystem.get_instance() as vmb:
                for cam in vmb.get_all_cameras():
                    result.append({
                        "id":    cam.get_id(),
                        "name":  cam.get_name(),
                        "model": cam.get_model(),
                    })
        except Exception as exc:
            print(f"[GigE] Scan mislukt: {exc}")
        return result

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
            self._fisheye_map1 = None
            self._fisheye_map2 = None
            self._fisheye_size = None

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

    def _debug(self, message: str) -> None:
        print(message)
        self.debug_signal.emit(message)

    def toggle_pause(self) -> bool:
        with QMutexLocker(self._mutex):
            self.paused = not self.paused
            return self.paused

    def stop(self) -> None:
        self.running = False

    # ── Hoofd-loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        self._debug(f"GigE | Verbinden met {self.camera_id}...")
        self.status_signal.emit(
            f"GigE | Verbinden met {self.camera_id}..."
        )
        try:
            with vmbpy.VmbSystem.get_instance() as vmb:
                try:
                    cam = vmb.get_camera_by_id(self.camera_id)
                except vmbpy.error.VmbCameraError:
                    ids = [c.get_id() for c in vmb.get_all_cameras()]
                    self.status_signal.emit(
                        f"GigE Fout | '{self.camera_id}' niet gevonden. "
                        f"Beschikbaar: {ids}"
                    )
                    return

                with cam:
                    # Zelfde volgorde als werkend script
                    self._setup_camera(cam)
                    self._setup_pixel_format(cam)
                    self._main_loop(cam)

        except vmbpy.error.VmbSystemError as e:
            self.status_signal.emit(f"GigE Fout | VmbSystem: {e}")
        except Exception as e:
            self.status_signal.emit(
                f"GigE Fout | {type(e).__name__}: {e}"
            )

        self.status_signal.emit("GigE | Camera gestopt")

    # ── Camera setup (exact zelfde als werkend script) ─────────────────

    def _setup_camera(self, cam: "vmbpy.Camera") -> None:
        """
        Identiek aan setup_camera() in het werkende script.
        Let op: GVSPAdjustPacketSize op het STREAM object, niet camera!
        """
        # Auto exposure
        try:
            cam.ExposureAuto.set("Continuous")
            self.status_signal.emit("GigE | Auto exposure: Continuous")
        except Exception:
            self.status_signal.emit("GigE | Auto exposure niet beschikbaar")

        # Auto white balance
        try:
            cam.BalanceWhiteAuto.set("Continuous")
            self.status_signal.emit("GigE | Auto white balance: Continuous")
        except Exception:
            pass

        # !! STREAM object gebruiken voor packet size – niet cam !!
        # Dit was de bug: cam.GVSPAdjustPacketSize bestaat niet,
        # het zit op cam.get_streams()[0]
        try:
            stream = cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass
            self.status_signal.emit("GigE | Packet size automatisch aangepast")
        except Exception as e:
            self.status_signal.emit(f"GigE | Packet size aanpassing mislukt: {e}")

    def _setup_pixel_format(self, cam: "vmbpy.Camera") -> None:
        """Gebruik exact dezelfde logica als het werkende VmbPy-voorbeeld."""
        target_fmt = vmbpy.PixelFormat.Bgr8
        cam_formats = cam.get_pixel_formats()

        cam_color_formats = vmbpy.intersect_pixel_formats(
            cam_formats, vmbpy.COLOR_PIXEL_FORMATS
        )
        convertible_color_formats = tuple(
            fmt for fmt in cam_color_formats
            if target_fmt in fmt.get_convertible_formats()
        )

        cam_mono_formats = vmbpy.intersect_pixel_formats(
            cam_formats, vmbpy.MONO_PIXEL_FORMATS
        )
        convertible_mono_formats = tuple(
            fmt for fmt in cam_mono_formats
            if target_fmt in fmt.get_convertible_formats()
        )

        if target_fmt in cam_formats:
            cam.set_pixel_format(target_fmt)
            self.status_signal.emit("GigE | Pixel format: Bgr8 (direct)")
            return

        if convertible_color_formats:
            selected = convertible_color_formats[0]
            cam.set_pixel_format(selected)
            self.status_signal.emit(
                f"GigE | Camera pixel format: {selected}"
            )
            return

        if convertible_mono_formats:
            selected = convertible_mono_formats[0]
            cam.set_pixel_format(selected)
            self.status_signal.emit(
                f"GigE | Camera pixel format: {selected}"
            )
            return

        self.status_signal.emit(
            "GigE | Waarschuwing: geen Bgr8-compatibel formaat gevonden"
        )

    # ── Asynchrone capture loop ────────────────────────────────────────
    def _main_loop(self, cam: "vmbpy.Camera") -> None:
        handler = _FrameHandler(queue_size=10)

        try:
            cam.start_streaming(handler=handler, buffer_count=10)
            self._debug("GigE | Streaming...")
            self.status_signal.emit("GigE | Streaming...")

            while self.running:
                with QMutexLocker(self._mutex):
                    paused = self.paused

                if paused:
                    self.msleep(50)
                    continue

                try:
                    img = handler.queue.get(timeout=0.2)
                    self._process_frame(img)
                except Empty:
                    continue

                if handler.frame_count % 30 == 0:
                    self.status_signal.emit(
                        f"GigE Live | OK: {handler.frame_count} | Error: {handler.incomplete}"
                    )
        finally:
            self._debug("GigE | stopping stream...")
            if cam.is_streaming():
                cam.stop_streaming()
            self._debug("GigE | stream stopped")
    # ── Verwerking ─────────────────────────────────────────────────────

    def _process_frame(self, bgr: np.ndarray) -> None:
        """
        bgr is gegarandeerd (H, W, 3) uint8 dankzij
        as_opencv_image() in de FrameHandler.
        """
        bgr = self._apply_fisheye_correction(bgr)
        robot_position, opponent_position = self.aruco_tracker.detect_mallets(bgr)
        blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        self._hsv_frame_lock.lockForWrite()
        self._current_hsv_frame = hsv
        self._hsv_frame_lock.unlock()

        with QMutexLocker(self._mutex):
            lower = np.array([self.h_min, self.s_min, self.v_min])
            upper = np.array([self.h_max, self.s_max, self.v_max])

        mask   = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = bgr.copy()
        if hasattr(self, "engine") and self.engine is not None:
            try:
                result, mask = self.engine.process_frame(bgr)
            except Exception:
                self._detect_and_draw(mask, result)
        else:
            self._detect_and_draw(mask, result)

        if self.engine is not None:
            self.sim_bridge.update_from_engine(
                self.engine.last_result,
                frame_width=bgr.shape[1],
                frame_height=bgr.shape[0],
                frame_rate=float(getattr(self.engine, "ESTIMATED_FPS", 30.0)),
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

        self.frame_processed.emit(
            self._mat_to_pixmap(result),
            self._mat_to_pixmap(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask),
        )

    # ── Detectie ───────────────────────────────────────────────────────

    def _best_contour_candidate(self, contours) -> tuple | None:
        best, best_score = None, float("-inf")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circ = 4 * np.pi * area / perimeter ** 2
            if circ < self.MIN_CIRCULARITY:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx), int(cy)
            score = circ * self.SCORE_CIRCULARITY_WEIGHT
            if self.last_px is not None:
                score -= (
                    np.hypot(cx - self.last_px, cy - self.last_py)
                    * self.SCORE_DISTANCE_WEIGHT
                )
            if score > best_score:
                best_score = score
                best = (cx, cy, radius, area, circ)
        return best

    def _detect_and_draw(
        self, mask: np.ndarray, result: np.ndarray
    ) -> None:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidate = self._best_contour_candidate(contours)

        if candidate is not None:
            x, y, radius, area, circ = candidate
            self.frames_without_detection = 0

            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            if not self.initialized:
                self.kalman.statePost = np.array(
                    [[x], [y], [0.], [0.]], dtype=np.float32
                )
                self.initialized = True

            self.kalman.correct(measurement)
            pred = self.kalman.predict()
            px, py = int(pred[0, 0]), int(pred[1, 0])
            vx, vy = float(pred[2, 0]), float(pred[3, 0])
            self.last_px, self.last_py = px, py

            cv2.circle(result, (x, y), int(radius), (0, 0, 255), 2)
            cv2.circle(result, (px, py), 8, (0, 255, 255), 2)
            cv2.arrowedLine(
                result, (px, py),
                (int(px + vx * 5), int(py + vy * 5)),
                (0, 255, 255), 2, tipLength=0.3,
            )
            cv2.putText(
                result,
                f"[GigE] circ:{circ:.2f}  area:{int(area)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2,
            )
            self.status_signal.emit(
                f"GigE ✓ | Pos:{x},{y} | Pred:{px},{py}"
            )
        else:
            self.frames_without_detection += 1
            if self.frames_without_detection > self.MAX_LOST_FRAMES:
                self.initialized = False
                self.last_px, self.last_py = None, None
                self.status_signal.emit("GigE | SEARCHING...")
            elif self.initialized:
                pred = self.kalman.predict()
                px, py = int(pred[0, 0]), int(pred[1, 0])
                cv2.circle(result, (px, py), 8, (0, 165, 255), 2)

    # ── Utility ───────────────────────────────────────────────────────

    @staticmethod
    def _mat_to_pixmap(mat: np.ndarray) -> QPixmap:
        """
        Mat is gegarandeerd (H, W, 3) BGR dankzij as_opencv_image().
        Veiligheidsnet blijft voor de mask die we zelf maken.
        """
        if mat.ndim == 2:
            mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
        elif mat.ndim == 3 and mat.shape[2] == 1:
            mat = cv2.cvtColor(mat.squeeze(axis=2), cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QPixmap.fromImage(
            QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        )