import numpy as np
import cv2

from queue import Queue, Empty, Full

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QReadWriteLock
from PyQt5.QtGui import QImage, QPixmap

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
        self._detect_and_draw(mask, result)

        self.frame_processed.emit(
            self._mat_to_pixmap(result),
            self._mat_to_pixmap(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
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