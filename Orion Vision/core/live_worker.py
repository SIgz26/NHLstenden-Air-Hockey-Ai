import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap


class LiveCameraWorker(QThread):
    frame_processed = pyqtSignal(QPixmap, QPixmap)  # (Result, Mask)
    status_signal = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.paused = False
        self.mutex = QMutex()

        # Essentieel voor de Pick Color optie in het Dashboard!
        self.current_hsv_frame = None

        # HSV parameters (standaard puck waarden)
        self.h_min, self.h_max = 35, 95
        self.s_min, self.s_max = 20, 255
        self.v_min, self.v_max = 40, 255
        self.min_area = 40
        self.max_area = 2500

        # Kalman Filter initialisatie
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.08
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kalman.statePost = np.array([[0.], [0.], [0.], [0.]], np.float32)

        self.initialized = False
        self.last_px, self.last_py = None, None
        self.frames_without_detection = 0
        self.MAX_LOST_FRAMES = 12

    @staticmethod
    def get_available_cameras(max_checks=5):
        available = []
        old_level = cv2.getLogLevel() if hasattr(cv2, 'getLogLevel') else None
        if hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(0)

        for i in range(max_checks):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()

        if hasattr(cv2, 'setLogLevel') and old_level is not None:
            cv2.setLogLevel(old_level)

        return available

    def set_camera_index(self, index):
        with QMutexLocker(self.mutex):
            self.camera_index = index

    def update_hsv(self, h_min, h_max, s_min, s_max, v_min=20, v_max=255):
        with QMutexLocker(self.mutex):
            self.h_min, self.h_max = h_min, h_max
            self.s_min, self.s_max = s_min, s_max
            self.v_min, self.v_max = v_min, v_max

    def toggle_pause(self):
        with QMutexLocker(self.mutex):
            self.paused = not self.paused
            return self.paused

    def stop(self):
        self.running = False

    def run(self):
        backend = cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_index, backend)

        if not cap.isOpened():
            self.status_signal.emit(f"Fout: Kan camera {self.camera_index} niet openen.")
            return

        self.status_signal.emit(f"Camera {self.camera_index} verbonden & actief")

        while self.running:
            with QMutexLocker(self.mutex):
                is_paused = self.paused

            if is_paused:
                self.msleep(50)
                continue

            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("Fout: Geen frame ontvangen van camera")
                self.msleep(100)
                continue

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # BEWAAR HET FRAME VOOR DE PICK COLOR PIPET
            with QMutexLocker(self.mutex):
                self.current_hsv_frame = hsv.copy()
                lower_bound = np.array([self.h_min, self.s_min, self.v_min])
                upper_bound = np.array([self.h_max, self.s_max, self.v_max])

            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = frame.copy()

            best_candidate = None
            best_score = -999

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > self.max_area:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.55:
                    continue

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                x, y = int(x), int(y)

                score = circularity * 120
                if self.last_px is not None:
                    dist = np.hypot(x - self.last_px, y - self.last_py)
                    score -= dist * 0.35

                if score > best_score:
                    best_score = score
                    best_candidate = (x, y, radius, area, circularity)

            if best_candidate is not None:
                x, y, radius, area, circ = best_candidate
                self.frames_without_detection = 0
                measurement = np.array([[np.float32(x)], [np.float32(y)]])

                if not self.initialized:
                    self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
                    self.initialized = True
                else:
                    self.kalman.correct(measurement)

                prediction = self.kalman.predict()
                px, py = int(prediction[0][0]), int(prediction[1][0])
                vx, vy = float(prediction[2][0]), float(prediction[3][0])

                self.last_px, self.last_py = px, py

                cv2.circle(result, (x, y), int(radius), (0, 0, 255), 2)
                cv2.circle(result, (px, py), 8, (0, 255, 255), 2)
                cv2.arrowedLine(result, (px, py), (int(px + vx * 5), int(py + vy * 5)), (0, 255, 255), 2, tipLength=0.3)
                cv2.putText(result, f"Puck circ:{circ:.2f} area:{int(area)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                self.status_signal.emit(f"LIVE | Pos: {x},{y} | Pred: {px},{py}")

            else:
                self.frames_without_detection += 1
                if self.frames_without_detection > self.MAX_LOST_FRAMES:
                    self.initialized = False
                    self.last_px, self.last_py = None, None
                    self.status_signal.emit("LIVE | Status: SEARCHING...")

                if self.initialized and self.frames_without_detection <= self.MAX_LOST_FRAMES:
                    prediction = self.kalman.predict()
                    px, py = int(prediction[0][0]), int(prediction[1][0])
                    cv2.circle(result, (px, py), 8, (0, 165, 255), 2)

            pixmap_result = self.mat_to_pixmap(result)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            pixmap_mask = self.mat_to_pixmap(mask_bgr)

            self.frame_processed.emit(pixmap_result, pixmap_mask)
            self.msleep(15)

        cap.release()
        self.status_signal.emit("Camera gestopt")

    def mat_to_pixmap(self, mat_img):
        rgb = cv2.cvtColor(mat_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)