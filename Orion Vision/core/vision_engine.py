import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap


class VisionWorker(QThread):
    position_changed = pyqtSignal(int, int)
    frame_processed = pyqtSignal(QPixmap, QPixmap)
    status_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True
        self.paused = False
        self.jump_to_frame = -1
        self.mutex = QMutex()

        # Essentieel voor de Pick Color optie in OrionDashboard
        self.current_hsv_frame = None

        # HSV parameters
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

    def update_hsv(self, h_min, h_max, s_min, s_max, v_min=20, v_max=255):
        with QMutexLocker(self.mutex):
            self.h_min, self.h_max = h_min, h_max
            self.s_min, self.s_max = s_min, s_max
            self.v_min, self.v_max = v_min, v_max

    def toggle_pause(self):
        with QMutexLocker(self.mutex):
            self.paused = not self.paused
            return self.paused

    def set_position(self, frame_number):
        with QMutexLocker(self.mutex):
            self.jump_to_frame = frame_number

    def stop(self):
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.running and cap.isOpened():
            with QMutexLocker(self.mutex):
                if self.jump_to_frame != -1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.jump_to_frame)
                    self.jump_to_frame = -1

                is_paused = self.paused

            if is_paused:
                self.msleep(50)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.position_changed.emit(current_frame, total_frames)

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

                self.status_signal.emit(f"Puck Pos: {x},{y} | Pred: {px},{py}")

            else:
                self.frames_without_detection += 1
                if self.frames_without_detection > self.MAX_LOST_FRAMES:
                    self.initialized = False
                    self.last_px, self.last_py = None, None
                    self.status_signal.emit("Status: LOST")

                if self.initialized and self.frames_without_detection <= self.MAX_LOST_FRAMES:
                    prediction = self.kalman.predict()
                    px, py = int(prediction[0][0]), int(prediction[1][0])
                    cv2.circle(result, (px, py), 8, (0, 165, 255), 2)

            pixmap_result = self.mat_to_pixmap(result)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            pixmap_mask = self.mat_to_pixmap(mask_bgr)

            self.frame_processed.emit(pixmap_result, pixmap_mask)
            self.msleep(30)

        cap.release()

    def mat_to_pixmap(self, mat_img):
        rgb = cv2.cvtColor(mat_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)