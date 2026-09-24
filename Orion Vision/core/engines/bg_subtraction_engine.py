import sys
from pathlib import Path

# Zorg dat de hoofdmap (Orion Vision) in het zoekpad staat
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from core.engines.base_engine import BaseEngine


class BackgroundSubtractionEngine(BaseEngine):
    """Puck detector using MOG2 background subtraction combined with circularity and size filtering."""

    name = "bg_subtraction"
    parameter_schema = (
        {"name": "history", "label": "History Frames", "type": "int", "min": int(10), "max": int(500), "default": int(50)},
        {"name": "var_threshold", "label": "Var Threshold", "type": "int", "min": int(5), "max": int(100), "default": int(16)},
        {"name": "min_area", "label": "Min Area", "type": "int", "min": int(10), "max": int(2000), "default": int(50)},
        {"name": "max_area", "label": "Max Area", "type": "int", "min": int(100), "max": int(10000), "default": int(2000)},
    )

    def __init__(self) -> None:
        super().__init__()
        self.history = 50
        self.var_threshold = 16
        self.min_area = 50
        self.max_area = 2000
        self.MIN_CIRCULARITY = 0.50

        # Maak de MOG2 background subtractor aan
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold, detectShadows=False
        )

        # Fysieke schaalfactor voor snelheid
        self.PIXEL_TO_METER = 0.0015 
        self.ESTIMATED_FPS = 30.0  

        # Kalman filter initialisatie
        self._init_kalman()
        self.initialized = False
        self._last_center = None

    def _init_kalman(self) -> None:
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
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.08
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kalman = kf

    def update_settings(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update de subtractor als de instellingen veranderen
        if "history" in kwargs or "var_threshold" in kwargs:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history, varThreshold=self.var_threshold, detectShadows=False
            )

    def process_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            empty = np.zeros((0, 0), dtype=np.uint8)
            standardized = self.standardize_result(
                frame, 0.0, 0.0, 0.0, 0.0,
                mask=empty, secondary=empty,
                metadata={"best": None, "in_range": False},
            )
            self.last_result = standardized
            return frame, empty

        # 1. Pas achtergrondsubstractie toe om bewegende delen te vinden
        fg_mask = self.bg_subtractor.apply(frame)

        # 2. Opschonen van het masker met morfologische operaties
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. Zoek contouren in het bewegingsmasker
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = frame.copy()

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

            # Scoring: beloon hoge circulariteit en nabijheid van de vorige positie
            score = circularity * 100.0
            if self._last_center is not None:
                dist = np.hypot(cx - self._last_center[0], cy - self._last_center[1])
                score -= dist * 0.4

            if score > best_score:
                best_score = score
                best_candidate = (cx, cy, int(radius), area, circularity)

        x = y = vx = vy = 0.0
        speed_ms = 0.0
        speed_kmh = 0.0

        if best_candidate is not None:
            cx, cy, radius, area, circ = best_candidate
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

            if not self.initialized:
                self.kalman.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
                self.initialized = True

            self.kalman.correct(measurement)
            prediction = self.kalman.predict()

            x = float(prediction[0, 0])
            y = float(prediction[1, 0])
            vx = float(prediction[2, 0]) 
            vy = float(prediction[3, 0]) 
            self._last_center = (cx, cy)

            speed_pixels_per_sec = np.hypot(vx, vy) * self.ESTIMATED_FPS
            speed_ms = speed_pixels_per_sec * self.PIXEL_TO_METER
            speed_kmh = speed_ms * 3.6

            # Teken elementen op het scherm
            cv2.circle(result, (cx, cy), radius, (0, 0, 255), 2)
            cv2.circle(result, (int(x), int(y)), 6, (0, 255, 255), 2)
            cv2.arrowedLine(
                result,
                (int(x), int(y)),
                (int(x + vx * 5), int(y + vy * 5)),
                (0, 255, 255), 2, tipLength=0.3,
            )
            
            cv2.putText(result, f"BG-Sub | Pos: ({int(x)}, {int(y)}) | Vel: ({vx:.1f}, {vy:.1f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result, f"Speed: {speed_ms:.2f} m/s ({speed_kmh:.1f} km/h) | Circ: {circ:.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            if self.initialized:
                prediction = self.kalman.predict()
                x = float(prediction[0, 0])
                y = float(prediction[1, 0])
                vx = float(prediction[2, 0])
                vy = float(prediction[3, 0])
                cv2.circle(result, (int(x), int(y)), 6, (0, 165, 255), 2)
                cv2.putText(result, "BG-Sub Tracking (Predicting)...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.putText(result, "Waiting for movement...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        standardized = self.standardize_result(
            result, x, y, vx, vy,
            mask=mask,
            secondary=mask_bgr,
            metadata={
                "best": best_candidate, 
                "in_range": best_candidate is not None,
                "speed_ms": speed_ms,
                "speed_kmh": speed_kmh
            },
        )
        self.last_result = standardized
        return result, mask_bgr


# Standalone test block om direct te testen op je .mov bestand
if __name__ == "__main__":
    print("Standalone test van BackgroundSubtractionEngine gestart...")
    engine = BackgroundSubtractionEngine()
    
    video_pad = r"C:\Users\timoz\Documents\00.school\appilicatie\NHLstenden-Air-Hockey-Ai\Orion Vision\dataset02\realtime_30fps_potje01.mov"
    print(f"Video laden vanuit dataset: {video_pad}")
    cap = cv2.VideoCapture(video_pad)

    if not cap.isOpened():
        print("Waarschuwing: Kan video niet openen.")
    else:
        print("Video geopend. Druk op 'q' om te stoppen.")
        cv2.namedWindow("BG Subtraction - Resultaat", cv2.WINDOW_NORMAL)
        cv2.namedWindow("BG Subtraction - Masker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BG Subtraction - Resultaat", 1024, 576)
        cv2.resizeWindow("BG Subtraction - Masker", 1024, 576)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Einde van de video bereikt.")
                break

            processed_frame, mask_frame = engine.process_frame(frame)

            cv2.imshow("BG Subtraction - Resultaat", processed_frame)
            cv2.imshow("BG Subtraction - Masker", mask_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    print("Test afgesloten.")