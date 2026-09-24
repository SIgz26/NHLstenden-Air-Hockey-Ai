import sys
from pathlib import Path

# Zorg dat de hoofdmap (Orion Vision) in het zoekpad staat
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from core.engines.base_engine import BaseEngine


class CircleContourEngine(BaseEngine):
    """Puck detector based on circular contours and an expected pixel size parameter."""

    name = "circle_contour"
    parameter_schema = (
        {"name": "expected_radius", "label": "Expected Radius (px)", "type": "int", "min": int(5), "max": int(150), "default": int(35)},
        {"name": "size_tolerance", "label": "Size Tolerance (px)", "type": "int", "min": int(2), "max": int(50), "default": int(15)},
        {"name": "min_circularity", "label": "Min Circularity", "type": "float", "min": float(0.3), "max": float(1.0), "default": float(0.65)},
        {"name": "blur_size", "label": "Blur Size", "type": "int", "min": int(1), "max": int(15), "default": int(5)},
    )

    def __init__(self) -> None:
        super().__init__()
        self.expected_radius = 35  # Verwachte straal van de puck in pixels
        self.size_tolerance = 18   # Speling in pixels (min_radius = expected - tolerance, etc.)
        self.min_circularity = 0.65
        self.blur_size = 5

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

        # 1. Pre-processing & Edge/Threshold detectie
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k_size = max(1, self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1)
        blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

        # Gebruik adaptive thresholding of Canny om objecten los te krijgen van de achtergrond
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morfologie om kleine rotzooi weg te poetsen
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = frame.copy()

        best_candidate = None
        best_score = float("-inf")

        min_r = max(1, self.expected_radius - self.size_tolerance)
        max_r = self.expected_radius + self.size_tolerance

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue

            # Bereken de omsloten cirkel van de contour
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            
            # Filter direct op de ingestelde pixel-grootte straal
            if not (min_r <= radius <= max_r):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue

            # Scoring: beloon cirkels die qua straal heel dicht bij de 'expected_radius' liggen
            size_diff = abs(radius - self.expected_radius)
            score = (circularity * 100.0) - (size_diff * 2.0)

            if self._last_center is not None:
                dist = np.hypot(cx - self._last_center[0], cy - self._last_center[1])
                score -= dist * 0.5

            if score > best_score:
                best_score = score
                best_candidate = (int(cx), int(cy), float(radius), area, circularity)

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

            # Teken de gedetecteerde cirkel op het resultaat
            cv2.circle(result, (cx, cy), int(radius), (0, 0, 255), 2)
            cv2.circle(result, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.arrowedLine(
                result,
                (int(x), int(y)),
                (int(x + vx * 5), int(y + vy * 5)),
                (0, 255, 255), 2, tipLength=0.3,
            )
            
            # Telemetrie op scherm
            cv2.putText(result, f"Puck Radius: {int(radius)}px (Exp: {self.expected_radius})", (10, 30),
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
                cv2.putText(result, "Circle Tracking (Predicting)...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.putText(result, "Searching for circle size...", (10, 30),
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







# Standalone test block om direct te testen met je .mov bestand
if __name__ == "__main__":
    print("Standalone test van CircleContourEngine gestart...")
    engine = CircleContourEngine()
    
    video_pad = r"C:\Users\timoz\Documents\00.school\appilicatie\NHLstenden-Air-Hockey-Ai\Orion Vision\dataset02\realtime_30fps_potje01.mov"
    print(f"Video laden vanuit dataset: {video_pad}")
    cap = cv2.VideoCapture(video_pad)

    if not cap.isOpened():
        print("Waarschuwing: Kan video niet openen.")
    else:
        print("Video geopend. Druk op 'q' om te stoppen.")
        cv2.namedWindow("Circle Engine - Resultaat", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Circle Engine - Masker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Circle Engine - Resultaat", 1024, 576)
        cv2.resizeWindow("Circle Engine - Masker", 1024, 576)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Einde van de video bereikt.")
                break

            processed_frame, mask_frame = engine.process_frame(frame)

            cv2.imshow("Circle Engine - Resultaat", processed_frame)
            cv2.imshow("Circle Engine - Masker", mask_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    print("Test afgesloten.")