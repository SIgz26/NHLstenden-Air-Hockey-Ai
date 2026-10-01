import sys
from pathlib import Path

# Zorg dat de hoofdmap (Orion Vision) in het zoekpad staat
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from core.engines.base_engine import BaseEngine


class HoughCircleEngine(BaseEngine):
    """Puck detector using OpenCV Hough Circle Transform with Kalman filtering and telemetry."""

    name = "hough_circles"
    parameter_schema = (
        {"name": "min_radius", "label": "Min Radius (px)", "type": "int", "min": int(1), "max": int(100), "default": int(15)},
        {"name": "max_radius", "label": "Max Radius (px)", "type": "int", "min": int(4), "max": int(150), "default": int(50)},
        {"name": "param2", "label": "Sensitivity (Acc Threshold)", "type": "int", "min": int(5), "max": int(100), "default": int(30)},
        {"name": "min_dist", "label": "Min Distance", "type": "int", "min": int(10), "max": int(200), "default": int(50)},
    )

    def __init__(self) -> None:
        super().__init__()
        self.min_radius = 15
        self.max_radius = 25
        self.param2 = 30        # Hoe lager, hoe meer cirkels worden gedetecteerd (gevoeliger)
        self.min_dist = 50      # Minimale afstand tussen gedetecteerde cirkelcentra

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

        # 1. Omzetten naar grijswaarden en vervagen om ruis te verminderen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # 2. Pas Hough Circle Transform toe
        # dp=1.2 (resolutie ratio), min_dist=self.min_dist, param1=100 (Canny drempel), param2=self.param2
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.min_dist,
            param1=100,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        result = frame.copy()
        mask = np.zeros_like(gray, dtype=np.uint8)

        best_candidate = None
        best_score = float("-inf")

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                # Score op basis van continuïteit (nabijheid van vorig centrum)
                score = 100.0
                if self._last_center is not None:
                    dist = np.hypot(cx - self._last_center[0], cy - self._last_center[1])
                    score -= dist * 0.5

                if score > best_score:
                    best_score = score
                    best_candidate = (int(cx), int(cy), int(r))

        x = y = vx = vy = 0.0
        speed_ms = 0.0
        speed_kmh = 0.0

        if best_candidate is not None:
            cx, cy, radius = best_candidate
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

            # Teken de gevonden cirkel en het Kalman-punt
            cv2.circle(result, (cx, cy), radius, (0, 0, 255), 2)
            cv2.circle(result, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.arrowedLine(
                result,
                (int(x), int(y)),
                (int(x + vx * 5), int(y + vy * 5)),
                (0, 255, 255), 2, tipLength=0.3,
            )
            
            # Teken ook op het masker voor de dual view
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            # Telemetrie op scherm
            cv2.putText(result, f"Hough Circle | Pos: ({int(x)}, {int(y)}) | Vel: ({vx:.1f}, {vy:.1f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result, f"Speed: {speed_ms:.2f} m/s ({speed_kmh:.1f} km/h) | R: {radius}px", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            if self.initialized:
                prediction = self.kalman.predict()
                x = float(prediction[0, 0])
                y = float(prediction[1, 0])
                vx = float(prediction[2, 0])
                vy = float(prediction[3, 0])
                cv2.circle(result, (int(x), int(y)), 6, (0, 165, 255), 2)
                cv2.putText(result, "Hough Tracking (Predicting)...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.putText(result, "Searching for circles...", (10, 30),
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
    print("Standalone test van HoughCircleEngine gestart...")
    engine = HoughCircleEngine()
    
    video_pad = r"C:\Users\timoz\Documents\00.school\appilicatie\NHLstenden-Air-Hockey-Ai\Orion Vision\dataset02\realtime_30fps_potje01.mov"
    print(f"Video laden vanuit dataset: {video_pad}")
    cap = cv2.VideoCapture(video_pad)

    if not cap.isOpened():
        print("Waarschuwing: Kan video niet openen.")
    else:
        print("Video geopend. Druk op 'q' om te stoppen.")
        cv2.namedWindow("Hough Engine - Resultaat", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Hough Engine - Masker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hough Engine - Resultaat", 1024, 576)
        cv2.resizeWindow("Hough Engine - Masker", 1024, 576)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Einde van de video bereikt.")
                break

            processed_frame, mask_frame = engine.process_frame(frame)

            cv2.imshow("Hough Engine - Resultaat", processed_frame)
            cv2.imshow("Hough Engine - Masker", mask_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    print("Test afgesloten.")