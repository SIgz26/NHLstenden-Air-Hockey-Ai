import sys
from pathlib import Path

# Zorg dat de hoofdmap (Orion Vision) in het zoekpad staat VÓÓRdat je core importeert
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from core.engines.base_engine import BaseEngine


class HsvEngine(BaseEngine):
    """HSV-based color filtering and contour detection engine with Kalman filtering and full telemetry display."""

    name = "hsv"
    parameter_schema = (
        {"name": "h_min", "label": "H Min", "type": "int", "min": int(0), "max": int(179), "default": int(0)},
        {"name": "h_max", "label": "H Max", "type": "int", "min": int(0), "max": int(179), "default": int(16)},
        {"name": "s_min", "label": "S Min", "type": "int", "min": int(0), "max": int(255), "default": int(154)},
        {"name": "s_max", "label": "S Max", "type": "int", "min": int(0), "max": int(255), "default": int(218)},
        {"name": "v_min", "label": "V Min", "type": "int", "min": int(0), "max": int(255), "default": int(40)},
        {"name": "v_max", "label": "V Max", "type": "int", "min": int(0), "max": int(255), "default": int(255)},
        {"name": "min_area", "label": "Min Area", "type": "int", "min": int(1), "max": int(5000), "default": int(40)},
        {"name": "max_area", "label": "Max Area", "type": "int", "min": int(100), "max": int(20000), "default": int(2500)},
    )

    def __init__(self) -> None:
        super().__init__()
        self.h_min, self.h_max = 0, 16
        self.s_min, self.s_max = 154, 218
        self.v_min, self.v_max = 40, 255
        self.min_area = 40
        self.max_area = 2500
        self.MIN_CIRCULARITY = 0.55
        self.SCORE_DISTANCE_WEIGHT = 0.35
        self.SCORE_CIRCULARITY_WEIGHT = 120.0
        
        self.PIXEL_TO_METER = 0.0015 
        self.ESTIMATED_FPS = 20.0  

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

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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

            score = circularity * self.SCORE_CIRCULARITY_WEIGHT
            if self._last_center is not None:
                dist = np.hypot(cx - self._last_center[0], cy - self._last_center[1])
                score -= dist * self.SCORE_DISTANCE_WEIGHT

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

            cv2.circle(result, (cx, cy), radius, (0, 0, 255), 2)
            cv2.circle(result, (int(x), int(y)), 6, (0, 255, 255), 2)
            cv2.arrowedLine(
                result,
                (int(x), int(y)),
                (int(x + vx * 5), int(y + vy * 5)),
                (0, 255, 255), 2, tipLength=0.3,
            )
            
            # Uitgebreide tekst met alle data (x, y, vx, vy, m/s, km/h)
            cv2.putText(result, f"Pos: ({int(x)}, {int(y)}) | Vel: ({vx:.1f}, {vy:.1f}) px/f", (10, 30),
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
                cv2.putText(result, f"Predicting... Pos: ({int(x)}, {int(y)}) | Vel: ({vx:.1f}, {vy:.1f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.putText(result, "HSV Searching...", (10, 30),
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




# Standalone test block (ondersteunt zowel webcam als dataset video)
if __name__ == "__main__":
    print("Standalone test van HsvEngine gestart...")
    engine = HsvEngine()
    
    # Kies hier of je de dataset video wilt testen (True) of de webcam (False)
    gebruik_dataset_video = True
    video_pad = r"C:\Users\timoz\Documents\00.school\appilicatie\NHLstenden-Air-Hockey-Ai\Orion Vision\dataset02\realtime_30fps_potje01.mov"

    if gebruik_dataset_video:
        print(f"Video laden vanuit dataset: {video_pad}")
        cap = cv2.VideoCapture(video_pad)
    else:
        print("Verbinden met webcam 0...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Waarschuwing: Kan video of webcam niet openen. Test wordt gedraaid met een zwart dummy-frame.")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        res, msk = engine.process_frame(dummy)
        print("Dummy frame succesvol verwerkt! Output shape:", res.shape)
    else:
        print("Bron succesvol geopend. Druk op de toets 'q' in het openstaande venster om de test te stoppen.")
        
        # Maak vensters aan met flag WINDOW_NORMAL zodat je ze zelf kunt schalen of kunt fixeren
        cv2.namedWindow("HSV Engine + Kalman - Resultaat", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Engine + Kalman - Masker", cv2.WINDOW_NORMAL)
        
        # Optioneel: geef het venster direct een vast handig formaat (bijv. 1280x720)
        cv2.resizeWindow("HSV Engine + Kalman - Resultaat", 1024, 576)
        cv2.resizeWindow("HSV Engine + Kalman - Masker", 1024, 576)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Einde van de video bereikt of geen frames meer van camera.")
                break

            # Verwerk het frame via de engine
            processed_frame, mask_frame = engine.process_frame(frame)

            # Toon de resultaten
            cv2.imshow("HSV Engine + Kalman - Resultaat", processed_frame)
            cv2.imshow("HSV Engine + Kalman - Masker", mask_frame)

            # Wacht 30ms per frame bij video, of 1ms bij webcam
            wacht_tijd = 30 if gebruik_dataset_video else 1
            if cv2.waitKey(wacht_tijd) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    print("Test afgesloten.")
