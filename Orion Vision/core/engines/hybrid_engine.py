import sys
from pathlib import Path

# Zorg dat de hoofdmap (Orion Vision) in het zoekpad staat
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from core.engines.base_engine import BaseEngine


class HybridEngine(BaseEngine):
    """Hybrid puck detector combining Hough Circle Transform, MOG2 occlusion handling, 
    and customizable table boundary walls with visual line rendering."""

    name = "hybrid"
    parameter_schema = (
        {"name": "min_radius", "label": "Min Radius (px)", "type": "int", "min": int(5), "max": int(100), "default": int(15)},
        {"name": "max_radius", "label": "Max Radius (px)", "type": "int", "min": int(10), "max": int(150), "default": int(50)},
        {"name": "param2", "label": "Hough Sensitivity", "type": "int", "min": int(5), "max": int(100), "default": int(30)},
        {"name": "table_left", "label": "Tafel Links (X)", "type": "int", "min": int(0), "max": int(500), "default": int(50)},
        {"name": "table_right", "label": "Tafel Rechts (X)", "type": "int", "min": int(100), "max": int(2000), "default": int(1230)},
        {"name": "table_top", "label": "Tafel Boven (Y)", "type": "int", "min": int(0), "max": int(500), "default": int(50)},
        {"name": "table_bottom", "label": "Tafel Onder (Y)", "type": "int", "min": int(100), "max": int(2000), "default": int(670)},
    )

    def __init__(self) -> None:
        super().__init__()
        self.min_radius = 15
        self.max_radius = 25
        self.param2 = 30
        
        # Configureerbare tafelgrenzen (muren) in pixels
        self.table_left = 380
        self.table_right = 1580
        self.table_top = 190
        self.table_bottom = 900

        # Achtergrondsubstractor voor beweging / occlusie detectie
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=16, detectShadows=False
        )

        # Fysieke schaalfactor en framerate (24 FPS)
        self.PIXEL_TO_METER = 0.0015 
        self.ESTIMATED_FPS = 24.0  

        # Kalman filter initialisatie
        self._init_kalman()
        self.initialized = False
        self._last_center = None
        self.occluded_frames = 0
        self.MAX_OCCLUSION_FRAMES = 45

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

    def _apply_wall_physics(self, x: float, y: float, vx: float, vy: float, radius: int):
        """Kaatst de puck af op basis van de ingestelde tafelgrenzen in plaats van de videorand."""
        bounced = False
        
        # Linker muur
        if x - radius <= self.table_left:
            x = float(self.table_left + radius)
            vx = -vx * 0.92  # Energieverlies / demping bij botsing
            bounced = True
        # Rechter muur
        elif x + radius >= self.table_right:
            x = float(self.table_right - radius)
            vx = -vx * 0.92
            bounced = True

        # Bovenste muur
        if y - radius <= self.table_top:
            y = float(self.table_top + radius)
            vy = -vy * 0.92
            bounced = True
        # Onderste muur
        elif y + radius >= self.table_bottom:
            y = float(self.table_bottom - radius)
            vy = -vy * 0.92
            bounced = True

        return x, y, vx, vy, bounced

    def process_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            empty = np.zeros((0, 0), dtype=np.uint8)
            standardized = self.standardize_result(
                frame, 0.0, 0.0, 0.0, 0.0,
                mask=empty, secondary=empty,
                metadata={"status": "EMPTY", "in_range": False},
            )
            self.last_result = standardized
            return frame, empty

        result = frame.copy()
        
        # Teken de fysieke tafelgrenzen (muren) op het scherm zodat je kunt kalibreren (Blauwe lijnen)
        cv2.rectangle(result, (self.table_left, self.table_top), (self.table_right, self.table_bottom), (255, 100, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # 1. Hough Circles detectie
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        best_circle = None
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            best_score = float("-inf")
            for (cx, cy, r) in circles:
                score = 100.0
                if self._last_center is not None:
                    dist = np.hypot(cx - self._last_center[0], cy - self._last_center[1])
                    score -= dist * 0.5
                if score > best_score:
                    best_score = score
                    best_circle = (int(cx), int(cy), int(r))

        # 2. Achtergrondsubstractie masker
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        x = y = vx = vy = 0.0
        speed_ms = speed_kmh = 0.0
        status = "SEARCHING"
        current_radius = self.min_radius

        if best_circle is not None:
            cx, cy, current_radius = best_circle
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

            # Pas wandfysica toe op basis van de ingestelde tafelgrenzen
            x, y, vx, vy, _ = self._apply_wall_physics(x, y, vx, vy, current_radius)
            self.kalman.statePost = np.array([[x], [y], [vx], [vy]], dtype=np.float32)

            self._last_center = (cx, cy)
            self.occluded_frames = 0
            status = "TRACKING"

            speed_pixels_per_sec = np.hypot(vx, vy) * self.ESTIMATED_FPS
            speed_ms = speed_pixels_per_sec * self.PIXEL_TO_METER
            speed_kmh = speed_ms * 3.6

            # Visualisatie
            cv2.circle(result, (cx, cy), current_radius, (0, 0, 255), 2)
            cv2.circle(result, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.arrowedLine(result, (int(x), int(y)), (int(x + vx * 5), int(y + vy * 5)), (0, 255, 255), 2, tipLength=0.3)
            
            cv2.putText(result, f"Status: TRACKING | Vel: ({vx:.1f}, {vy:.1f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result, f"Speed: {speed_ms:.2f} m/s ({speed_kmh:.1f} km/h)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            if self.initialized and self.occluded_frames < self.MAX_OCCLUSION_FRAMES:
                self.occluded_frames += 1
                status = "BLOCKED / OCCLUDED (Physics Prediction)"

                prediction = self.kalman.predict()
                x = float(prediction[0, 0])
                y = float(prediction[1, 0])
                vx = float(prediction[2, 0])
                vy = float(prediction[3, 0])

                # Voorspel kaatsing tegen de ingestelde tafelranden tijdens blokkade
                x, y, vx, vy, bounced = self._apply_wall_physics(x, y, vx, vy, current_radius)
                self.kalman.statePost = np.array([[x], [y], [vx], [vy]], dtype=np.float32)

                speed_pixels_per_sec = np.hypot(vx, vy) * self.ESTIMATED_FPS
                speed_ms = speed_pixels_per_sec * self.PIXEL_TO_METER
                speed_kmh = speed_ms * 3.6

                cv2.circle(result, (int(x), int(y)), current_radius, (0, 165, 255), 2)
                bounce_txt = " | [BOUNCE!]" if bounced else ""
                cv2.putText(result, f"Status: BLOCKED{bounce_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(result, f"Est. Speed: {speed_ms:.2f} m/s ({speed_kmh:.1f} km/h)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                status = "SEARCHING"
                self.initialized = False
                cv2.putText(result, "Status: SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        standardized = self.standardize_result(
            result, x, y, vx, vy,
            mask=mask,
            secondary=mask_bgr,
            metadata={
                "status": status,
                "in_range": best_circle is not None or self.occluded_frames > 0,
                "speed_ms": speed_ms,
                "speed_kmh": speed_kmh
            },
        )
        self.last_result = standardized
        return result, mask_bgr


# Standalone test block
if __name__ == "__main__":
    print("Standalone test van HybridEngine met aangepaste muren gestart...")
    engine = HybridEngine()
    
    video_pad = r"C:\Users\timoz\Documents\00.school\appilicatie\NHLstenden-Air-Hockey-Ai\Orion Vision\dataset02\realtime_30fps_potje01.mov"
    cap = cv2.VideoCapture(video_pad)

    if not cap.isOpened():
        print("Kan video niet openen.")
    else:
        print("Video geopend. Druk op 'q' om te stoppen.")
        cv2.namedWindow("Hybrid Engine - Resultaat", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Hybrid Engine - Masker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hybrid Engine - Resultaat", 1024, 576)
        cv2.resizeWindow("Hybrid Engine - Masker", 1024, 576)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, mask_frame = engine.process_frame(frame)

            cv2.imshow("Hybrid Engine - Resultaat", processed_frame)
            cv2.imshow("Hybrid Engine - Masker", mask_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    print("Test afgesloten.")