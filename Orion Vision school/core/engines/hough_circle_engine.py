import sys
from pathlib import Path

# Zorg dat de hoofdmap (Orion Vision) in het zoekpad staat
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from core.engines.base_engine import BaseEngine


class RedPuckContourEngine(BaseEngine):
    """
    High-Speed Red Puck Detector for Air Hockey (290 FPS target).
    Uses LAB color space (A-channel), contour filtering, and Kalman tracking.
    """

    name = "red_puck_contour"
    parameter_schema = (
        {"name": "a_threshold", "label": "Red Sensitivity (LAB-A)", "type": "int", "min": 120, "max": 200, "default": 145},
        {"name": "min_area", "label": "Min Area (px)", "type": "int", "min": 10, "max": 2000, "default": 80},
        {"name": "max_area", "label": "Max Area (px)", "type": "int", "min": 100, "max": 8000, "default": 2500},
        {"name": "min_circularity", "label": "Min Circularity (%)", "type": "int", "min": 20, "max": 100, "default": 50},
        {"name": "max_dist_cm", "label": "Max Dist/Frame (cm)", "type": "int", "min": 1, "max": 30, "default": 8},
        {"name": "pixels_per_cm", "label": "Pixels per CM", "type": "int", "min": 1, "max": 100, "default": 10},
        {"name": "max_lost_frames", "label": "Max Lost Frames", "type": "int", "min": 1, "max": 60, "default": 15},
    )

    def __init__(self) -> None:
        super().__init__()
        # Slider parameters (Standaardwaarden)
        self.a_threshold = 145
        self.min_area = 80
        self.max_area = 2500
        self.min_circularity = 50  # Wordt later gedeeld door 100 (0.50)
        self.max_dist_cm = 8
        self.pixels_per_cm = 10
        self.max_lost_frames = 15

        # Fysieke schaalfactor
        self.ESTIMATED_FPS = 290.0  # Ingesteld op je snelle camera

        # Kalman filter & State tracking
        self._init_kalman()
        self.tracking_state = "LOST"  # 'TRACKING' of 'LOST'
        self.lost_frames = 0
        self._last_center = None

    def _init_kalman(self) -> None:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kalman = kf

    def update_settings(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                # UI-controls kunnen waarden als strings teruggeven.
                # Houd de numerieke parameters bruikbaar voor vergelijkingen.
                if key in {
                    "a_threshold",
                    "min_area",
                    "max_area",
                    "min_circularity",
                    "max_dist_cm",
                    "pixels_per_cm",
                    "max_lost_frames",
                }:
                    value = int(float(value))
                    if key in {"pixels_per_cm", "max_lost_frames"}:
                        value = max(1, value)
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

        result = frame.copy()

        # 1. LAB Kleurconversie (Ongevoelig voor schaduwen)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1]  # A-kanaal = Groen-tot-Rood as

        # Drempelwaarde toepassen op Rood
        _, mask = cv2.threshold(a_channel, self.a_threshold, 255, cv2.THRESH_BINARY)

        # 2. Ruis-verwijdering (Morphological Opening)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 3. Zoek contouren (Extreem snel op binarized beeld)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Kalman voorspelling ophalen voor dit frame
        prediction = self.kalman.predict()
        pred_x, pred_y = float(prediction[0, 0]), float(prediction[1, 0])

        best_candidate = None
        min_dist_to_pred = float("inf")
        candidates_found = 0

        # Bereken maximale verplaatsing in pixels
        max_dist_px = self.max_dist_cm * self.pixels_per_cm

        # 5. Filter contouren
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter 1: Grootte Check
            if self.min_area <= area <= self.max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                
                # Filter 2: Rondheid Check (Circularity)
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                min_circ_float = self.min_circularity / 100.0

                if circularity >= min_circ_float:
                    candidates_found += 1
                    
                    # Bereken middelpunt van de contour
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        continue

                    # Filter 3: Afstands Check (Alleen actief als we de puck AL VOLGEN)
                    if self.tracking_state == "TRACKING":
                        dist = np.hypot(cx - pred_x, cy - pred_y)
                        if dist <= max_dist_px:
                            if dist < min_dist_to_pred:
                                min_dist_to_pred = dist
                                best_candidate = (cx, cy, area, circularity)
                        else:
                            # Teken afgekeurde kandidaat die te ver weg ligt (Rood dun)
                            cv2.circle(result, (cx, cy), 5, (0, 0, 155), 1)
                    else:
                        # Als we de puck KWLIJT zijn, accepteer de eerste/beste geldige ronde vorm op het hele veld
                        best_candidate = (cx, cy, area, circularity)
                        break

        # 6. Kalman Update & State Machine
        x = y = vx = vy = 0.0
        speed_ms = speed_kmh = 0.0

        if best_candidate is not None:
            cx, cy, area, circ = best_candidate
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

            # Als we opnieuw opstarten, reset de Kalman positie direct naar de puck
            if self.tracking_state == "LOST":
                self.kalman.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
                self.tracking_state = "TRACKING"

            self.kalman.correct(measurement)
            self.lost_frames = 0

            x, y = float(cx), float(cy)
            vx = float(prediction[2, 0])
            vy = float(prediction[3, 0])

            # Teken de Puck (Groen) & Richting (Gele pijl)
            radius = int(np.sqrt(area / np.pi))
            cv2.circle(result, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 3, (0, 255, 0), -1)
            cv2.arrowedLine(result, (cx, cy), (int(cx + vx * 5), int(cy + vy * 5)), (0, 255, 255), 2)

        else:
            # Geen kandidaat gevonden in dit frame
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.tracking_state = "LOST"  # Geef zoekruimte weer vrij!

            if self.tracking_state == "TRACKING":
                # Blijf voorspellen via Kalman (Oranje cirkel)
                x, y = pred_x, pred_y
                vx = float(prediction[2, 0])
                vy = float(prediction[3, 0])
                cv2.circle(result, (int(x), int(y)), 8, (0, 165, 255), 2)

        # Telemetrie Berekeningen
        speed_px_per_sec = np.hypot(vx, vy) * self.ESTIMATED_FPS
        pixels_per_cm = max(1, int(self.pixels_per_cm))
        speed_ms = (speed_px_per_sec / pixels_per_cm) / 100.0  # cm/s naar m/s
        speed_kmh = speed_ms * 3.6

        # Teken Kalman Voorspelling (Blauw kruisje)
        cv2.drawMarker(result, (int(pred_x), int(pred_y)), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)

        # Visualiseer Status en Debug Info
        status_color = (0, 255, 0) if self.tracking_state == "TRACKING" else (0, 0, 255)
        cv2.putText(result, f"State: {self.tracking_state} | Lost: {self.lost_frames}/{self.max_lost_frames} | Candidates: {candidates_found}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        cv2.putText(result, f"Speed: {speed_ms:.2f} m/s ({speed_kmh:.1f} km/h) | Pos: ({int(x)}, {int(y)})",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        standardized = self.standardize_result(
            result, x, y, vx, vy,
            mask=mask,
            secondary=mask_bgr,
            metadata={
                "best": best_candidate,
                "in_range": best_candidate is not None,
                "speed_ms": speed_ms,
                "speed_kmh": speed_kmh,
                "state": self.tracking_state
            },
        )
        self.last_result = standardized
        return result, mask_bgr


# Compatibility alias: some engine loaders derive the class name from the
# module filename (``hough_circle_engine.py``).
HoughCircleEngine = RedPuckContourEngine


# Standalone test block
if __name__ == "__main__":
    print("Standalone test van RedPuckContourEngine gestart...")
    engine = RedPuckContourEngine()

    # Zorg dat het pad naar jouw video klopt
    video_pad = r"C:\Users\timoz\Documents\00.school\appilicatie\NHLstenden-Air-Hockey-Ai\Orion Vision\dataset02\realtime_30fps_potje01.mov"
    cap = cv2.VideoCapture(video_pad)

    if not cap.isOpened():
        print(f"Fout: Kan video niet openen op {video_pad}")
    else:
        cv2.namedWindow("Red Puck Engine - Resultaat", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Red Puck Engine - Masker (LAB)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Red Puck Engine - Resultaat", 1024, 576)
        cv2.resizeWindow("Red Puck Engine - Masker (LAB)", 1024, 576)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Einde van de video bereikt.")
                break

            processed_frame, mask_frame = engine.process_frame(frame)

            cv2.imshow("Red Puck Engine - Resultaat", processed_frame)
            cv2.imshow("Red Puck Engine - Masker (LAB)", mask_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    print("Test afgesloten.")
