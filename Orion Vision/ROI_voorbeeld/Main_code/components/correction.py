"""
Fisheye correctie pipeline stap.

Twee modi:
    Automatisch  - optimaliseer k1/k2 via ellips detectie (langzaam, eenmalig)
    Handmatig    - gebruik slider waarden (instant)

Na calibratie/update is correct() O(n_pixels) via cv2.remap().
Dit is de snelste mogelijke CPU methode.

Snelheidsoverzicht:
    cv2.remap()     ~ 1-3 ms voor 800x550    ← wij gebruiken dit
    cv2.undistort() ~ 5-10 ms                ← was langzamer
    GPU (CUDA)      ~ 0.1-0.3 ms             ← mogelijk later
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from scipy.optimize import minimize

from settings.config import K1_START, K2_START
from result import StepResult


class FisheyeCorrector:
    """
    Beheert fisheye correctie via remap tabellen.

    Attributen die voor main.py interessant zijn:
        corrector.k1          huidige k1 waarde
        corrector.k2          huidige k2 waarde
        corrector.calibrated  is er een geldige correctie actief?
    """

    def __init__(self):
        self.k1: float = K1_START
        self.k2: float = K2_START
        self.calibrated: bool = False

        # Intern
        self._camera_matrix     = None
        self._dist_coeffs       = None
        self._new_camera_matrix = None
        self._map1              = None
        self._map2              = None
        self._img_size          = None

    # ── Remap tabellen bouwen ────────────────────────────────

    def _build_maps(self, w: int, h: int, k1: float, k2: float) -> None:
        """
        Bouw remap-tabellen voor gegeven parameters.
        Wordt aangeroepen bij elke k1/k2 wijziging.

        Tijdsduur: ~10-50 ms (eenmalig per wijziging)
        Daarna is correct() slechts ~1-3 ms per frame.
        """
        fx = fy = w * 0.9
        cx, cy = w / 2, h / 2

        self._camera_matrix = np.array(
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]],
            dtype=np.float64
        )
        self._dist_coeffs = np.array(
            [k1, k2, 0.0, 0.0, 0.0],
            dtype=np.float64
        )
        self._new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix,
            self._dist_coeffs,
            (w, h), 1, (w, h)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._camera_matrix,
            self._dist_coeffs,
            None,
            self._new_camera_matrix,
            (w, h),
            cv2.CV_16SC2   # Integer lookup tabel = snelst
        )
        self._img_size = (w, h)

    # ── Publieke update methode ──────────────────────────────

    def update(self, k1: float, k2: float, frame_size: tuple) -> StepResult:
        """
        Pas k1/k2 aan en herbouw remap tabellen.

        Args:
            k1:         distortiecoëfficiënt k1
            k2:         distortiecoëfficiënt k2
            frame_size: (breedte, hoogte)
        """
        self.k1 = k1
        self.k2 = k2
        w, h = frame_size
        self._build_maps(w, h, k1, k2)
        self.calibrated = True

        return StepResult.ok(
            message=f"Correctie bijgewerkt: k1={k1:.4f}, k2={k2:.4f}"
        )

    # ── Automatische kalibratie ──────────────────────────────

    def calibrate(self, frame: np.ndarray) -> StepResult:
        """
        Bepaal automatisch optimale k1/k2 via ellips detectie.

        Tijdsduur: 5-30 seconden (eenmalig).
        Daarna is correct() snel.
        """
        h, w = frame.shape[:2]

        print("\n[CORRECTIE] Ellipsen zoeken...")
        ellipses = self._detect_ellipses(frame)
        print(f"[CORRECTIE] {len(ellipses)} ellips(en) gevonden.")

        if len(ellipses) < 2:
            return StepResult.fail(
                message="Automatische kalibratie mislukt: te weinig ellipsen.",
                details=(
                    "Zorg dat de cirkels op het speelveld goed zichtbaar zijn.\n"
                    "Gebruik handmatige modus [S] als alternatief."
                )
            )

        for i, e in enumerate(ellipses):
            cx, cy = e['center']
            a, b   = e['axes']
            r      = e['roundness']
            print(
                f"  Ellips {i+1}: "
                f"centrum=({cx:.0f},{cy:.0f}), "
                f"assen=({a:.0f},{b:.0f}), "
                f"ronding={r:.3f}"
            )

        print("[CORRECTIE] Optimaliseren (kan 10-30 sec duren)...")

        result = minimize(
            self._score,
            x0=[self.k1, self.k2],
            args=(frame,),
            method='Nelder-Mead',
            options={'xatol': 1e-4, 'fatol': 1e-5, 'maxiter': 300}
        )

        k1_opt, k2_opt = result.x

        # Bouw remap tabellen met gevonden waarden
        self._build_maps(w, h, k1_opt, k2_opt)
        self.k1 = k1_opt
        self.k2 = k2_opt
        self.calibrated = True

        return StepResult.ok(
            message=(
                f"Automatische kalibratie geslaagd: "
                f"k1={k1_opt:.5f}, k2={k2_opt:.5f} "
                f"(score={result.fun:.5f})"
            ),
            data={'k1': k1_opt, 'k2': k2_opt, 'score': result.fun}
        )

    def reset(self, frame_size: tuple) -> StepResult:
        """Zet correctie op nul (geen distortie)."""
        return self.update(0.0, 0.0, frame_size)

    # ── Correctie toepassen (per frame, moet snel zijn) ──────

    def correct(self, frame: np.ndarray) -> np.ndarray:
        """
        Pas fisheye correctie toe op frame.

        Tijdsduur: ~1-3 ms per frame (800x550).
        Geeft origineel frame terug als niet gekalibreerd.
        """
        if not self.calibrated:
            return frame

        return cv2.remap(
            frame,
            self._map1,
            self._map2,
            cv2.INTER_LINEAR
        )

    # ── Ellips detectie (intern) ─────────────────────────────

    def _detect_ellipses(self, frame: np.ndarray) -> list:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        edges   = cv2.Canny(blurred, 20, 80)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        ellipses = []
        h, w = frame.shape[:2]
        margin = 20

        for cnt in contours:
            if len(cnt) < 5:
                continue

            area = cv2.contourArea(cnt)
            if not (500 < area < 60000):
                continue

            ellipse        = cv2.fitEllipse(cnt)
            (cx, cy), (a, b), _ = ellipse

            if cx < margin or cx > w - margin:
                continue
            if cy < margin or cy > h - margin:
                continue
            if max(a, b) == 0:
                continue

            roundness = min(a, b) / max(a, b)
            if not (0.35 < roundness < 0.98):
                continue

            ellipses.append({
                'ellipse':   ellipse,
                'center':    (cx, cy),
                'axes':      (a, b),
                'roundness': roundness,
                'area':      area
            })

        ellipses.sort(key=lambda e: e['area'], reverse=True)
        return ellipses

    def _score(self, params: list, frame: np.ndarray) -> float:
        """Doelfunctie: lagere score = ronder na correctie."""
        k1, k2 = params
        h, w = frame.shape[:2]

        fx = fy = w * 0.9
        cx, cy = w / 2, h / 2
        cam_mat = np.array(
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]],
            dtype=np.float64
        )
        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
        new_mat, _ = cv2.getOptimalNewCameraMatrix(
            cam_mat, dist, (w, h), 1, (w, h)
        )
        corrected = cv2.undistort(frame, cam_mat, dist, None, new_mat)
        ellipses  = self._detect_ellipses(corrected)

        if len(ellipses) < 2:
            return 10.0

        return float(np.mean([(1.0 - e['roundness'])**2 for e in ellipses]))

    # ── Debug ────────────────────────────────────────────────

    def draw_ellipses(self, frame: np.ndarray) -> np.ndarray:
        """Teken gevonden ellipsen op frame (voor debug)."""
        debug    = frame.copy()
        ellipses = self._detect_ellipses(frame)

        for e in ellipses:
            cv2.ellipse(debug, e['ellipse'], (0, 255, 0), 2)
            cx, cy = int(e['center'][0]), int(e['center'][1])
            cv2.circle(debug, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                debug,
                f"{e['roundness']:.2f}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255), 1
            )

        return debug