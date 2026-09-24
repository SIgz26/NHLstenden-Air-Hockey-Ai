"""
Display pipeline stap.

Verantwoordelijk voor:
    - Camera venster tonen
    - Slider venster voor handmatige correctie
    - Toetsinvoer verwerken
    - Status overlay op frame tekenen
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from settings.config import (
    WINDOW_NAME, SLIDER_WINDOW_NAME,
    K1_MIN, K1_MAX, K2_MIN, K2_MAX, SLIDER_STEPS
)
from result import StepResult
from components.correction import FisheyeCorrector


# ── Slider hulpfuncties ──────────────────────────────────────

def slider_to_k(val: int, k_min: float, k_max: float) -> float:
    return k_min + (val / SLIDER_STEPS) * (k_max - k_min)


def k_to_slider(k: float, k_min: float, k_max: float) -> int:
    val = int((k - k_min) / (k_max - k_min) * SLIDER_STEPS)
    return max(0, min(SLIDER_STEPS, val))


# ============================================================
# SLIDER VENSTER
# ============================================================

class SliderWindow:

    def __init__(self, corrector: FisheyeCorrector):
        self._corrector    = corrector
        self._frame_size   = None
        self._open         = False
        self._last_k1      = None
        self._last_k2      = None

    def open(self, frame_size: tuple) -> StepResult:
        self._frame_size = frame_size
        cv2.namedWindow(SLIDER_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(SLIDER_WINDOW_NAME, 500, 120)

        k1_pos = k_to_slider(self._corrector.k1, K1_MIN, K1_MAX)
        k2_pos = k_to_slider(self._corrector.k2, K2_MIN, K2_MAX)

        cv2.createTrackbar(
            "k1 (barrel<0)", SLIDER_WINDOW_NAME,
            k1_pos, SLIDER_STEPS, self._on_change
        )
        cv2.createTrackbar(
            "k2 (fijn)", SLIDER_WINDOW_NAME,
            k2_pos, SLIDER_STEPS, self._on_change
        )

        self._open = True
        return StepResult.ok("Slider venster geopend")

    def _on_change(self, _):
        if not self._open or not self._frame_size:
            return

        k1_pos = cv2.getTrackbarPos("k1 (barrel<0)", SLIDER_WINDOW_NAME)
        k2_pos = cv2.getTrackbarPos("k2 (fijn)", SLIDER_WINDOW_NAME)

        if k1_pos == self._last_k1 and k2_pos == self._last_k2:
            return

        self._last_k1 = k1_pos
        self._last_k2 = k2_pos

        k1 = slider_to_k(k1_pos, K1_MIN, K1_MAX)
        k2 = slider_to_k(k2_pos, K2_MIN, K2_MAX)

        self._corrector.update(k1, k2, self._frame_size)
        print(f"[SLIDER] k1={k1:.4f}  k2={k2:.4f}", flush=True)

    def sync_from_corrector(self):
        """Zet sliders op huidige corrector waarden (na auto-kalibratie)."""
        if not self._open:
            return
        cv2.setTrackbarPos(
            "k1 (barrel<0)", SLIDER_WINDOW_NAME,
            k_to_slider(self._corrector.k1, K1_MIN, K1_MAX)
        )
        cv2.setTrackbarPos(
            "k2 (fijn)", SLIDER_WINDOW_NAME,
            k_to_slider(self._corrector.k2, K2_MIN, K2_MAX)
        )


# ============================================================
# HOOFDVENSTER + TOETSEN
# ============================================================

class DisplayManager:
    """
    Beheert het camera venster en verwerkt toetsen.

    Gebruik:
        dm = DisplayManager(corrector)
        dm.setup()

        for frame in frame_generator:
            action = dm.update(frame)
            if action == 'quit':
                break
    """

    # Toets → actie mapping (leesbaar in main.py)
    KEY_ACTIONS = {
        ord('k'): 'calibrate',
        ord('K'): 'calibrate',
        ord('s'): 'sliders',
        ord('S'): 'sliders',
        ord('c'): 'toggle_correction',
        ord('C'): 'toggle_correction',
        ord('d'): 'toggle_debug',
        ord('D'): 'toggle_debug',
        ord('r'): 'reset',
        ord('R'): 'reset',
        13:       'quit',   # Enter
        27:       'quit',   # ESC
    }

    def __init__(self, corrector: FisheyeCorrector):
        self._corrector       = corrector
        self._slider_win      = SliderWindow(corrector)
        self._show_corrected  = True
        self._debug_mode      = False
        self._slider_open     = False

    def setup(self) -> StepResult:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        return StepResult.ok("Display venster aangemaakt")

    def update(self, raw_frame: np.ndarray) -> str:
        """
        Verwerk één frame: corrigeer, teken overlay, toon, lees toets.

        Returns:
            'quit'      → stop de lus
            'calibrate' → start automatische kalibratie
            ''          → geen actie
        """
        h, w = raw_frame.shape[:2]
        frame_size = (w, h)

        # ── Correctie ────────────────────────────────────────
        if self._corrector.calibrated and self._show_corrected:
            display = self._corrector.correct(raw_frame)
        else:
            display = raw_frame.copy()

        # ── Debug overlay ────────────────────────────────────
        if self._debug_mode:
            display = self._corrector.draw_ellipses(raw_frame)

        # ── Status tekst ─────────────────────────────────────
        self._draw_status(display)

        # ── Tonen ────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, display)

        # ── Toets ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        action = self.KEY_ACTIONS.get(key, '')

        return self._handle_action(action, raw_frame, frame_size)

    def _handle_action(
        self, action: str, raw_frame: np.ndarray, frame_size: tuple
    ) -> str:

        if action == 'toggle_correction':
            self._show_corrected = not self._show_corrected
            staat = "AAN" if self._show_corrected else "UIT"
            print(f"[DISPLAY] Correctie: {staat}")
            return ''

        elif action == 'toggle_debug':
            self._debug_mode = not self._debug_mode
            staat = "AAN" if self._debug_mode else "UIT"
            print(f"[DISPLAY] Debug modus: {staat}")
            return ''

        elif action == 'sliders':
            if not self._slider_open:
                self._slider_win.open(frame_size)
                self._slider_open = True
                # Activeer correctie met huidige k-waarden
                if not self._corrector.calibrated:
                    self._corrector.update(
                        self._corrector.k1,
                        self._corrector.k2,
                        frame_size
                    )
                self._show_corrected = True
            return ''

        elif action == 'reset':
            self._corrector.reset(frame_size)
            if self._slider_open:
                self._slider_win.sync_from_corrector()
            print("[DISPLAY] Correctie gereset (k1=0, k2=0)")
            return ''

        elif action == 'quit':
            return 'quit'

        # Venster handmatig gesloten?
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return 'quit'
        except cv2.error:
            return 'quit'

        return action  # Geef 'calibrate' etc. door aan main.py

    def after_calibration(self):
        """Aanroepen na succesvolle auto-kalibratie."""
        self._show_corrected = True
        if self._slider_open:
            self._slider_win.sync_from_corrector()

    def _draw_status(self, frame: np.ndarray):
        if self._corrector.calibrated and self._show_corrected:
            tekst = (
                f"Gecorrigeerd | "
                f"k1={self._corrector.k1:.3f} "
                f"k2={self._corrector.k2:.3f}"
            )
            kleur = (0, 255, 0)
        else:
            tekst = "GEEN correctie"
            kleur = (0, 0, 255)

        cv2.putText(
            frame,
            f"[K]=Auto [S]=Slider [C]=Toggle [R]=Reset | {tekst}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, kleur, 2
        )

    def destroy(self):
        cv2.destroyAllWindows()