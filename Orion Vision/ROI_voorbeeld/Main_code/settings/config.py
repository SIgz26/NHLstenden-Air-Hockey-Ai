"""
Alle instellingen op één plek.
Pas hier aan, niet in de pipeline bestanden.
"""

from vmbpy import PixelFormat

# ── Camera ──────────────────────────────────────────────────
PIXEL_FORMAT        = PixelFormat.Bgr8
CAMERA_BUFFER_COUNT = 10
AUTO_MAX_FRAMERATE  = True

# ── ROI (Region of Interest) ────────────────────────────────
ROI_WIDTH    = 800
ROI_HEIGHT   = 550
ROI_OFFSET_X = 310
ROI_OFFSET_Y = 285

# ── Fisheye correctie ────────────────────────────────────────
# Startwaarden voor handmatige slider modus
K1_START = -0.40
K2_START =  0.05

# Slider bereik
K1_MIN = -1.00
K1_MAX =  0.50
K2_MIN = -0.30
K2_MAX =  0.30

SLIDER_STEPS = 1000

# ── Display ──────────────────────────────────────────────────
WINDOW_NAME         = "Air Hockey"
SLIDER_WINDOW_NAME  = "Fisheye Instellingen"