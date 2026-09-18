import os
import time

import cv2
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from core.live_worker import LiveCameraWorker
from core.gige_worker import GigECameraWorker, VMBPY_AVAILABLE
from ui.debug_window import CameraDebugDialog

# Type alias zodat de type-checker beide workers accepteert
CameraWorker = LiveCameraWorker | GigECameraWorker


class ClickableVideoLabel(QLabel):
    """
    QLabel met kleur-pik modus.

    Bij een klik in picking-modus wordt ``color_picked`` geëmit met
    genormaliseerde (x, y) coördinaten t.o.v. het *getoonde* beeld
    (letterbox/pillarbox offset wordt verrekend).
    """

    color_picked = pyqtSignal(float, float)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._picking_mode = False

    def set_picking_mode(self, enabled: bool) -> None:
        self._picking_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        if self._picking_mode and event.button() == Qt.LeftButton:
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                pix_w, pix_h = pixmap.width(), pixmap.height()

                # Offset door Qt's KeepAspectRatio scaling
                off_x = (self.width()  - pix_w) / 2.0
                off_y = (self.height() - pix_h) / 2.0

                click_x = event.x() - off_x
                click_y = event.y() - off_y

                if 0 <= click_x <= pix_w and 0 <= click_y <= pix_h:
                    self.color_picked.emit(click_x / pix_w, click_y / pix_h)
                    self.set_picking_mode(False)

        super().mousePressEvent(event)


class OrionLiveDashboard(QWidget):
    """Live camera dashboard – ondersteunt USB (OpenCV) én GigE (VmbPy)."""

    def __init__(self, on_back_callback=None) -> None:
        super().__init__()
        # Expliciete union-type: kan LiveCameraWorker of GigECameraWorker zijn
        self.live_worker: CameraWorker | None = None
        self.on_back = on_back_callback
        self.recording_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "recordings",
        )
        self._last_feed_refresh = 0.0
        self._init_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI opbouw
    # ──────────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        layout.addWidget(self._build_header())
        layout.addLayout(self._build_content())
        self.refresh_cameras()

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("HeaderFrame")
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(30, 15, 30, 15)

        if self.on_back:
            btn_back = QPushButton("◀ MENU")
            btn_back.clicked.connect(self.go_back)
            hbox.addWidget(btn_back)

        title = QLabel("ORION")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("AI VISION // LIVE STREAM FEED")
        subtitle.setObjectName("SubTitleLabel")

        hbox.addWidget(title)
        hbox.addWidget(subtitle)
        hbox.addStretch()
        return header

    def _build_content(self) -> QHBoxLayout:
        content = QHBoxLayout()
        content.setContentsMargins(30, 10, 30, 30)
        content.setSpacing(20)
        content.addWidget(self._build_control_card(), stretch=1)
        content.addWidget(self._build_display_card(),  stretch=3)
        return content

    # ── Control card ──────────────────────────────────────────────────

    def _build_control_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(self._section_label("CAMERA DEVICE"))

        self.combo_camera = QComboBox()
        self.btn_refresh_cams = QPushButton("REFRESH CAMERAS")
        self.btn_refresh_cams.clicked.connect(self.refresh_cameras)

        self.btn_start = QPushButton("START LIVE STREAM")
        self.btn_start.clicked.connect(self.toggle_stream)

        self.btn_debug = QPushButton("DEBUG")
        self.btn_debug.clicked.connect(self._open_debug_window)

        self.btn_record = QPushButton("START RECORDING")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self._toggle_recording)

        self.fps_label = QLabel("Record FPS: 20")
        self.combo_record_fps = QComboBox()
        self.combo_record_fps.addItems(["5", "10", "15", "20", "25", "30", "60"])
        self.combo_record_fps.setCurrentText("20")
        self.combo_record_fps.currentIndexChanged.connect(self._apply_fps_settings)

        self.recording_dir_label = QLabel(f"Opname map: {self.recording_dir}")
        self.recording_dir_label.setWordWrap(True)

        self.btn_select_record_folder = QPushButton("SELECT SAVE FOLDER")
        self.btn_select_record_folder.clicked.connect(self._select_recording_dir)

        self.status_label = QLabel("Status: Selecteer een camera")
        self.status_label.setWordWrap(True)

        for widget in (
            self.combo_camera, self.btn_refresh_cams,
            self.btn_start, self.btn_debug, self.btn_record,
            self.fps_label, self.combo_record_fps,
            self.recording_dir_label, self.btn_select_record_folder,
            self.status_label,
        ):
            layout.addWidget(widget)

        layout.addWidget(self._divider())

        # View mode
        layout.addWidget(self._section_label("VIEW MODE"))
        self.combo_view = QComboBox()
        self.combo_view.addItems([
            "Single View (Live Result)",
            "Dual View (Result + Mask)",
        ])
        self.combo_view.currentIndexChanged.connect(self._change_view_mode)
        layout.addWidget(self.combo_view)

        layout.addWidget(self._divider())

        # HSV tuning
        layout.addWidget(self._section_label("HSV TUNING"))
        layout.addLayout(self._build_color_picker_row())

        self.slider_h_min, self.lbl_h_min = self._add_slider("Hue Min:", 0, 179,  35, layout)
        self.slider_h_max, self.lbl_h_max = self._add_slider("Hue Max:", 0, 179,  95, layout)
        self.slider_s_min, self.lbl_s_min = self._add_slider("Sat Min:", 0, 255,  20, layout)
        self.slider_s_max, self.lbl_s_max = self._add_slider("Sat Max:", 0, 255, 255, layout)

        self._update_color_preview()
        layout.addStretch()
        return card

    def _build_color_picker_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Target Color:"))

        self.color_preview = QFrame()
        self.color_preview.setFixedSize(30, 20)
        self.color_preview.setStyleSheet(
            "border: 1px solid #503422; border-radius: 3px;"
        )
        row.addWidget(self.color_preview)

        self.btn_pick_color = QPushButton("PICK COLOR")
        self.btn_pick_color.clicked.connect(self._enable_color_picker)
        row.addWidget(self.btn_pick_color)
        row.addStretch()
        return row

    # ── Display card ──────────────────────────────────────────────────

    def _build_display_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        btn_row = QHBoxLayout()
        self.btn_pause = QPushButton("FREEZE FRAME")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_freeze)
        btn_row.addWidget(self.btn_pause)
        layout.addLayout(btn_row)

        self.feed_primary = ClickableVideoLabel("Live Camera Feed (Offline)")
        self.feed_primary.setAlignment(Qt.AlignCenter)
        self.feed_primary.setStyleSheet(
            "background-color: #0c0a0a; border: 1px solid #503422;"
        )
        self.feed_primary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.feed_primary.color_picked.connect(self._handle_color_picked)

        self.feed_secondary = QLabel("Live Mask Feed")
        self.feed_secondary.setAlignment(Qt.AlignCenter)
        self.feed_secondary.setStyleSheet(
            "background-color: #0c0a0a; border: 1px solid #503422;"
        )
        self.feed_secondary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.feed_secondary.hide()

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.feed_primary,   0, 0)
        grid.addWidget(self.feed_secondary, 0, 1)
        layout.addLayout(grid)
        return card

    # ──────────────────────────────────────────────────────────────────
    # Camera beheer  ← ENIGE definitie van refresh_cameras & toggle_stream
    # ──────────────────────────────────────────────────────────────────

    def refresh_cameras(self) -> None:
        """Vult de combo met USB- én GigE-camera's."""
        self.combo_camera.clear()

        # USB camera's via OpenCV
        for cam_id in LiveCameraWorker.get_available_cameras():
            self.combo_camera.addItem(
                f"USB  | Camera {cam_id}",
                {"type": "usb", "id": cam_id},
            )

        # GigE camera's via VmbPy (optioneel)
        if VMBPY_AVAILABLE:
            for cam in GigECameraWorker.get_available_gige_cameras():
                self.combo_camera.addItem(
                    f"GigE | {cam['name']}  [{cam['id']}]",
                    {"type": "gige", "id": cam["id"]},
                )
        else:
            # Informatief item – niet selecteerbaar
            self.combo_camera.addItem(
                "⚠ GigE niet beschikbaar (installeer VmbPy)", None
            )

        total = self.combo_camera.count()
        has_valid = any(
            self.combo_camera.itemData(i) is not None
            for i in range(total)
        )
        self.status_label.setText(
            f"{total} camera('s) gevonden" if has_valid
            else "Geen camera gevonden – sluit een camera aan"
        )
        self.btn_start.setEnabled(has_valid)

    def toggle_stream(self) -> None:
        """Start of stop de actieve camera worker."""
        if self.live_worker is None or not self.live_worker.isRunning():
            self._start_stream()
        else:
            self._stop_stream()

    def _start_stream(self) -> None:
        payload = self.combo_camera.currentData()

        # Sla items zonder payload over (bv. het GigE-waarschuwings-item)
        if not isinstance(payload, dict):
            self.status_label.setText("Selecteer een geldige camera.")
            return

        # Kies de juiste worker op basis van het type
        if payload["type"] == "gige":
            self.live_worker = GigECameraWorker(camera_id=payload["id"])
        else:
            self.live_worker = LiveCameraWorker(camera_index=payload["id"])

        # Signalen koppelen – interface is identiek voor beide workers
        self.live_worker.frame_processed.connect(self._update_feed)
        self.live_worker.status_signal.connect(self.status_label.setText)
        if hasattr(self.live_worker, "debug_signal"):
            self.live_worker.debug_signal.connect(self._append_debug)

        self._sync_vision_settings()
        if hasattr(self.live_worker, "set_record_fps"):
            self.live_worker.set_record_fps(float(self.combo_record_fps.currentText()))
        self._apply_fps_settings()
        self.live_worker.start()

        self.btn_start.setText("STOP LIVE STREAM")
        self.btn_refresh_cams.setEnabled(False)
        self.combo_camera.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_record.setEnabled(True)

    def _stop_stream(self) -> None:
        if self.live_worker is None:
            return

        self.live_worker.stop()

        # Wacht maximaal 3 seconden netjes op de thread
        if not self.live_worker.wait(3000):
            self.live_worker.terminate()

        self.live_worker = None

        self.btn_start.setText("START LIVE STREAM")
        self.btn_refresh_cams.setEnabled(True)
        self.combo_camera.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("FREEZE FRAME")
        self.btn_record.setEnabled(False)
        self.btn_record.setText("START RECORDING")

    # ──────────────────────────────────────────────────────────────────
    # Slots & event handlers
    # ──────────────────────────────────────────────────────────────────

    def _append_debug(self, message: str) -> None:
        if hasattr(self, "debug_window") and self.debug_window is not None:
            self.debug_window.append_log(message)

    def _open_debug_window(self) -> None:
        self.debug_window = CameraDebugDialog(self)
        self.debug_window.show()
        self.debug_window.append_log("Camera debug window opened")
        if self.live_worker is not None:
            self.debug_window.append_log(f"Active worker: {type(self.live_worker).__name__}")
            if hasattr(self.live_worker, "debug_signal"):
                self.live_worker.debug_signal.connect(self._append_debug)
        else:
            self.debug_window.append_log("No active worker running")

    def _enable_color_picker(self) -> None:
        self.btn_pick_color.setText("SELECTING…")
        self.feed_primary.set_picking_mode(True)

    def _handle_color_picked(self, norm_x: float, norm_y: float) -> None:
        self.btn_pick_color.setText("PICK COLOR")

        if self.live_worker is None:
            self.status_label.setText("Fout: Start eerst de live stream.")
            return

        hsv_frame = self.live_worker.current_hsv_frame   # thread-safe property
        if hsv_frame is None:
            self.status_label.setText("Fout: Nog geen HSV frame beschikbaar.")
            return

        h_img, w_img = hsv_frame.shape[:2]
        px = int(np.clip(norm_x * w_img, 1, w_img - 2))
        py = int(np.clip(norm_y * h_img, 1, h_img - 2))

        # 3×3 patch voor ruis-robuustheid
        roi = hsv_frame[py - 1:py + 2, px - 1:px + 2]
        h_val, s_val, v_val = np.mean(roi, axis=(0, 1)).astype(int)

        h_margin = 15 if s_val > 40 else 30
        s_margin = 50

        sliders_values = [
            (self.slider_h_min, max(0,   h_val - h_margin)),
            (self.slider_h_max, min(179, h_val + h_margin)),
            (self.slider_s_min, max(10,  s_val - s_margin)),
            (self.slider_s_max, min(255, s_val + s_margin)),
        ]

        # Bulk-update zonder tussentijdse sync-aanroepen
        for slider, _ in sliders_values:
            slider.blockSignals(True)
        for slider, value in sliders_values:
            slider.setValue(value)
        for slider, _ in sliders_values:
            slider.blockSignals(False)

        # Label-tekst bijwerken
        for slider, lbl in zip(
            [self.slider_h_min, self.slider_h_max,
             self.slider_s_min, self.slider_s_max],
            [self.lbl_h_min,    self.lbl_h_max,
             self.lbl_s_min,    self.lbl_s_max],
        ):
            lbl.setText(str(slider.value()))

        self._sync_vision_settings()
        self.status_label.setText(
            f"Kleur ingesteld | H:{h_val}  S:{s_val}  V:{v_val}"
        )

    def _select_recording_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Kies een map voor opnames",
            self.recording_dir,
        )
        if folder:
            self.recording_dir = folder
            self.recording_dir_label.setText(f"Opname map: {self.recording_dir}")
            self.status_label.setText(f"Opname map ingesteld: {self.recording_dir}")

    def _apply_fps_settings(self) -> None:
        self.fps_label.setText(f"Record FPS: {self.combo_record_fps.currentText()}")

        if self.live_worker is not None and self.live_worker.isRunning():
            if hasattr(self.live_worker, "set_record_fps"):
                self.live_worker.set_record_fps(float(self.combo_record_fps.currentText()))

    def _toggle_recording(self) -> None:
        if self.live_worker is None or not self.live_worker.isRunning():
            self.status_label.setText("Start eerst de live stream voordat je opneemt.")
            return

        if self.live_worker.recording:
            path = self.live_worker.stop_recording()
            if path:
                self.status_label.setText(f"Opname opgeslagen: {path}")
            self.btn_record.setText("START RECORDING")
            return

        if not os.path.isdir(self.recording_dir):
            os.makedirs(self.recording_dir, exist_ok=True)

        if hasattr(self.live_worker, "set_record_fps"):
            self.live_worker.set_record_fps(float(self.combo_record_fps.currentText()))

        path = self.live_worker.start_recording(self.recording_dir)
        if path:
            self.status_label.setText(f"Opname gestart: {path}")
            self.btn_record.setText("STOP RECORDING")

    def go_back(self) -> None:
        # Stop de thread volledig voor we de UI verlaten
        if self.live_worker is not None and self.live_worker.isRunning():
            self.live_worker.stop_recording()
        self._stop_stream()
        if self.on_back:
            self.on_back()

    def toggle_freeze(self) -> None:
        if self.live_worker and self.live_worker.isRunning():
            paused = self.live_worker.toggle_pause()
            self.btn_pause.setText("UNFREEZE" if paused else "FREEZE FRAME")

    # ──────────────────────────────────────────────────────────────────
    # Interne helpers
    # ──────────────────────────────────────────────────────────────────

    def _update_color_preview(self) -> None:
        avg_h = (self.slider_h_min.value() + self.slider_h_max.value()) // 2
        avg_s = (self.slider_s_min.value() + self.slider_s_max.value()) // 2
        hsv_px = np.uint8([[[avg_h, avg_s, 200]]])
        r, g, b = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2RGB)[0, 0]
        self.color_preview.setStyleSheet(
            f"background-color: rgb({r},{g},{b});"
            " border: 1px solid #503422; border-radius: 3px;"
        )

    def _sync_vision_settings(self) -> None:
        self._update_color_preview()
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.update_hsv(
                self.slider_h_min.value(), self.slider_h_max.value(),
                self.slider_s_min.value(), self.slider_s_max.value(),
                v_min=20, v_max=255,
            )

    def _change_view_mode(self, index: int) -> None:
        self.feed_secondary.setVisible(index != 0)

    def _update_feed(self, pixmap_result: QPixmap, pixmap_mask: QPixmap) -> None:
        if not pixmap_result.isNull():
            self.feed_primary.setPixmap(
                pixmap_result.scaled(
                    self.feed_primary.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        if self.feed_secondary.isVisible() and not pixmap_mask.isNull():
            self.feed_secondary.setPixmap(
                pixmap_mask.scaled(
                    self.feed_secondary.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    # ── UI factory methodes ───────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SubTitleLabel")
        return lbl

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #503422;")
        return line

    def _add_slider(
        self,
        label_text: str,
        min_v: int,
        max_v: int,
        default_v: int,
        parent_layout: QVBoxLayout,
    ) -> tuple[QSlider, QLabel]:
        row = QHBoxLayout()

        lbl_title = QLabel(label_text)
        lbl_val   = QLabel(str(default_v))
        lbl_val.setStyleSheet("color: #b86230; font-weight: bold;")

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default_v)
        slider.valueChanged.connect(lambda v, l=lbl_val: l.setText(str(v)))
        slider.valueChanged.connect(self._sync_vision_settings)

        row.addWidget(lbl_title)
        row.addWidget(slider)
        row.addWidget(lbl_val)
        parent_layout.addLayout(row)
        return slider, lbl_val