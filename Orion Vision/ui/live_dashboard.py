import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QSlider, QSizePolicy, QComboBox, QGridLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSignal

from core.live_worker import LiveCameraWorker
from ui.dashboard import ClickableVideoLabel

class ClickableVideoLabel(QLabel):
    color_picked = pyqtSignal(float, float)

    def __init__(self, text=""):
        super().__init__(text)
        self.picking_mode = False

    def set_picking_mode(self, enabled: bool):
        self.picking_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event):
        if self.picking_mode and event.button() == Qt.LeftButton:
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                # Bereken exacte verhouding van het getoonde beeld t.o.v. het Label
                label_w, label_h = self.width(), self.height()
                pix_w, pix_h = pixmap.width(), pixmap.height()

                # Vondst van zwarte balken (KeepAspectRatio)
                off_x = (label_w - pix_w) / 2.0
                off_y = (label_h - pix_h) / 2.0

                click_x = event.x() - off_x
                click_y = event.y() - off_y

                if 0 <= click_x <= pix_w and 0 <= click_y <= pix_h:
                    norm_x = click_x / float(pix_w)
                    norm_y = click_y / float(pix_h)
                    self.color_picked.emit(norm_x, norm_y)
                    self.set_picking_mode(False)
        super().mousePressEvent(event)

class OrionLiveDashboard(QWidget):
    def __init__(self, on_back_callback=None):
        super().__init__()
        self.live_worker = None
        self.on_back = on_back_callback
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # 1. Header Frame
        header_frame = QFrame()
        header_frame.setObjectName("HeaderFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(30, 15, 30, 15)

        if self.on_back:
            btn_back = QPushButton("◀ MENU")
            btn_back.clicked.connect(self.go_back)
            header_layout.addWidget(btn_back)

        title_label = QLabel("ORION")
        title_label.setObjectName("TitleLabel")
        subtitle_label = QLabel("AI VISION // LIVE STREAM FEED")
        subtitle_label.setObjectName("SubTitleLabel")

        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        header_layout.addStretch()
        layout.addWidget(header_frame)

        # 2. Content Layout
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(30, 10, 30, 30)
        content_layout.setSpacing(20)

        # --- Card 1: Control Panel ---
        card1 = QFrame()
        card1.setObjectName("CardFrame")
        c1_layout = QVBoxLayout(card1)
        c1_layout.setContentsMargins(20, 20, 20, 20)
        c1_layout.setSpacing(12)

        c1_title = QLabel("CAMERA DEVICE")
        c1_title.setObjectName("SubTitleLabel")

        self.combo_camera = QComboBox()
        self.btn_refresh_cams = QPushButton("REFRESH CAMERAS")
        self.btn_refresh_cams.clicked.connect(self.refresh_cameras)

        self.btn_start = QPushButton("START LIVE STREAM")
        self.btn_start.clicked.connect(self.toggle_stream)

        self.status_label = QLabel("Status: Selecteer een camera")
        self.status_label.setWordWrap(True)

        c1_layout.addWidget(c1_title)
        c1_layout.addWidget(self.combo_camera)
        c1_layout.addWidget(self.btn_refresh_cams)
        c1_layout.addWidget(self.btn_start)
        c1_layout.addWidget(self.status_label)

        # Divider 1
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("color: #503422;")
        c1_layout.addWidget(line1)

        # View Mode Selector
        view_title = QLabel("VIEW MODE")
        view_title.setObjectName("SubTitleLabel")
        c1_layout.addWidget(view_title)

        self.combo_view = QComboBox()
        self.combo_view.addItems(["Single View (Live Result)", "Dual View (Result + Mask)"])
        self.combo_view.currentIndexChanged.connect(self.change_view_mode)
        c1_layout.addWidget(self.combo_view)

        # Divider 2
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color: #503422;")
        c1_layout.addWidget(line2)

        # Vision Settings Sliders & Preview
        settings_title = QLabel("HSV TUNING")
        settings_title.setObjectName("SubTitleLabel")
        c1_layout.addWidget(settings_title)

        preview_layout = QHBoxLayout()
        lbl_preview_text = QLabel("Target Color:")
        self.color_preview = QFrame()
        self.color_preview.setFixedSize(30, 20)
        self.color_preview.setStyleSheet("border: 1px solid #503422; border-radius: 3px;")

        self.btn_pick_color = QPushButton("PICK COLOR")
        self.btn_pick_color.clicked.connect(self.enable_color_picker)

        preview_layout.addWidget(lbl_preview_text)
        preview_layout.addWidget(self.color_preview)
        preview_layout.addWidget(self.btn_pick_color)
        preview_layout.addStretch()
        c1_layout.addLayout(preview_layout)

        self.slider_h_min, self.lbl_h_min = self.create_slider("Hue Min:", 0, 179, 35, c1_layout)
        self.slider_h_max, self.lbl_h_max = self.create_slider("Hue Max:", 0, 179, 95, c1_layout)
        self.slider_s_min, self.lbl_s_min = self.create_slider("Sat Min:", 0, 255, 20, c1_layout)
        self.slider_s_max, self.lbl_s_max = self.create_slider("Sat Max:", 0, 255, 255, c1_layout)

        self.update_color_preview()
        c1_layout.addStretch()

        # --- Card 2: Live Displays ---
        card2 = QFrame()
        card2.setObjectName("CardFrame")
        c2_layout = QVBoxLayout(card2)
        c2_layout.setContentsMargins(10, 10, 10, 10)
        c2_layout.setSpacing(10)

        self.grid_display = QGridLayout()
        self.grid_display.setContentsMargins(0, 0, 0, 0)

        timeline_layout = QHBoxLayout()
        self.btn_pause = QPushButton("FREEZE FRAME")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_freeze)
        timeline_layout.addWidget(self.btn_pause)

        c2_layout.addLayout(timeline_layout)
        c2_layout.addLayout(self.grid_display)

        self.feed_primary = ClickableVideoLabel("Live Camera Feed (Offline)")
        self.feed_primary.setAlignment(Qt.AlignCenter)
        self.feed_primary.setStyleSheet("background-color: #0c0a0a; border: 1px solid #503422;")
        self.feed_primary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.feed_primary.color_picked.connect(self.handle_color_picked)

        self.feed_secondary = QLabel("Live Mask Feed")
        self.feed_secondary.setAlignment(Qt.AlignCenter)
        self.feed_secondary.setStyleSheet("background-color: #0c0a0a; border: 1px solid #503422;")
        self.feed_secondary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.feed_secondary.hide()

        self.grid_display.addWidget(self.feed_primary, 0, 0)
        self.grid_display.addWidget(self.feed_secondary, 0, 1)

        content_layout.addWidget(card1, 1)
        content_layout.addWidget(card2, 3)

        layout.addLayout(content_layout)
        self.refresh_cameras()

    def enable_color_picker(self):
        self.btn_pick_color.setText("SELECTING...")
        self.feed_primary.set_picking_mode(True)

    def handle_color_picked(self, norm_x, norm_y):
        self.btn_pick_color.setText("PICK COLOR")
        
        if not self.live_worker or not hasattr(self.live_worker, 'current_hsv_frame') or self.live_worker.current_hsv_frame is None:
            self.status_label.setText("Fout: Start eerst de live stream om een kleur te kiezen!")
            return

        hsv_frame = self.live_worker.current_hsv_frame
        h_img, w_img = hsv_frame.shape[:2]
        px = max(1, min(w_img - 2, int(norm_x * w_img)))
        py = max(1, min(h_img - 2, int(norm_y * h_img)))

        roi = hsv_frame[py-1:py+2, px-1:px+2]
        avg_hsv = np.mean(roi, axis=(0, 1))
        h_val, s_val, v_val = int(avg_hsv[0]), int(avg_hsv[1]), int(avg_hsv[2])

        h_margin = 15 if s_val > 40 else 30
        s_margin = 50

        # Tijdelijk signals blokkeren voor nette GUI update
        self.slider_h_min.blockSignals(True)
        self.slider_h_max.blockSignals(True)
        self.slider_s_min.blockSignals(True)
        self.slider_s_max.blockSignals(True)

        self.slider_h_min.setValue(max(0, h_val - h_margin))
        self.slider_h_max.setValue(min(179, h_val + h_margin))
        self.slider_s_min.setValue(max(10, s_val - s_margin))
        self.slider_s_max.setValue(min(255, s_val + s_margin))

        self.slider_h_min.blockSignals(False)
        self.slider_h_max.blockSignals(False)
        self.slider_s_min.blockSignals(False)
        self.slider_s_max.blockSignals(False)

        # Update de tekstlabels
        self.lbl_h_min.setText(str(self.slider_h_min.value()))
        self.lbl_h_max.setText(str(self.slider_h_max.value()))
        self.lbl_s_min.setText(str(self.slider_s_min.value()))
        self.lbl_s_max.setText(str(self.slider_s_max.value()))

        # Direct doorschakelen naar de live camera worker
        self.sync_vision_settings()

        self.status_label.setText(f"Color Set | Target H:{h_val} S:{s_val} V:{v_val}")

    def refresh_cameras(self):
        self.combo_camera.clear()
        cams = LiveCameraWorker.get_available_cameras()
        if cams:
            for cam_id in cams:
                self.combo_camera.addItem(f"Camera Device {cam_id}", cam_id)
            self.status_label.setText(f"{len(cams)} camera('s) gevonden")
            self.btn_start.setEnabled(True)
        else:
            self.combo_camera.addItem("Geen camera gevonden")
            self.status_label.setText("Sluit een USB camera aan")
            self.btn_start.setEnabled(False)

    def go_back(self):
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.stop()
        if self.on_back:
            self.on_back()

    def toggle_stream(self):
        if self.live_worker is None or not self.live_worker.isRunning():
            cam_id = self.combo_camera.currentData()
            if cam_id is None:
                cam_id = 0

            self.live_worker = LiveCameraWorker(camera_index=cam_id)
            self.live_worker.frame_processed.connect(self.update_feed)
            self.live_worker.status_signal.connect(self.update_status)

            self.sync_vision_settings()
            self.live_worker.start()

            self.btn_start.setText("STOP LIVE STREAM")
            self.btn_refresh_cams.setEnabled(False)
            self.combo_camera.setEnabled(False)
            self.btn_pause.setEnabled(True)
        else:
            self.live_worker.stop()
            self.btn_start.setText("START LIVE STREAM")
            self.btn_refresh_cams.setEnabled(True)
            self.combo_camera.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_pause.setText("FREEZE FRAME")

    def toggle_freeze(self):
        if self.live_worker and self.live_worker.isRunning():
            is_paused = self.live_worker.toggle_pause()
            self.btn_pause.setText("UNFREEZE" if is_paused else "FREEZE FRAME")

    def update_color_preview(self):
        avg_h = int((self.slider_h_min.value() + self.slider_h_max.value()) / 2)
        avg_s = int((self.slider_s_min.value() + self.slider_s_max.value()) / 2)
        avg_v = 200

        hsv_pixel = np.uint8([[[avg_h, avg_s, avg_v]]])
        rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]

        r, g, b = rgb_pixel[0], rgb_pixel[1], rgb_pixel[2]
        self.color_preview.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #503422; border-radius: 3px;"
        )
    def sync_vision_settings(self):
        self.update_color_preview()
        # Gebruik self.live_worker in plaats van self.vision_worker
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.update_hsv(
                self.slider_h_min.value(), self.slider_h_max.value(),
                self.slider_s_min.value(), self.slider_s_max.value(),
                20, 255
            )

    def change_view_mode(self, index):
        if index == 0:
            self.feed_secondary.hide()
        else:
            self.feed_secondary.show()

    def create_slider(self, text, min_v, max_v, default_v, parent_layout):
        row = QHBoxLayout()
        lbl_title = QLabel(text)
        lbl_val = QLabel(str(default_v))
        lbl_val.setStyleSheet("color: #b86230; font-weight: bold;")
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default_v)
        slider.valueChanged.connect(lambda v, l=lbl_val: l.setText(str(v)))
        slider.valueChanged.connect(self.sync_vision_settings)

        row.addWidget(lbl_title)
        row.addWidget(slider)
        row.addWidget(lbl_val)
        parent_layout.addLayout(row)
        return slider, lbl_val

    def update_feed(self, pixmap_result, pixmap_mask):
        if not pixmap_result.isNull():
            scaled_res = pixmap_result.scaled(
                self.feed_primary.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.feed_primary.setPixmap(scaled_res)

        if self.feed_secondary.isVisible() and not pixmap_mask.isNull():
            scaled_mask = pixmap_mask.scaled(
                self.feed_secondary.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.feed_secondary.setPixmap(scaled_mask)

    def update_status(self, text):
        self.status_label.setText(text)