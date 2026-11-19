import cv2
import json
import numpy as np
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QSettings, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QFileDialog, QLabel, QMessageBox, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from core.live_worker import LiveCameraWorker
from core.gige_worker import GigECameraWorker, VMBPY_AVAILABLE
from core.auto_calibration import ManualTableCalibrator
from core.sim_bridge import SimBridge
from ai.agent_worker import SACAgentWorker
from ai.scripted_worker import ScriptedOpponentWorker
from ai.sac_controller import LiveSACController
from core.opcua_worker import OpcUaWorker, limit_velocity
from ui.debug_window import CameraDebugDialog
from ui.widgets.digital_twin_widget import DigitalTwinWindow

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
    table_corner_picked = pyqtSignal(float, float)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._picking_mode = False
        self._table_calibration_mode = False

    def set_picking_mode(self, enabled: bool) -> None:
        self._picking_mode = enabled
        self._set_interaction_cursor()

    def set_table_calibration_mode(self, enabled: bool) -> None:
        """Enable normalized clicks for ordered manual table calibration."""
        self._table_calibration_mode = enabled
        self._set_interaction_cursor()

    def _set_interaction_cursor(self) -> None:
        active = self._picking_mode or self._table_calibration_mode
        self.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        if (self._table_calibration_mode or self._picking_mode) and event.button() == Qt.LeftButton:
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                pix_w, pix_h = pixmap.width(), pixmap.height()

                # Offset door Qt's KeepAspectRatio scaling
                off_x = (self.width()  - pix_w) / 2.0
                off_y = (self.height() - pix_h) / 2.0

                click_x = event.x() - off_x
                click_y = event.y() - off_y

                if 0 <= click_x <= pix_w and 0 <= click_y <= pix_h:
                    normalized_x = click_x / pix_w
                    normalized_y = click_y / pix_h
                    if self._table_calibration_mode:
                        self.table_corner_picked.emit(normalized_x, normalized_y)
                    else:
                        self.color_picked.emit(normalized_x, normalized_y)
                        self.set_picking_mode(False)
                    event.accept()
                    return

        super().mousePressEvent(event)


class OrionLiveDashboard(QWidget):
    """Live camera dashboard – ondersteunt USB (OpenCV) én GigE (VmbPy)."""

    sac_action_ready = pyqtSignal(float, float)

    def __init__(self, on_back_callback=None) -> None:
        super().__init__()
        # Expliciete union-type: kan LiveCameraWorker of GigECameraWorker zijn
        self.live_worker: CameraWorker | None = None
        self.digital_twin_window: DigitalTwinWindow | None = None
        self._manual_table_calibrator = ManualTableCalibrator()
        self._calibration_frame_size: tuple[int, int] | None = None
        self._saved_calibration_restored = False
        self._settings = QSettings("Orion Vision", "NHLstenden Air Hockey AI")
        self.sac_controller: LiveSACController | None = None
        self.sac_worker: SACAgentWorker | None = None
        self.sac_model_path: str | None = None
        self.sac_action_sink = None
        self._latest_sac_action: tuple[float, float] | None = None
        self.scripted_worker: ScriptedOpponentWorker | None = None
        self.opcua_worker: OpcUaWorker | None = None
        self._controller_generation = 0
        self._action_lock = threading.Lock()
        self._latest_action_mm_s = (0.0, 0.0)
        self._restart_controller_after_stop = False
        self._stream_stopping = False
        self._opcua_connected = False
        self._pending_zero_revision: int | None = None
        self._plc_homed = False
        self._plc_pose_valid = False
        self._plc_has_error = False
        self._alignment_target_pending: tuple[float, float] | None = None
        self._alignment_target_table_mm: tuple[float, float] | None = None
        self._alignment_target_plc_mm: tuple[float, float] | None = None
        self._alignment_active = False
        self._manual_twin_control_active = False
        self._manual_twin_signal_connected = False
        self._alignment_timer = QTimer(self)
        self._alignment_timer.setInterval(20)
        self._alignment_timer.timeout.connect(self._tick_alignment)
        self.on_back = on_back_callback
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
        self._refresh_alignment_enabled()

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

        self.combo_engine = QComboBox()
        self.combo_engine.addItems([
            "HSV Detection",
            "Hough Circle",
            "Background Subtraction",
            "Circle Contour",
            "Hybrid",
        ])
        self.combo_engine.setCurrentText("HSV Detection")
        self.combo_engine.currentIndexChanged.connect(self._refresh_engine_settings_ui)

        self.btn_start = QPushButton("START LIVE STREAM")
        self.btn_start.clicked.connect(self.toggle_stream)

        self.btn_debug = QPushButton("DEBUG")
        self.btn_debug.clicked.connect(self._open_debug_window)

        self.status_label = QLabel("Status: Selecteer een camera")
        self.status_label.setWordWrap(True)

        for widget in (
            self.combo_camera, self.combo_engine,
            self.btn_refresh_cams, self.btn_start,
            self.btn_debug, self.status_label,
        ):
            layout.addWidget(widget)

        self.chk_show_table_overlay = QCheckBox("Show Table Overlay")
        self.chk_show_table_overlay.setChecked(True)
        self.chk_show_table_overlay.setStyleSheet(
            "QCheckBox { color: #E0E0E0; font-weight: bold; }"
        )
        self.chk_show_table_overlay.toggled.connect(self._on_toggle_table_overlay)
        layout.addWidget(self.chk_show_table_overlay)

        self.btn_calibrate_table_corners = QPushButton("CALIBRATE TABLE CORNERS")
        self.btn_calibrate_table_corners.clicked.connect(self._start_table_calibration)
        layout.addWidget(self.btn_calibrate_table_corners)
        self.btn_reset_table_corners = QPushButton("RESET CORNERS")
        self.btn_reset_table_corners.clicked.connect(self._reset_table_corners)
        layout.addWidget(self.btn_reset_table_corners)

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

        # Fisheye correction
        layout.addWidget(self._section_label("FISHEYE CORRECTION"))
        self.fisheye_enabled = QCheckBox("Apply correction")
        self.fisheye_enabled.setChecked(True)
        self.fisheye_enabled.stateChanged.connect(self._sync_vision_settings)
        layout.addWidget(self.fisheye_enabled)

        self.fisheye_k1_slider = self._add_numeric_slider("k1", -1000, 500, -400, 1000, layout)
        self.fisheye_k2_slider = self._add_numeric_slider("k2", -300, 300, 50, 1000, layout)

        layout.addWidget(self._section_label("ENGINE SETTINGS"))
        self.engine_settings_widget = QWidget()
        self.engine_settings_layout = QVBoxLayout(self.engine_settings_widget)
        self.engine_settings_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.engine_settings_widget)

        self.engine_setting_controls = {}
        self.hsv_slider_map = {}
        self._engine_setting_widgets = set()
        self._refresh_engine_settings_ui()
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
        self.btn_open_digital_twin = QPushButton("OPEN DIGITAL TWIN")
        self.btn_open_digital_twin.clicked.connect(self._open_digital_twin)
        btn_row.addWidget(self.btn_open_digital_twin)
        layout.addLayout(btn_row)

        agent_panel = QFrame()
        agent_panel.setObjectName("CardFrame")
        agent_layout = QVBoxLayout(agent_panel)
        agent_layout.setContentsMargins(12, 8, 12, 8)
        agent_layout.setSpacing(6)
        agent_layout.addWidget(self._section_label("AI AGENT CONTROL"))

        self.combo_controller = QComboBox()
        self.combo_controller.addItems(
            [
                "None (Idle)",
                "SAC Agent",
                "Complex Opponent",
                "Attacker",
                "Defender",
                "Striker Defender",
                "Muisbesturing (Virtual Twin)",
            ]
        )
        self.combo_controller.currentTextChanged.connect(
            self._on_controller_selected
        )
        agent_layout.addWidget(self.combo_controller)

        agent_buttons = QHBoxLayout()
        self.btn_load_sac_model = QPushButton("LOAD MODEL")
        self.btn_load_sac_model.clicked.connect(self._load_sac_model)
        agent_buttons.addWidget(self.btn_load_sac_model)
        self.btn_toggle_sac_agent = QPushButton("START SAC AGENT")
        self.btn_toggle_sac_agent.setEnabled(False)
        self.btn_toggle_sac_agent.clicked.connect(self._toggle_sac_agent)
        agent_buttons.addWidget(self.btn_toggle_sac_agent)
        agent_layout.addLayout(agent_buttons)

        self.lbl_sac_model = QLabel("No model loaded")
        self.lbl_sac_action = QLabel("Target Vx: --    Target Vy: --")
        self.lbl_sac_status = QLabel("Agent stopped")
        agent_layout.addWidget(self.lbl_sac_model)
        agent_layout.addWidget(self.lbl_sac_action)
        agent_layout.addWidget(self.lbl_sac_status)

        opcua_row = QHBoxLayout()
        self.btn_toggle_opcua = QPushButton("CONNECT OPC UA")
        self.btn_toggle_opcua.setCheckable(True)
        self.btn_toggle_opcua.toggled.connect(self._toggle_opcua_connection)
        opcua_row.addWidget(self.btn_toggle_opcua)
        self.lbl_opcua_status = QLabel("OPC UA: Disconnected")
        self.lbl_opcua_status.setWordWrap(True)
        opcua_row.addWidget(self.lbl_opcua_status, stretch=1)
        agent_layout.addLayout(opcua_row)
        self.lbl_plc_machine_status = QLabel(
            "PLC: Homed -- | Tracking -- | Error -- | State --"
        )
        self.lbl_plc_machine_status.setWordWrap(True)
        agent_layout.addWidget(self.lbl_plc_machine_status)

        layout.addWidget(agent_panel)
        layout.addWidget(self._build_alignment_panel())

        self.feed_primary = ClickableVideoLabel("Live Camera Feed (Offline)")
        self.feed_primary.setAlignment(Qt.AlignCenter)
        self.feed_primary.setStyleSheet(
            "background-color: #0c0a0a; border: 1px solid #503422;"
        )
        self.feed_primary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.feed_primary.color_picked.connect(self._handle_color_picked)
        self.feed_primary.table_corner_picked.connect(self._on_table_corner_picked)

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
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        return card

    def _build_alignment_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("CardFrame")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(5)
        layout.addWidget(self._section_label("Veld-uitlijning (Stip-kalibratie)"))

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Stip X (mm)"))
        self.alignment_x_spin = QDoubleSpinBox()
        self.alignment_x_spin.setRange(-416.0, 416.0)
        self.alignment_x_spin.setDecimals(1)
        self.alignment_x_spin.setSingleStep(1.0)
        self.alignment_x_spin.setValue(0.0)
        target_row.addWidget(self.alignment_x_spin)
        target_row.addWidget(QLabel("Y (mm)"))
        self.alignment_y_spin = QDoubleSpinBox()
        self.alignment_y_spin.setRange(0.0, 735.0)
        self.alignment_y_spin.setDecimals(1)
        self.alignment_y_spin.setSingleStep(1.0)
        self.alignment_y_spin.setValue(420.0)
        target_row.addWidget(self.alignment_y_spin)

        self.btn_go_to_stip = QPushButton("Ga naar Stip")
        self.btn_go_to_stip.clicked.connect(self._start_alignment)
        target_row.addWidget(self.btn_go_to_stip)
        self.btn_stip_left = QPushButton("Ga naar Stip 1 (-281, 430 mm)")
        self.btn_stip_left.clicked.connect(
            lambda: self._go_to_preset_stip(-281.0, 430.0)
        )
        target_row.addWidget(self.btn_stip_left)
        self.btn_stip_right = QPushButton("Ga naar Stip 2 (+260, 430 mm)")
        self.btn_stip_right.clicked.connect(
            lambda: self._go_to_preset_stip(260.0, 430.0)
        )
        target_row.addWidget(self.btn_stip_right)
        self.btn_mirror_stip_x = QPushButton("Spiegel X-zijde")
        self.btn_mirror_stip_x.clicked.connect(self._mirror_stip_x)
        target_row.addWidget(self.btn_mirror_stip_x)
        self.btn_stop_alignment = QPushButton("Stop Uitlijning")
        self.btn_stop_alignment.clicked.connect(self._stop_alignment)
        self.btn_stop_alignment.setEnabled(False)
        target_row.addWidget(self.btn_stop_alignment)
        layout.addLayout(target_row)

        self._trim_sliders: dict[str, QSlider] = {}
        self._trim_value_labels: dict[str, QLabel] = {}
        for key, label in (
            ("robot_trim_x_mm", "Robot X"),
            ("robot_trim_y_mm", "Robot Y"),
            ("puck_trim_x_mm", "Puck X"),
            ("puck_trim_y_mm", "Puck Y"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            decrease = QPushButton("-")
            decrease.setFixedWidth(30)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-70, 70)
            initial = float(self._settings.value(key, 0.0))
            slider.setValue(int(round(np.clip(initial, -35.0, 35.0) * 2.0)))
            increase = QPushButton("+")
            increase.setFixedWidth(30)
            value_label = QLabel(f"{slider.value() / 2.0:+.1f} mm")
            value_label.setMinimumWidth(60)
            decrease.clicked.connect(
                lambda _checked=False, trim_key=key: self._adjust_trim(trim_key, -1)
            )
            increase.clicked.connect(
                lambda _checked=False, trim_key=key: self._adjust_trim(trim_key, 1)
            )
            slider.valueChanged.connect(
                lambda value, trim_key=key: self._on_trim_changed(trim_key, value)
            )
            row.addWidget(decrease)
            row.addWidget(slider, stretch=1)
            row.addWidget(increase)
            row.addWidget(value_label)
            layout.addLayout(row)
            self._trim_sliders[key] = slider
            self._trim_value_labels[key] = value_label

        trim_buttons = QHBoxLayout()
        save_x = QPushButton("X Opslaan")
        save_x.clicked.connect(lambda: self._save_trim_axis("x"))
        save_y = QPushButton("Y Opslaan")
        save_y.clicked.connect(lambda: self._save_trim_axis("y"))
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset_trims)
        trim_buttons.addWidget(save_x)
        trim_buttons.addWidget(save_y)
        trim_buttons.addWidget(reset)
        layout.addLayout(trim_buttons)
        self.lbl_alignment_status = QLabel("Uitlijning inactief")
        layout.addWidget(self.lbl_alignment_status)
        return panel

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

        self._apply_bridge_trims(self.live_worker.sim_bridge)
        self._refresh_alignment_enabled()
        engine_name = self.combo_engine.currentText()
        self.live_worker.set_engine(engine_name)
        self.live_worker.set_show_table_overlay(self.chk_show_table_overlay.isChecked())
        self._saved_calibration_restored = False
        if self.digital_twin_window is not None:
            self.digital_twin_window.update_bridge(self.live_worker.sim_bridge)

        # Signalen koppelen – interface is identiek voor beide workers
        self.live_worker.frame_processed.connect(self._update_feed)
        self.live_worker.status_signal.connect(self.status_label.setText)
        if isinstance(self.live_worker, LiveCameraWorker):
            self.live_worker.cycle_metrics_updated.connect(
                self._on_camera_metrics_updated
            )
        if hasattr(self.live_worker, "debug_signal"):
            self.live_worker.debug_signal.connect(self._append_debug)

        self._sync_vision_settings()
        self.live_worker.start()

        self.btn_start.setText("STOP LIVE STREAM")
        self.btn_refresh_cams.setEnabled(False)
        self.combo_camera.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self._refresh_alignment_enabled()
        self._start_selected_controller()

    def _stop_stream(self) -> None:
        if self.live_worker is None:
            return

        self._stream_stopping = True
        self._stop_active_controller()

        self.live_worker.stop()

        # Wacht maximaal 3 seconden netjes op de thread
        if not self.live_worker.wait(3000):
            self.live_worker.terminate()

        self.live_worker = None
        self._stream_stopping = False
        self._refresh_alignment_enabled()
        self.feed_primary.set_table_calibration_mode(False)
        self._manual_table_calibrator.reset()
        self._calibration_frame_size = None
        if self.digital_twin_window is not None:
            self.digital_twin_window.update_bridge(None)

        self.btn_start.setText("START LIVE STREAM")
        self.btn_refresh_cams.setEnabled(True)
        self.combo_camera.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("FREEZE FRAME")

    def _load_sac_model(self) -> None:
        """Select and load a Stable-Baselines3 SAC archive."""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load SAC model",
            "",
            "Stable-Baselines3 model (*.zip)",
        )
        if not model_path:
            return
        try:
            controller = LiveSACController(
                self.live_worker.sim_bridge if self.live_worker else SimBridge()
            )
            controller.load_model(model_path)
            self.sac_controller = controller
            self.sac_model_path = model_path
            self.lbl_sac_model.setText(f"Model: {model_path}")
            self.lbl_sac_status.setText("Model loaded; start live camera to run agent")
            self.btn_toggle_sac_agent.setEnabled(True)
            if self.combo_controller.currentText() == "SAC Agent":
                self._start_sac_agent()
        except Exception as exc:
            self.sac_controller = None
            self.sac_model_path = None
            self.btn_toggle_sac_agent.setEnabled(False)
            self.lbl_sac_model.setText("No model loaded")
            self.lbl_sac_status.setText(f"Model load failed: {exc}")

    def _toggle_sac_agent(self) -> None:
        """Select SAC, or toggle it when it is already the active controller."""
        if self.combo_controller.currentText() != "SAC Agent":
            self.combo_controller.setCurrentText("SAC Agent")
            return
        if self.sac_worker is not None and self.sac_worker.isRunning():
            self.combo_controller.setCurrentText("None (Idle)")
            return
        self._start_sac_agent()

    def _start_sac_agent(self) -> None:
        """Start the SAC inference worker with the common mm/s action sink."""
        if self._pending_zero_revision is not None:
            return
        if self.live_worker is None or not self.live_worker.isRunning():
            self.lbl_sac_status.setText("Start the live camera before the agent.")
            return
        if not self.live_worker.sim_bridge.physical_table_calibrated:
            self.lbl_sac_status.setText(
                "Calibrate all four table corners before enabling physical control."
            )
            return
        if self.sac_controller is None or self.sac_controller.model is None:
            if not self._offer_dummy_sac_model():
                self.lbl_sac_status.setText("Load a trained model before starting the agent.")
                return

        self.sac_controller.sim_bridge = self.live_worker.sim_bridge
        generation = self._controller_generation
        self.sac_worker = SACAgentWorker(
            self.sac_controller,
            action_sink=self._make_action_sink(generation),
            frequency_hz=100.0,
            parent=self,
        )
        self.sac_worker.action_updated.connect(self._on_sac_action)
        self.sac_worker.error_occurred.connect(self._on_sac_error)
        self.sac_worker.finished.connect(self._on_sac_finished)
        self.sac_worker.start()
        self.btn_toggle_sac_agent.setText("STOP SAC AGENT")
        self.lbl_sac_status.setText("SAC agent running at 100 Hz")

    def _on_controller_selected(self, controller_name: str) -> None:
        self._stop_active_controller()
        if controller_name != "None (Idle)":
            self._start_selected_controller()

    def _start_selected_controller(self) -> None:
        if (
            self._stream_stopping
            or self._pending_zero_revision is not None
            or self.live_worker is None
            or not self.live_worker.isRunning()
            or self.sac_worker is not None
            or self.scripted_worker is not None
        ):
            return

        controller_name = self.combo_controller.currentText()
        if controller_name == "None (Idle)":
            self.lbl_sac_status.setText("Controller idle")
            return
        if not self.live_worker.sim_bridge.physical_table_calibrated:
            self.lbl_sac_status.setText(
                "Calibrate all four table corners before enabling physical control."
            )
            return
        if controller_name == "Muisbesturing (Virtual Twin)":
            if not self._alignment_is_ready():
                self.lbl_sac_status.setText(
                    "Muisbesturing vereist actieve OPC UA, gehomede PLC en geldige pose."
                )
                return
            self._start_manual_twin_control()
            return

        generation = self._controller_generation
        action_sink = self._make_action_sink(generation)
        if controller_name == "SAC Agent":
            if self.sac_controller is None or self.sac_controller.model is None:
                self.lbl_sac_status.setText(
                    "Load a SAC model before selecting the SAC Agent."
                )
                return
            self._start_sac_agent()
            return
        try:
            worker = ScriptedOpponentWorker(
                self.live_worker.sim_bridge,
                controller_name,
                action_sink=action_sink,
                frequency_hz=100.0,
                parent=self,
            )
        except ValueError as exc:
            self.lbl_sac_status.setText(str(exc))
            return
        self.scripted_worker = worker
        worker.action_updated.connect(self._on_scripted_action)
        worker.error_occurred.connect(self._on_sac_error)
        worker.finished.connect(self._on_scripted_finished)
        worker.start()
        self.lbl_sac_status.setText(f"{controller_name} running at 100 Hz")
        self.btn_toggle_sac_agent.setText("START SAC AGENT")

    def _make_action_sink(self, generation: int):
        def enqueue(vx_mm_s: float, vy_mm_s: float) -> None:
            with self._action_lock:
                if generation != self._controller_generation:
                    return
                self._latest_action_mm_s = limit_velocity(vx_mm_s, vy_mm_s)
                worker = self.opcua_worker
                if worker is not None and worker.accepting_commands:
                    worker.set_target_velocity(*self._latest_action_mm_s)

        return enqueue

    def _stop_active_controller(self) -> None:
        self._cancel_alignment(clear_pending=True)
        self._manual_twin_control_active = False
        if self.digital_twin_window is not None:
            self.digital_twin_window.twin_widget.set_manual_control_enabled(False)
        self._refresh_alignment_enabled()
        with self._action_lock:
            self._controller_generation += 1
            worker = self.opcua_worker
            if worker is not None and worker.accepting_commands:
                revision = worker.set_target_velocity(0.0, 0.0)
                self._pending_zero_revision = (
                    revision if self._opcua_connected else None
                )
            self._latest_action_mm_s = (0.0, 0.0)
        if self.digital_twin_window is not None:
            self.digital_twin_window.update_sac_action(0.0, 0.0)

        scripted_worker = self.scripted_worker
        if scripted_worker is not None:
            scripted_worker.stop()
            if scripted_worker.isRunning():
                scripted_worker.wait(250)
            if not scripted_worker.isRunning():
                self.scripted_worker = None
                self._restart_controller_after_stop = False
            else:
                self._restart_controller_after_stop = True

        sac_worker = self.sac_worker
        if sac_worker is not None:
            sac_worker.set_action_sink(None)
            sac_worker.stop()
            if sac_worker.isRunning():
                sac_worker.wait(500)
            if not sac_worker.isRunning():
                self.sac_worker = None
                self._restart_controller_after_stop = False
            else:
                self._restart_controller_after_stop = True
        self.btn_toggle_sac_agent.setText("START SAC AGENT")

    def _start_manual_twin_control(self) -> None:
        self._open_digital_twin()
        if self.digital_twin_window is None:
            self.lbl_sac_status.setText("Virtual Twin kon niet worden geopend.")
            return
        self._manual_twin_control_active = True
        self.digital_twin_window.twin_widget.set_manual_control_enabled(True)
        self.lbl_sac_status.setText(
            "Muisbesturing actief: sleep de mallet in de Virtual Twin."
        )
        self.lbl_alignment_status.setText(
            "Sleep de mallet naar een doel; loslaten houdt de doelpositie vast."
        )
        self._refresh_alignment_enabled()

    def _on_manual_twin_target(self, table_x_mm: float, table_y_mm: float) -> None:
        if not self._manual_twin_control_active or not self._alignment_is_ready():
            self._abort_alignment(
                "Muisbesturing afgebroken: PLC, OPC UA of camera niet gereed."
            )
            return
        self._alignment_target_table_mm = (table_x_mm, table_y_mm)
        self._alignment_target_plc_mm, _ = self._bounded_alignment_target(
            self._alignment_target_table_mm
        )
        self._alignment_active = True
        if not self._alignment_timer.isActive():
            self._alignment_timer.start()
        self.lbl_alignment_status.setText(
            f"Muisdoel: X={table_x_mm:.1f}, Y={table_y_mm:.1f} mm"
        )
        self._refresh_alignment_enabled()

    def _go_to_preset_stip(self, table_x_mm: float, table_y_mm: float) -> None:
        self.alignment_x_spin.setValue(table_x_mm)
        self.alignment_y_spin.setValue(table_y_mm)
        self._start_alignment()

    def _mirror_stip_x(self) -> None:
        self.alignment_x_spin.setValue(-self.alignment_x_spin.value())

    def _alignment_is_ready(self) -> bool:
        worker = self.live_worker
        opcua_worker = self.opcua_worker
        if not (
            worker is not None
            and worker.isRunning()
            and worker.sim_bridge.physical_table_calibrated
            and self._opcua_connected
            and opcua_worker is not None
            and opcua_worker.accepting_commands
            and self._plc_homed
            and not self._plc_has_error
            and self._plc_pose_valid
        ):
            return False
        pose = worker.sim_bridge.plc_robot_position_mm
        return bool(
            pose is not None
            and worker.sim_bridge.PLC_X_BOUNDS_MM[0] <= pose[0]
            <= worker.sim_bridge.PLC_X_BOUNDS_MM[1]
            and worker.sim_bridge.PLC_Y_BOUNDS_MM[0] <= pose[1]
            <= worker.sim_bridge.PLC_Y_BOUNDS_MM[1]
        )

    def _refresh_alignment_enabled(self) -> None:
        if not hasattr(self, "btn_go_to_stip"):
            return
        ready = self._alignment_is_ready()
        self.btn_go_to_stip.setEnabled(ready and not self._alignment_active)
        self.btn_stip_left.setEnabled(ready and not self._alignment_active)
        self.btn_stip_right.setEnabled(ready and not self._alignment_active)
        self.btn_stop_alignment.setEnabled(
            self._alignment_active
            or self._alignment_target_pending is not None
            or self._manual_twin_control_active
        )

    def _apply_bridge_trims(self, bridge: SimBridge) -> None:
        bridge.set_robot_trim_mm(
            self._trim_sliders["robot_trim_x_mm"].value() / 2.0,
            self._trim_sliders["robot_trim_y_mm"].value() / 2.0,
        )
        bridge.set_puck_trim_mm(
            self._trim_sliders["puck_trim_x_mm"].value() / 2.0,
            self._trim_sliders["puck_trim_y_mm"].value() / 2.0,
        )

    def _on_trim_changed(self, key: str, slider_value: int) -> None:
        trim_mm = slider_value / 2.0
        self._trim_value_labels[key].setText(f"{trim_mm:+.1f} mm")
        if self.live_worker is not None:
            self._apply_bridge_trims(self.live_worker.sim_bridge)

    def _adjust_trim(self, key: str, step: int) -> None:
        slider = self._trim_sliders[key]
        slider.setValue(slider.value() + step)

    def _save_trim_axis(self, axis: str) -> None:
        for component in ("robot", "puck"):
            key = f"{component}_trim_{axis}_mm"
            self._settings.setValue(key, self._trim_sliders[key].value() / 2.0)
        self._settings.sync()
        self.lbl_alignment_status.setText(f"{axis.upper()}-trim opgeslagen.")

    def _reset_trims(self) -> None:
        for slider in self._trim_sliders.values():
            slider.setValue(0)
        for key in self._trim_sliders:
            self._settings.setValue(key, 0.0)
        self._settings.sync()
        if self.live_worker is not None:
            self._apply_bridge_trims(self.live_worker.sim_bridge)
        self.lbl_alignment_status.setText("Alle trims teruggezet naar 0.0 mm.")
        self._refresh_alignment_enabled()

    def _start_alignment(self) -> None:
        if not self._alignment_is_ready():
            self.lbl_alignment_status.setText(
                "Uitlijning vereist actieve camera, tafelkalibratie, OPC UA, "
                "geldige PLC-pose en bHomed."
            )
            self._refresh_alignment_enabled()
            return
        if self._manual_twin_control_active:
            self.combo_controller.setCurrentText("None (Idle)")

        table_target = (
            self.alignment_x_spin.value(),
            self.alignment_y_spin.value(),
        )
        _, target_was_clamped = self._bounded_alignment_target(table_target)

        self._stop_active_controller()
        self._alignment_target_pending = table_target
        if target_was_clamped:
            self.lbl_alignment_status.setText(
                "Stip wordt begrensd binnen het PLC-werkgebied; wachten op 0 mm/s..."
            )
        else:
            self.lbl_alignment_status.setText("Wachten op bevestiging van 0 mm/s...")
        if self._pending_zero_revision is None:
            target = self._alignment_target_pending
            self._alignment_target_pending = None
            self._activate_alignment(target)
        self._refresh_alignment_enabled()

    def _bounded_alignment_target(
        self, table_target: tuple[float, float]
    ) -> tuple[tuple[float, float], bool]:
        bridge = self.live_worker.sim_bridge
        raw_target = bridge.physical_target_to_plc_mm(*table_target)
        bounded_target = (
            float(np.clip(raw_target[0], *bridge.PLC_X_BOUNDS_MM)),
            float(np.clip(raw_target[1], *bridge.PLC_Y_BOUNDS_MM)),
        )
        return bounded_target, bounded_target != raw_target

    def _activate_alignment(self, table_target: tuple[float, float]) -> None:
        if not self._alignment_is_ready():
            self._abort_alignment(
                "Uitlijning afgebroken: PLC/camera-voorwaarden zijn gewijzigd"
            )
            return
        self._alignment_target_plc_mm, target_was_clamped = (
            self._bounded_alignment_target(table_target)
        )
        self._alignment_target_table_mm = table_target
        self._alignment_active = True
        self._restart_controller_after_stop = False
        self._alignment_timer.start()
        clamp_note = " (begrensd op PLC-werkgebied)" if target_was_clamped else ""
        self.lbl_alignment_status.setText(
            f"Uitlijnen naar X={table_target[0]:.1f}, "
            f"Y={table_target[1]:.1f} mm{clamp_note}"
        )
        self._refresh_alignment_enabled()

    def _tick_alignment(self) -> None:
        if not self._alignment_active or not self._alignment_is_ready():
            self._abort_alignment(
                "Uitlijning afgebroken: camera, OPC UA, homing of PLC-pose verloren"
            )
            return

        bridge = self.live_worker.sim_bridge
        current = bridge.plc_robot_position_mm
        table_target = self._alignment_target_table_mm
        if current is None or table_target is None:
            self._abort_alignment("Uitlijning afgebroken: PLC-pose ontbreekt")
            return
        target, target_was_clamped = self._bounded_alignment_target(table_target)
        self._alignment_target_plc_mm = target
        if not (
            bridge.PLC_X_BOUNDS_MM[0] <= current[0] <= bridge.PLC_X_BOUNDS_MM[1]
            and bridge.PLC_Y_BOUNDS_MM[0] <= current[1] <= bridge.PLC_Y_BOUNDS_MM[1]
        ):
            self._abort_alignment("Uitlijning afgebroken: PLC-pose buiten werkgebied")
            return

        error_x = target[0] - current[0]
        error_y = target[1] - current[1]
        error_mm = float(np.hypot(error_x, error_y))
        if error_mm <= 0.3:
            velocity = (0.0, 0.0)
            status = "Stip bereikt; positie wordt vastgehouden."
        else:
            speed_mm_s = float(np.clip(error_mm * 1.5, 20.0, 850.0))
            velocity = limit_velocity(
                speed_mm_s * error_x / error_mm,
                speed_mm_s * error_y / error_mm,
            )
            status = (
                f"Uitlijnen: fout {error_mm:.1f} mm | "
                f"Vx={velocity[0]:+.1f}, Vy={velocity[1]:+.1f} mm/s"
            )
        if target_was_clamped:
            status += " (PLC-grens actief)"
        self.lbl_alignment_status.setText(status)

        opcua_worker = self.opcua_worker
        if opcua_worker is None or not opcua_worker.accepting_commands:
            self._abort_alignment("Uitlijning afgebroken: OPC UA niet beschikbaar")
            return
        opcua_worker.set_target_velocity(*velocity)
        self._latest_action_mm_s = velocity
        if self.digital_twin_window is not None:
            self.digital_twin_window.update_sac_action(
                velocity[0] / 1000.0, -velocity[1] / 1000.0
            )

    def _cancel_alignment(self, *, clear_pending: bool) -> None:
        self._alignment_timer.stop()
        self._alignment_active = False
        self._alignment_target_table_mm = None
        self._alignment_target_plc_mm = None
        if clear_pending:
            if self._alignment_target_pending is not None:
                self._pending_zero_revision = None
            self._alignment_target_pending = None
        if hasattr(self, "btn_stop_alignment"):
            self._refresh_alignment_enabled()

    def _stop_alignment(self) -> None:
        was_active = (
            self._alignment_active
            or self._alignment_target_pending is not None
            or self._manual_twin_control_active
        )
        self._cancel_alignment(clear_pending=True)
        worker = self.opcua_worker
        if was_active and worker is not None and worker.accepting_commands:
            worker.set_target_velocity(0.0, 0.0)
        self._latest_action_mm_s = (0.0, 0.0)
        if self.digital_twin_window is not None:
            self.digital_twin_window.update_sac_action(0.0, 0.0)
        self.lbl_alignment_status.setText("Uitlijning gestopt.")
        if self.combo_controller.currentText() != "None (Idle)":
            self.combo_controller.setCurrentText("None (Idle)")

    def _abort_alignment(self, message: str, *, send_zero: bool = True) -> None:
        was_active = (
            self._alignment_active
            or self._alignment_target_pending is not None
            or self._manual_twin_control_active
        )
        self._cancel_alignment(clear_pending=True)
        self._manual_twin_control_active = False
        if self.digital_twin_window is not None:
            self.digital_twin_window.twin_widget.set_manual_control_enabled(False)
        worker = self.opcua_worker
        if was_active and send_zero and worker is not None and worker.accepting_commands:
            worker.set_target_velocity(0.0, 0.0)
        if was_active:
            self._latest_action_mm_s = (0.0, 0.0)
            self.lbl_alignment_status.setText(message)
            if self.digital_twin_window is not None:
                self.digital_twin_window.update_sac_action(0.0, 0.0)
            self._restart_controller_after_stop = False
            if self.combo_controller.currentText() != "None (Idle)":
                self.combo_controller.setCurrentText("None (Idle)")

    def _on_scripted_action(self, vx_mm_s: float, vy_mm_s: float) -> None:
        self.lbl_sac_action.setText(
            f"Target Vx: {vx_mm_s:+.1f} mm/s    Target Vy: {vy_mm_s:+.1f} mm/s"
        )
        self._latest_action_mm_s = (vx_mm_s, vy_mm_s)
        if self.digital_twin_window is not None:
            self.digital_twin_window.update_sac_action(
                vx_mm_s / 1000.0, -vy_mm_s / 1000.0
            )

    def _on_scripted_finished(self) -> None:
        if self.sender() is not self.scripted_worker:
            return
        if self.scripted_worker is not None and not self.scripted_worker.isRunning():
            self.scripted_worker = None
        self.btn_toggle_sac_agent.setText("START SAC AGENT")
        if self._restart_controller_after_stop and self._alignment_target_pending is None:
            self._restart_controller_after_stop = False
            self._start_selected_controller()
            return
        if self.combo_controller.currentText() != "None (Idle)":
            self.lbl_sac_status.setText("Controller stopped")

    def _toggle_opcua_connection(self, connect: bool) -> None:
        if connect:
            if self.opcua_worker is not None:
                self.lbl_opcua_status.setText(
                    "OPC UA: Previous connection is still stopping"
                )
                self.btn_toggle_opcua.setChecked(False)
                return
            worker = OpcUaWorker(parent=self)
            worker.connection_status.connect(self._on_opcua_status)
            worker.cycle_metrics_updated.connect(self._on_opcua_metrics_updated)
            worker.error_occurred.connect(self._on_opcua_error)
            worker.plc_pose_updated.connect(self._on_plc_pose_updated)
            worker.plc_pose_validity_updated.connect(self._on_plc_pose_validity_updated)
            worker.plc_status_updated.connect(self._on_plc_status_updated)
            worker.zero_written.connect(self._on_opcua_zero_written)
            worker.finished.connect(self._on_opcua_finished)
            self.opcua_worker = worker
            worker.start()
            self.btn_toggle_opcua.setText("DISCONNECT OPC UA")
            self._refresh_alignment_enabled()
            return

        self._abort_alignment("OPC UA wordt verbroken", send_zero=False)
        worker = self.opcua_worker
        if worker is not None:
            worker.stop()
            self.lbl_opcua_status.setText("OPC UA: Disconnecting (sending zero)")
        self._opcua_connected = False
        self._plc_homed = False
        self._plc_pose_valid = False
        self._plc_has_error = False
        if self.live_worker is not None:
            self.live_worker.sim_bridge.clear_robot_position_mm()
        self._pending_zero_revision = None
        self.btn_toggle_opcua.setText("CONNECT OPC UA")
        self._refresh_alignment_enabled()

    def _on_opcua_status(self, status: str) -> None:
        self.lbl_opcua_status.setText(f"OPC UA: {status}")
        if status.startswith("Connected"):
            self._opcua_connected = True
        elif status in ("Disconnected", "Error"):
            self._opcua_connected = False
            self._pending_zero_revision = None
            self._plc_homed = False
            self._plc_pose_valid = False
            self._plc_has_error = status == "Error"
            self._abort_alignment(
                f"Uitlijning afgebroken: OPC UA {status.lower()}",
                send_zero=False,
            )
            self._start_selected_controller()
        self._refresh_alignment_enabled()

    def _on_opcua_metrics_updated(self, roundtrip_ms: float, loop_hz: float) -> None:
        if self._opcua_connected:
            self.lbl_opcua_status.setText(
                f"OPC UA: {roundtrip_ms:.2f} ms roundtrip | {loop_hz:.1f} Hz"
            )

    def _on_camera_metrics_updated(
        self, capture_ms: float, vision_ms: float, fps: float
    ) -> None:
        calibration_text = ""
        if self.live_worker is not None:
            offset_x, offset_y = (
                self.live_worker.sim_bridge.calibration_offset_mm
            )
            calibration_text = (
                f" | Calib offset: X={offset_x:.1f}, Y={offset_y:.1f} mm"
            )
        self.status_label.setText(
            f"Camera: {fps:.1f} FPS | capture {capture_ms:.2f} ms | "
            f"vision {vision_ms:.2f} ms{calibration_text}"
        )

    def _on_opcua_error(self, message: str) -> None:
        self.lbl_opcua_status.setText(f"OPC UA error: {message}")
        self._abort_alignment("Uitlijning afgebroken: OPC UA-fout")
        self._refresh_alignment_enabled()

    def _on_opcua_zero_written(self, revision: int) -> None:
        if revision != self._pending_zero_revision:
            return
        self._pending_zero_revision = None
        if self._alignment_target_pending is not None:
            target = self._alignment_target_pending
            self._alignment_target_pending = None
            self._activate_alignment(target)
            return
        self._start_selected_controller()

    def _on_opcua_finished(self) -> None:
        self._abort_alignment("Uitlijning afgebroken: OPC UA gestopt", send_zero=False)
        if self.opcua_worker is not None and not self.opcua_worker.isRunning():
            self.opcua_worker = None
        self._opcua_connected = False
        self._plc_homed = False
        self._plc_pose_valid = False
        self._plc_has_error = False
        if self.live_worker is not None:
            self.live_worker.sim_bridge.clear_robot_position_mm()
        self._pending_zero_revision = None
        self.btn_toggle_opcua.setChecked(False)
        self.btn_toggle_opcua.setText("CONNECT OPC UA")
        if self.lbl_opcua_status.text() == "OPC UA: Disconnected":
            self._refresh_alignment_enabled()
            return
        self.lbl_opcua_status.setText("OPC UA: Disconnected")
        self._refresh_alignment_enabled()

    def _on_plc_pose_updated(self, x_mm: float, y_mm: float) -> None:
        if self.live_worker is not None:
            self.live_worker.sim_bridge.set_robot_position_mm(x_mm, y_mm)
        self._plc_pose_valid = True
        self._refresh_alignment_enabled()

    def _on_plc_pose_validity_updated(self, valid: bool) -> None:
        self._plc_pose_valid = valid
        if not valid and self.live_worker is not None:
            self.live_worker.sim_bridge.clear_robot_position_mm()
            self._abort_alignment("Uitlijning afgebroken: PLC-pose ongeldig")
        self._refresh_alignment_enabled()

    def _on_plc_status_updated(
        self,
        homed: bool,
        tracking: bool,
        has_error: bool,
        state_text: str,
    ) -> None:
        self._plc_homed = homed
        self._plc_has_error = has_error
        self.lbl_plc_machine_status.setText(
            f"PLC: Homed {'Yes' if homed else 'No'} | "
            f"Tracking {'Yes' if tracking else 'No'} | "
            f"Error {'Yes' if has_error else 'No'} | State {state_text}"
        )
        if (
            self._alignment_active
            or self._alignment_target_pending is not None
            or self._manual_twin_control_active
        ):
            if has_error or not homed:
                reason = "PLC-fout" if has_error else "PLC niet gehomed"
                self._abort_alignment(f"Uitlijning afgebroken: {reason}")
        self._refresh_alignment_enabled()

    def _offer_dummy_sac_model(self) -> bool:
        """Offer the bundled untrained model when no policy has been selected."""
        dummy_model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "dummy_sac_model.zip"
        )
        if not dummy_model_path.is_file():
            self.lbl_sac_status.setText(
                "No model loaded. Generate models/dummy_sac_model.zip or load a model."
            )
            return False

        answer = QMessageBox.warning(
            self,
            "Untrained Dummy SAC Model",
            "No model is loaded. Load the bundled untrained dummy model for a "
            "software smoke test? Its actions are random/untrained and must "
            "not be used to control a real table or robot.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False

        try:
            controller = LiveSACController(self.live_worker.sim_bridge)
            controller.load_model(dummy_model_path)
        except Exception as exc:
            self.lbl_sac_status.setText(f"Dummy model load failed: {exc}")
            return False

        self.sac_controller = controller
        self.sac_model_path = str(dummy_model_path)
        self.lbl_sac_model.setText(f"UNTRAINED TEST MODEL: {dummy_model_path.name}")
        return True

    def _on_sac_action(self, target_vx: float, target_vy: float) -> None:
        """Forward actions to actuator adapters and refresh the GUI labels."""
        self._latest_sac_action = (target_vx, target_vy)
        self.sac_action_ready.emit(target_vx, target_vy)
        vx_mm_s, vy_mm_s = limit_velocity(
            target_vy * SACAgentWorker.MAX_ACTION_SPEED_MM_S,
            -target_vx * SACAgentWorker.MAX_ACTION_SPEED_MM_S,
        )
        self.lbl_sac_action.setText(
            f"Target Vx: {vx_mm_s:+.1f} mm/s    Target Vy: {vy_mm_s:+.1f} mm/s"
        )
        self._latest_action_mm_s = (vx_mm_s, vy_mm_s)
        if self.digital_twin_window is not None:
            self.digital_twin_window.update_sac_action(
                -vy_mm_s / 1000.0, vx_mm_s / 1000.0
            )

    def _on_sac_error(self, message: str) -> None:
        self.lbl_sac_status.setText(message)

    def _on_sac_finished(self) -> None:
        if self.sac_worker is not None and not self.sac_worker.isRunning():
            self.sac_worker = None
        self.btn_toggle_sac_agent.setText("START SAC AGENT")
        if (
            self._restart_controller_after_stop
            and self.sac_worker is None
            and self._alignment_target_pending is None
        ):
            self._restart_controller_after_stop = False
            self._start_selected_controller()

    # ──────────────────────────────────────────────────────────────────
    # Slots & event handlers
    # ──────────────────────────────────────────────────────────────────

    def _append_debug(self, message: str) -> None:
        if hasattr(self, "debug_window") and self.debug_window is not None:
            self.debug_window.append_log(message)

    def _on_toggle_table_overlay(self, checked: bool) -> None:
        """Apply the visible table overlay preference to the active worker."""
        if self.live_worker is not None:
            self.live_worker.set_show_table_overlay(checked)

    def _start_table_calibration(self) -> None:
        """Begin collecting table corners in TL, TR, BR, BL order."""
        self._stop_active_controller()
        worker = self.live_worker
        if worker is None or not worker.isRunning():
            self.status_label.setText("Start eerst de live camera voor kalibratie.")
            return
        frame = worker.current_hsv_frame
        if frame is None:
            self.status_label.setText("Wacht tot het eerste camerabeeld beschikbaar is.")
            return

        height, width = frame.shape[:2]
        self._calibration_frame_size = (width, height)
        self._manual_table_calibrator.reset()
        worker.set_table_calibration_preview(())
        self.feed_primary.set_table_calibration_mode(True)
        self.btn_calibrate_table_corners.setText("CLICK TABLE CORNERS (0/4)")
        self.status_label.setText("Klik op de tafelpunten in volgorde: TL, TR, BR, BL.")

    def _on_table_corner_picked(self, norm_x: float, norm_y: float) -> None:
        """Record one normalized live-image click and commit after four points."""
        worker = self.live_worker
        if worker is None or self._calibration_frame_size is None:
            return
        frame = worker.current_hsv_frame
        if frame is None:
            self.status_label.setText("Geen camerabeeld beschikbaar voor dit punt.")
            return

        height, width = frame.shape[:2]
        if (width, height) != self._calibration_frame_size:
            self.status_label.setText("Beeldformaat gewijzigd; start de kalibratie opnieuw.")
            self._cancel_table_calibration()
            return

        point = (float(norm_x) * (width - 1), float(norm_y) * (height - 1))
        try:
            label = self._manual_table_calibrator.add_corner(point)
            worker.set_table_calibration_preview(self._manual_table_calibrator.corners)
            count = len(self._manual_table_calibrator.corners)
            self.btn_calibrate_table_corners.setText(f"CLICK TABLE CORNERS ({count}/4)")

            if self._manual_table_calibrator.is_complete:
                self._manual_table_calibrator.apply(worker.sim_bridge)
                worker.sim_bridge.configure_physical_table_bounds()
                self._refresh_alignment_enabled()
                self._save_table_calibration(
                    self._manual_table_calibrator.corners,
                    self._calibration_frame_size,
                )
                self._cancel_table_calibration()
                self.status_label.setText("Tafelkalibratie opgeslagen.")
                self._start_selected_controller()
                return

            next_label = self._manual_table_calibrator.next_corner_label
            self.status_label.setText(f"{label} ingesteld. Klik nu op {next_label}.")
        except (ValueError, RuntimeError) as exc:
            self._manual_table_calibrator.reset()
            worker.set_table_calibration_preview(())
            self.btn_calibrate_table_corners.setText("CLICK TABLE CORNERS (0/4)")
            self.status_label.setText(f"Ongeldige tafelpunten: {exc}. Klik opnieuw vanaf TL.")

    def _cancel_table_calibration(self, clear_preview: bool = True) -> None:
        self.feed_primary.set_table_calibration_mode(False)
        self.btn_calibrate_table_corners.setText("CALIBRATE TABLE CORNERS")
        if clear_preview and self.live_worker is not None:
            self.live_worker.set_table_calibration_preview(())
        self._calibration_frame_size = None

    def _save_table_calibration(
        self,
        corners: tuple[tuple[float, float], ...],
        frame_size: tuple[int, int],
    ) -> None:
        calibration = {
            "corners": corners,
            "width": frame_size[0],
            "height": frame_size[1],
        }
        self._settings.setValue("table_calibration", json.dumps(calibration))
        self._settings.sync()

    def _restore_saved_table_calibration(self, frame_size: tuple[int, int]) -> None:
        """Restore persisted four-corner calibration, scaling for frame size."""
        worker = self.live_worker
        if worker is None or self._saved_calibration_restored:
            return
        stored = self._settings.value("table_calibration", "")
        self._saved_calibration_restored = True
        if not stored:
            return

        try:
            calibration = json.loads(str(stored))
            saved_width = int(calibration["width"])
            saved_height = int(calibration["height"])
            corners = calibration["corners"]
            if len(corners) != 4 or saved_width <= 1 or saved_height <= 1:
                raise ValueError("stored corner calibration is incomplete")
            scale_x = (frame_size[0] - 1) / (saved_width - 1)
            scale_y = (frame_size[1] - 1) / (saved_height - 1)
            scaled_corners = tuple(
                (float(point[0]) * scale_x, float(point[1]) * scale_y)
                for point in corners
            )
            self._manual_table_calibrator.reset()
            for point in scaled_corners:
                self._manual_table_calibrator.add_corner(point)
            self._manual_table_calibrator.apply(worker.sim_bridge)
            worker.sim_bridge.configure_physical_table_bounds()
            self._refresh_alignment_enabled()
            self._manual_table_calibrator.reset()
            self.status_label.setText("Opgeslagen tafelkalibratie geladen.")
            self._start_selected_controller()
        except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self.status_label.setText(f"Opgeslagen tafelkalibratie ongeldig: {exc}")

    def _reset_table_corners(self) -> None:
        """Clear stored calibration and restore full-frame table bounds."""
        self._stop_active_controller()
        self._settings.remove("table_calibration")
        self._settings.sync()
        self._manual_table_calibrator.reset()
        worker = self.live_worker
        if worker is not None:
            worker.set_table_calibration_preview(())
            frame = worker.current_hsv_frame
            if frame is not None:
                height, width = frame.shape[:2]
                worker.sim_bridge.reset_table_corners(width, height)
            else:
                worker.sim_bridge.reset_table_corners()
        self._cancel_table_calibration()
        self.status_label.setText("Tafelhoeken gewist; volledige camera-ROI wordt gebruikt.")
        self._refresh_alignment_enabled()

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

    def _open_digital_twin(self) -> None:
        """Show the field simulation in its own resizable window."""
        bridge = self.live_worker.sim_bridge if self.live_worker is not None else None
        if self.digital_twin_window is None:
            self.digital_twin_window = DigitalTwinWindow(bridge, self)
        else:
            self.digital_twin_window.update_bridge(bridge)
        if not self._manual_twin_signal_connected:
            self.digital_twin_window.twin_widget.manual_target_requested.connect(
                self._on_manual_twin_target
            )
            self._manual_twin_signal_connected = True
        self.digital_twin_window.update_sac_action(
            -self._latest_action_mm_s[1] / 1000.0,
            self._latest_action_mm_s[0] / 1000.0,
        )
        self.digital_twin_window.show()
        self.digital_twin_window.raise_()
        self.digital_twin_window.activateWindow()

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

    def go_back(self) -> None:
        # Stop de thread volledig voor we de UI verlaten
        self._stop_stream()
        if self.on_back:
            self.on_back()

    def shutdown(self) -> None:
        """Stop control, camera, and OPC UA workers before application exit."""
        self._stop_stream()
        worker = self.opcua_worker
        if worker is not None:
            worker.stop()
            worker.wait(2500)

    def toggle_freeze(self) -> None:
        if self.live_worker and self.live_worker.isRunning():
            paused = self.live_worker.toggle_pause()
            self.btn_pause.setText("UNFREEZE" if paused else "FREEZE FRAME")

    # ──────────────────────────────────────────────────────────────────
    # Interne helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _slider_is_live(slider) -> bool:
        if slider is None:
            return False
        try:
            slider.value()
            return True
        except RuntimeError:
            return False

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue

            widget = item.widget()
            if widget is not None:
                self._engine_setting_widgets.discard(widget)
                if isinstance(widget, QSlider):
                    try:
                        widget.valueChanged.disconnect(self._sync_vision_settings)
                    except TypeError:
                        pass
                elif hasattr(self, "color_preview") and widget is self.color_preview:
                    self.color_preview = None
                elif hasattr(self, "btn_pick_color") and widget is self.btn_pick_color:
                    self.btn_pick_color = None
                widget.deleteLater()
                continue

            child_layout = item.layout()
            if child_layout is not None:
                self._clear_layout(child_layout)
                child_layout.deleteLater()

    def _safe_read_slider(self, slider, default=0):
        if not self._slider_is_live(slider):
            return default
        try:
            return int(slider.value())
        except RuntimeError:
            return default

    def _safe_set_preview_color(self, r, g, b) -> None:
        preview = getattr(self, "color_preview", None)
        if preview is None:
            return
        try:
            preview.setStyleSheet(
                f"background-color: rgb({r},{g},{b});"
                " border: 1px solid #503422; border-radius: 3px;"
            )
        except RuntimeError:
            self.color_preview = None

    def _update_color_preview(self) -> None:
        h_min = getattr(self, "slider_h_min", None)
        h_max = getattr(self, "slider_h_max", None)
        s_min = getattr(self, "slider_s_min", None)
        s_max = getattr(self, "slider_s_max", None)

        if (
            self._slider_is_live(h_min)
            and self._slider_is_live(h_max)
            and self._slider_is_live(s_min)
            and self._slider_is_live(s_max)
        ):
            avg_h = (self._safe_read_slider(h_min) + self._safe_read_slider(h_max)) // 2
            avg_s = (self._safe_read_slider(s_min) + self._safe_read_slider(s_max)) // 2
        else:
            avg_h = 35
            avg_s = 120

        hsv_px = np.uint8([[[avg_h, avg_s, 200]]])
        r, g, b = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2RGB)[0, 0]
        self._safe_set_preview_color(int(r), int(g), int(b))

    def _engine_name_to_class(self, name: str):
        mapping = {
            "HSV Detection": "core.engines.hsv_engine.HsvEngine",
            "Hough Circle": "core.engines.hough_circle_engine.HoughCircleEngine",
            "Background Subtraction": "core.engines.bg_subtraction_engine.BackgroundSubtractionEngine",
            "Circle Contour": "core.engines.circle_contour_engine.CircleContourEngine",
            "Hybrid": "core.engines.hybrid_engine.HybridEngine",
        }
        module_name, class_name = mapping.get(name, mapping["HSV Detection"]).rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)

    def _current_engine_instance(self):
        if self.live_worker and self.live_worker.engine is not None:
            return self.live_worker.engine
        return self._engine_name_to_class(self.combo_engine.currentText())()

    def _refresh_engine_settings_ui(self) -> None:
        if hasattr(self, "engine_settings_layout"):
            self._clear_layout(self.engine_settings_layout)

        for attr in (
            "slider_h_min", "slider_h_max", "slider_s_min", "slider_s_max",
            "lbl_h_min", "lbl_h_max", "lbl_s_min", "lbl_s_max",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)

        self.engine_setting_controls.clear()
        self.hsv_slider_map.clear()

        engine = self._current_engine_instance()
        schema = getattr(engine, "parameter_schema", ())

        if self.combo_engine.currentText() == "HSV Detection":
            self.engine_settings_layout.addLayout(self._build_color_picker_row())
            self.slider_h_min, self.lbl_h_min = self._add_slider("Hue Min:", 0, 179, 35, self.engine_settings_layout)
            self.slider_h_max, self.lbl_h_max = self._add_slider("Hue Max:", 0, 179, 95, self.engine_settings_layout)
            self.slider_s_min, self.lbl_s_min = self._add_slider("Sat Min:", 0, 255, 20, self.engine_settings_layout)
            self.slider_s_max, self.lbl_s_max = self._add_slider("Sat Max:", 0, 255, 255, self.engine_settings_layout)
            self.hsv_slider_map = {
                "h_min": self.slider_h_min,
                "h_max": self.slider_h_max,
                "s_min": self.slider_s_min,
                "s_max": self.slider_s_max,
            }

        for param in schema:
            name = param["name"]
            if self.combo_engine.currentText() == "HSV Detection" and name in self.hsv_slider_map:
                continue

            label_text = param["label"]
            min_v = int(param.get("min", 0))
            max_v = int(param.get("max", 255))
            default_v = int(param.get("default", min_v))
            current_value = getattr(engine, name, default_v)

            row = QHBoxLayout()
            lbl_title = QLabel(label_text)
            lbl_value = QLabel(str(current_value))
            lbl_value.setStyleSheet("color: #b86230; font-weight: bold;")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(int(current_value))
            slider.valueChanged.connect(lambda value, label=lbl_value: label.setText(str(value)))
            slider.valueChanged.connect(self._sync_vision_settings)

            row.addWidget(lbl_title)
            row.addWidget(slider)
            row.addWidget(lbl_value)
            self.engine_settings_layout.addLayout(row)
            self.engine_setting_controls[name] = slider
            self._engine_setting_widgets.add(slider)

    def _sync_vision_settings(self) -> None:
        self._update_color_preview()

        engine_params = {}
        for name, slider in self.engine_setting_controls.items():
            if not self._slider_is_live(slider):
                continue
            engine_params[name] = int(slider.value())

        if self.live_worker and self.live_worker.engine is not None:
            self.live_worker.engine.update_settings(**engine_params)

        if self.live_worker and self.live_worker.isRunning():
            if hasattr(self, "fisheye_k1_slider") and self._slider_is_live(self.fisheye_k1_slider):
                k1 = self.fisheye_k1_slider.value() / 1000.0
            else:
                k1 = -0.4
            if hasattr(self, "fisheye_k2_slider") and self._slider_is_live(self.fisheye_k2_slider):
                k2 = self.fisheye_k2_slider.value() / 1000.0
            else:
                k2 = 0.05
            self.live_worker.set_fisheye_correction(
                getattr(self, "fisheye_enabled", None).isChecked() if hasattr(self, "fisheye_enabled") else False,
                k1,
                k2,
            )
            if self.combo_engine.currentText() == "HSV Detection":
                if (
                    self._slider_is_live(getattr(self, "slider_h_min", None))
                    and self._slider_is_live(getattr(self, "slider_h_max", None))
                    and self._slider_is_live(getattr(self, "slider_s_min", None))
                    and self._slider_is_live(getattr(self, "slider_s_max", None))
                ):
                    self.live_worker.update_hsv(
                        self.slider_h_min.value(), self.slider_h_max.value(),
                        self.slider_s_min.value(), self.slider_s_max.value(),
                        v_min=20, v_max=255,
                    )
            self.live_worker.set_engine(self.combo_engine.currentText())
            if self.live_worker.engine is not None:
                self.live_worker.engine.update_settings(**engine_params)

    def _change_view_mode(self, index: int) -> None:
        self.feed_secondary.setVisible(index != 0)

    def _update_feed(self, frame_result, frame_mask) -> None:
        if not self._saved_calibration_restored and self.live_worker is not None:
            frame = self.live_worker.current_hsv_frame
            if frame is not None:
                self._restore_saved_table_calibration((frame.shape[1], frame.shape[0]))
        pixmap_result = (
            LiveCameraWorker._mat_to_pixmap(frame_result)
            if isinstance(frame_result, np.ndarray)
            else frame_result
        )
        pixmap_mask = (
            LiveCameraWorker._mat_to_pixmap(frame_mask)
            if self.feed_secondary.isVisible() and isinstance(frame_mask, np.ndarray)
            else frame_mask
        )
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

    def _add_numeric_slider(
        self,
        label_text: str,
        min_v: int,
        max_v: int,
        default_v: int,
        divider: int,
        parent_layout: QVBoxLayout,
    ) -> QSlider:
        row = QHBoxLayout()

        lbl_title = QLabel(f"{label_text}:")
        lbl_val = QLabel(f"{default_v / divider:.3f}")
        lbl_val.setStyleSheet("color: #b86230; font-weight: bold;")

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default_v)
        slider.valueChanged.connect(
            lambda v, l=lbl_val: l.setText(f"{v / divider:.3f}")
        )
        slider.valueChanged.connect(self._sync_vision_settings)

        row.addWidget(lbl_title)
        row.addWidget(slider)
        row.addWidget(lbl_val)
        parent_layout.addLayout(row)
        return slider

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