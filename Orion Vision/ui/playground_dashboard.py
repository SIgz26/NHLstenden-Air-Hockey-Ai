"""Interactive PyQt playground for human and policy air-hockey matches."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QElapsedTimer, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ai.air_hockey_env import AirHockeyGymEnv
from core.match_simulator import (
    AI_VS_AI,
    AI_VS_SCRIPTED,
    HUMAN_VS_AI,
    MATCH_MODES,
    MatchSimulator,
    mouse_position_to_world,
)
from ui.widgets.digital_twin_widget import DigitalTwinWidget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"
MODE_LABELS = {
    HUMAN_VS_AI: "Human vs AI",
    AI_VS_AI: "AI vs AI",
    AI_VS_SCRIPTED: "AI vs Scripted Opponent",
}


class MatchFieldWidget(DigitalTwinWidget):
    """Digital-twin field that turns mouse movement into a left-mallet target."""

    target_changed = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(refresh_hz=30.0, parent=parent)
        self.setMouseTracking(True)
        self._control_enabled = False
        self._mouse_target: tuple[float, float] | None = None
        self.setCursor(Qt.CrossCursor)

    def set_control_enabled(self, enabled: bool) -> None:
        self._control_enabled = bool(enabled)
        self._mouse_target = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._control_enabled:
            field = self._field_rect()
            target = mouse_position_to_world(
                event.pos().x(),
                event.pos().y(),
                field.left(),
                field.top(),
                field.width(),
                field.height(),
            )
            self._mouse_target = target
            self.target_changed.emit(*target)
            self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._control_enabled or self._mouse_target is None:
            return
        field = self._field_rect()
        normalized_x = 2.0 * self._mouse_target[0] / AirHockeyGymEnv.FIELD_LENGTH_M
        normalized_y = 2.0 * self._mouse_target[1] / AirHockeyGymEnv.FIELD_WIDTH_M
        point = self._field_point(field, normalized_x, normalized_y)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor("#9be564"), 2.0, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(point, 13.0, 13.0)
        painter.end()


class PlaygroundDashboard(QWidget):
    """Configure and run a local 120 Hz match simulation."""

    SIM_DT = 1.0 / 120.0

    def __init__(self, on_back_callback=None) -> None:
        super().__init__()
        self.on_back = on_back_callback
        self.simulator: MatchSimulator | None = None
        self.left_score = 0
        self.right_score = 0
        self._accumulator = 0.0
        self._clock = QElapsedTimer()
        self._build_ui()

        self._simulation_timer = QTimer(self)
        self._simulation_timer.setTimerType(Qt.PreciseTimer)
        self._simulation_timer.setInterval(4)
        self._simulation_timer.timeout.connect(self._advance_simulation)

        self._hud_timer = QTimer(self)
        self._hud_timer.setInterval(33)
        self._hud_timer.timeout.connect(self._refresh_hud)

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "QWidget { background: #111715; color: #e6e9dd; }"
            "QFrame#PlayPanel { background: #18211e; border: 1px solid #34463e; border-radius: 6px; }"
            "QPushButton { background: #202b26; color: #eef4e9; border: 1px solid #547366; padding: 8px 12px; border-radius: 4px; }"
            "QPushButton:hover { background: #2a3b33; }"
            "QComboBox { background: #111715; color: #eef4e9; border: 1px solid #547366; padding: 6px; }"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 24)
        root.setSpacing(12)

        header = QHBoxLayout()
        if self.on_back is not None:
            self.btn_back = QPushButton("BACK TO MENU")
            self.btn_back.clicked.connect(self._back_to_menu)
            header.addWidget(self.btn_back)
        title = QLabel("PLAYGROUND // MATCH SIMULATOR")
        title.setStyleSheet("color: #b7e477; font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        setup_panel = QFrame()
        setup_panel.setObjectName("PlayPanel")
        setup = QVBoxLayout(setup_panel)
        setup.setContentsMargins(14, 14, 14, 14)
        setup.setSpacing(9)
        setup.addWidget(self._section_title("MATCH SETUP"))

        self.mode_selector = QComboBox()
        for mode in MATCH_MODES:
            self.mode_selector.addItem(MODE_LABELS[mode], mode)
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        setup.addWidget(QLabel("Mode"))
        setup.addWidget(self.mode_selector)

        self.model_a_selector = QComboBox()
        self.model_b_selector = QComboBox()
        self._refresh_model_lists()
        setup.addWidget(QLabel("Model A / Left"))
        setup.addWidget(self.model_a_selector)
        self.btn_model_a = QPushButton("LOAD MODEL A (.ZIP)")
        self.btn_model_a.clicked.connect(lambda: self._choose_model(self.model_a_selector))
        setup.addWidget(self.btn_model_a)

        setup.addWidget(QLabel("Model B / Right"))
        setup.addWidget(self.model_b_selector)
        self.btn_model_b = QPushButton("LOAD MODEL B (.ZIP)")
        self.btn_model_b.clicked.connect(lambda: self._choose_model(self.model_b_selector))
        setup.addWidget(self.btn_model_b)

        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setStyleSheet("color: #a9b8ad;")
        setup.addWidget(self.mode_hint)

        self.btn_start = QPushButton("START MATCH")
        self.btn_start.clicked.connect(self.start_match)
        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_reset_puck = QPushButton("RESET PUCK")
        self.btn_reset_puck.setEnabled(False)
        self.btn_reset_puck.clicked.connect(self.reset_puck)
        setup.addWidget(self.btn_start)
        setup.addWidget(self.btn_pause)
        setup.addWidget(self.btn_reset_puck)
        setup.addStretch()
        body.addWidget(setup_panel, stretch=0)

        arena_panel = QFrame()
        arena_panel.setObjectName("PlayPanel")
        arena_layout = QVBoxLayout(arena_panel)
        arena_layout.setContentsMargins(14, 14, 14, 14)
        arena_layout.setSpacing(10)

        hud = QHBoxLayout()
        self.left_score_label = QLabel("LEFT  0")
        self.left_score_label.setStyleSheet("color: #71d6e5; font-size: 20px; font-weight: bold;")
        self.match_time_label = QLabel("00:00.00")
        self.match_time_label.setStyleSheet("color: #e6e9dd; font-size: 17px; font-weight: bold;")
        self.right_score_label = QLabel("0  RIGHT")
        self.right_score_label.setStyleSheet("color: #ed796e; font-size: 20px; font-weight: bold;")
        self.puck_speed_label = QLabel("PUCK  0.00 m/s")
        hud.addWidget(self.left_score_label)
        hud.addStretch()
        hud.addWidget(self.match_time_label)
        hud.addStretch()
        hud.addWidget(self.right_score_label)
        hud.addSpacing(18)
        hud.addWidget(self.puck_speed_label)
        arena_layout.addLayout(hud)

        self.field = MatchFieldWidget()
        self.field.setMinimumSize(480, 340)
        self.field.target_changed.connect(self._set_human_target)
        arena_layout.addWidget(self.field, stretch=1)
        self.status_label = QLabel("Choose a mode and press START MATCH.")
        self.status_label.setWordWrap(True)
        arena_layout.addWidget(self.status_label)
        body.addWidget(arena_panel, stretch=1)

        self._on_mode_changed()

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #b7e477; font-weight: bold;")
        return label

    def _refresh_model_lists(self) -> None:
        models = sorted(MODEL_ROOT.rglob("*.zip")) if MODEL_ROOT.exists() else []
        for selector in (self.model_a_selector, self.model_b_selector):
            selector.clear()
            for model_path in models:
                try:
                    label = str(model_path.relative_to(PROJECT_ROOT))
                except ValueError:
                    label = str(model_path)
                selector.addItem(label, str(model_path))

    def _choose_model(self, selector: QComboBox) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Stable-Baselines3 model",
            str(MODEL_ROOT),
            "Model archive (*.zip)",
        )
        if not path:
            return
        file_path = Path(path)
        existing = selector.findData(str(file_path))
        if existing < 0:
            selector.addItem(file_path.name, str(file_path))
            existing = selector.count() - 1
        selector.setCurrentIndex(existing)

    def _on_mode_changed(self, *_args) -> None:
        mode = self.mode_selector.currentData()
        needs_model_a = mode in (AI_VS_AI, AI_VS_SCRIPTED)
        needs_model_b = mode in (HUMAN_VS_AI, AI_VS_AI)
        self.model_a_selector.setEnabled(needs_model_a)
        self.btn_model_a.setEnabled(needs_model_a)
        self.model_b_selector.setEnabled(needs_model_b)
        self.btn_model_b.setEnabled(needs_model_b)
        self.field.set_control_enabled(mode == HUMAN_VS_AI and self.simulator is not None)
        self.mode_hint.setText(
            {
                HUMAN_VS_AI: "Left mallet: mouse. Right mallet: selected AI model.",
                AI_VS_AI: "Select two policies. Model B is mirrored to play from the right side.",
                AI_VS_SCRIPTED: "Model A plays from the left against the built-in ComplexOpponent.",
            }[mode]
        )

    def _selected_model_path(self, selector: QComboBox) -> str | None:
        data = selector.currentData()
        return str(data) if data else None

    def start_match(self) -> None:
        self._simulation_timer.stop()
        self._hud_timer.stop()
        if self.simulator is not None:
            self.simulator.close()
            self.simulator = None
        mode = self.mode_selector.currentData()
        try:
            self.simulator = MatchSimulator(
                mode=mode,
                model_a_path=self._selected_model_path(self.model_a_selector),
                model_b_path=self._selected_model_path(self.model_b_selector),
                seed=int(time.time_ns() % (2**32)),
            )
        except (ValueError, OSError) as exc:
            self.btn_pause.setEnabled(False)
            self.btn_reset_puck.setEnabled(False)
            self._set_setup_enabled(True)
            QMessageBox.warning(self, "Could not start match", str(exc))
            return

        self.left_score = 0
        self.right_score = 0
        self._accumulator = 0.0
        self.field.update_observation_vector(self.simulator.observation)
        self.field.set_control_enabled(mode == HUMAN_VS_AI)
        self._set_setup_enabled(False)
        self.btn_pause.setText("PAUSE")
        self.btn_pause.setEnabled(True)
        self.btn_reset_puck.setEnabled(True)
        self.status_label.setText(f"Match running: {MODE_LABELS[mode]}")
        self._clock.start()
        self._simulation_timer.start()
        self._hud_timer.start()
        self._refresh_hud()

    def toggle_pause(self) -> None:
        if self.simulator is None:
            return
        if self._simulation_timer.isActive():
            self._simulation_timer.stop()
            self._hud_timer.stop()
            self.btn_pause.setText("RESUME")
            self.status_label.setText("Match paused")
        else:
            self._clock.start()
            self._simulation_timer.start()
            self._hud_timer.start()
            self.btn_pause.setText("PAUSE")
            self.status_label.setText(f"Match running: {MODE_LABELS[self.simulator.mode]}")

    def reset_puck(self) -> None:
        if self.simulator is None:
            return
        self.simulator.reset_puck()
        self.field.update_observation_vector(self.simulator.observation)
        self._refresh_hud()
        self.status_label.setText("Puck reset to center; score and match time retained.")

    def _back_to_menu(self) -> None:
        self._simulation_timer.stop()
        self._hud_timer.stop()
        if self.simulator is not None:
            self.simulator.close()
            self.simulator = None
        self.btn_pause.setEnabled(False)
        self.btn_reset_puck.setEnabled(False)
        self.btn_pause.setText("PAUSE")
        self._set_setup_enabled(True)
        self._on_mode_changed()
        if self.on_back is not None:
            self.on_back()

    def _set_setup_enabled(self, enabled: bool) -> None:
        self.mode_selector.setEnabled(enabled)
        self.model_a_selector.setEnabled(
            enabled and self.mode_selector.currentData() in (AI_VS_AI, AI_VS_SCRIPTED)
        )
        self.model_b_selector.setEnabled(
            enabled and self.mode_selector.currentData() in (HUMAN_VS_AI, AI_VS_AI)
        )
        self.btn_model_a.setEnabled(
            enabled and self.mode_selector.currentData() in (AI_VS_AI, AI_VS_SCRIPTED)
        )
        self.btn_model_b.setEnabled(
            enabled and self.mode_selector.currentData() in (HUMAN_VS_AI, AI_VS_AI)
        )

    def _set_human_target(self, x_m: float, y_m: float) -> None:
        if self.simulator is not None and self.simulator.mode == HUMAN_VS_AI:
            self.simulator.set_mouse_target(x_m, y_m)

    def _advance_simulation(self) -> None:
        if self.simulator is None:
            return
        elapsed = min(self._clock.restart() / 1000.0, 0.1)
        self._accumulator += elapsed
        while self._accumulator >= self.SIM_DT:
            _, _, terminated, truncated, info = self.simulator.step()
            self._accumulator -= self.SIM_DT
            if terminated or truncated:
                scored_for = info.get("scored_for")
                if scored_for == "robot":
                    self.left_score += 1
                elif scored_for == "opponent":
                    self.right_score += 1
                if terminated:
                    self.simulator.reset_after_goal(scored_for)
                else:
                    self.simulator.observation, _ = self.simulator.env.reset()
                self.status_label.setText(
                    f"Goal: {'Left' if scored_for == 'robot' else 'Right'} player"
                    if terminated
                    else "Round time limit reached"
                )
        self._refresh_hud()

    def _refresh_hud(self) -> None:
        if self.simulator is None:
            return
        self.field.update_observation_vector(self.simulator.observation)
        self.left_score_label.setText(f"LEFT  {self.left_score}")
        self.right_score_label.setText(f"{self.right_score}  RIGHT")
        elapsed = self.simulator.elapsed_seconds
        minutes = int(elapsed // 60)
        seconds = elapsed % 60.0
        self.match_time_label.setText(f"{minutes:02d}:{seconds:05.2f}")
        self.puck_speed_label.setText(f"PUCK  {self.simulator.puck_speed_mps:.2f} m/s")

    def closeEvent(self, event) -> None:
        self._simulation_timer.stop()
        self._hud_timer.stop()
        if self.simulator is not None:
            self.simulator.close()
        super().closeEvent(event)
