"""PyQt dashboard for training an air-hockey SAC policy."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ai.trainer_worker import SACTrainerWorker
from ui.widgets.digital_twin_widget import DigitalTwinWidget

TENSORBOARD_URL = "http://localhost:6006"


class OrionTrainerDashboard(QWidget):
    """Configure, run, monitor, and save a background SAC training session."""

    def __init__(self, on_back_callback=None) -> None:
        super().__init__()
        self.on_back = on_back_callback
        self.trainer_worker: SACTrainerWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "QWidget { background: #120d0b; color: #e9c89d; }"
            "QFrame#TrainerPanel { background: #17110f; border: 1px solid #503422; border-radius: 8px; }"
            "QPushButton { background: #1a1412; color: #f5d9b8; border: 1px solid #80512f; padding: 7px 12px; border-radius: 5px; }"
            "QPushButton:hover { background: #2a1e18; }"
            "QSpinBox, QDoubleSpinBox { background: #0c0a09; color: #f5d9b8; border: 1px solid #503422; padding: 4px; }"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(14)

        header = QHBoxLayout()
        if self.on_back is not None:
            self.btn_back = QPushButton("BACK TO MENU")
            self.btn_back.clicked.connect(self.on_back)
            header.addWidget(self.btn_back)
        title = QLabel("AI TRAINER // SAC MODEL BUILD")
        title.setStyleSheet("color: #d77d42; font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        root.addLayout(columns, stretch=1)

        controls = QFrame()
        controls.setObjectName("TrainerPanel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self._section_title("HYPERPARAMETERS"))

        self.total_steps_input = QSpinBox()
        self.total_steps_input.setRange(100, 100_000_000)
        self.total_steps_input.setSingleStep(10_000)
        self.total_steps_input.setValue(500_000)
        controls_layout.addWidget(QLabel("Total Timesteps"))
        controls_layout.addWidget(self.total_steps_input)

        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setDecimals(6)
        self.learning_rate_input.setRange(0.000001, 1.0)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setValue(0.0003)
        controls_layout.addWidget(QLabel("Learning Rate"))
        controls_layout.addWidget(self.learning_rate_input)

        self.buffer_size_input = QSpinBox()
        self.buffer_size_input.setRange(1_000, 10_000_000)
        self.buffer_size_input.setSingleStep(10_000)
        self.buffer_size_input.setValue(100_000)
        controls_layout.addWidget(QLabel("Replay Buffer Size"))
        controls_layout.addWidget(self.buffer_size_input)

        self.reward_mode_selector = QComboBox()
        self.reward_mode_selector.addItems(["aggressive", "balanced", "defensive"])
        self.reward_mode_selector.setCurrentText("aggressive")
        controls_layout.addWidget(QLabel("Reward Variant"))
        controls_layout.addWidget(self.reward_mode_selector)

        self.output_path_label = QLabel("models/trained_sac_model.zip")
        self.output_path_label.setWordWrap(True)
        self.btn_choose_output = QPushButton("CHOOSE MODEL OUTPUT")
        self.btn_choose_output.clicked.connect(self._choose_output_path)
        controls_layout.addWidget(QLabel("Output Model"))
        controls_layout.addWidget(self.output_path_label)
        controls_layout.addWidget(self.btn_choose_output)

        self.btn_start = QPushButton("START TRAINING")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop_save = QPushButton("STOP & SAVE MODEL")
        self.btn_stop_save.setEnabled(False)
        self.btn_stop_save.clicked.connect(self.stop_and_save)
        self.btn_tensorboard = QPushButton("Open TensorBoard")
        self.btn_tensorboard.clicked.connect(self.open_tensorboard)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_stop_save)
        controls_layout.addWidget(self.btn_tensorboard)
        controls_layout.addStretch()
        columns.addWidget(controls, stretch=1)

        monitor = QFrame()
        monitor.setObjectName("TrainerPanel")
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(16, 16, 16, 16)
        monitor_layout.setSpacing(10)
        monitor_layout.addWidget(self._section_title("LIVE TRAINING MONITOR"))

        self.progress_label = QLabel("0 / 0 steps")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        monitor_layout.addWidget(self.progress_label)
        monitor_layout.addWidget(self.progress_bar)

        metrics = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.reward_label = QLabel("Mean reward: --")
        self.loss_label = QLabel("Critic loss: --")
        metrics.addWidget(self.fps_label)
        metrics.addWidget(self.reward_label)
        metrics.addWidget(self.loss_label)
        monitor_layout.addLayout(metrics)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        monitor_layout.addWidget(self.status_label)
        self.device_label = QLabel("Compute device: not started")
        self.device_label.setWordWrap(True)
        self.device_label.setStyleSheet("color: #c4d2cb;")
        monitor_layout.addWidget(self.device_label)

        monitor_layout.addWidget(self._section_title("SIMULATION PREVIEW"))
        self.preview_widget = DigitalTwinWidget(refresh_hz=30.0)
        self.preview_widget.setMinimumSize(360, 220)
        monitor_layout.addWidget(self.preview_widget, stretch=1)
        columns.addWidget(monitor, stretch=2)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #d77d42; font-weight: bold;")
        return label

    def _choose_output_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SAC model",
            str(Path(self.output_path_label.text())),
            "Stable-Baselines3 model (*.zip)",
        )
        if path:
            destination = Path(path)
            if destination.suffix.lower() != ".zip":
                destination = destination.with_suffix(".zip")
            self.output_path_label.setText(str(destination))

    def start_training(self) -> None:
        """Start SAC training in its worker thread."""
        if self.trainer_worker is not None and self.trainer_worker.isRunning():
            return
        try:
            self.trainer_worker = SACTrainerWorker(
                total_timesteps=self.total_steps_input.value(),
                learning_rate=self.learning_rate_input.value(),
                buffer_size=self.buffer_size_input.value(),
                output_path=self.output_path_label.text(),
                reward_mode=self.reward_mode_selector.currentText(),
                parent=self,
            )
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        worker = self.trainer_worker
        worker.progress_changed.connect(self._update_progress)
        worker.metrics_updated.connect(self._update_metrics)
        worker.model_saved.connect(self._on_model_saved)
        worker.observation_updated.connect(self.preview_widget.update_observation_vector)
        worker.status_changed.connect(self._on_training_status)
        worker.training_error.connect(self._on_training_error)
        worker.finished.connect(self._on_worker_finished)
        self.device_label.setText("Compute device: checking CUDA...")
        self._set_training_controls(running=True)
        worker.start()

    def toggle_pause(self) -> None:
        """Pause or resume the active trainer."""
        worker = self.trainer_worker
        if worker is None or not worker.isRunning():
            return
        if worker._pause_event.is_set():
            worker.resume_training()
            self.btn_pause.setText("PAUSE")
        else:
            worker.pause_training()
            self.btn_pause.setText("RESUME")

    def stop_and_save(self) -> None:
        """Stop after the next training callback and save the model."""
        if self.trainer_worker is not None:
            self.status_label.setText("Stopping; saving model after current training step...")
            self.trainer_worker.stop_training()

    def open_tensorboard(self) -> None:
        """Open the local TensorBoard UI when its server is reachable."""
        try:
            with urlopen(TENSORBOARD_URL, timeout=1):
                pass
        except (OSError, URLError) as exc:
            message = (
                "TensorBoard is not running at localhost:6006. Start it with: "
                "python -m tensorboard.main --logdir .\\sac_air_hockey_tensorboard"
            )
            self.status_label.setText(message)
            QMessageBox.warning(self, "TensorBoard unavailable", f"{message}\n\n{exc}")
            return

        try:
            if not webbrowser.open(TENSORBOARD_URL):
                raise webbrowser.Error("No browser accepted the TensorBoard URL")
        except (webbrowser.Error, OSError) as exc:
            message = f"Could not open TensorBoard in a browser: {exc}"
            self.status_label.setText(message)
            QMessageBox.warning(self, "Could not open TensorBoard", message)

    def _set_training_controls(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_stop_save.setEnabled(running)
        self.total_steps_input.setEnabled(not running)
        self.learning_rate_input.setEnabled(not running)
        self.buffer_size_input.setEnabled(not running)
        self.btn_choose_output.setEnabled(not running)

    def _update_progress(self, current_step: int, total_steps: int) -> None:
        percentage = current_step / max(total_steps, 1)
        self.progress_bar.setValue(int(np_clip_0_1(percentage) * 1000))
        self.progress_label.setText(f"{current_step:,} / {total_steps:,} steps")

    def _update_metrics(self, mean_reward: float, fps: float, loss: float) -> None:
        self.reward_label.setText(f"Mean reward: {mean_reward:.3f}")
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.loss_label.setText(f"Critic loss: {loss:.5f}")

    def _on_training_status(self, message: str) -> None:
        self.status_label.setText(message)
        prefix = "Training running on "
        if message.startswith(prefix):
            self.device_label.setText(f"Compute device: {message[len(prefix):]}")

    def _on_model_saved(self, file_path: str) -> None:
        self.status_label.setText(f"Model saved: {file_path}")

    def _on_training_error(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_worker_finished(self) -> None:
        self._set_training_controls(running=False)
        self.btn_pause.setText("PAUSE")

    def closeEvent(self, event) -> None:
        worker = self.trainer_worker
        if worker is not None and worker.isRunning():
            worker.stop_training()
            worker.wait(3000)
        super().closeEvent(event)


def np_clip_0_1(value: float) -> float:
    """Clamp dashboard progress to the closed unit interval."""
    return max(0.0, min(1.0, float(value)))
