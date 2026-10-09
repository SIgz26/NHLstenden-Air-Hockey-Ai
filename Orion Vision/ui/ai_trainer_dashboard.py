"""PyQt dashboard for training an air-hockey SAC policy."""

from __future__ import annotations

import subprocess
import sys
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

from ai.trainer_worker import PPOCurriculumWorker, SACTrainerWorker
from ui.widgets.digital_twin_widget import DigitalTwinWidget

TENSORBOARD_URL = "http://localhost:6006"
TENSORBOARD_LOG_DIR = Path(__file__).resolve().parents[1] / "sac_air_hockey_tensorboard"


class OrionTrainerDashboard(QWidget):
    """Configure, run, monitor, and save a background SAC training session."""

    def __init__(self, on_back_callback=None) -> None:
        super().__init__()
        self.on_back = on_back_callback
        self.trainer_worker: SACTrainerWorker | PPOCurriculumWorker | None = None
        self._sac_output_path = "models/trained_sac_model.zip"
        self._ppo_output_path = "models/final_curriculum_ppo.zip"
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
        title = QLabel("AI TRAINER // POLICY TRAINING")
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

        self.training_mode_selector = QComboBox()
        self.training_mode_selector.addItem("Single Model (SAC)", "sac")
        self.training_mode_selector.addItem("Curriculum Training (PPO)", "ppo")
        self.training_mode_selector.currentIndexChanged.connect(self._on_training_mode_changed)
        controls_layout.addWidget(QLabel("Training Mode"))
        controls_layout.addWidget(self.training_mode_selector)

        self.total_steps_input = QSpinBox()
        self.total_steps_input.setRange(100, 100_000_000)
        self.total_steps_input.setSingleStep(10_000)
        self.total_steps_input.setValue(500_000)
        self.total_steps_label = QLabel("Total Timesteps")
        controls_layout.addWidget(self.total_steps_label)
        controls_layout.addWidget(self.total_steps_input)

        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setDecimals(6)
        self.learning_rate_input.setRange(0.000001, 1.0)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setValue(0.0003)
        controls_layout.addWidget(QLabel("Learning Rate (SAC)"))
        controls_layout.addWidget(self.learning_rate_input)

        self.buffer_size_input = QSpinBox()
        self.buffer_size_input.setRange(1_000, 10_000_000)
        self.buffer_size_input.setSingleStep(10_000)
        self.buffer_size_input.setValue(100_000)
        controls_layout.addWidget(QLabel("Replay Buffer Size (SAC)"))
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
        self.output_model_title = QLabel("Output Model")
        controls_layout.addWidget(self.output_model_title)
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
        self.stage_label = QLabel("Current stage: --")
        monitor_layout.addWidget(self.stage_label)
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
        self._on_training_mode_changed()

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #d77d42; font-weight: bold;")
        return label

    def _choose_output_path(self) -> None:
        is_ppo = self.training_mode_selector.currentData() == "ppo"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PPO curriculum model" if is_ppo else "Save SAC model",
            str(Path(self.output_path_label.text())),
            "Stable-Baselines3 model (*.zip)",
        )
        if path:
            destination = Path(path)
            if destination.suffix.lower() != ".zip":
                destination = destination.with_suffix(".zip")
            if is_ppo:
                self._ppo_output_path = str(destination)
            else:
                self._sac_output_path = str(destination)
            self.output_path_label.setText(str(destination))

    def start_training(self) -> None:
        """Start the selected training mode in its worker thread."""
        if self.trainer_worker is not None and self.trainer_worker.isRunning():
            return
        try:
            if self.training_mode_selector.currentData() == "ppo":
                self.trainer_worker = PPOCurriculumWorker(
                    output_path=self.output_path_label.text(),
                    parent=self,
                )
                self.stage_label.setText("Current stage: waiting to start")
            else:
                self.trainer_worker = SACTrainerWorker(
                    total_timesteps=self.total_steps_input.value(),
                    learning_rate=self.learning_rate_input.value(),
                    buffer_size=self.buffer_size_input.value(),
                    output_path=self.output_path_label.text(),
                    reward_mode=self.reward_mode_selector.currentText(),
                    parent=self,
                )
                self.stage_label.setText("Current stage: Single Model (SAC)")
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        worker = self.trainer_worker
        worker.progress_changed.connect(self._update_progress)
        worker.metrics_updated.connect(self._update_metrics)
        worker.model_saved.connect(self._on_model_saved)
        worker.observation_updated.connect(self.preview_widget.update_observation_vector)
        worker.status_changed.connect(self._on_training_status)
        if isinstance(worker, PPOCurriculumWorker):
            worker.stage_changed.connect(self._on_stage_changed)
        worker.training_error.connect(self._on_training_error)
        worker.finished.connect(self._on_worker_finished)
        self.device_label.setText(
            "Compute device: Stable-Baselines3 auto-selection"
            if isinstance(worker, PPOCurriculumWorker)
            else "Compute device: checking CUDA..."
        )
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
        """Launch TensorBoard for the shared training log directory if needed."""
        try:
            with urlopen(TENSORBOARD_URL, timeout=1):
                pass
        except (OSError, URLError):
            TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tensorboard.main",
                        "--logdir",
                        str(TENSORBOARD_LOG_DIR),
                        "--port",
                        "6006",
                    ],
                    cwd=str(TENSORBOARD_LOG_DIR.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as launch_error:
                message = f"Could not start TensorBoard: {launch_error}"
                self.status_label.setText(message)
                QMessageBox.warning(self, "TensorBoard unavailable", message)
                return
            self.status_label.setText(f"Starting TensorBoard for {TENSORBOARD_LOG_DIR}")

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
        self.training_mode_selector.setEnabled(not running)
        self.reward_mode_selector.setEnabled(
            not running and self.training_mode_selector.currentData() != "ppo"
        )

    def _on_training_mode_changed(self, *_args) -> None:
        is_ppo = self.training_mode_selector.currentData() == "ppo"
        self.learning_rate_input.setEnabled(not is_ppo)
        self.buffer_size_input.setEnabled(not is_ppo)
        self.reward_mode_selector.setEnabled(not is_ppo)
        self.total_steps_input.setEnabled(not is_ppo)
        self.btn_choose_output.setEnabled(self.btn_start.isEnabled())
        self.total_steps_label.setText(
            "Total Timesteps (fixed: 1,100,000)" if is_ppo else "Total Timesteps"
        )
        self.output_model_title.setText("Final PPO Model" if is_ppo else "Output Model")
        self.output_path_label.setText(
            self._ppo_output_path if is_ppo else self._sac_output_path
        )
        self.loss_label.setText("Value loss: --" if is_ppo else "Critic loss: --")
        if not self.btn_start.isEnabled():
            return
        self.status_label.setText("Ready for PPO curriculum" if is_ppo else "Ready for SAC training")

    def _on_stage_changed(self, stage: str) -> None:
        self.stage_label.setText(f"Current stage: {stage}")

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
