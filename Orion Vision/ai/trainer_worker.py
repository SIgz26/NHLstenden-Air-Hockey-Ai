"""Background Stable-Baselines3 SAC training worker with Qt telemetry."""

from __future__ import annotations

import threading
import time
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ai.air_hockey_env import AirHockeyGymEnv
from ai.sac_controller import get_torch_runtime

logger = logging.getLogger(__name__)


class SACTrainerWorker(QThread):
    """Train SAC away from the GUI thread and stream training metrics."""

    progress_changed = pyqtSignal(int, int)
    metrics_updated = pyqtSignal(float, float, float)
    model_saved = pyqtSignal(str)
    observation_updated = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    training_error = pyqtSignal(str)

    def __init__(
        self,
        total_timesteps: int = 500_000,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        output_path: str | Path = "models/trained_sac_model.zip",
        parent=None,
    ) -> None:
        super().__init__(parent)
        if total_timesteps < 1:
            raise ValueError("total_timesteps must be at least 1")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero")
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")

        self.total_timesteps = int(total_timesteps)
        self.learning_rate = float(learning_rate)
        self.buffer_size = int(buffer_size)
        self.output_path = Path(output_path)
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._model: Any | None = None
        self._env: AirHockeyGymEnv | None = None
        self._saved_path: Path | None = None
        self.device_used: str | None = None
        self.device_name: str | None = None

    def pause_training(self) -> None:
        """Pause at the next callback boundary."""
        self._pause_event.set()
        self.status_changed.emit("Training paused")

    def resume_training(self) -> None:
        """Resume a paused training run."""
        self._pause_event.clear()
        self.status_changed.emit("Training running")

    def stop_training(self) -> None:
        """Request an orderly stop and final model save."""
        self._stop_event.set()
        self._pause_event.clear()

    def save_model(self) -> str | None:
        """Save the current model when one has been initialized."""
        if self._model is None:
            return None
        destination = self._save_current_model()
        self.model_saved.emit(str(destination))
        return str(destination)

    def run(self) -> None:
        try:
            torch = get_torch_runtime()
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import BaseCallback

            worker = self

            class QtTrainingCallback(BaseCallback):
                def __init__(self) -> None:
                    super().__init__(verbose=0)
                    self.started_at = time.perf_counter()
                    self.last_metrics_at = self.started_at
                    self.recent_rewards: list[float] = []
                    self.last_episode_count = 0

                def _on_step(self) -> bool:
                    if worker._stop_event.is_set():
                        return False
                    while worker._pause_event.is_set() and not worker._stop_event.is_set():
                        time.sleep(0.05)
                    if worker._stop_event.is_set():
                        return False

                    current_step = int(self.num_timesteps)
                    worker.progress_changed.emit(current_step, worker.total_timesteps)
                    if worker._env is not None:
                        worker.observation_updated.emit(worker._env.observation)

                    now = time.perf_counter()
                    if now - self.last_metrics_at >= 0.5:
                        episode_rewards = self.locals.get("ep_info_buffer", [])
                        mean_reward = (
                            float(np.mean([entry["r"] for entry in episode_rewards]))
                            if episode_rewards else 0.0
                        )
                        elapsed = max(now - self.started_at, 1e-6)
                        fps = float(current_step / elapsed)
                        logger_values = getattr(self.model.logger, "name_to_value", {})
                        loss = float(logger_values.get("train/critic_loss", 0.0))
                        worker.metrics_updated.emit(mean_reward, fps, loss)
                        self.last_metrics_at = now
                    return True

            self._env = AirHockeyGymEnv()
            if torch.cuda.is_available():
                device = "cuda"
                self.device_name = torch.cuda.get_device_name(0)
                self.device_used = f"cuda:0 ({self.device_name})"
            else:
                device = "cpu"
                self.device_name = "CPU fallback"
                self.device_used = device

            logger.info("Starting SAC training on %s", self.device_used)

            self._model = SAC(
                "MlpPolicy",
                self._env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                verbose=0,
                device=device,
            )
            self.status_changed.emit(f"Training running on {self.device_used}")
            self._model.learn(
                total_timesteps=self.total_timesteps,
                callback=QtTrainingCallback(),
                progress_bar=False,
            )
            self.progress_changed.emit(
                min(int(self._model.num_timesteps), self.total_timesteps),
                self.total_timesteps,
            )
            saved_path = self._save_current_model()
            self._saved_path = saved_path
            self.model_saved.emit(str(saved_path))
            self.status_changed.emit("Training stopped and model saved")
        except Exception as exc:
            self.training_error.emit(f"Training failed: {type(exc).__name__}: {exc}")
            self.status_changed.emit("Training failed")
        finally:
            if self._env is not None:
                self._env.close()
                self._env = None

    def _save_current_model(self) -> Path:
        if self._model is None:
            raise RuntimeError("No SAC model is available to save")
        destination = self.output_path.expanduser()
        if not destination.is_absolute():
            destination = Path(__file__).resolve().parents[1] / destination
        if destination.suffix.lower() != ".zip":
            destination = destination.with_suffix(".zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(destination.with_suffix("")))
        return destination
