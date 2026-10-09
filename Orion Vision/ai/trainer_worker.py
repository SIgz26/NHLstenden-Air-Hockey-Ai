"""Background Stable-Baselines3 SAC training worker with Qt telemetry."""

from __future__ import annotations

import threading
import time
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ai.air_hockey_env import AirHockeyGymEnv
from ai.sac_controller import get_torch_runtime

logger = logging.getLogger(__name__)


def linear_schedule(initial_value: float, final_value: float = 1e-5):
    """Linearly decay the learning rate as training progresses."""
    def schedule(progress_remaining: float) -> float:
        # progress_remaining gaat van 1.0 (start) naar 0.0 (einde)
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


class AirHockeyMetricsCallback(BaseCallback):
    """Record readable behavior, score, and episode metrics for TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.wall_violation_steps = 0
        self.defensive_steps = 0
        self.environment_steps = 0
        self.robot_contacts = 0
        self.goals_for = 0
        self.goals_against = 0
        self._episode_rewards = np.zeros(0, dtype=np.float64)
        self._episode_lengths = np.zeros(0, dtype=np.int64)
        self._recent_episode_rewards: deque[float] = deque(maxlen=100)
        self._recent_episode_lengths: deque[int] = deque(maxlen=100)

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._episode_rewards = np.zeros(n_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        observations = np.asarray(self.locals.get("new_obs", []), dtype=np.float64)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)):
            infos = []

        n_envs = min(len(rewards), len(self._episode_rewards))
        self.environment_steps += n_envs
        self._episode_rewards[:n_envs] += rewards[:n_envs]
        self._episode_lengths[:n_envs] += 1

        for env_index in range(n_envs):
            info = infos[env_index] if env_index < len(infos) else {}
            if info.get("robot_contact", False):
                self.robot_contacts += 1
            if info.get("scored_for") == "robot":
                self.goals_for += 1
            elif info.get("scored_for") == "opponent":
                self.goals_against += 1

            observation = info.get("terminal_observation") if dones[env_index] else None
            if observation is None and env_index < len(observations):
                observation = observations[env_index]
            if observation is not None and len(observation) >= 8:
                half_length = AirHockeyGymEnv.FIELD_LENGTH_M / 2.0
                half_width = AirHockeyGymEnv.FIELD_WIDTH_M / 2.0
                robot_x = float(observation[6]) * half_length
                robot_y = float(observation[7]) * half_width
                wall_limit = (
                    half_width
                    - AirHockeyGymEnv.MALLET_RADIUS_M
                    - 0.05
                )
                if abs(robot_y) > wall_limit:
                    self.wall_violation_steps += 1

                in_defensive_zone = (
                    -half_length < robot_x < -half_length * 0.5
                    and abs(robot_y) <= AirHockeyGymEnv.GOAL_WIDTH_M / 2.0
                )
                if in_defensive_zone:
                    self.defensive_steps += 1

            if env_index < len(dones) and dones[env_index]:
                episode_info = info.get("episode", {})
                episode_reward = float(episode_info.get("r", self._episode_rewards[env_index]))
                episode_length = int(episode_info.get("l", self._episode_lengths[env_index]))
                self._recent_episode_rewards.append(episode_reward)
                self._recent_episode_lengths.append(episode_length)
                self._episode_rewards[env_index] = 0.0
                self._episode_lengths[env_index] = 0

        self.logger.record("1_Gedrag/Muurovertredingen", self.wall_violation_steps)
        defensive_coverage = (
            100.0 * self.defensive_steps / self.environment_steps
            if self.environment_steps
            else 0.0
        )
        self.logger.record("1_Gedrag/Verdediging_Dekking", defensive_coverage)
        self.logger.record("1_Gedrag/Puck_Balcontact", self.robot_contacts)
        self.logger.record("2_Prestaties/Doelpunten_Voor", self.goals_for)
        self.logger.record("2_Prestaties/Doelpunten_Tegen", self.goals_against)

        if self._recent_episode_rewards:
            self.logger.record(
                "0_Overzicht/Gemiddelde_Beloning",
                float(np.mean(self._recent_episode_rewards)),
            )
            self.logger.record(
                "0_Overzicht/Gemiddelde_Episode_Duur_Stappen",
                float(np.mean(self._recent_episode_lengths)),
            )
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps:
            self.logger.dump(step=self.num_timesteps)


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
        buffer_size: int = 200_000,
        output_path: str | Path = "models/trained_sac_model.zip",
        n_envs: int = 8,
        reward_mode: str = "aggressive",
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
        self.n_envs = int(n_envs)
        if self.n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        self.output_path = Path(output_path)
        self.reward_mode = AirHockeyGymEnv._resolve_reward_mode(reward_mode)
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
        eval_env: AirHockeyGymEnv | None = None
        try:
            torch = get_torch_runtime()
            from stable_baselines3 import SAC

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
                        if hasattr(worker._env, "get_attr"):
                            try:
                                observations = worker._env.get_attr("observation")
                                if observations:
                                    worker.observation_updated.emit(observations[0])
                            except Exception:
                                pass
                        elif hasattr(worker._env, "observation"):
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

            if self.n_envs > 1:
                try:
                    self._env = make_vec_env(
                        AirHockeyGymEnv,
                        n_envs=self.n_envs,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs={"reward_mode": self.reward_mode},
                    )
                except Exception as exc:  # pragma: no cover - safety fallback for PyQt multiprocess limits
                    logger.warning(
                        "Subprocess vectorized env failed; falling back to single env (%s)", exc
                    )
                    self._env = AirHockeyGymEnv(reward_mode=self.reward_mode)
                    self.n_envs = 1
            else:
                self._env = AirHockeyGymEnv(reward_mode=self.reward_mode)
            if torch.cuda.is_available():
                device = "cuda"
                self.device_name = torch.cuda.get_device_name(0)
                self.device_used = f"cuda:0 ({self.device_name})"
            else:
                device = "cpu"
                self.device_name = "CPU fallback"
                self.device_used = device

            logger.info("Starting SAC training on %s with %d parallel envs", self.device_used, self.n_envs)

            project_root = Path(__file__).resolve().parents[1]
            tensorboard_log = Path(__file__).resolve().parents[1] / "sac_air_hockey_tensorboard"
            best_model_dir = project_root / "models" / "best_model"
            eval_log_dir = project_root / "models" / "best_model_eval_logs"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            eval_log_dir.mkdir(parents=True, exist_ok=True)

            eval_env = AirHockeyGymEnv(reward_mode=self.reward_mode)
            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=10,
                min_evals=5,
                verbose=1,
            )
            eval_callback = EvalCallback(
                eval_env,
                callback_after_eval=stop_callback,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_log_dir),
                eval_freq=max(10_000 // self.n_envs, 1),
                n_eval_episodes=10,
                deterministic=True,
                render=False,
                verbose=1,
            )
            self._model = SAC(
                "MlpPolicy",
                self._env,
                learning_rate=linear_schedule(self.learning_rate, 1e-5), # Expliciet 1e-5 als eindwaarde
                batch_size=1024,
                gamma=0.98,
                tau=0.001,
                buffer_size=self.buffer_size,
                verbose=0,
                device=device,
                tensorboard_log=str(tensorboard_log),
            )
            
            self.status_changed.emit(f"Training running on {self.device_used}")
            self._model.learn(
                total_timesteps=self.total_timesteps,
                tb_log_name=f"SAC_{self.reward_mode}",
                callback=CallbackList(
                    [QtTrainingCallback(), AirHockeyMetricsCallback(), eval_callback]
                ),
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
            if eval_env is not None:
                eval_env.close()
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


class PPOCurriculumWorker(QThread):
    """Run the staged PPO curriculum and stream dashboard telemetry."""

    progress_changed = pyqtSignal(int, int)
    metrics_updated = pyqtSignal(float, float, float)
    model_saved = pyqtSignal(str)
    observation_updated = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    stage_changed = pyqtSignal(str)
    training_error = pyqtSignal(str)

    STAGES = (
        ("Stage 1: No Opponent", "none", 300_000, "stage1_ppo.zip", "Stage1_Puck_Raken", "aggressive"),
        ("Stage 2: Static Opponent", "static", 300_000, "stage2_ppo.zip", "Stage2_Static_Opponent", "defensive"),
        ("Stage 3: Complex Opponent", "complex", 500_000, "final_curriculum_ppo.zip", "Stage3_Complex_Opponent", "defensive"),
    )

    def __init__(
        self,
        output_path: str | Path = "models/final_curriculum_ppo.zip",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.total_timesteps = sum(stage[2] for stage in self.STAGES)
        self.output_path = Path(output_path).expanduser()
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._model: Any | None = None
        self._env: Any | None = None
        self._started_at = 0.0

    def pause_training(self) -> None:
        self._pause_event.set()
        self.status_changed.emit("Training paused")

    def resume_training(self) -> None:
        self._pause_event.clear()
        self.status_changed.emit("Training running")

    def stop_training(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()

    def run(self) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        from ai.train_curriculum import ROOT_DIR, train_stage

        worker = self
        completed_steps = 0
        previous_checkpoint: str | None = None
        self._started_at = time.perf_counter()
        output_path = self.output_path
        if not output_path.is_absolute():
            output_path = ROOT_DIR / output_path

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            for stage_index, (stage_label, opponent_type, stage_steps, _checkpoint, run_name, reward_mode) in enumerate(self.STAGES):
                if self._stop_event.is_set():
                    break
                stage_output_path = (
                    output_path
                    if stage_index == len(self.STAGES) - 1
                    else output_path.with_name(f"{output_path.stem}_stage{stage_index + 1}.zip")
                )
                self.stage_changed.emit(stage_label)
                self.status_changed.emit(f"{stage_label} running")
                stage_start = int(self._model.num_timesteps) if self._model is not None else 0

                class DashboardCallback(BaseCallback):
                    def _on_step(self) -> bool:
                        if worker._stop_event.is_set():
                            return False
                        while worker._pause_event.is_set() and not worker._stop_event.is_set():
                            time.sleep(0.05)
                        if worker._stop_event.is_set():
                            return False

                        stage_progress = max(0, int(self.num_timesteps) - stage_start)
                        worker.progress_changed.emit(
                            min(completed_steps + stage_progress, worker.total_timesteps),
                            worker.total_timesteps,
                        )
                        try:
                            observations = self.training_env.get_attr("observation")
                            if observations:
                                worker.observation_updated.emit(observations[0])
                        except (AttributeError, IndexError, RuntimeError):
                            pass

                        now = time.perf_counter()
                        if now - self.last_metrics_at >= 0.5:
                            rewards = getattr(self.model, "ep_info_buffer", [])
                            mean_reward = float(np.mean([item["r"] for item in rewards])) if rewards else 0.0
                            elapsed = max(now - worker._started_at, 1e-6)
                            fps = float(self.num_timesteps / elapsed)
                            values = getattr(self.model.logger, "name_to_value", {})
                            value_loss = float(values.get("train/value_loss", 0.0))
                            worker.metrics_updated.emit(mean_reward, fps, value_loss)
                            self.last_metrics_at = now
                        return True

                    def _on_training_start(self) -> None:
                        self.last_metrics_at = time.perf_counter()

                callback = DashboardCallback()
                self._model = train_stage(
                    opponent_type=opponent_type,
                    total_timesteps=stage_steps,
                    save_name=str(stage_output_path),
                    previous_model_path=previous_checkpoint,
                    stage_name=run_name,
                    seed=42,
                    reward_mode=reward_mode,
                    callback=callback,
                )
                completed_steps += min(stage_steps, max(0, int(self._model.num_timesteps) - stage_start))
                self.progress_changed.emit(min(completed_steps, self.total_timesteps), self.total_timesteps)
                previous_checkpoint = str(stage_output_path)
                self.model_saved.emit(previous_checkpoint)

            self.status_changed.emit("Curriculum stopped and latest checkpoint saved" if self._stop_event.is_set() else "Curriculum training complete")
        except Exception as exc:
            self.training_error.emit(f"Training failed: {type(exc).__name__}: {exc}")
            self.status_changed.emit("Training failed")
