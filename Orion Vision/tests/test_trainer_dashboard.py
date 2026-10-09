"""Smoke tests for the SAC trainer environment, worker, and launcher."""

import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import URLError

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton

from ai.air_hockey_env import AirHockeyGymEnv
from ai.trainer_worker import (
    AirHockeyMetricsCallback,
    PPOCurriculumWorker,
    SACTrainerWorker,
    linear_schedule,
)
from ui.ai_trainer_dashboard import OrionTrainerDashboard
from ui.main_menu import OrionMainMenu


def test_air_hockey_env_observation_action_and_step_contract():
    env = AirHockeyGymEnv(seed=7, max_episode_steps=10)
    observation, _ = env.reset(seed=11)

    assert observation.shape == (8,)
    assert env.observation_space.shape == (8,)
    assert env.action_space.shape == (2,)
    assert np.all(observation >= -1.0) and np.all(observation <= 1.0)

    next_observation, reward, terminated, truncated, info = env.step(
        np.asarray([0.2, -0.3], dtype=np.float32)
    )
    assert next_observation.shape == (8,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "goal_scored" in info
    env.close()


def test_reward_mode_is_configurable_and_kept_on_environment():
    env = AirHockeyGymEnv(seed=7, max_episode_steps=10, reward_mode="balanced")
    assert env.reward_mode == "balanced"
    env.reset(seed=11)
    _, reward, _, _, _ = env.step(np.asarray([0.2, -0.3], dtype=np.float32))
    assert np.isfinite(reward)
    env.close()

    worker = SACTrainerWorker(total_timesteps=10, reward_mode="defensive")
    assert worker.reward_mode == "defensive"


def test_contact_threshold_and_clear_bonus_reward_logic():
    env = AirHockeyGymEnv(seed=12, max_episode_steps=5, opponent_type="none")
    env.reset(seed=13)

    env._robot_position[:] = (-0.55, 0.0)
    env._robot_velocity[:] = (0.0, 0.0)
    env._puck_position[:] = (-0.80, 0.0)
    env._puck_velocity[:] = (1.6, 0.0)

    _, reward, _, _, _ = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert reward > 2.0
    assert env.opponent_type == "none"
    env.close()


def test_robot_is_clamped_to_15_cm_before_center_line():
    env = AirHockeyGymEnv(seed=12, max_episode_steps=5, opponent_type="none")
    env.reset(seed=13)
    env._robot_position[:] = (-0.16, 0.0)
    env._robot_velocity[:] = (2.5, 0.0)
    env._puck_position[:] = (0.5, 0.3)
    env._puck_velocity[:] = 0.0

    env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert env.robot_position_m[0] <= -0.15
    env.close()


def test_clear_bonus_requires_active_puck_crossing():
    quiet_env = AirHockeyGymEnv(seed=21, opponent_type="none")
    quiet_env.reset(seed=21)
    quiet_env._robot_position[:] = (-0.7, 0.3)
    quiet_env._puck_position[:] = (-0.000001, 0.0)
    quiet_env._puck_velocity[:] = (0.001, 0.0)
    _, quiet_reward, _, _, _ = quiet_env.step(np.zeros(2, dtype=np.float32))

    active_env = AirHockeyGymEnv(seed=21, opponent_type="none")
    active_env.reset(seed=21)
    active_env._robot_position[:] = (-0.7, 0.3)
    active_env._puck_position[:] = (-0.001, 0.0)
    active_env._puck_velocity[:] = (0.7, 0.0)
    _, active_reward, _, _, _ = active_env.step(np.zeros(2, dtype=np.float32))

    assert quiet_env.puck_position_m[0] >= 0.0
    assert quiet_reward < 1.0
    assert active_env.puck_position_m[0] >= 0.0
    assert active_reward > 1.0
    quiet_env.close()
    active_env.close()


def test_robot_is_penalized_for_forward_position_when_puck_is_opponent_side():
    forward_env = AirHockeyGymEnv(seed=31, opponent_type="none")
    forward_env.reset(seed=31)
    forward_env._robot_position[:] = (-0.30, 0.0)
    forward_env._puck_position[:] = (0.5, 0.3)
    forward_env._puck_velocity[:] = 0.0
    _, forward_reward, _, _, _ = forward_env.step(np.zeros(2, dtype=np.float32))

    home_env = AirHockeyGymEnv(seed=31, opponent_type="none")
    home_env.reset(seed=31)
    home_env._robot_position[:] = (-0.40, 0.0)
    home_env._puck_position[:] = (0.5, 0.3)
    home_env._puck_velocity[:] = 0.0
    _, home_reward, _, _, _ = home_env.step(np.zeros(2, dtype=np.float32))

    assert np.isclose(home_reward - forward_reward, 0.02)
    forward_env.close()
    home_env.close()


def test_learning_rate_schedule_decays_from_initial_to_final_value():
    schedule = linear_schedule(3e-4)

    assert np.isclose(schedule(1.0), 3e-4)
    assert np.isclose(schedule(0.5), (3e-4 + 1e-5) / 2.0)
    assert np.isclose(schedule(0.0), 1e-5)


def test_ppo_curriculum_uses_defensive_rewards_in_later_stages():
    assert PPOCurriculumWorker.STAGES[0][-1] == "aggressive"
    assert PPOCurriculumWorker.STAGES[1][-1] == "defensive"
    assert PPOCurriculumWorker.STAGES[2][-1] == "defensive"


def test_air_hockey_metrics_callback_records_behavior_score_and_episode_metrics():
    class FakeEnv:
        num_envs = 1

    class FakeLogger:
        def __init__(self):
            self.values = {}

        def record(self, key, value):
            self.values[key] = value

    class FakeModel:
        num_timesteps = 1
        logger = FakeLogger()

        def get_env(self):
            return FakeEnv()

    model = FakeModel()
    callback = AirHockeyMetricsCallback()
    callback.init_callback(model)
    callback.on_training_start(locals_={}, globals_={})
    terminal_observation = np.zeros(8, dtype=np.float32)
    terminal_observation[6] = -0.72
    callback.update_locals(
        {
            "rewards": np.asarray([1.5]),
            "dones": np.asarray([True]),
            "new_obs": terminal_observation.reshape(1, -1),
            "infos": [
                {
                    "robot_contact": True,
                    "scored_for": "robot",
                    "terminal_observation": terminal_observation,
                    "episode": {"r": 1.5, "l": 1},
                }
            ],
        }
    )

    assert callback.on_step()
    assert model.logger.values["1_Gedrag/Muurovertredingen"] == 0
    assert model.logger.values["1_Gedrag/Verdediging_Dekking"] == 100.0
    assert model.logger.values["1_Gedrag/Puck_Balcontact"] == 1
    assert model.logger.values["2_Prestaties/Doelpunten_Voor"] == 1
    assert model.logger.values["2_Prestaties/Doelpunten_Tegen"] == 0
    assert model.logger.values["0_Overzicht/Gemiddelde_Beloning"] == 1.5
    assert model.logger.values["0_Overzicht/Gemiddelde_Episode_Duur_Stappen"] == 1.0


def test_dashboard_exposes_training_modes_and_stage_monitor():
    app = QApplication.instance() or QApplication([])
    dashboard = OrionTrainerDashboard()
    assert hasattr(dashboard, "reward_mode_selector")
    assert dashboard.reward_mode_selector.currentText() == "aggressive"
    assert dashboard.training_mode_selector.currentData() == "sac"
    dashboard.training_mode_selector.setCurrentIndex(1)
    assert dashboard.training_mode_selector.currentData() == "ppo"
    assert "1,100,000" in dashboard.total_steps_label.text()
    dashboard._on_stage_changed("Stage 2: Static Opponent")
    assert dashboard.stage_label.text() == "Current stage: Stage 2: Static Opponent"
    dashboard.close()
    app.processEvents()


def test_dashboard_starts_ppo_worker_for_curriculum_mode(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class FakeSignal:
        def connect(self, _slot):
            pass

    class FakeWorker:
        def __init__(self, output_path=None, parent=None):
            self.output_path = output_path
            self.signals = [FakeSignal() for _ in range(8)]
            (
                self.progress_changed,
                self.metrics_updated,
                self.model_saved,
                self.observation_updated,
                self.status_changed,
                self.training_error,
                self.stage_changed,
                self.finished,
            ) = self.signals
            self.started = False

        def isRunning(self):
            return False

        def start(self):
            self.started = True

    monkeypatch.setattr("ui.ai_trainer_dashboard.PPOCurriculumWorker", FakeWorker)
    dashboard = OrionTrainerDashboard()
    dashboard.training_mode_selector.setCurrentIndex(1)

    dashboard.start_training()

    assert isinstance(dashboard.trainer_worker, FakeWorker)
    assert dashboard.trainer_worker.started
    assert dashboard.trainer_worker.output_path == dashboard.output_path_label.text()
    dashboard._on_worker_finished()
    dashboard.close()
    app.processEvents()


def test_dashboard_allows_custom_ppo_output_path(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    chosen_path = tmp_path / "runs" / "custom_ppo.zip"
    dialogs = []

    def choose_save_path(*args):
        dialogs.append(args)
        return str(chosen_path), "Stable-Baselines3 model (*.zip)"

    monkeypatch.setattr(
        "ui.ai_trainer_dashboard.QFileDialog.getSaveFileName",
        choose_save_path,
    )
    dashboard = OrionTrainerDashboard()
    dashboard.training_mode_selector.setCurrentIndex(1)

    assert dashboard.btn_choose_output.isEnabled()
    dashboard.btn_choose_output.click()

    assert dashboard.output_path_label.text() == str(chosen_path)
    assert dashboard._ppo_output_path == str(chosen_path)
    assert dialogs[0][2].endswith("final_curriculum_ppo.zip")
    dashboard.close()
    app.processEvents()


def test_ppo_worker_saves_stage_checkpoints_beside_selected_final_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ai.train_curriculum import train_stage

    output_path = tmp_path / "models" / "custom_agent.zip"
    calls = []
    cumulative_timesteps = 0

    def fake_train_stage(**kwargs):
        nonlocal cumulative_timesteps
        calls.append(kwargs)
        cumulative_timesteps += kwargs["total_timesteps"]
        return SimpleNamespace(num_timesteps=cumulative_timesteps)

    monkeypatch.setattr("ai.train_curriculum.train_stage", fake_train_stage)
    worker = PPOCurriculumWorker(output_path=output_path)

    worker.run()

    expected_paths = [
        tmp_path / "models" / "custom_agent_stage1.zip",
        tmp_path / "models" / "custom_agent_stage2.zip",
        output_path,
    ]
    assert [Path(call["save_name"]) for call in calls] == expected_paths
    assert calls[0]["previous_model_path"] is None
    assert calls[1]["previous_model_path"] == str(expected_paths[0])
    assert calls[2]["previous_model_path"] == str(expected_paths[1])


def test_dashboard_opens_tensorboard_when_server_is_available(monkeypatch):
    app = QApplication.instance() or QApplication([])
    opened_urls = []
    monkeypatch.setattr("ui.ai_trainer_dashboard.urlopen", lambda *args, **kwargs: BytesIO())
    monkeypatch.setattr(
        "ui.ai_trainer_dashboard.webbrowser.open",
        lambda url: opened_urls.append(url) or True,
    )
    dashboard = OrionTrainerDashboard()

    dashboard.btn_tensorboard.click()

    assert opened_urls == ["http://localhost:6006"]
    dashboard.close()
    app.processEvents()


def test_dashboard_reports_when_tensorboard_server_is_unavailable(monkeypatch):
    app = QApplication.instance() or QApplication([])
    opened_urls = []
    launched_commands = []

    def unavailable(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr("ui.ai_trainer_dashboard.urlopen", unavailable)
    monkeypatch.setattr(
        "ui.ai_trainer_dashboard.webbrowser.open",
        lambda url: opened_urls.append(url) or True,
    )
    monkeypatch.setattr(
        "ui.ai_trainer_dashboard.subprocess.Popen",
        lambda command, **kwargs: launched_commands.append((command, kwargs)),
    )
    monkeypatch.setattr("ui.ai_trainer_dashboard.QMessageBox.warning", lambda *args: None)
    dashboard = OrionTrainerDashboard()

    dashboard.btn_tensorboard.click()

    assert opened_urls == ["http://localhost:6006"]
    assert launched_commands
    command, options = launched_commands[0]
    assert command[command.index("--logdir") + 1] == str(
        Path(__file__).resolve().parents[1] / "sac_air_hockey_tensorboard"
    )
    assert options["cwd"] == str(Path(command[command.index("--logdir") + 1]).parent)
    assert "Starting TensorBoard" in dashboard.status_label.text()
    dashboard.close()
    app.processEvents()


def test_circle_impact_uses_restitution_and_separates_overlap():
    env = AirHockeyGymEnv(seed=3)
    env.reset(seed=3)
    env._robot_position[:] = (0.0, 0.0)
    env._robot_velocity[:] = 0.0
    env._puck_position[:] = (0.07, 0.0)
    env._puck_velocity[:] = (-1.0, 0.0)

    collided = env._resolve_mallet_puck_collision(
        env._robot_position,
        env._robot_velocity,
    )

    combined_radius = env.PUCK_RADIUS_M + env.MALLET_RADIUS_M
    assert collided
    assert env._puck_position[0] >= combined_radius
    assert env._puck_velocity[0] > 0.0
    post_impact_relative_speed = env._puck_velocity[0] - env._robot_velocity[0]
    np.testing.assert_allclose(post_impact_relative_speed, 0.85, atol=1e-6)
    env.close()


def test_puck_wall_bounce_uses_radius_and_damping():
    env = AirHockeyGymEnv(seed=4)
    env.reset(seed=4)
    env._puck_position[:] = (0.0, env.FIELD_WIDTH_M / 2.0 - env.PUCK_RADIUS_M - 0.001)
    env._puck_velocity[:] = (0.0, 1.0)
    env._robot_position[:] = (-0.7, -0.3)
    env._opponent_position[:] = (0.7, 0.3)

    env.step(np.zeros(2, dtype=np.float32))

    y_limit = env.FIELD_WIDTH_M / 2.0 - env.PUCK_RADIUS_M
    assert env._puck_position[1] <= y_limit
    assert env._puck_velocity[1] < 0.0
    assert abs(float(env._puck_velocity[1])) < 0.91
    env.close()


def test_robot_motor_acceleration_is_limited_to_12_mps2():
    env = AirHockeyGymEnv(seed=5)
    env.reset(seed=5)
    initial_velocity = env.robot_velocity_mps

    env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    acceleration = np.linalg.norm(env.robot_velocity_mps - initial_velocity) / env.DT
    assert acceleration <= env.MAX_MALLET_ACCELERATION_MPS2 + 1e-6
    np.testing.assert_allclose(acceleration, 12.0, atol=1e-5)
    env.close()


def test_physics_stays_finite_in_bounds_and_over_100_steps_per_second():
    import time

    env = AirHockeyGymEnv(seed=6, max_episode_steps=10_000)
    env.reset(seed=6)
    action = np.asarray([0.45, -0.35], dtype=np.float32)
    started = time.perf_counter()
    for _ in range(2000):
        observation, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert np.all(observation >= -1.0) and np.all(observation <= 1.0)
        assert abs(float(env._puck_position[1])) <= (
            env.FIELD_WIDTH_M / 2.0 - env.PUCK_RADIUS_M + 1e-9
        )
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - started
    steps_per_second = 2000 / max(elapsed, 1e-9)

    assert steps_per_second >= 100.0
    env.close()


def test_short_sac_training_emits_progress_metrics_and_saves(tmp_path):
    output = tmp_path / "short_training_model.zip"
    project_root = Path(__file__).resolve().parents[1]
    tensorboard_root = project_root / "sac_air_hockey_tensorboard"
    existing_event_files = set(tensorboard_root.rglob("events.out.tfevents.*"))
    child_script = f"""
import torch
from PyQt5.QtCore import QCoreApplication
from ai.trainer_worker import SACTrainerWorker

app = QCoreApplication([])
worker = SACTrainerWorker(
    total_timesteps=100,
    learning_rate=3e-4,
    buffer_size=1000,
    output_path={str(output)!r},
)
progress = []
saved = []
observations = []
errors = []
worker.progress_changed.connect(lambda step, total: progress.append((step, total)))
worker.model_saved.connect(saved.append)
worker.observation_updated.connect(observations.append)
worker.training_error.connect(errors.append)
worker.start()
if not worker.wait(90000):
    raise TimeoutError('trainer worker did not finish within 90 seconds')
app.processEvents()
assert not errors, '; '.join(errors)
assert progress, 'no progress signals were emitted'
assert progress[-1][1] == 100
assert observations and len(observations[-1]) == 8
assert saved == [{str(output)!r}]
expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert worker._model.device.type == expected_device
print(f'short SAC run saved model on {{worker._model.device}}')
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=110,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file()
    assert "saved model on" in completed.stdout

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = [
        path for path in tensorboard_root.rglob("events.out.tfevents.*")
        if path not in existing_event_files
    ]
    assert event_files, "training did not create a TensorBoard event file"
    latest_event_file = max(event_files, key=lambda path: path.stat().st_mtime)
    event_accumulator = EventAccumulator(str(latest_event_file))
    event_accumulator.Reload()
    scalar_tags = set(event_accumulator.Tags().get("scalars", []))
    assert {
        "1_Gedrag/Muurovertredingen",
        "1_Gedrag/Verdediging_Dekking",
        "1_Gedrag/Puck_Balcontact",
        "2_Prestaties/Doelpunten_Voor",
        "2_Prestaties/Doelpunten_Tegen",
    } <= scalar_tags


def test_launch_builder_button_requests_trainer_module():
    app = QApplication.instance() or QApplication([])
    requested_modules = []
    menu = OrionMainMenu(requested_modules.append)

    launch_button = next(
        button for button in menu.findChildren(QPushButton)
        if button.text() == "LAUNCH BUILDER"
    )
    assert launch_button.isEnabled()
    launch_button.click()

    assert requested_modules == ["ai_trainer"]
    menu.close()
    app.processEvents()
