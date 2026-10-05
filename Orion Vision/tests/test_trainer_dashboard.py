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
from ai.trainer_worker import AirHockeyMetricsCallback, SACTrainerWorker, linear_schedule
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


def test_learning_rate_schedule_decays_from_initial_to_final_value():
    schedule = linear_schedule(3e-4)

    assert np.isclose(schedule(1.0), 3e-4)
    assert np.isclose(schedule(0.5), (3e-4 + 1e-5) / 2.0)
    assert np.isclose(schedule(0.0), 1e-5)


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


def test_dashboard_exposes_reward_selector():
    app = QApplication.instance() or QApplication([])
    dashboard = OrionTrainerDashboard()
    assert hasattr(dashboard, "reward_mode_selector")
    assert dashboard.reward_mode_selector.currentText() == "aggressive"
    dashboard.close()
    app.processEvents()


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

    def unavailable(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr("ui.ai_trainer_dashboard.urlopen", unavailable)
    monkeypatch.setattr(
        "ui.ai_trainer_dashboard.webbrowser.open",
        lambda url: opened_urls.append(url) or True,
    )
    monkeypatch.setattr("ui.ai_trainer_dashboard.QMessageBox.warning", lambda *args: None)
    dashboard = OrionTrainerDashboard()

    dashboard.btn_tensorboard.click()

    assert not opened_urls
    assert "TensorBoard is not running" in dashboard.status_label.text()
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
