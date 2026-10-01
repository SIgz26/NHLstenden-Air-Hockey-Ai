"""Smoke tests for the SAC trainer environment, worker, and launcher."""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton

from ai.air_hockey_env import AirHockeyGymEnv
from ai.trainer_worker import SACTrainerWorker
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
