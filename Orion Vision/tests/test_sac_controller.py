"""Tests for live SAC action inference and its 100 Hz worker loop."""

import os
import subprocess
import sys
import time
from unittest.mock import patch
from pathlib import Path

import numpy as np

from ai.agent_worker import SACAgentWorker
from ai.air_hockey_env import AirHockeyGymEnv
from ai.sac_controller import LiveSACController
from core.coordinate_contract import action_to_velocity_mps
from core.sim_bridge import SimBridge


class DummySACModel:
    observation_space = type("Space", (), {"shape": (8,)})()
    action_space = type("Space", (), {"shape": (2,)})()

    def __init__(self, action=(0.25, -0.5)):
        self.action = np.asarray(action, dtype=np.float32)
        self.last_observation = None
        self.deterministic = None

    def predict(self, observation, deterministic=True):
        self.last_observation = np.asarray(observation).copy()
        self.deterministic = deterministic
        return self.action.copy(), None


def test_controller_reads_live_vector_and_returns_two_actions():
    bridge = SimBridge()
    bridge.update(0.5, 0.25, 0.0, 0.0, frame_rate=60.0)
    model = DummySACModel()
    controller = LiveSACController(bridge, model=model)

    action = controller.predict_action()

    assert action == (0.25, -0.5)
    assert model.last_observation.shape == (8,)
    assert model.deterministic is True


def test_controller_predicts_without_gradients():
    import torch

    class GradientCheckingModel(DummySACModel):
        def predict(self, observation, deterministic=True):
            assert not torch.is_grad_enabled()
            return super().predict(observation, deterministic)

    controller = LiveSACController(SimBridge(), model=GradientCheckingModel())

    controller.predict_action()


def test_worker_runs_approximately_at_100_hz_and_calls_action_sink():
    bridge = SimBridge()
    bridge.set_table_corners(
        top_left=(0.0, 0.0),
        top_right=(1000.0, 0.0),
        bottom_right=(1000.0, 800.0),
        bottom_left=(0.0, 800.0),
    )
    bridge.configure_physical_table_bounds()
    bridge.update_from_engine(
        {
            "x": 500.0,
            "y": 400.0,
            "vx": 0.0,
            "vy": 0.0,
            "metadata": {"in_range": True},
        },
        frame_width=1000,
        frame_height=800,
        frame_rate=60.0,
    )
    controller = LiveSACController(bridge, model=DummySACModel((0.1, 0.1)))
    sent_actions = []
    worker = SACAgentWorker(
        controller,
        action_sink=lambda vx, vy: sent_actions.append((vx, vy)),
        frequency_hz=100.0,
    )

    worker.start()
    time.sleep(0.16)
    worker.stop()
    assert worker.wait(1000)

    assert 8 <= worker.completed_steps <= 25
    assert len(sent_actions) == worker.completed_steps
    assert sent_actions
    np.testing.assert_allclose(
        sent_actions[-1], (85.0, -100.0), atol=1e-5
    )


def test_worker_applies_asymmetric_plc_speed_limits():
    assert SACAgentWorker._to_plc_velocity(1.0, 0.0) == (0.0, -1000.0)
    assert SACAgentWorker._to_plc_velocity(-1.0, 0.0) == (0.0, 850.0)
    assert SACAgentWorker._to_plc_velocity(0.0, 1.0) == (850.0, 0.0)


def test_controller_reorders_calibrated_live_axes_for_training_policy():
    bridge = SimBridge()
    bridge.set_table_corners(
        top_left=(0.0, 0.0),
        top_right=(1000.0, 0.0),
        bottom_right=(1000.0, 800.0),
        bottom_left=(0.0, 800.0),
    )
    bridge.configure_physical_table_bounds()
    bridge.update_from_engine(
        {
            "x": 250.0,
            "y": 200.0,
            "vx": 0.0,
            "vy": 0.0,
            "metadata": {"in_range": True},
        },
        frame_width=1000,
        frame_height=800,
        frame_rate=60.0,
    )
    bridge.update_telemetry(
        robot_position=(750.0, 600.0),
        opponent_position=(100.0, 100.0),
    )
    model = DummySACModel()

    LiveSACController(bridge, model=model).predict_action()

    np.testing.assert_allclose(
        model.last_observation[[0, 1, 4, 5, 6, 7]],
        [0.5, -0.5, 0.75, -0.8, -0.5, 0.5],
        atol=1e-6,
    )


def test_training_and_live_bridge_share_exact_physical_observation_and_action(monkeypatch):
    bridge = SimBridge(invert_y=True)
    bridge.set_table_corners(
        top_left=(0.0, 0.0),
        top_right=(1020.0, 0.0),
        bottom_right=(1020.0, 1940.0),
        bottom_left=(0.0, 1940.0),
    )
    bridge.configure_physical_table_bounds()
    now = [5.0]
    monkeypatch.setattr("core.sim_bridge.time.monotonic", lambda: now[0])

    bridge.set_robot_position_mm(93.65, 533.7)
    now[0] += 0.1
    bridge.set_robot_position_mm(103.65, 553.7)
    bridge.update(
        puck_x=610.0,
        puck_y=1270.0,
        puck_vx=1.5,
        puck_vy=2.0,
        frame_rate=20.0,
        opponent_x=700.0,
        opponent_y=300.0,
    )

    physical_state = bridge.get_training_physical_state()
    assert physical_state is not None
    puck_position, puck_velocity, opponent_position, robot_position, robot_velocity = (
        physical_state
    )
    np.testing.assert_allclose(puck_position, (-0.3, 0.1), atol=1e-6)
    np.testing.assert_allclose(puck_velocity, (-0.04, 0.03), atol=1e-6)
    np.testing.assert_allclose(opponent_position, (0.67, 0.19), atol=1e-6)
    np.testing.assert_allclose(robot_position, (-0.5, 0.15), atol=1e-6)
    np.testing.assert_allclose(robot_velocity, (-0.2, 0.1), atol=1e-6)

    env = AirHockeyGymEnv(opponent_type="none")
    env._puck_position[:] = puck_position
    env._puck_velocity[:] = puck_velocity
    env._opponent_position[:] = opponent_position
    env._robot_position[:] = robot_position
    env._robot_velocity[:] = robot_velocity
    expected_observation = env._make_observation()

    model = DummySACModel()
    model.observation_space = type("Space", (), {"shape": (12,)})()
    controller = LiveSACController(bridge, model=model)
    live_observation = controller._model_observation(
        bridge.get_observation_vector()
    )
    np.testing.assert_array_equal(live_observation, expected_observation)

    action = np.asarray([0.6, -0.4], dtype=np.float32)
    training_velocity = action_to_velocity_mps(action)
    plc_vx_mm_s, plc_vy_mm_s = SACAgentWorker._to_plc_velocity(*action)
    live_physical_velocity = np.asarray(
        [-plc_vy_mm_s, plc_vx_mm_s], dtype=np.float64
    ) / 1000.0
    np.testing.assert_allclose(live_physical_velocity, training_velocity, atol=1e-7)
    env.close()


def test_controller_rejects_wrong_action_dimension():
    bridge = SimBridge()
    controller = LiveSACController(bridge, model=DummySACModel((0.1, 0.2, 0.3)))

    try:
        controller.predict_action()
    except ValueError as exc:
        assert "2D action" in str(exc)
    else:
        raise AssertionError("Expected non-2D policy output to fail validation")


def test_controller_rejects_unknown_observation_dimension(tmp_path):
    class UnknownDimensionModel:
        observation_space = type("Space", (), {"shape": (10,)})()
        action_space = type("Space", (), {"shape": (2,)})()

    class FakeSAC:
        @staticmethod
        def load(*args, **kwargs):
            return UnknownDimensionModel()

    model_path = tmp_path / "unknown_dimension.zip"
    model_path.touch()
    controller = LiveSACController(SimBridge())

    with (
        patch("ai.sac_controller._SAC_CLASS", FakeSAC),
        patch("ai.sac_controller.check_sac_runtime"),
    ):
        try:
            controller.load_model(model_path)
        except ValueError as exc:
            assert "supports only (8,) and (12,)" in str(exc)
        else:
            raise AssertionError("Expected unknown observation dimension to be rejected")

    assert controller.model is None


def test_generated_dummy_model_loads_and_predicts_valid_action(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "dummy_sac_model.zip"
    trained_model_path = (
        project_root / "models" / "Sessie06" / "trained_sac_model_stage2.zip"
    )
    child_script = f"""
import faulthandler
faulthandler.enable()
import PyQt5
import cv2
import vmbpy
from PyQt5.QtWidgets import QApplication, QFileDialog

app = QApplication([])
dialog = QFileDialog()
dialog.setOption(QFileDialog.DontUseNativeDialog)
assert dialog.testOption(QFileDialog.DontUseNativeDialog)

import numpy as np
from ai.generate_dummy_model import generate_dummy_model
from ai.agent_worker import SACAgentWorker
from ai.sac_controller import LiveSACController
from core.sim_bridge import SimBridge

model_path = generate_dummy_model({str(model_path)!r})
bridge = SimBridge()
bridge.update(0.25, -0.2, 0.1, -0.1, frame_rate=60.0,
              opponent_x=0.4, opponent_y=0.3,
              robot_x=-0.4, robot_y=-0.3)
controller = LiveSACController(bridge)
controller.load_model(model_path)
action = controller.predict_action()
assert model_path.is_file()
assert controller.model.device.type == "cpu"
assert len(action) == 2 and np.isfinite(action).all()
print(f"valid two-dimensional action: {{action}}")

trained_bridge = SimBridge()
trained_bridge.set_table_corners(
    top_left=(0.0, 0.0),
    top_right=(1000.0, 0.0),
    bottom_right=(1000.0, 800.0),
    bottom_left=(0.0, 800.0),
)
trained_bridge.configure_physical_table_bounds()
trained_bridge.update(500.0, 524.0, 0.0, 15.0, frame_rate=30.0)
trained_bridge.set_robot_position_mm(0.0, 250.0)
live_observation = trained_bridge.get_observation_vector()
assert live_observation.shape == (8,)
trained_controller = LiveSACController(trained_bridge)
trained_controller.load_model({str(trained_model_path)!r})
assert trained_controller.observation_adapter_status.startswith(
    "12D SAC model; shared physical-state 8->12 adapter active"
)
captured = {{}}
model_predict = trained_controller.model.predict
def capture_model_input(observation, deterministic=True):
    captured["observation"] = np.asarray(observation).copy()
    return model_predict(observation, deterministic=deterministic)
trained_controller.model.predict = capture_model_input
trained_action = trained_controller.predict_action()
model_observation = captured["observation"]
assert model_observation.shape == (12,)
np.testing.assert_allclose(
    model_observation,
    [
        -0.31, 0.0, -0.21825, 0.0,
        0.72, 0.0, -0.20237114, 0.09088235,
        -0.05381444, -0.04544118, -0.181875, 0.0,
    ],
    atol=1e-5,
)
assert np.isfinite(trained_action).all()
assert trained_action[0] < 0.0
plc_velocity = SACAgentWorker._to_plc_velocity(*trained_action)
assert plc_velocity[1] > 0.0
print(
    "12D model inference OK; incoming puck and mallet both move toward "
    f"our goal: action={{trained_action}}, PLC velocity={{plc_velocity}}"
)
"""
    child_env = os.environ.copy()
    child_env["PYTHONFAULTHANDLER"] = "1"
    child_env["QT_QPA_PLATFORM"] = "offscreen"
    child_env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=project_root,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "valid two-dimensional action" in completed.stdout
    assert "12D model inference OK" in completed.stdout
