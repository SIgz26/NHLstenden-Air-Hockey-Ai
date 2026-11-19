"""Tests for live SAC action inference and its 100 Hz worker loop."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ai.agent_worker import SACAgentWorker
from ai.sac_controller import LiveSACController
from core.sim_bridge import SimBridge


class DummySACModel:
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
    np.testing.assert_allclose(sent_actions[-1], (250.0, -250.0), atol=1e-7)


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


def test_controller_rejects_wrong_action_dimension():
    bridge = SimBridge()
    controller = LiveSACController(bridge, model=DummySACModel((0.1, 0.2, 0.3)))

    try:
        controller.predict_action()
    except ValueError as exc:
        assert "2D action" in str(exc)
    else:
        raise AssertionError("Expected non-2D policy output to fail validation")


def test_generated_dummy_model_loads_and_predicts_valid_action(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "dummy_sac_model.zip"
    child_script = f"""
import numpy as np
from ai.generate_dummy_model import generate_dummy_model
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
assert len(action) == 2 and np.isfinite(action).all()
print(f"valid two-dimensional action: {{action}}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "valid two-dimensional action" in completed.stdout
