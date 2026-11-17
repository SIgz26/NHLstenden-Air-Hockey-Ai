import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ai.opponents.base_opponent import BaseOpponent
from ai.scripted_worker import OPPONENTS, ScriptedOpponentWorker
from core.sim_bridge import SimBridge


class _RecordingOpponent(BaseOpponent):
    received: tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float] | None = None

    def get_action(
        self,
        puck_pos: np.ndarray,
        puck_vel: np.ndarray,
        opponent_pos: np.ndarray,
        field_length: float,
        field_width: float,
        mallet_radius: float,
    ) -> np.ndarray:
        type(self).received = (
            puck_pos.copy(),
            puck_vel.copy(),
            opponent_pos.copy(),
            field_length,
            field_width,
            mallet_radius,
        )
        return np.asarray([0.2, 0.3], dtype=np.float64)


def _calibrated_bridge(*, detected: bool = True) -> SimBridge:
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
            "metadata": {"in_range": detected},
        },
        frame_width=1000,
        frame_height=800,
        frame_rate=20.0,
    )
    bridge.update_telemetry(robot_position=(500.0, 400.0))
    bridge.set_robot_position_mm(-46.35, 421.6)
    return bridge


def test_scripted_worker_converts_physical_axes_and_limits_speed(monkeypatch) -> None:
    monkeypatch.setitem(OPPONENTS, "Recording", _RecordingOpponent)
    _RecordingOpponent.received = None
    worker = ScriptedOpponentWorker(
        _calibrated_bridge(),
        "Recording",
    )

    velocity = worker._calculate_action()

    assert velocity == (300.0, 200.0)
    received = _RecordingOpponent.received
    assert received is not None
    puck_pos, puck_vel, mallet_pos, field_length, field_width, mallet_radius = received
    np.testing.assert_allclose(puck_pos, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(puck_vel, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(mallet_pos, [0.485, -0.04635], atol=1e-6)
    assert field_length == 1.94
    assert field_width == 1.02
    assert mallet_radius == 0.048


def test_scripted_worker_maps_our_goal_to_positive_half_length(monkeypatch) -> None:
    monkeypatch.setitem(OPPONENTS, "Recording", _RecordingOpponent)
    _RecordingOpponent.received = None
    bridge = _calibrated_bridge()
    bridge.set_robot_position_mm(0.0, 789.5)
    worker = ScriptedOpponentWorker(bridge, "Recording")

    worker._calculate_action()

    received = _RecordingOpponent.received
    assert received is not None
    np.testing.assert_allclose(received[2], [0.97, 0.0], atol=1e-6)


def test_scripted_worker_maps_forward_velocity_to_negative_plc_y(monkeypatch) -> None:
    class ForwardOpponent(_RecordingOpponent):
        def get_action(self, **kwargs) -> np.ndarray:
            return np.asarray([-0.2, 0.3], dtype=np.float64)

    monkeypatch.setitem(OPPONENTS, "Forward", ForwardOpponent)
    worker = ScriptedOpponentWorker(_calibrated_bridge(), "Forward")

    assert worker._calculate_action() == (300.0, -200.0)


def test_scripted_worker_clamps_converted_velocity_vector_to_500_mm_s(monkeypatch) -> None:
    class FastOpponent(_RecordingOpponent):
        def get_action(self, **kwargs) -> np.ndarray:
            return np.asarray([-3.5, 2.5], dtype=np.float64)

    monkeypatch.setitem(OPPONENTS, "Fast", FastOpponent)
    worker = ScriptedOpponentWorker(_calibrated_bridge(), "Fast")

    vx_mm_s, vy_mm_s = worker._calculate_action()

    assert vx_mm_s > 0.0
    assert vy_mm_s < 0.0
    assert np.isclose(np.hypot(vx_mm_s, vy_mm_s), 500.0)


def test_opponent_physical_state_uses_full_table_and_plc_robot_coordinates() -> None:
    bridge = _calibrated_bridge()
    observation_before = bridge.get_observation_vector()

    physical_state = bridge.get_opponent_physical_state()

    assert physical_state is not None
    puck_pos, puck_vel, robot_pos = physical_state
    np.testing.assert_array_equal(bridge.get_observation_vector(), observation_before)
    np.testing.assert_allclose(puck_pos, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(puck_vel, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(robot_pos, [0.485, -0.04635], atol=1e-6)


def test_opponent_physical_state_maps_full_table_and_forward_velocity() -> None:
    bridge = _calibrated_bridge()
    bridge.update_from_engine(
        {
            "x": 500.0,
            "y": 400.0,
            "vx": 0.0,
            "vy": -1.0,
            "metadata": {"in_range": True},
        },
        frame_width=1000,
        frame_height=800,
        frame_rate=20.0,
    )

    physical_state = bridge.get_opponent_physical_state()

    assert physical_state is not None
    puck_pos, puck_vel, _ = physical_state
    assert puck_vel[0] < 0.0
    top_left = bridge.update_from_engine(
        {
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "metadata": {"in_range": True},
        },
        frame_width=1000,
        frame_height=800,
        frame_rate=20.0,
    )
    assert top_left is not None
    top_left_state = bridge.get_opponent_physical_state()
    assert top_left_state is not None
    np.testing.assert_allclose(top_left_state[0], [-0.97, -0.51], atol=1e-5)
    bottom_right = bridge.update_from_engine(
        {
            "x": 1000.0,
            "y": 800.0,
            "vx": 0.0,
            "vy": 0.0,
            "metadata": {"in_range": True},
        },
        frame_width=1000,
        frame_height=800,
        frame_rate=20.0,
    )
    assert bottom_right is not None
    bottom_right_state = bridge.get_opponent_physical_state()
    assert bottom_right_state is not None
    np.testing.assert_allclose(bottom_right_state[0], [0.97, 0.51], atol=1e-5)


def test_scripted_worker_outputs_zero_when_puck_is_not_detected() -> None:
    worker = ScriptedOpponentWorker(_calibrated_bridge(detected=False), "Defender")

    assert worker._calculate_action() == (0.0, 0.0)


def test_scripted_worker_outputs_zero_for_stale_puck_detection(monkeypatch) -> None:
    bridge = _calibrated_bridge()
    worker = ScriptedOpponentWorker(bridge, "Defender")
    assert bridge._last_puck_update is not None
    stale_time = bridge._last_puck_update + 1.0
    monkeypatch.setattr("core.sim_bridge.time.monotonic", lambda: stale_time)

    assert worker._calculate_action() == (0.0, 0.0)


def test_physical_calibration_maps_width_and_depth_to_plc_mm() -> None:
    bridge = _calibrated_bridge()
    bridge.set_robot_position_mm(-46.35, 421.6)
    observation = bridge.get_observation_vector()

    assert bridge.physical_table_calibrated
    assert bridge.table_x_min == -463.2
    assert bridge.table_x_max == 370.5
    assert bridge.table_y_min == 53.7
    assert bridge.table_y_max == 789.5
    np.testing.assert_allclose(observation[6:8], [0.0, 0.0], atol=1e-6)

    top_left = bridge.update_from_engine(
        {
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "metadata": {"in_range": True},
        },
        frame_width=1000,
        frame_height=800,
        frame_rate=20.0,
    )
    assert top_left is not None
    np.testing.assert_allclose(top_left[:2], [-1.0, 1.0], atol=1e-6)
