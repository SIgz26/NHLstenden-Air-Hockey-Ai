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
    return bridge


def test_scripted_worker_converts_physical_axes_and_limits_speed(monkeypatch) -> None:
    monkeypatch.setitem(OPPONENTS, "Recording", _RecordingOpponent)
    _RecordingOpponent.received = None
    worker = ScriptedOpponentWorker(
        _calibrated_bridge(),
        "Recording",
    )

    velocity = worker._calculate_action()

    assert velocity == (300.0, -200.0)
    received = _RecordingOpponent.received
    assert received is not None
    puck_pos, puck_vel, mallet_pos, field_length, field_width, mallet_radius = received
    np.testing.assert_allclose(puck_pos, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(puck_vel, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(mallet_pos, [0.0, 0.0], atol=1e-6)
    assert field_length == 0.7358
    assert field_width == 0.8337
    assert mallet_radius == 0.048


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
