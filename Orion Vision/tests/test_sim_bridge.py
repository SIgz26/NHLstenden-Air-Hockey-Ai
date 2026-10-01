import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.sim_bridge import SimBridge


def test_update_normalizes_positions_and_velocity_per_simulation_step():
    bridge = SimBridge(
        x_bounds=(0.0, 100.0),
        y_bounds=(0.0, 100.0),
        sim_dt=0.1,
        invert_y=True,
    )

    vector = bridge.update(
        puck_x=25.0,
        puck_y=75.0,
        puck_vx=5.0,
        puck_vy=2.0,
        frame_rate=10.0,
        opponent_x=75.0,
        opponent_y=25.0,
        robot_x=50.0,
        robot_y=50.0,
    )

    np.testing.assert_allclose(
        vector,
        [-0.5, -0.5, 0.1, -0.04, 0.5, 0.5, 0.0, 0.0],
        atol=1e-6,
    )
    assert vector.dtype == np.float32
    np.testing.assert_array_equal(bridge.get_observation_vector(), vector)


def test_update_from_engine_uses_frame_size_and_replaces_stale_queue_item():
    bridge = SimBridge(sim_dt=1.0 / 30.0)
    engine_result = {"x": 49.5, "y": 24.5, "vx": 1.0, "vy": 0.0}

    vector = bridge.update_from_engine(
        engine_result,
        frame_width=100,
        frame_height=50,
        frame_rate=30.0,
    )

    assert vector is not None
    np.testing.assert_allclose(vector[:4], [0.0, 0.0, 2.0 / 99.0, 0.0], atol=1e-6)
    queued = bridge.get_latest_queued_observation()
    assert queued is not None
    np.testing.assert_array_equal(queued, vector)

    bridge.update(10, 10, 0, 0, frame_rate=30.0)
    latest = bridge.update(90, 40, 0, 0, frame_rate=30.0)
    queued_latest = bridge.get_latest_queued_observation()
    assert queued_latest is not None
    np.testing.assert_array_equal(queued_latest, latest)
    assert bridge.get_latest_queued_observation() is None


def test_update_from_engine_without_telemetry_returns_none():
    bridge = SimBridge()

    assert bridge.update_from_engine(None, 356, 288, frame_rate=30.0) is None


def test_invalid_simulation_rate_is_rejected():
    bridge = SimBridge()

    try:
        bridge.update(0, 0, 0, 0, frame_rate=0.0)
    except ValueError as exc:
        assert "frame_rate" in str(exc)
    else:
        raise AssertionError("Expected a non-positive frame rate to be rejected")


def test_calibrated_table_corners_map_to_simulation_corners():
    bridge = SimBridge(sim_dt=1.0 / 60.0, invert_y=True)
    bridge.set_table_bounds(x_min=40.0, y_min=30.0, x_max=300.0, y_max=250.0)

    top_left = bridge.update(40.0, 30.0, 0.0, 0.0, frame_rate=60.0)
    bottom_right = bridge.update(300.0, 250.0, 0.0, 0.0, frame_rate=60.0)
    bottom_left = bridge.update(40.0, 250.0, 0.0, 0.0, frame_rate=60.0)
    top_right = bridge.update(300.0, 30.0, 0.0, 0.0, frame_rate=60.0)

    np.testing.assert_array_equal(top_left[:2], [-1.0, 1.0])
    np.testing.assert_array_equal(bottom_right[:2], [1.0, -1.0])
    np.testing.assert_array_equal(bottom_left[:2], [-1.0, -1.0])
    np.testing.assert_array_equal(top_right[:2], [1.0, 1.0])

    assert bridge.table_x_min == 40.0
    assert bridge.table_y_min == 30.0
    assert bridge.table_x_max == 300.0
    assert bridge.table_y_max == 250.0


def test_calibration_is_not_reset_to_full_frame_on_engine_updates():
    bridge = SimBridge()
    bridge.set_table_bounds(x_min=20.0, y_min=10.0, x_max=80.0, y_max=40.0)

    result = bridge.update_from_engine(
        {"x": 20.0, "y": 10.0, "vx": 0.0, "vy": 0.0},
        frame_width=100,
        frame_height=50,
        frame_rate=30.0,
    )

    assert result is not None
    np.testing.assert_array_equal(bridge.get_observation_vector()[:2], [-1.0, 1.0])


def test_four_point_perspective_maps_camera_corners_to_simulation_edges():
    bridge = SimBridge(sim_dt=1.0 / 60.0, invert_y=True)
    bridge.set_table_corners(
        top_left=(30.0, 20.0),
        top_right=(280.0, 45.0),
        bottom_right=(250.0, 240.0),
        bottom_left=(50.0, 220.0),
    )

    corners = (
        ((30.0, 20.0), [-1.0, 1.0]),
        ((280.0, 45.0), [1.0, 1.0]),
        ((250.0, 240.0), [1.0, -1.0]),
        ((50.0, 220.0), [-1.0, -1.0]),
    )
    for (x, y), expected in corners:
        vector = bridge.update(x, y, 0.0, 0.0, frame_rate=60.0)
        np.testing.assert_allclose(vector[:2], expected, atol=1e-6)


def test_velocity_along_tilted_table_edge_is_rectified():
    bridge = SimBridge(sim_dt=1.0 / 60.0, invert_y=False)
    top_left = np.asarray([30.0, 20.0])
    top_right = np.asarray([280.0, 45.0])
    bridge.set_table_corners(
        tuple(top_left),
        tuple(top_right),
        (250.0, 240.0),
        (50.0, 220.0),
    )

    point = top_left + (top_right - top_left) * 0.5
    frame_velocity = (top_right - top_left) * 0.001
    vector = bridge.update(
        float(point[0]),
        float(point[1]),
        float(frame_velocity[0]),
        float(frame_velocity[1]),
        frame_rate=60.0,
    )

    assert vector[2] > 0.0
    assert abs(float(vector[3])) < 1e-5


def test_puck_trajectory_reflects_from_top_and_bottom_wall():
    bridge = SimBridge(
        x_bounds=(0.0, 2.0),
        y_bounds=(0.0, 2.0),
        sim_dt=1.0,
        invert_y=False,
    )
    bridge.update(1.0, 1.0, 0.0, 0.8, frame_rate=1.0)

    trajectory = bridge.predict_puck_trajectory(max_bounces=3, time_horizon=2.0)

    np.testing.assert_allclose(
        trajectory,
        [(0.0, 0.0), (0.0, 1.0), (0.0, 0.4)],
        atol=1e-6,
    )


def test_puck_trajectory_stops_at_goal_line():
    bridge = SimBridge(
        x_bounds=(0.0, 2.0),
        y_bounds=(0.0, 2.0),
        sim_dt=1.0,
        invert_y=False,
    )
    bridge.update(1.0, 1.0, 0.5, 0.0, frame_rate=1.0)

    trajectory = bridge.predict_puck_trajectory(time_horizon=5.0)

    np.testing.assert_allclose(trajectory, [(0.0, 0.0), (1.0, 0.0)], atol=1e-6)


def test_puck_trajectory_respects_max_bounces():
    bridge = SimBridge(
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 2.0),
        sim_dt=1.0,
        invert_y=False,
    )
    bridge.update(5.0, 1.0, 0.0, 0.8, frame_rate=1.0)

    trajectory = bridge.predict_puck_trajectory(max_bounces=1, time_horizon=10.0)

    assert len(trajectory) == 2
    np.testing.assert_allclose(trajectory[-1], (0.0, 1.0), atol=1e-6)


def test_opponent_velocity_is_estimated_in_normalized_table_units(monkeypatch):
    bridge = SimBridge(x_bounds=(0.0, 2.0), y_bounds=(0.0, 2.0), invert_y=False)
    current_time = [5.0]
    monkeypatch.setattr("core.sim_bridge.time.monotonic", lambda: current_time[0])

    bridge.update_telemetry(opponent_position=(0.5, 0.5))
    current_time[0] += 0.1
    bridge.update_telemetry(opponent_position=(0.7, 0.6))

    np.testing.assert_allclose(bridge.get_opponent_velocity(), (2.0, 1.0), atol=1e-6)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
