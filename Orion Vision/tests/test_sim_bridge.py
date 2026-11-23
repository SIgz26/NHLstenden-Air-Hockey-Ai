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


def test_puck_trajectory_reflects_from_left_and_right_wall():
    bridge = SimBridge(
        x_bounds=(0.0, 2.0),
        y_bounds=(0.0, 2.0),
        sim_dt=1.0,
        invert_y=False,
    )
    bridge.update(1.0, 1.0, 0.8, 0.0, frame_rate=1.0)

    trajectory = bridge.predict_puck_trajectory(max_bounces=3, time_horizon=2.0)

    np.testing.assert_allclose(
        trajectory,
        [(0.0, 0.0), (1.0, 0.0), (0.4, 0.0)],
        atol=1e-6,
    )


def test_puck_trajectory_stops_at_top_or_bottom_goal_line():
    bridge = SimBridge(
        x_bounds=(0.0, 2.0),
        y_bounds=(0.0, 2.0),
        sim_dt=1.0,
        invert_y=False,
    )
    bridge.update(1.0, 1.0, 0.0, 0.5, frame_rate=1.0)

    trajectory = bridge.predict_puck_trajectory(time_horizon=5.0)

    np.testing.assert_allclose(trajectory, [(0.0, 0.0), (0.0, 1.0)], atol=1e-6)


def test_puck_trajectory_respects_max_bounces():
    bridge = SimBridge(
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 2.0),
        sim_dt=1.0,
        invert_y=False,
    )
    bridge.update(5.0, 1.0, 0.8, 0.0, frame_rate=1.0)

    trajectory = bridge.predict_puck_trajectory(max_bounces=1, time_horizon=10.0)

    assert len(trajectory) == 2
    np.testing.assert_allclose(trajectory[-1], (1.0, 0.0), atol=1e-6)


def test_opponent_velocity_is_estimated_in_normalized_table_units(monkeypatch):
    bridge = SimBridge(x_bounds=(0.0, 2.0), y_bounds=(0.0, 2.0), invert_y=False)
    current_time = [5.0]
    monkeypatch.setattr("core.sim_bridge.time.monotonic", lambda: current_time[0])

    bridge.update_telemetry(opponent_position=(0.5, 0.5))
    current_time[0] += 0.1
    bridge.update_telemetry(opponent_position=(0.7, 0.6))

    np.testing.assert_allclose(bridge.get_opponent_velocity(), (2.0, 1.0), atol=1e-6)


def test_plc_robot_velocity_reports_signed_axis_motion(monkeypatch):
    bridge = SimBridge()
    current_time = [5.0]
    monkeypatch.setattr("core.sim_bridge.time.monotonic", lambda: current_time[0])

    bridge.set_robot_position_mm(10.0, 20.0)
    current_time[0] += 0.1
    bridge.set_robot_position_mm(30.0, 60.0)

    np.testing.assert_allclose(bridge.plc_robot_velocity_mm_s, (200.0, 400.0))


def _calibrated_bias_bridge(monkeypatch):
    bridge = SimBridge(invert_y=False)
    bridge.set_table_corners(
        top_left=(0.0, 0.0),
        top_right=(1020.0, 0.0),
        bottom_right=(1020.0, 1940.0),
        bottom_left=(0.0, 1940.0),
    )
    bridge.configure_physical_table_bounds()
    current_time = [1.0]
    monkeypatch.setattr("core.sim_bridge.time.monotonic", lambda: current_time[0])
    bridge.set_robot_position_mm(-46.35, 53.7)
    current_time[0] += 0.1
    bridge.set_robot_position_mm(-46.35, 53.7)
    return bridge, current_time


def test_mallet_bias_updates_at_rest_and_corrects_opponent_puck_state(monkeypatch):
    bridge, _current_time = _calibrated_bias_bridge(monkeypatch)
    bridge.update(530.0, 990.0, 0.0, 0.0, frame_rate=30.0)

    assert bridge.update_mallet_bias(520.0, 980.0)
    np.testing.assert_allclose(bridge.calibration_offset_mm, (0.5, 0.5))

    physical_state = bridge.get_opponent_physical_state()
    assert physical_state is not None
    puck_pos_m, _puck_vel_mps, _robot_pos_m = physical_state
    np.testing.assert_allclose(puck_pos_m, (0.0195, 0.0195), atol=1e-6)


def test_mallet_bias_is_frozen_at_or_above_robot_speed_threshold(monkeypatch):
    bridge, current_time = _calibrated_bias_bridge(monkeypatch)
    current_time[0] += 0.1
    bridge.set_robot_position_mm(-43.25, 53.7)

    assert bridge.plc_robot_speed_mm_s >= 30.0
    assert not bridge.update_mallet_bias(520.0, 980.0)
    np.testing.assert_array_equal(bridge.calibration_offset_mm, (0.0, 0.0))


def test_mallet_bias_is_clamped_to_fifteen_millimeters(monkeypatch):
    bridge, _current_time = _calibrated_bias_bridge(monkeypatch)

    for _ in range(100):
        assert bridge.update_mallet_bias(2000.0, -1000.0)

    np.testing.assert_array_equal(bridge.calibration_offset_mm, (15.0, -15.0))


def test_robot_trim_converts_between_plc_and_table_coordinates(monkeypatch):
    bridge, _current_time = _calibrated_bias_bridge(monkeypatch)
    bridge.set_robot_trim_mm(10.0, -5.0)
    bridge.update(510.0, 970.0, 0.0, 0.0, frame_rate=30.0)

    np.testing.assert_allclose(bridge.robot_position_mm(), (-10.0, 5.0))
    np.testing.assert_allclose(
        bridge.physical_target_to_plc_mm(20.0, 40.0),
        (-16.35, 88.7),
    )
    observation = bridge.get_observation_vector()
    expected_robot_x = 2.0 * (-56.35 - bridge.PLC_X_BOUNDS_MM[0]) / (
        bridge.PLC_X_BOUNDS_MM[1] - bridge.PLC_X_BOUNDS_MM[0]
    ) - 1.0
    expected_robot_y = 2.0 * (58.7 - bridge.PLC_Y_BOUNDS_MM[0]) / (
        bridge.PLC_Y_BOUNDS_MM[1] - bridge.PLC_Y_BOUNDS_MM[0]
    ) - 1.0
    np.testing.assert_allclose(observation[6:8], (expected_robot_x, expected_robot_y))

    physical_state = bridge.get_opponent_physical_state()
    assert physical_state is not None
    robot_pos_m = physical_state[2]
    np.testing.assert_allclose(robot_pos_m, (0.005, -0.01))


def test_puck_trim_shifts_puck_observation_without_changing_velocity():
    bridge = SimBridge(sim_dt=1.0 / 60.0, invert_y=False)
    bridge.set_table_corners(
        top_left=(0.0, 0.0),
        top_right=(100.0, 0.0),
        bottom_right=(100.0, 100.0),
        bottom_left=(0.0, 100.0),
    )
    bridge.configure_physical_table_bounds()
    untrimmed = bridge.update(50.0, 50.0, 0.5, 0.25, frame_rate=10.0)
    bridge.set_puck_trim_mm(10.0, -5.0)
    trimmed = bridge.get_observation_vector()

    np.testing.assert_allclose(trimmed[:2], (10.0 / 510.0, -10.0 / 1940.0), atol=1e-6)
    np.testing.assert_allclose(trimmed[2:4], untrimmed[2:4], atol=1e-6)


def test_robot_and_puck_trims_are_clamped_to_thirty_five_millimeters():
    bridge = SimBridge()
    bridge.set_robot_trim_mm(45.0, -50.0)
    bridge.set_puck_trim_mm(-40.0, 35.0)

    assert (bridge.robot_trim_x_mm, bridge.robot_trim_y_mm) == (35.0, -35.0)
    assert (bridge.puck_trim_x_mm, bridge.puck_trim_y_mm) == (-35.0, 35.0)




if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
