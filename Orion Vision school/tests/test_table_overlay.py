import cv2
import numpy as np

from core.sim_bridge import SimBridge
from core.table_overlay import draw_table_overlay


def test_overlay_draws_calibrated_corner_dots_and_outline():
    bridge = SimBridge()
    corners = ((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0))
    bridge.set_table_corners(*corners)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    rendered = draw_table_overlay(frame, bridge.get_table_corners())

    assert rendered is frame
    np.testing.assert_array_equal(frame[20, 20], [0, 255, 255])
    np.testing.assert_array_equal(frame[20, 50], [255, 255, 0])
    assert bridge.get_table_corners() == corners


def test_overlay_can_be_disabled_or_skips_uncalibrated_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()

    draw_table_overlay(frame, ((20, 20), (80, 20), (80, 80), (20, 80)), enabled=False)
    np.testing.assert_array_equal(frame, original)

    draw_table_overlay(frame, None)
    np.testing.assert_array_equal(frame, original)


def test_manual_corners_are_used_by_perspective_transform():
    bridge = SimBridge()
    corners = ((25.0, 20.0), (275.0, 38.0), (260.0, 220.0), (42.0, 205.0))
    bridge.set_table_corners(*corners)

    expected_positions = (
        (corners[0], [-1.0, 1.0]),
        (corners[1], [1.0, 1.0]),
        (corners[2], [1.0, -1.0]),
        (corners[3], [-1.0, -1.0]),
    )
    for (x, y), expected in expected_positions:
        observation = bridge.update(x, y, 0.0, 0.0, frame_rate=60.0)
        np.testing.assert_allclose(observation[:2], expected, atol=1e-5)


def test_reset_table_corners_clears_transform_and_restores_full_roi():
    bridge = SimBridge()
    bridge.set_table_corners((10, 10), (90, 15), (85, 90), (15, 85))

    bridge.reset_table_corners(frame_width=100, frame_height=100)

    assert bridge.get_table_corners() is None
    assert bridge._perspective_matrix is None
    observation = bridge.update(0.0, 0.0, 0.0, 0.0, frame_rate=60.0)
    np.testing.assert_allclose(observation[:2], [-1.0, 1.0], atol=1e-6)


def test_reset_without_frame_size_relearns_full_roi_on_next_engine_frame():
    bridge = SimBridge()
    bridge.set_table_corners((10, 10), (90, 15), (85, 90), (15, 85))

    bridge.reset_table_corners()
    result = bridge.update_from_engine(
        {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        frame_width=100,
        frame_height=80,
        frame_rate=30.0,
    )

    assert result is not None
    np.testing.assert_allclose(result[:2], [-1.0, 1.0], atol=1e-6)


def test_manual_point_order_rejects_crossed_quadrilateral():
    bridge = SimBridge()
    crossed = ((10, 10), (90, 10), (40, 40), (10, 90))

    try:
        bridge.set_table_corners(*crossed)
    except ValueError as exc:
        assert "convex" in str(exc)
    else:
        raise AssertionError("Expected crossed corners to be rejected")
