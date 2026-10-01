"""Synthetic marker tests for the ArUco mallet tracker."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core.engines.aruco_tracker import ArUcoMalletTracker
from core.sim_bridge import SimBridge


def make_marker_scene(marker_specs, size=(320, 520), marker_size=100):
    """Create a white synthetic frame and paste generated dictionary markers."""
    canvas = np.full(size, 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, (x, y) in marker_specs:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        canvas[y:y + marker_size, x:x + marker_size] = marker
    return canvas


def test_detect_mallets_returns_centers_by_marker_id():
    frame = make_marker_scene([(0, (60, 80)), (1, (340, 180))])
    tracker = ArUcoMalletTracker()

    robot_position, opponent_position = tracker.detect_mallets(frame)

    assert robot_position is not None
    assert opponent_position is not None
    np.testing.assert_allclose(robot_position, (109.5, 129.5), atol=1.0)
    np.testing.assert_allclose(opponent_position, (389.5, 229.5), atol=1.0)


def test_unknown_ids_are_ignored_and_missing_markers_return_none():
    tracker = ArUcoMalletTracker()
    frame = make_marker_scene([(7, (180, 100))])

    assert tracker.detect_mallets(frame) == (None, None)


def test_marker_positions_are_perspective_transformed_into_bridge_vector():
    bridge = SimBridge()
    bridge.set_table_corners(
        top_left=(50.0, 50.0),
        top_right=(450.0, 70.0),
        bottom_right=(430.0, 270.0),
        bottom_left=(70.0, 250.0),
    )
    bridge.update(250.0, 160.0, 0.0, 0.0, frame_rate=60.0)

    bridge.update_telemetry(
        robot_position=(100.0, 100.0),
        opponent_position=(400.0, 220.0),
    )
    observation = bridge.get_observation_vector()

    assert observation.shape == (8,)
    assert -1.0 <= observation[4] <= 1.0
    assert -1.0 <= observation[5] <= 1.0
    assert -1.0 <= observation[6] <= 1.0
    assert -1.0 <= observation[7] <= 1.0
    assert observation[6] < 0.0
    assert observation[4] > 0.0
