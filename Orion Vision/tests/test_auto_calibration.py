"""Tests for ordered manual table-corner collection."""

import pytest

from core.auto_calibration import ManualTableCalibrator
from core.sim_bridge import SimBridge


CORNERS = (
    (22.0, 18.0),
    (282.0, 33.0),
    (265.0, 220.0),
    (40.0, 205.0),
)


def test_manual_calibrator_collects_and_commits_corners_in_order():
    calibrator = ManualTableCalibrator()
    bridge = SimBridge()

    for point, label in zip(CORNERS, calibrator.CORNER_LABELS):
        assert calibrator.next_corner_label == label
        assert calibrator.add_corner(point) == label

    assert calibrator.is_complete
    calibrator.apply(bridge)

    assert bridge.get_table_corners() == CORNERS
    assert calibrator.next_corner_label is None


def test_manual_calibrator_requires_four_points_before_commit():
    calibrator = ManualTableCalibrator()
    calibrator.add_corner(CORNERS[0])

    with pytest.raises(RuntimeError, match="four table corners"):
        calibrator.apply(SimBridge())


def test_manual_calibrator_reset_discards_partial_points():
    calibrator = ManualTableCalibrator()
    calibrator.add_corner(CORNERS[0])
    calibrator.reset()

    assert calibrator.corners == ()
    assert calibrator.next_corner_label == "TL"
