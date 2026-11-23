"""Tests for ordered manual table-corner collection."""

import cv2
import numpy as np
import pytest

from core.auto_calibration import (
    ManualTableCalibrator,
    auto_detect_table_corners,
)
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


@pytest.mark.parametrize("landscape", [False, True])
@pytest.mark.parametrize("is_rgb", [False, True])
def test_auto_calibration_finds_rotated_table_with_red_puck_over_circle(
    landscape: bool, is_rgb: bool
):
    physical_corners = np.asarray(
        [[-510.0, -970.0], [510.0, -970.0], [510.0, 970.0], [-510.0, 970.0]],
        dtype=np.float32,
    )
    if landscape:
        frame_corners = np.asarray(
            [[80.0, 750.0], [65.0, 90.0], [930.0, 55.0], [950.0, 720.0]],
            dtype=np.float32,
        )
        frame = np.full((800, 1000, 3), 35, dtype=np.uint8)
    else:
        frame_corners = np.asarray(
            [[155.0, 45.0], [650.0, 70.0], [620.0, 950.0], [110.0, 925.0]],
            dtype=np.float32,
        )
        frame = np.full((1000, 800, 3), 35, dtype=np.uint8)
    physical_to_frame = cv2.getPerspectiveTransform(
        physical_corners, frame_corners
    )
    cv2.fillConvexPoly(
        frame, np.rint(frame_corners).astype(np.int32), (220, 220, 220)
    )

    physical_circle_centers = np.asarray(
        [
            [-270.0, -560.0],
            [270.0, -560.0],
            [270.0, 560.0],
            [-270.0, 560.0],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    circle_centers = cv2.perspectiveTransform(
        physical_circle_centers, physical_to_frame
    )[0]
    for center in circle_centers:
        cv2.circle(
            frame,
            tuple(np.rint(center).astype(int)),
            24,
            (15, 15, 15),
            thickness=6,
            lineType=cv2.LINE_AA,
        )
    puck_center = tuple(np.rint(circle_centers[0] + [12.0, -4.0]).astype(int))
    cv2.circle(frame, puck_center, 15, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    detector_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if is_rgb else frame
    detected = auto_detect_table_corners(detector_frame, is_rgb=is_rgb)

    assert detected is not None
    np.testing.assert_allclose(
        detected, frame_corners[[3, 2, 1, 0]], atol=2.0
    )
