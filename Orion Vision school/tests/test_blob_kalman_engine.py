import cv2
import numpy as np
import pytest

from core.engines.blob_kalman_engine import BlobKalmanEngine


def test_blob_kalman_engine_detects_red_puck_and_returns_app_frames():
    engine = BlobKalmanEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 7, (0, 0, 255), -1)

    result, mask = engine.process_frame(frame)

    assert result.shape == frame.shape
    assert mask.shape == frame.shape
    assert engine.last_result["metadata"]["best"] == (320, 240)
    assert engine.last_result["metadata"]["in_range"] is True
    assert engine.last_result["metadata"]["status"] == "TRACKING"
    assert engine.last_result["x"] == 320.0
    assert engine.last_result["y"] == 240.0
    assert engine.last_result["mask"].ndim == 2


def test_blob_kalman_engine_predicts_brief_occlusion_and_bounces_at_bounds():
    engine = BlobKalmanEngine()
    engine.kf.statePost = np.array([[588], [240], [5], [0]], dtype=np.float32)
    path = engine.predict_path(
        (588, 240),
        (5, 0),
        steps=2,
        bounds={"left": 50, "right": 590, "top": 50, "bottom": 430},
    )
    assert path == [(590, 240), (585, 240)]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.initialized = True
    result, mask = engine.process_frame(frame)

    assert result.shape == frame.shape
    assert mask.shape == frame.shape
    assert engine.last_result["metadata"]["status"] == "PREDICTING (OCCLUSION)"
    assert engine.last_result["metadata"]["future_path"]


def test_calibrated_table_corners_define_prediction_bounce_edges():
    engine = BlobKalmanEngine()
    corners = (
        (100.0, 100.0),
        (500.0, 120.0),
        (480.0, 380.0),
        (120.0, 360.0),
    )
    engine.set_table_corners(corners)

    table_start = np.array([[[390.0, 130.0]]], dtype=np.float32)
    table_next = np.array([[[405.0, 130.0]]], dtype=np.float32)
    image_start = cv2.perspectiveTransform(
        table_start, engine._table_to_image
    )[0, 0]
    image_next = cv2.perspectiveTransform(
        table_next, engine._table_to_image
    )[0, 0]

    path, intercept = engine._predict_calibrated_path(
        (float(image_start[0]), float(image_start[1])),
        (
            float(image_next[0] - image_start[0]),
            float(image_next[1] - image_start[1]),
        ),
        steps=3,
    )
    predicted_table_points = cv2.perspectiveTransform(
        np.asarray(path, dtype=np.float32).reshape((-1, 1, 2)),
        engine._image_to_table,
    ).reshape((-1, 2))

    assert predicted_table_points[0, 0] == pytest.approx(
        engine._calibrated_table_size[0] - 1,
        abs=1.0,
    )
    assert predicted_table_points[1, 0] < predicted_table_points[0, 0]
    assert intercept is not None
    assert engine._table_corners == corners
