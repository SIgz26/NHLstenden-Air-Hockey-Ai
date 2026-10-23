import numpy as np
import pytest

from core.fps_overlay import FpsOverlay


def test_fps_overlay_measures_camera_frames_before_processing():
    overlay = FpsOverlay(smoothing=0.5)
    frame = np.zeros((120, 320, 3), dtype=np.uint8)

    overlay.record_frame(now=1.0)
    assert overlay.fps is None

    overlay.record_frame(now=1.1)
    assert overlay.fps == pytest.approx(10.0)

    third_result = overlay.draw(frame.copy())
    assert overlay.fps == pytest.approx(10.0)
    assert third_result.any()

    overlay.record_frame(now=1.3)
    assert overlay.fps == pytest.approx(7.5)


def test_fps_overlay_uses_small_single_line_annotation():
    overlay = FpsOverlay()
    overlay.record_frame(now=1.0)
    frame = np.zeros((120, 320, 3), dtype=np.uint8)

    result = overlay.draw(frame)
    changed_y, changed_x = np.where(np.any(result != 0, axis=2))

    assert changed_x.min() > 200
    assert changed_y.max() < 30


def test_fps_overlay_rejects_invalid_smoothing():
    with pytest.raises(ValueError, match="smoothing"):
        FpsOverlay(smoothing=0.0)
