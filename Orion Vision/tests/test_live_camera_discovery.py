from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.live_worker import LiveCameraWorker


def test_live_camera_defaults_to_90_fps_and_caps_requested_rate():
    worker = LiveCameraWorker()

    assert worker.target_fps == 90.0
    worker.set_target_fps(120.0)
    assert worker.target_fps == 90.0


def test_robot_marker_detection_is_throttled_and_requires_slow_plc_pose(monkeypatch):
    worker = LiveCameraWorker()
    detected = []

    class Detector:
        def detectMarkers(self, _gray):
            detected.append(True)
            corners = [np.asarray([[[10, 10], [20, 10], [20, 20], [10, 20]]])]
            return corners, np.asarray([[0]]), ()

    worker._calibration_marker_detector = Detector()
    worker.sim_bridge._plc_robot_speed_mm_s = 0.0
    bias_updates = []
    worker.sim_bridge.update_mallet_bias = (
        lambda x, y, **kwargs: bias_updates.append((x, y, kwargs)) or True
    )
    now = [1.0]
    monkeypatch.setattr("core.live_worker.time.monotonic", lambda: now[0])

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    result = np.zeros_like(frame)
    worker._update_calibration_marker(frame, result)
    worker._update_calibration_marker(frame, result)

    assert len(detected) == 1
    assert bias_updates[0][:2] == (15.0, 15.0)
    assert bias_updates[0][2]["max_speed_mm_s"] == 30.0

    now[0] += 0.11
    worker.sim_bridge._plc_robot_speed_mm_s = 30.0
    worker._update_calibration_marker(frame, result)
    assert len(detected) == 1


def test_linux_camera_discovery_probes_only_existing_device_indices(monkeypatch):
    device_nodes = [Path("/dev/video2"), Path("/dev/video7"), Path("/dev/video7")]
    probed_indices = []

    class Camera:
        def isOpened(self):
            return True

        def read(self):
            return True, object()

        def release(self):
            pass

    monkeypatch.setattr(
        Path,
        "glob",
        lambda self, pattern: iter(device_nodes),
    )
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda index, backend: (probed_indices.append(index) or Camera()),
    )

    assert LiveCameraWorker.get_available_cameras() == [2, 7]
    assert probed_indices == [2, 7]


def test_linux_camera_discovery_does_not_probe_when_no_device_nodes_exist(monkeypatch):
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(()))
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected camera probe")),
    )

    assert LiveCameraWorker.get_available_cameras() == []


def test_gige_camera_discovery_returns_empty_without_physical_cameras(monkeypatch):
    import core.gige_worker as gige_worker

    class CameraSystem:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_all_cameras(self):
            return []

    monkeypatch.setattr(gige_worker, "VMBPY_AVAILABLE", True)
    monkeypatch.setattr(
        gige_worker,
        "vmbpy",
        SimpleNamespace(VmbSystem=SimpleNamespace(get_instance=CameraSystem)),
        raising=False,
    )

    assert gige_worker.GigECameraWorker.get_available_gige_cameras() == []
