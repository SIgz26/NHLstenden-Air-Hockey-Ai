from pathlib import Path
from types import SimpleNamespace

import cv2

from core.live_worker import LiveCameraWorker


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
