from core.live_worker import LiveCameraWorker


def test_live_worker_can_switch_to_hough_engine():
    worker = LiveCameraWorker(camera_index=0)
    worker.set_engine("Hough Circle")

    assert worker.engine_name == "hough_circle"
    assert worker.engine.__class__.__name__ == "HoughCircleEngine"


def test_live_worker_can_switch_to_blob_kalman_engine():
    worker = LiveCameraWorker(camera_index=0)
    worker.set_engine("Blob + Kalman")

    assert worker.engine_name == "blob_kalman"
    assert worker.engine.__class__.__name__ == "BlobKalmanEngine"


def test_live_worker_can_switch_to_only_camera_engine():
    worker = LiveCameraWorker(camera_index=0)
    worker.set_engine("Only Camera")

    assert worker.engine_name == "only_camera"
    assert worker.engine.__class__.__name__ == "OnlyCameraEngine"
