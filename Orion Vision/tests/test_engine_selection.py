from core.live_worker import LiveCameraWorker


def test_live_worker_can_switch_to_hough_engine():
    worker = LiveCameraWorker(camera_index=0)
    worker.set_engine("Hough Circle")

    assert worker.engine_name == "hough_circle"
    assert worker.engine.__class__.__name__ == "HoughCircleEngine"
