from unittest.mock import MagicMock, call
import pytest
from core.gige_worker import GigECameraWorker, VMBPY_AVAILABLE


class MockFeature:
    def __init__(self, name: str, value=0, rng=(0, 1000), inc=1):
        self.name = name
        self.value = value
        self.rng = rng
        self.inc = inc

    def get(self):
        return self.value

    def set(self, val):
        self.value = val

    def get_range(self):
        return self.rng

    def get_increment(self):
        return self.inc


@pytest.mark.skipif(not VMBPY_AVAILABLE, reason="VmbPy niet geïnstalleerd")
def test_gige_worker_configures_hardware_roi_and_framerate():
    worker = GigECameraWorker(camera_id="DEV_MOCK")

    features = {
        "OffsetX": MockFeature("OffsetX", value=100, rng=(0, 800), inc=2),
        "OffsetY": MockFeature("OffsetY", value=50, rng=(0, 600), inc=2),
        "Width": MockFeature("Width", value=1440, rng=(32, 1440), inc=16),
        "Height": MockFeature("Height", value=1080, rng=(32, 1080), inc=4),
        "ExposureAuto": MockFeature("ExposureAuto", value="Continuous"),
        "BalanceWhiteAuto": MockFeature("BalanceWhiteAuto", value="Continuous"),
        "ExposureTime": MockFeature("ExposureTime", value=20000.0, rng=(10.0, 100000.0)),
        "AcquisitionFrameRateEnable": MockFeature("AcquisitionFrameRateEnable", value=False),
        "AcquisitionFrameRate": MockFeature("AcquisitionFrameRate", value=60.0, rng=(1.0, 290.0)),
    }

    mock_cam = MagicMock()
    mock_cam.get_feature_by_name.side_effect = lambda name: features[name]
    mock_stream = MagicMock()
    mock_cam.get_streams.return_value = [mock_stream]

    # Voer _setup_camera uit
    worker._setup_camera(mock_cam)

    # 1. Hardware ROI controle
    assert features["Width"].value == 416
    assert features["Height"].value == 260
    assert features["OffsetX"].value == 520
    assert features["OffsetY"].value == 440

    # 2. Auto exposure & auto white balance Off
    assert features["ExposureAuto"].value == "Off"
    assert features["BalanceWhiteAuto"].value == "Off"

    # 3. Sluitertijd verlaagd voor high-speed framerate
    assert features["ExposureTime"].value <= 3000.0

    # 4. Framerate ontgrendeld op maximale range (290 FPS)
    assert features["AcquisitionFrameRate"].value == 290.0
    assert worker.camera_fps == 290.0
