import numpy as np


class BaseEngine:
    """Abstract base class / interface for all Orion Vision processing engines."""

    name = "base"
    parameter_schema = ()

    def __init__(self) -> None:
        self.last_result = None

    def update_settings(self, **kwargs) -> None:
        """Update engine parameters dynamically from the UI or worker."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def process_frame(self, frame: np.ndarray):
        """
        Process the incoming camera frame.
        Must return a tuple: (result_frame, secondary_frame).
        """
        raise NotImplementedError("Elke engine moet de method 'process_frame' implementeren.")

    def standardize_result(self, result_frame, x: float, y: float, vx: float, vy: float, mask: np.ndarray, secondary: np.ndarray, metadata: dict) -> dict:
        """
        Gegarandeerd uniform contract voor de output, zodat de live_worker, 
        het Kalman-filter en de UI altijd dezelfde datastructuur ontvangen.
        """
        return {
            "result_frame": result_frame,
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
            "mask": mask,
            "secondary": secondary,
            "metadata": metadata,
        }