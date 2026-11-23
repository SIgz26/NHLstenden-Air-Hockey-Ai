"""Stable-Baselines3 SAC inference for live SimBridge observations."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    np: Any = import_module("numpy")
except Exception as exc:
    np = None
    _NUMPY_IMPORT_ERROR: Exception | None = exc
else:
    _NUMPY_IMPORT_ERROR = None

if TYPE_CHECKING:
    from core.sim_bridge import SimBridge

_torch: Any | None = None
_SAC_CLASS: Any | None = None
_TORCH_IMPORT_ERROR: Exception | None = None
_SB3_IMPORT_ERROR: Exception | None = None
_SAC_RUNTIME_CHECKED = False


def preload_sac_runtime(
    torch_module: Any | None = None,
    torch_error: Exception | None = None,
    *,
    torch_import_attempted: bool = False,
    sac_class: Any | None = None,
    sac_import_error: Exception | None = None,
    sac_import_attempted: bool = False,
) -> bool:
    """Initialize Torch/SB3 once, preferably before Qt loads its DLLs.

    ``main.py`` passes its pre-Qt Torch/SB3 imports. Other callers may omit
    arguments to perform a guarded lazy check when SAC is first requested.
    Failures are cached and returned as ``False`` rather than retried later in
    a potentially conflicting Windows DLL state.
    """
    global _torch, _SAC_CLASS, _TORCH_IMPORT_ERROR, _SB3_IMPORT_ERROR
    global _SAC_RUNTIME_CHECKED

    if _SAC_RUNTIME_CHECKED:
        return _torch is not None and _SAC_CLASS is not None

    _SAC_RUNTIME_CHECKED = True
    if torch_import_attempted:
        _torch = torch_module
        _TORCH_IMPORT_ERROR = torch_error
    else:
        try:
            import torch as loaded_torch
        except Exception as exc:
            _torch = None
            _TORCH_IMPORT_ERROR = exc
        else:
            _torch = loaded_torch

    if _torch is None:
        _SAC_CLASS = None
        _SB3_IMPORT_ERROR = (
            sac_import_error if sac_import_attempted else _TORCH_IMPORT_ERROR
        )
        return False

    if sac_import_attempted:
        _SAC_CLASS = sac_class
        _SB3_IMPORT_ERROR = sac_import_error
        return _SAC_CLASS is not None

    try:
        from stable_baselines3 import SAC as loaded_sac
    except Exception as exc:
        _SAC_CLASS = None
        _SB3_IMPORT_ERROR = exc
        return False

    _SAC_CLASS = loaded_sac
    _SB3_IMPORT_ERROR = None
    return True


def check_sac_runtime() -> None:
    """Raise a recoverable error if Torch or Stable-Baselines3 failed to load."""
    preload_sac_runtime()
    if _torch is None:
        raise RuntimeError(
            f"PyTorch could not initialize; SAC inference is unavailable: {_TORCH_IMPORT_ERROR}"
        ) from _TORCH_IMPORT_ERROR
    if _SAC_CLASS is None:
        raise RuntimeError(
            f"Stable-Baselines3 could not initialize; SAC inference is unavailable: {_SB3_IMPORT_ERROR}"
        ) from _SB3_IMPORT_ERROR


def get_torch_runtime() -> Any:
    """Return the initialized Torch module without reloading its DLLs."""
    check_sac_runtime()
    return _torch


class LiveSACController:
    """Load a trained SAC policy and map the live observation to two actions."""

    FIELD_LENGTH_M = 1.95
    FIELD_WIDTH_M = 0.98
    MAX_PUCK_SPEED_MPS = 5.0
    MAX_RELATIVE_SPEED_MPS = 7.5
    DEFAULT_OPPONENT_LENGTH_NORM = 0.72

    def __init__(self, sim_bridge: SimBridge, model: Any | None = None) -> None:
        self.sim_bridge = sim_bridge
        self.model = model
        self.observation_adapter_status = ""

    def load_model(self, model_path: str | Path) -> None:
        """Load a Stable-Baselines3 SAC model from its ``.zip`` archive."""
        path = Path(model_path)
        if path.suffix.lower() != ".zip":
            raise ValueError("SAC model path must point to a .zip file")
        if not path.is_file():
            raise FileNotFoundError(f"SAC model not found: {path}")

        check_sac_runtime()
        try:
            model = _SAC_CLASS.load(
                str(path),
                device="cpu",
                custom_objects={
                    "learning_rate": 3e-4,
                    "lr_schedule": lambda _progress_remaining: 3e-4,
                },
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize/load the SAC archive: {exc}") from exc

        observation_shape = model.observation_space.shape
        if observation_shape not in ((8,), (12,)):
            raise ValueError(
                "Unsupported SAC observation shape "
                f"{observation_shape}; Live View supports only (8,) and (12,)"
            )
        if model.action_space.shape != (2,):
            raise ValueError(
                f"SAC model expects action shape {model.action_space.shape}; Live View requires (2,)"
            )
        self.model = model
        if observation_shape == (12,):
            self.observation_adapter_status = (
                "12D SAC model; 8->12 adapter active (training axes/side aligned, "
                "field and SI-velocity scales applied, opponent fallback enabled)"
            )
        else:
            self.observation_adapter_status = (
                "8D SAC model; native observations"
                if not self.sim_bridge.physical_table_calibrated
                else "8D SAC model; legacy calibrated-axis mapping"
            )

    def _model_observation(self, observation: Any) -> Any:
        """Map SimBridge's 8D width/length vector to the training 12D state."""
        if np is None:
            raise RuntimeError(
                f"NumPy could not initialize; SAC inference is unavailable: {_NUMPY_IMPORT_ERROR}"
            ) from _NUMPY_IMPORT_ERROR

        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape != (8,) or not np.isfinite(vector).all():
            raise ValueError("SimBridge must provide a finite 8D observation vector")
        if self.model.observation_space.shape == (8,):
            if self.sim_bridge.physical_table_calibrated:
                return vector[[1, 0, 3, 2, 5, 4, 7, 6]]
            return vector
        if self.model.observation_space.shape != (12,):
            raise ValueError(
                "Unsupported SAC observation shape "
                f"{self.model.observation_space.shape}; Live View supports only (8,) and (12,)"
            )
        if not self.sim_bridge.physical_table_calibrated:
            raise RuntimeError(
                "The 8->12 SAC observation adapter requires calibrated physical table bounds"
            )

        bridge = self.sim_bridge
        # Bridge index 1 already points toward the training opponent when
        # invert_y is enabled; reverse it only for non-inverted image axes.
        length_sign = 1.0 if bridge.invert_y else -1.0

        half_length_m = bridge.PHYSICAL_TABLE_LENGTH_MM / 2000.0
        half_width_m = bridge.PHYSICAL_TABLE_WIDTH_MM / 2000.0
        training_half_length_m = self.FIELD_LENGTH_M / 2.0
        training_half_width_m = self.FIELD_WIDTH_M / 2.0
        if bridge.sim_dt <= 0.0:
            raise ValueError("SimBridge sim_dt must be greater than zero")

        puck_length = (
            length_sign
            * float(vector[1])
            * half_length_m
            / training_half_length_m
        )
        puck_width = (
            float(vector[0]) * half_width_m / training_half_width_m
        )
        puck_length_speed = (
            length_sign * float(vector[3]) * half_length_m / bridge.sim_dt
        )
        puck_width_speed = float(vector[2]) * half_width_m / bridge.sim_dt

        if bridge.has_opponent_position:
            opponent_length = (
                length_sign
                * float(vector[5])
                * half_length_m
                / training_half_length_m
            )
            opponent_width = (
                float(vector[4]) * half_width_m / training_half_width_m
            )
        else:
            opponent_length = self.DEFAULT_OPPONENT_LENGTH_NORM
            opponent_width = 0.0

        robot_position_mm = bridge.robot_position_mm()
        if robot_position_mm is None:
            robot_length = (
                length_sign
                * float(vector[7])
                * half_length_m
                / training_half_length_m
            )
            robot_width = (
                float(vector[6]) * half_width_m / training_half_width_m
            )
        else:
            robot_length = -robot_position_mm[1] / (
                1000.0 * training_half_length_m
            )
            robot_width = robot_position_mm[0] / (
                1000.0 * training_half_width_m
            )

        robot_velocity = bridge.plc_robot_velocity_mm_s
        if robot_velocity is None:
            robot_length_speed = 0.0
            robot_width_speed = 0.0
        else:
            robot_width_speed = robot_velocity[0] / 1000.0
            robot_length_speed = -robot_velocity[1] / 1000.0

        adapted = np.asarray(
            [
                puck_length,
                puck_width,
                puck_length_speed / self.MAX_PUCK_SPEED_MPS,
                puck_width_speed / self.MAX_PUCK_SPEED_MPS,
                opponent_length,
                opponent_width,
                robot_length,
                robot_width,
                (puck_length - robot_length) * 0.5,
                (puck_width - robot_width) * 0.5,
                (puck_length_speed - robot_length_speed)
                / self.MAX_RELATIVE_SPEED_MPS,
                (puck_width_speed - robot_width_speed)
                / self.MAX_RELATIVE_SPEED_MPS,
            ],
            dtype=np.float32,
        )
        return np.clip(adapted, -1.0, 1.0)

    def predict_action(self) -> tuple[float, float]:
        """Return ``(target_vx, target_vy)`` for the newest 8D observation.

        The model is called deterministically for live control. The returned
        action is intentionally validated instead of silently clipping it;
        map/clamp model outputs to actuator units at the controller boundary.
        """
        if self.model is None:
            raise RuntimeError("No SAC model is loaded")
        if np is None:
            raise RuntimeError(
                f"NumPy could not initialize; SAC inference is unavailable: {_NUMPY_IMPORT_ERROR}"
            ) from _NUMPY_IMPORT_ERROR

        observation = self._model_observation(
            self.sim_bridge.get_observation_vector()
        )

        with get_torch_runtime().no_grad():
            prediction = self.model.predict(observation, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (2,) or not np.isfinite(action_array).all():
            raise ValueError("SAC model must return a finite 2D action")
        return float(action_array[0]), float(action_array[1])
