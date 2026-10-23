"""Stable-Baselines3 SAC inference for live SimBridge observations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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
) -> bool:
    """Initialize Torch/SB3 once, preferably before Qt loads its DLLs.

    ``main.py`` passes the result of its pre-Qt Torch import. Other callers may
    omit arguments to perform a guarded lazy check when SAC is first requested.
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
        _SB3_IMPORT_ERROR = _TORCH_IMPORT_ERROR
        return False

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

    def __init__(self, sim_bridge: SimBridge, model: Any | None = None) -> None:
        self.sim_bridge = sim_bridge
        self.model = model

    def load_model(self, model_path: str | Path) -> None:
        """Load a Stable-Baselines3 SAC model from its ``.zip`` archive."""
        path = Path(model_path)
        if path.suffix.lower() != ".zip":
            raise ValueError("SAC model path must point to a .zip file")
        if not path.is_file():
            raise FileNotFoundError(f"SAC model not found: {path}")

        check_sac_runtime()
        try:
            self.model = _SAC_CLASS.load(str(path), device="auto")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize/load the SAC archive: {exc}") from exc

    def predict_action(self) -> tuple[float, float]:
        """Return ``(target_vx, target_vy)`` for the newest 8D observation.

        The model is called deterministically for live control. The returned
        action is intentionally validated instead of silently clipping it;
        map/clamp model outputs to actuator units at the controller boundary.
        """
        if self.model is None:
            raise RuntimeError("No SAC model is loaded")

        observation = np.asarray(
            self.sim_bridge.get_observation_vector(), dtype=np.float32
        ).reshape(-1)
        if observation.shape != (8,) or not np.isfinite(observation).all():
            raise ValueError("SimBridge must provide a finite 8D observation vector")

        prediction = self.model.predict(observation, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (2,) or not np.isfinite(action_array).all():
            raise ValueError("SAC model must return a finite 2D action")
        return float(action_array[0]), float(action_array[1])
